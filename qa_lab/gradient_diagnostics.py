"""TRAIN-only CE/KL parameter-gradient geometry; no updates or generation.

Primary: three fixed initial states. Endpoint probes are descriptive secondary
observations, never cross-checkpoint causal comparisons or quality evaluation.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import subprocess
import time

from .confirmation_training import confirmation_losses, trainable_state_sha256
from .logits_distillation import (_load_cache_manifest, _torch_load, sequence_config,
    tokenizer_fingerprint, validate_cache_lineage, validate_cached_record)
from .model_identity import (_hash_file, load_verified_model,
                             validate_cache_identities, verify_local_model)

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("ce", "full_kl", "gated_kl")
VECTORS = (*COMPONENTS, "combined_full", "combined_gated")
PARAMETER_PATTERN = re.compile(r"\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.default\.weight$")


def save(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data)
    os.replace(temporary, path)


def sha(path):
    return _hash_file(path)["sha256"]


def tensor_sha(tensor):
    """SHA256 over original contiguous CPU float32 bytes, no float64 cast."""
    import torch
    if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
        raise ValueError("Gradient digest requires CPU float32")
    view = memoryview(tensor.detach().contiguous().numpy()).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start:start + 1024 * 1024])
    return digest.hexdigest()


def parameter_state(model):
    """Separate frozen-base and trainable-adapter parameter-byte digests."""
    result = {}
    for kind, trainable in (("base", False), ("adapter", True)):
        digest = hashlib.sha256()
        count = 0
        for name, tensor in sorted(model.named_parameters()):
            if tensor.requires_grad != trainable:
                continue
            digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
            view = memoryview(tensor.detach().contiguous().numpy()).cast("B")
            for start in range(0, len(view), 1024 * 1024):
                digest.update(view[start:start + 1024 * 1024])
            count += tensor.numel()
        result[kind] = {"sha256": digest.hexdigest(), "numel": count}
    return result


def parameter_metadata(name, tensor):
    match = PARAMETER_PATTERN.search(name)
    if match is None:
        raise ValueError("Unexpected trainable parameter name")
    return {"name": name, "layer": int(match[1]), "module": match[2],
            "matrix": match[3], "shape": list(tensor.shape), "numel": tensor.numel()}


def gradient_check(actual, reference, atol, rtol):
    """Compare saved FP32 values in FP64; relative term uses reference."""
    import torch
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Invalid gradient comparison")
    a, b = actual.detach().double().reshape(-1), reference.detach().double().reshape(-1)
    delta = a - b
    failed = delta.abs() > (atol + rtol * b.abs())
    reference_norm = float(torch.linalg.vector_norm(b))
    residual = float(torch.linalg.vector_norm(delta))
    bad = torch.nonzero(failed).flatten()
    return {"allclose": not bool(failed.any()), "failed_elements": int(failed.sum()),
            "first_failure_flat": int(bad[0]) if len(bad) else None,
            "max_abs_error": float(delta.abs().max()), "l2_error": residual,
            "relative_l2_error": residual / reference_norm if reference_norm else None,
            "reference_norm": reference_norm, "reconstructed_norm": float(torch.linalg.vector_norm(a)),
            "atol": atol, "rtol": rtol,
            "comparison": "abs(actual-reference)<=atol+rtol*abs(reference), evaluated in float64"}


def geometry(gram):
    """Descriptive geometry from unrounded FP64 block-summed Gram entries."""
    if (len(gram) != 3 or any(len(row) != 3 for row in gram)
            or any(not math.isfinite(x) for row in gram for x in row)
            or any(gram[i][i] < 0 for i in range(3))):
        raise ValueError("Invalid Gram matrix")
    norms = {key: math.sqrt(gram[i][i]) for i, key in enumerate(COMPONENTS)}
    pairs = {}
    for i, key in enumerate(COMPONENTS[1:], 1):
        zero = [name for name in ("ce", key) if norms[name] == 0]
        pairs[key] = {"dot": gram[0][i],
                      "cosine": gram[0][i] / (norms["ce"] * norms[key]) if not zero else None,
                      "cosine_undefined_reason": "zero_gradient:" + ",".join(zero) if zero else None,
                      "norm_ratio_to_ce": norms[key] / norms["ce"] if norms["ce"] else None,
                      "norm_ratio_undefined_reason": "zero_ce_gradient" if not norms["ce"] else None,
                      "norm_larger_than_ce": gram[i][i] > gram[0][0],
                      "negative_dot": gram[0][i] < 0}
    return {"gram": gram, "norms": norms, "pairs": pairs}


def sum_grams(parameters):
    return [[math.fsum(p["gram"][i][j] for p in parameters) for j in range(3)] for i in range(3)]


def grouped_geometry(parameters):
    result = {"all": geometry(sum_grams(parameters))}
    for layer in sorted({p["layer"] for p in parameters}):
        result[f"layer:{layer}"] = geometry(sum_grams([p for p in parameters if p["layer"] == layer]))
    for module in ("q_proj", "v_proj"):
        result[f"module:{module}"] = geometry(sum_grams([p for p in parameters if p["module"] == module]))
    for matrix in ("A", "B"):
        result[f"matrix:{matrix}"] = geometry(sum_grams([p for p in parameters if p["matrix"] == matrix]))
    return result


def weighted_objectives(selected_logits, targets, teacher_log_probs):
    """Preserve old cached FP16 target mass; all KL denominators are N."""
    import torch
    import torch.nn.functional as functional
    if (selected_logits.ndim != 2 or targets.ndim != 1 or len(targets) == 0
            or selected_logits.shape != teacher_log_probs.shape or len(targets) != len(selected_logits)
            or selected_logits.dtype != torch.float32 or teacher_log_probs.dtype != torch.float16
            or targets.dtype != torch.long or selected_logits.device.type != "cpu"
            or teacher_log_probs.device.type != "cpu" or targets.device.type != "cpu"
            or not torch.isfinite(selected_logits).all() or not torch.isfinite(teacher_log_probs).all()
            or bool((targets < 0).any()) or bool((targets >= selected_logits.shape[1]).any())):
        raise ValueError("Invalid diagnostic logits/target contract")
    teacher = teacher_log_probs.float()
    log_student = functional.log_softmax(selected_logits / 2.0, dim=-1)
    token_kl = functional.kl_div(log_student, teacher, reduction="none", log_target=True).sum(dim=-1)
    gate = teacher.argmax(dim=-1).eq(targets)
    ce = .5 * functional.cross_entropy(selected_logits, targets)
    full = 2.0 * token_kl.sum() / len(targets)
    gated = 2.0 * (token_kl * gate).sum() / len(targets)
    losses = {"ce": ce, "full_kl": full, "gated_kl": gated,
              "combined_full": ce + full, "combined_gated": ce + gated}
    if any(not torch.isfinite(value) for value in losses.values()):
        raise ValueError("Nonfinite diagnostic objective")
    return losses, gate


def analytic_logit_gradients(selected_logits, targets, teacher_log_probs):
    """Independent float64 formula includes teacher mass m, not presumed 1."""
    import torch
    z = selected_logits.detach().double()
    teacher = teacher_log_probs.detach().double().exp()
    n = len(targets)
    one_hot = torch.zeros_like(z)
    one_hot.scatter_(1, targets[:, None], 1.)
    ce = .5 / n * (torch.softmax(z, dim=-1) - one_hot)
    # d(2 KL)/dz = (2/T)/N * (m softmax(z/T) - p), T=2.
    full = (teacher.sum(dim=-1, keepdim=True) * torch.softmax(z / 2., dim=-1) - teacher) / n
    gate = teacher_log_probs.argmax(dim=-1).eq(targets).double()[:, None]
    gated = full * gate
    return {"ce": ce, "full_kl": full, "gated_kl": gated,
            "combined_full": ce + full, "combined_gated": ce + gated}


def inspect_probe(model, ids, labels, teacher, tolerance, public_block=False, observer=None):
    """Five autograd VJPs at one theta; never populate .grad or step optimizer."""
    import torch
    parameters = sorted((name, p) for name, p in model.named_parameters() if p.requires_grad)
    metadata = [parameter_metadata(name, p) for name, p in parameters]
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    mask = labels[:, 1:] != -100
    selected = logits[:, :-1, :][mask]
    targets = labels[:, 1:][mask]
    if observer is not None:
        observer("inputs", {"selected_logits": selected, "targets": targets,
                 "teacher_log_probs": teacher, "input_ids": ids, "labels": labels,
                 "supervised_positions": torch.nonzero(mask)}, metadata)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite model logits")
    losses, gate = weighted_objectives(selected, targets, teacher)
    legacy_full = confirmation_losses(logits, labels, teacher, "full_kd")
    legacy_gate = confirmation_losses(logits, labels, teacher, "gated_kd")
    loss_checks = {"full": gradient_check(losses["combined_full"].reshape(1), legacy_full["loss"].reshape(1), **tolerance),
                   "gated": gradient_check(losses["combined_gated"].reshape(1), legacy_gate["loss"].reshape(1), **tolerance)}
    analytic = analytic_logit_gradients(selected, targets, teacher)
    gradients = {}
    logit_checks = {}
    for index, key in enumerate(VECTORS):
        values = torch.autograd.grad(losses[key], (selected, *[p for _, p in parameters]),
                                     retain_graph=index < len(VECTORS) - 1, allow_unused=False)
        if observer is not None:
            observer(key, {"logit_actual": values[0], "logit_reference": analytic[key],
                     **{f"parameter_{i:03d}": value for i, value in enumerate(values[1:])}}, metadata)
        logit_checks[key] = gradient_check(values[0], analytic[key], **tolerance)
        gradients[key] = [value.detach() for value in values[1:]]
    rows, raw, raw_parameter = [], {}, None
    for index, info in enumerate(metadata):
        vectors = {key: gradients[key][index] for key in VECTORS}
        if any(not torch.isfinite(v).all() for v in vectors.values()):
            raise ValueError("Nonfinite parameter gradient")
        flat = [vectors[key].double().reshape(-1) for key in COMPONENTS]
        gram = [[float(torch.dot(a, b)) for b in flat] for a in flat]
        reconstruction = {"full": gradient_check(vectors["ce"] + vectors["full_kl"], vectors["combined_full"], **tolerance),
                          "gated": gradient_check(vectors["ce"] + vectors["gated_kl"], vectors["combined_gated"], **tolerance)}
        rows.append({**info, "gram": gram,
                     "gradient_sha256": {key: tensor_sha(v) for key, v in vectors.items()},
                     "zero_gradient": {key: bool(torch.count_nonzero(v) == 0) for key, v in vectors.items()},
                     "reconstruction": reconstruction})
        if public_block and info["layer"] == 0 and info["module"] == "q_proj" and info["matrix"] == "B":
            raw = {key: value.cpu().contiguous().numpy() for key, value in vectors.items()}
            raw_parameter = info["name"]
    if public_block and not raw:
        raise ValueError("Fixed public gradient block is unavailable")
    if any(p.grad is not None for _, p in parameters):
        raise ValueError("Diagnostic unexpectedly accumulated parameter .grad")
    passed = (all(v["allclose"] for v in loss_checks.values())
              and all(v["allclose"] for v in logit_checks.values())
              and all(v["allclose"] for p in rows for v in p["reconstruction"].values()))
    return {"losses": {key: float(value.detach()) for key, value in losses.items()},
            "input_tokens": int(ids.shape[1]), "supervised_tokens": len(targets),
            "teacher_top1_correct": int(gate.sum()), "parameters": rows,
            "groups": grouped_geometry(rows), "logit_gradient_checks": logit_checks,
            "legacy_loss_checks": loss_checks, "correctness_passed": passed,
            "public_parameter": raw_parameter}, raw


def summarize(probes):
    def describe(rows):
        return {"probes": len(rows), "distinct_train_ids": len({p["train_id"] for p in rows}),
                "full_norm_larger_count": sum(p["groups"]["all"]["pairs"]["full_kl"]["norm_larger_than_ce"] for p in rows),
                "full_negative_dot_count": sum(p["groups"]["all"]["pairs"]["full_kl"]["negative_dot"] for p in rows),
                "gated_negative_dot_count": sum(p["groups"]["all"]["pairs"]["gated_kl"]["negative_dot"] for p in rows),
                "full_cosine_undefined_count": sum(p["groups"]["all"]["pairs"]["full_kl"]["cosine"] is None for p in rows),
                "gated_cosine_undefined_count": sum(p["groups"]["all"]["pairs"]["gated_kl"]["cosine"] is None for p in rows)}
    initial = [p for p in probes if p["state"] == "initial"]
    primary = describe(initial)
    complete = len(initial) == 48 and len(probes) == 192 and all(p["correctness_passed"] for p in probes)
    primary["hypotheses"] = {
        "initial_full_kl_larger_than_ce_majority": primary["full_norm_larger_count"] > 24 if complete else None,
        "initial_gate_eliminates_negative_dot": (primary["gated_negative_dot_count"] == 0 and primary["gated_cosine_undefined_count"] == 0) if complete else None}
    return {"diagnostic_complete": complete, "primary_initial": primary,
            "by_state": {state: describe([p for p in probes if p["state"] == state]) for state in ("initial", "gold", "full_kd", "gated_kd")},
            "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
            "quality_or_performance_conclusion": False}


def _read(path):
    return json.loads(path.read_text())


def frozen_binding(args):
    protocol = _read(args.protocol)
    if (protocol["study"] != "ce-kl-gradient-v1" or protocol["status"] != "frozen_before_execution"
            or protocol["seeds"] != [20261009, 20261010, 20261011]
            or protocol["states"] != ["initial", "gold", "full_kd", "gated_kd"]
            or len(protocol["selected_ids"]) != 16 or len(set(protocol["selected_ids"])) != 16
            or protocol["runtime"] != {"device": "cpu", "dtype": "float32", "attention_implementation": "eager", "cpu_threads": 8}
            or protocol["weights"] != {"ce": .5, "kl": 2., "temperature": 2., "gated_denominator": "all supervised positions; do not renormalize by retained count"}):
        raise ValueError("Unexpected frozen diagnostic protocol")
    sources = sorted((ROOT / "qa_lab").glob("*.py")) + [args.protocol.resolve()]
    source_sha = {}
    (args.output / "source").mkdir()
    for path in sources:
        name = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT, stderr=subprocess.PIPE)
        if committed != path.read_bytes():
            raise ValueError("Protocol/source is not identical to committed HEAD")
        source_sha[name] = sha(path)
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(committed)
    return protocol, {"protocol_sha256": sha(args.protocol),
        "protocol_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_sha}


def preflight(args, protocol):
    inputs = {"student_config": args.student_config, "cache_manifest": args.cache / "manifest.json",
              "artifact_manifest": args.artifact / "manifest.json", "artifact_train": args.artifact / "train.jsonl",
              "source_train": args.data_dir / "train.jsonl"}
    for key, path in inputs.items():
        if sha(path) != protocol["hashes"][key]:
            raise ValueError("Frozen diagnostic input hash differs")
    config = _read(args.student_config)
    if any(config[key] != value for key, value in protocol["runtime"].items() if key != "cpu_threads"):
        raise ValueError("Runtime configuration differs")
    cache = _load_cache_manifest(args.cache)
    if (cache["status"] != "complete" or cache["temperature"] != 2.
            or cache["student_model_config"] != config
            or cache["student_prompt_config"] != sequence_config(config)
            or validate_cache_identities(cache) != "bound_cache_receipt"):
        raise ValueError("Bound TRAIN cache differs")
    rows = validate_cache_lineage(cache, args.artifact, args.data_dir)
    source_rows = [json.loads(line) for line in (args.data_dir / "train.jsonl").read_text().splitlines() if line]
    selected = []
    for flag in (False, True):
        ids = sorted((row["id"] for row in source_rows if row["is_impossible"] is flag),
                     key=lambda key: hashlib.sha256(("ce-kl-gradient-v1|" + key).encode()).hexdigest())[:8]
        selected.extend(ids)
    if selected != protocol["selected_ids"] or not set(selected) <= rows.keys():
        raise ValueError("TRAIN selection differs from frozen hash order")
    source_by_id = {row["id"]: row for row in source_rows}
    endpoints = {}
    input_dir = args.output / "inputs"
    input_dir.mkdir()
    for key, path in inputs.items():
        if key != "cache_manifest":
            (input_dir / (key + path.suffix)).write_bytes(path.read_bytes())
    safe_cache = dict(cache)
    safe_cache.pop("artifact_path", None)
    save(input_dir / "cache.receipt.json", {"original_manifest_sha256": sha(inputs["cache_manifest"]),
        "redactions": ["artifact_path"], "redacted_manifest": safe_cache})
    for seed in protocol["seeds"]:
        initial = set()
        for state in protocol["states"][1:]:
            key = f"{state}:{seed}"
            folder = args.training_root / f"{state}-{seed}"
            file = folder / "training.json"
            if sha(file) != protocol["hashes"]["endpoint_training"][key]:
                raise ValueError("Endpoint training receipt changed")
            receipt = _read(file)
            if (receipt["status"] != "complete" or receipt["arm"] != state or receipt["seed"] != seed
                    or receipt["steps_completed"] != 242
                    or receipt["cache_manifest_sha256"] != protocol["hashes"]["cache_manifest"]
                    or receipt["model_identities"] != cache["model_identities"]):
                raise ValueError("Endpoint identity/status differs")
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                if _hash_file(folder / name) != receipt["adapter_files"][name]:
                    raise ValueError("Endpoint adapter file changed")
            adapter = _read(folder / "adapter_config.json")
            if adapter["base_model_name_or_path"] != config["model_id"] or adapter["revision"] != config["revision"]:
                raise ValueError("Endpoint base identity differs")
            initial.add(receipt["initial_trainable_state_sha256"])
            endpoints[key] = receipt
            save(input_dir / f"endpoint-{state}-{seed}.json", receipt)
        if len(initial) != 1:
            raise ValueError("Existing initial-state digests differ within seed")
    snapshot = verify_local_model(config, args.cache_dir)
    if snapshot.receipt != cache["model_identities"]["student"]:
        raise ValueError("Loaded student identity differs from cache")
    return config, cache, rows, source_by_id, endpoints, snapshot


def token_classes(labels, tokenizer):
    ids = labels[labels != -100].tolist()
    eos = tokenizer.eos_token_id
    special = set(tokenizer.all_special_ids)
    return {"supervised_tokens": len(ids), "tokenizer_eos_token_id": eos,
            "eos_count": sum(value == eos for value in ids),
            "other_special_count": sum(value in special and value != eos for value in ids),
            "other_count": sum(value not in special and value != eos for value in ids),
            "scope": "Token-ID classes only. Other tokens are not assigned to answer text versus chat-template suffix."}


def _budget(started, protocol, output, run):
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if platform.system() == "Darwin" else value * 1024)
    run["peak_process_rss_bytes"] = max(run.get("peak_process_rss_bytes", 0), rss)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    run["output_bytes_last_check"] = size
    if (time.perf_counter() - started > protocol["budget"]["wall_seconds"]
            or run["peak_process_rss_bytes"] > protocol["budget"]["max_rss_bytes"]
            or size > protocol["budget"]["max_output_bytes"]):
        raise RuntimeError("Frozen diagnostic resource budget exceeded")


def _artifact_manifest(output):
    return {path.relative_to(output).as_posix(): sha(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != output / "run.json"}


def run_diagnostic(args):
    if not __debug__:
        raise RuntimeError("Optimized Python execution is forbidden")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run = {"schema": 1, "status": "running", "study": "ce-kl-gradient-v1",
           "started_utc": datetime.now(timezone.utc).isoformat(), "phase": "protocol",
           "probes_completed": 0, "states_completed": 0,
           "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
           "budget_scope": "Wall/RSS/disk checked at phases and probe boundaries; ru_maxrss is process high-water RSS. No speed conclusion or hard interruption of a blocking call."}
    probes, state_files = [], []
    save(args.output / "run.json", run)
    state_receipt = state_file = model = None
    active_probe_file = active_receipt = None
    try:
        protocol, binding = frozen_binding(args)
        run.update(binding, protocol=protocol)
        check = lambda: _budget(started, protocol, args.output, run)
        check()
        run["phase"] = "identity_cache_preflight"
        config, cache, rows, source_rows, endpoints, snapshot = preflight(args, protocol)
        run["model_identities"] = cache["model_identities"]
        run["input_sha256"] = protocol["hashes"]
        check()
        import torch
        import numpy as np
        from peft import LoraConfig, PeftModel, get_peft_model
        torch.set_num_threads(protocol["runtime"]["cpu_threads"])
        run["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}
        (args.output / "probes").mkdir()
        (args.output / "states").mkdir()
        (args.output / "arrays").mkdir()
        metadata = {row["id"]: row for row in cache["records"]}
        tolerance = {"atol": protocol["correctness"]["gradient_atol"], "rtol": protocol["correctness"]["gradient_rtol"]}
        for seed in protocol["seeds"]:
            for state in protocol["states"]:
                state_key = f"{state}:{seed}"
                run["phase"] = "load_state"
                run["active_state"] = state_key
                state_file = args.output / "states" / f"{state}-{seed}.json"
                state_receipt = {"status": "running", "state": state, "seed": seed, "probes_completed": 0,
                                 "started_utc": datetime.now(timezone.utc).isoformat()}
                save(state_file, state_receipt)
                save(args.output / "run.json", run)
                random.seed(seed)
                torch.manual_seed(seed)
                tokenizer, base = load_verified_model(config, snapshot)
                if tokenizer_fingerprint(tokenizer) != cache["student_tokenizer_sha256"]:
                    raise ValueError("Diagnostic tokenizer differs from cache")
                for key in protocol["selected_ids"]:
                    meta = metadata[key]
                    validate_cached_record(_torch_load(args.cache / meta["path"]), meta, rows[key], tokenizer,
                                           config, protocol["training"]["max_tokens"], base.get_output_embeddings().weight.shape[0])
                cfg = protocol["training"]
                if state == "initial":
                    model = get_peft_model(base, LoraConfig(revision=config["revision"], r=cfg["lora_r"],
                        lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                        target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
                    expected = endpoints[f"gold:{seed}"]["initial_trainable_state_sha256"]
                    state_receipt["initial_trainable_state_sha256"] = trainable_state_sha256(model)
                    if state_receipt["initial_trainable_state_sha256"] != expected:
                        raise ValueError("Recreated initial parameters differ from frozen training receipt")
                else:
                    folder = args.training_root / f"{state}-{seed}"
                    for name in ("adapter_config.json", "adapter_model.safetensors"):
                        if _hash_file(folder / name) != endpoints[state_key]["adapter_files"][name]:
                            raise ValueError("Adapter changed after preflight")
                    model = PeftModel.from_pretrained(base, str(folder), is_trainable=True, local_files_only=True)
                    state_receipt["adapter_files"] = endpoints[state_key]["adapter_files"]
                model.eval()
                model.config.use_cache = False
                if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
                    raise ValueError("Parameter device/dtype contract differs")
                params = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
                if len(params) != 96 or sum(value.numel() for _, value in params) != 540672:
                    raise ValueError("Trainable parameter coverage differs")
                for name, value in params:
                    parameter_metadata(name, value)
                state_receipt["parameters_before"] = parameter_state(model)
                state_receipt["trainable_parameters"] = 540672
                state_receipt["parameter_blocks"] = 96
                save(state_file, state_receipt)
                check()
                for selected_index, key in enumerate(protocol["selected_ids"]):
                    run["phase"] = "gradient_probe"
                    run["active_probe"] = len(probes)
                    probe_file = args.output / "probes" / f"{len(probes):03d}.json"
                    receipt = {"schema": 1, "status": "running", "index": len(probes), "state": state,
                               "seed": seed, "train_id": key, "is_impossible": source_rows[key]["is_impossible"],
                               "role": "primary_initial" if state == "initial" else "secondary_endpoint"}
                    active_probe_file, active_receipt = probe_file, receipt
                    receipt["phase"] = "cache_input_check"
                    save(probe_file, receipt)
                    meta = metadata[key]
                    if sha(args.cache / meta["path"]) != meta["sha256"]:
                        raise ValueError("Selected cache record changed")
                    saved = _torch_load(args.cache / meta["path"])
                    ids = saved["input_ids"].long().unsqueeze(0)
                    labels = saved["labels"].long().unsqueeze(0)
                    receipt["cache_record_sha256"] = meta["sha256"]
                    receipt["token_classes"] = token_classes(labels[:, 1:], tokenizer)
                    receipt["phase"] = "autograd"
                    save(probe_file, receipt)
                    values, raw = inspect_probe(model, ids, labels, saved["teacher_log_probs"], tolerance,
                                               public_block=selected_index == 0)
                    receipt.update(values, raw_artifact=None, phase="serialize_gradient_receipt")
                    if raw:
                        array_file = args.output / "arrays" / f"{state}-{seed}.npz"
                        np.savez_compressed(array_file, **raw)
                        receipt["raw_artifact"] = {"path": array_file.relative_to(args.output).as_posix(),
                                                   "sha256": sha(array_file), "parameter": values["public_parameter"],
                                                   "keys": list(VECTORS), "scope": "Layer 0 q_proj lora_B only; other parameter vectors not published"}
                    receipt["status"] = "complete" if values["correctness_passed"] else "failed"
                    save(probe_file, receipt)
                    if not values["correctness_passed"]:
                        raise ValueError("Gradient correctness check failed; probe preserved")
                    receipt["phase"] = "probe_budget_check"
                    check()
                    receipt["phase"] = "complete"
                    save(probe_file, receipt)
                    probes.append(receipt)
                    state_receipt["probes_completed"] += 1
                    run["probes_completed"] = len(probes)
                    active_probe_file = active_receipt = None
                state_receipt["parameters_after"] = parameter_state(model)
                state_receipt["parameters_unchanged"] = state_receipt["parameters_before"] == state_receipt["parameters_after"]
                if not state_receipt["parameters_unchanged"]:
                    raise ValueError("Diagnostic changed model parameters")
                state_receipt.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
                save(state_file, state_receipt)
                state_files.append(state_file.relative_to(args.output).as_posix())
                run["states_completed"] += 1
                print(json.dumps({"state": state_key, "probes_completed": len(probes)}), flush=True)
                state_receipt = state_file = None
                model = None
                del base, params
                gc.collect()
                check()
        run["phase"] = "final_integrity"
        # Source, original manifest, endpoint receipts and adapter bytes must
        # remain unchanged after all probes as well as before the first load.
        for name, expected in run["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError("Source changed during diagnostic")
        final_inputs = {"student_config": args.student_config, "cache_manifest": args.cache / "manifest.json",
                        "artifact_manifest": args.artifact / "manifest.json", "artifact_train": args.artifact / "train.jsonl",
                        "source_train": args.data_dir / "train.jsonl"}
        for key, path in final_inputs.items():
            if sha(path) != protocol["hashes"][key]:
                raise ValueError("Frozen input changed during diagnostic")
        for key in protocol["selected_ids"]:
            meta = metadata[key]
            if sha(args.cache / meta["path"]) != meta["sha256"]:
                raise ValueError("Selected cache record changed after probes")
        for key, expected in protocol["hashes"]["endpoint_training"].items():
            folder = args.training_root / key.replace(":", "-")
            if sha(folder / "training.json") != expected:
                raise ValueError("Endpoint receipt changed during diagnostic")
            for name, value in endpoints[key]["adapter_files"].items():
                if _hash_file(folder / name) != value:
                    raise ValueError("Endpoint adapter changed during diagnostic")
        summary = summarize(probes)
        if not summary["diagnostic_complete"]:
            raise ValueError("Incomplete diagnostic coverage")
        save(args.output / "summary.json", summary)
        check()
        run.update(status="complete", phase="complete", diagnostic_complete=True)
    except BaseException as error:
        run.update(status="failed", error_type=type(error).__name__, diagnostic_complete=False,
                   error="Diagnostic stopped; see preserved phase and partial receipts")
        if active_receipt is not None:
            active_receipt.update(status="failed", error_type=type(error).__name__,
                                  error_code="diagnostic_probe_" + active_receipt["phase"])
            save(active_probe_file, active_receipt)
        if state_receipt is not None:
            if model is not None and "parameters_before" in state_receipt:
                try:
                    state_receipt["parameters_after"] = parameter_state(model)
                    state_receipt["parameters_unchanged"] = state_receipt["parameters_before"] == state_receipt["parameters_after"]
                except Exception as hash_error:
                    state_receipt["parameter_audit_error_type"] = type(hash_error).__name__
            state_receipt.update(status="failed", error_type=type(error).__name__, finished_utc=datetime.now(timezone.utc).isoformat())
            save(state_file, state_receipt)
        raise
    finally:
        recorded_states = list(state_files)
        if state_file is not None and state_file.exists():
            active_name = state_file.relative_to(args.output).as_posix()
            if active_name not in recorded_states:
                recorded_states.append(active_name)
        run.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started,
                   state_files=recorded_states,
                   probe_files=[p.relative_to(args.output).as_posix() for p in sorted((args.output / "probes").glob("*.json"))],
                   artifacts=_artifact_manifest(args.output))
        save(args.output / "run.json", run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "cache", "training-root", "artifact", "data-dir", "student-config", "cache-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run_diagnostic(parser.parse_args())


if __name__ == "__main__":
    main()
