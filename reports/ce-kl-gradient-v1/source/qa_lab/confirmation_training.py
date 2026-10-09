"""Frozen, CPU-only gold/full-KD/teacher-agreement KD training.

Only TRAIN tokens enter this module. Cost records are not a performance study.
The parent experiment controls nine-run order and independent prediction/scoring.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time

from .logits_distillation import (_load_cache_manifest, _torch_load, sequence_config,
    supervised_prediction_mask, tokenizer_fingerprint, validate_cache_lineage,
    validate_cached_record)
from .model_identity import (_hash_file, load_verified_model,
                             validate_cache_identities, verify_local_model)

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("gold", "full_kd", "gated_kd")
SEEDS = (20261009, 20261010, 20261011)
FIXED_TRAINING = {
    "rows": 242, "steps": 242, "learning_rate": 0.0001, "lora_r": 8,
    "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
    "temperature": 2.0, "hard_ce_weight": 0.5, "soft_kl_weight": 0.5,
    "max_tokens": 2048, "cpu_threads": 8, "gradient_clip_norm": 1.0,
    "weight_decay": 0.01, "device": "cpu", "dtype": "float32",
    "attention_implementation": "eager", "batch_size": 1,
}


def _sha(path):
    return _hash_file(path)["sha256"]


def _save(path, value):
    """Atomic receipts; non-finite data never becomes non-standard JSON."""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _git(*arguments):
    return subprocess.check_output(["git", "-C", str(ROOT), *arguments], stderr=subprocess.PIPE)


def frozen_protocol(path, arm, seed):
    """Require committed protocol, committed Python sources and frozen inputs."""
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Arm or seed is outside the frozen study")
    relative = path.resolve().relative_to(ROOT).as_posix()
    content = path.read_bytes()
    if _git("show", f"HEAD:{relative}") != content:
        raise ValueError("Protocol differs from committed HEAD")
    protocol = json.loads(content)
    if (protocol["study"] != "teacher-gated-confirmation-v1"
            or protocol["status"] != "frozen_before_execution"
            or protocol["arms"] != list(ARMS) or protocol["seeds"] != list(SEEDS)):
        raise ValueError("Unexpected confirmation protocol")
    for key, value in FIXED_TRAINING.items():
        if protocol["training"].get(key) != value:
            raise ValueError(f"Frozen training setting differs: {key}")
    for name, expected in protocol["frozen_inputs"].items():
        candidate = (ROOT / name).resolve()
        if not candidate.is_relative_to(ROOT) or _sha(candidate) != expected:
            raise ValueError("Frozen protocol input changed")
    sources = {}
    for source in sorted((ROOT / "qa_lab").glob("*.py")):
        name = source.relative_to(ROOT).as_posix()
        if _git("show", f"HEAD:{name}") != source.read_bytes():
            raise ValueError("Python source differs from committed HEAD")
        sources[name] = _sha(source)
    return protocol, {"protocol_sha256": hashlib.sha256(content).hexdigest(),
                      "protocol_commit": _git("rev-parse", "HEAD").decode().strip(),
                      "source_sha256": sources}


def confirmation_losses(student_logits, labels, teacher_log_probs, arm):
    """Return loss tensors/counts on shifted, unmasked answer positions.

    Shapes: logits [1,L,V], labels [1,L], teacher [N,V]. All arms reduce
    across the same N positions; gated KL never divides by the retained count.
    Float16 teacher storage is converted to float32 without renormalization.
    """
    import torch
    import torch.nn.functional as functional

    if arm not in ARMS:
        raise ValueError("Unknown confirmation arm")
    if (student_logits.ndim != 3 or labels.ndim != 2 or labels.shape[0] != 1
            or student_logits.shape[:2] != labels.shape
            or student_logits.dtype != torch.float32 or labels.dtype != torch.long):
        raise ValueError("Invalid CPU-float32 logits/labels contract")
    if student_logits.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("This study requires CPU tensors")
    mask = supervised_prediction_mask(labels)
    selected = student_logits[:, :-1, :][mask]
    targets = labels[:, 1:][mask]
    n = int(selected.shape[0])
    if (n == 0 or teacher_log_probs.shape != selected.shape
            or teacher_log_probs.dtype != torch.float16
            or teacher_log_probs.device.type != "cpu"
            or not torch.isfinite(student_logits).all()
            or not torch.isfinite(teacher_log_probs).all()
            or torch.any(targets < 0) or torch.any(targets >= selected.shape[1])):
        raise ValueError("Invalid teacher/student distribution or target")
    teacher = teacher_log_probs.float()
    hard_ce = functional.cross_entropy(selected, targets)
    log_student = functional.log_softmax(selected / 2.0, dim=-1)
    per_token = functional.kl_div(log_student, teacher, reduction="none", log_target=True).sum(dim=-1)
    gate = teacher.argmax(dim=-1).eq(targets)  # First vocabulary index wins ties.
    full_kl = per_token.sum() / n
    optimized_kl = ((per_token * gate).sum() / n if arm == "gated_kd"
                    else full_kl if arm == "full_kd" else full_kl * 0.0)
    total = hard_ce if arm == "gold" else 0.5 * hard_ce + 0.5 * 4.0 * optimized_kl
    if any(not torch.isfinite(value) for value in (hard_ce, full_kl, optimized_kl, total)):
        raise ValueError("Non-finite confirmation loss")
    return {"loss": total, "hard_ce": hard_ce, "full_kl": full_kl,
            "optimized_kl": optimized_kl, "supervised_tokens": n,
            "teacher_top1_correct": int(gate.sum().item()),
            "kl_retained_tokens": int(gate.sum().item()) if arm == "gated_kd" else n if arm == "full_kd" else 0}


def trainable_state_sha256(model):
    """Digest ordered trainable names, shapes, dtypes and initial bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad:
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _budget_check(started, budget, output, receipt):
    elapsed = time.perf_counter() - started
    receipt["max_observed_rss_bytes"] = max(receipt.get("max_observed_rss_bytes", 0), _rss_bytes())
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    receipt["output_bytes_at_last_check"] = size
    if elapsed > budget["training_run_seconds"]:
        raise RuntimeError("Training run wall budget exceeded")
    if receipt["max_observed_rss_bytes"] > budget["max_sampled_rss_bytes"]:
        raise RuntimeError("Training RSS budget exceeded")
    if size > budget["max_output_bytes"]:
        raise RuntimeError("Training output disk budget exceeded")


def _bound_preflight(args, protocol):
    config = json.loads(args.student_config.read_text())
    frozen = protocol["frozen_inputs"]
    prefix = "configs/teacher-gated-confirmation-v1/"
    if _sha(args.student_config) != frozen[prefix + "student.json"]:
        raise ValueError("Student configuration differs from frozen protocol")
    for key in ("device", "dtype", "attention_implementation"):
        if config[key] != protocol["training"][key]:
            raise ValueError("Student runtime differs from frozen protocol")
    artifact_name, source_name = protocol["training"]["artifact"], protocol["training"]["source"]
    for actual, name in ((args.artifact / "manifest.json", artifact_name + "/manifest.json"),
                         (args.artifact / "train.jsonl", artifact_name + "/train.jsonl"),
                         (args.data_dir / "train.jsonl", source_name + "/train.jsonl")):
        if _sha(actual) != frozen[name]:
            raise ValueError("Relocated TRAIN input differs from frozen protocol")
    cache = _load_cache_manifest(args.cache)
    if (cache["status"] != "complete" or cache["temperature"] != 2.0
            or cache["student_prompt_config"] != sequence_config(config)
            or cache["student_model_config"] != config
            or cache["teacher_model_config"] != json.loads((ROOT / (prefix + "teacher.json")).read_text())):
        raise ValueError("Cache configuration differs from frozen protocol")
    if validate_cache_identities(cache) != "bound_cache_receipt":
        raise ValueError("Confirmation requires a newly bound cache")
    rows = validate_cache_lineage(cache, args.artifact, args.data_dir)
    if len(rows) != protocol["training"]["rows"] or cache["artifact_method"] != "gold_sft":
        raise ValueError("Confirmation requires the complete frozen gold TRAIN artifact")
    identity = verify_local_model(config, args.cache_dir)
    if identity.receipt != cache["model_identities"]["student"]:
        raise ValueError("Student bytes differ from bound cache")
    return config, cache, rows, identity


def train(args):
    if not __debug__:
        raise RuntimeError("Optimized Python execution is forbidden")
    # Own the directory before any operation that can fail. A second attempt
    # always fails; partial evidence is retained under its original run path.
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log = []
    receipt = {"schema": 1, "status": "running", "arm": args.arm, "seed": args.seed,
               "started_utc": datetime.now(timezone.utc).isoformat(), "phase": "protocol",
               "steps_completed": 0, "max_observed_rss_bytes": 0,
               "cost_scope": "run_total_seconds includes identity/cache preflight, model load, training and adapter save; training_loop_seconds excludes preflight/load. Cost only, not a speed comparison.",
               "budget_scope": "Checked at phase/step boundaries; ru_maxrss is process high-water RSS, not GPU or current RSS. A single blocking call is not preempted."}
    _save(args.output / "steps.json", log)
    _save(args.output / "training.json", receipt)
    loop_start = load_start = None
    try:
        protocol, binding = frozen_protocol(args.protocol, args.arm, args.seed)
        receipt.update(binding, training_config=protocol["training"], budget=protocol["budget"])
        check = lambda: _budget_check(started, protocol["budget"], args.output, receipt)
        check()
        receipt["phase"] = "identity_cache_preflight"
        config, cache, rows, identity = _bound_preflight(args, protocol)
        receipt.update(student_config=config, model_identities=cache["model_identities"],
            teacher_identity_binding="bound_cache_receipt", legacy_teacher_unbound=False,
            cache_manifest_sha256=_sha(args.cache / "manifest.json"),
            verified_artifact_manifest_sha256=_sha(args.artifact / "manifest.json"),
            verified_train_source_sha256=_sha(args.data_dir / "train.jsonl"))
        check()
        import torch
        from peft import LoraConfig, get_peft_model
        cfg = protocol["training"]
        torch.set_num_threads(cfg["cpu_threads"])
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        receipt["phase"] = "model_load"
        load_start = time.perf_counter()
        tokenizer, model = load_verified_model(config, identity)
        receipt["model_load_seconds"] = time.perf_counter() - load_start
        load_start = None
        check()
        receipt["phase"] = "cached_tensor_preflight"
        if tokenizer_fingerprint(tokenizer) != cache["student_tokenizer_sha256"]:
            raise ValueError("Student tokenizer differs from bound cache")
        vocabulary = model.get_output_embeddings().weight.shape[0]
        for meta in cache["records"]:
            validate_cached_record(_torch_load(args.cache / meta["path"]), meta, rows[meta["id"]],
                                   tokenizer, config, cfg["max_tokens"], vocabulary)
            check()
        model = get_peft_model(model, LoraConfig(revision=config["revision"],
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Loaded parameters differ from CPU float32 contract")
        model.train()
        model.config.use_cache = False
        receipt["initial_trainable_state_sha256"] = trainable_state_sha256(model)
        receipt["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["weight_decay"])
        records = list(cache["records"])
        random.Random(args.seed).shuffle(records)
        order = [r["id"] for r in records]
        receipt["training_id_order"] = order
        receipt["training_id_order_sha256"] = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
        receipt["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}
        receipt["phase"] = "training"
        _save(args.output / "training.json", receipt)
        loop_start = time.perf_counter()
        for step in range(cfg["steps"]):
            check()
            meta = records[step % len(records)]
            # Recheck bytes consumed by this step after preflight. No concurrent
            # mutation is supported, but a changed record cannot silently train.
            if _sha(args.cache / meta["path"]) != meta["sha256"]:
                raise ValueError("Cache record changed after preflight")
            saved = _torch_load(args.cache / meta["path"])
            ids = saved["input_ids"].to(dtype=torch.long, device="cpu").unsqueeze(0)
            labels = saved["labels"].to(dtype=torch.long, device="cpu").unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            values = confirmation_losses(student_logits, labels, saved["teacher_log_probs"], args.arm)
            values["loss"].backward()
            grad_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            grad = float(grad_tensor)
            clip_scale = float(torch.clamp(cfg["gradient_clip_norm"] / (grad_tensor + 1e-6), max=1.0))
            if not math.isfinite(grad):
                raise ValueError("Non-finite gradient norm")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                raise ValueError("Non-finite updated trainable parameter")
            row = {key: float(value.detach()) if hasattr(value, "detach") else value for key, value in values.items()}
            row.update(step=step + 1, id=meta["id"], input_tokens=int(ids.shape[1]),
                       grad_norm_before_clip=grad, gradient_clip_applied=clip_scale < 1.0,
                       clip_scale=clip_scale)
            log.append(row)
            receipt["steps_completed"] = len(log)
            _save(args.output / "steps.json", log)
            check()
            if (step + 1) % 50 == 0 or step + 1 == cfg["steps"]:
                print(json.dumps({"arm": args.arm, "seed": args.seed, "steps_completed": step + 1}), flush=True)
        receipt["training_loop_seconds"] = time.perf_counter() - loop_start
        loop_start = None
        receipt["phase"] = "adapter_save"
        model.save_pretrained(args.output, safe_serialization=True)
        receipt["adapter_files"] = {name: _hash_file(args.output / name) for name in ("adapter_config.json", "adapter_model.safetensors")}
        adapter_config = json.loads((args.output / "adapter_config.json").read_text())
        if (adapter_config.get("base_model_name_or_path") != config["model_id"]
                or adapter_config.get("revision") != config["revision"]):
            raise ValueError("Adapter base identity is not pinned and portable")
        check()
        receipt.update(status="complete", phase="complete")
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__,
                       error="Confirmation training stopped; inspect the recorded phase and completed steps")
        raise
    finally:
        now = time.perf_counter()
        if loop_start is not None:
            receipt["training_loop_seconds"] = now - loop_start
        if load_start is not None:
            receipt["model_load_seconds"] = now - load_start
        receipt.update(run_total_seconds=now - started, finished_utc=datetime.now(timezone.utc).isoformat(),
                       total_input_tokens=sum(r["input_tokens"] for r in log),
                       total_supervised_tokens=sum(r["supervised_tokens"] for r in log),
                       total_teacher_top1_correct=sum(r["teacher_top1_correct"] for r in log),
                       total_kl_retained_tokens=sum(r["kl_retained_tokens"] for r in log))
        _save(args.output / "training.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    for name in ("student-config", "cache", "artifact", "output", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/complexity-v1"))
    parser.add_argument("--cache-dir", type=Path, help="Existing Hugging Face cache; never downloads")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
