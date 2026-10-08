"""One frozen TRAIN probe with atomic forensic capture, never an update or retry.

Both raw output directories stay below ignored local work/. Derivatives are
separate from logits, teacher cache values and token IDs; none are published.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time

from . import gradient_diagnostics as diagnostic

ROOT = diagnostic.ROOT
VECTORS = diagnostic.VECTORS
STUDY = "ce-kl-gradient-capture-v1"
INPUT_KEYS = ("selected_logits", "targets", "teacher_log_probs", "input_ids", "labels", "supervised_positions")


def _array_metadata(array):
    import numpy as np
    return {"dtype": str(array.dtype), "shape": list(array.shape), "numel": int(array.size),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "finite": bool(np.isfinite(array).all())}


def atomic_arrays(path, tensors):
    """Persist exact dtype/shape/bytes before any finite or numerical gate."""
    import numpy as np
    if path.exists():
        raise FileExistsError("An archived tensor file cannot be overwritten")
    arrays = {key: np.ascontiguousarray(value.detach().cpu().numpy()) for key, value in tensors.items()}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"sha256": diagnostic.sha(path), "bytes": path.stat().st_size,
            "arrays": {key: _array_metadata(value) for key, value in arrays.items()}}


class CaptureObserver:
    """Stage-level write-ahead artifacts; inputs stay separate from the derivative archive."""
    def __init__(self, output, private_output, check_budget=lambda: None):
        self.output, self.private_output, self.check_budget = output, private_output, check_budget
        (output / "arrays").mkdir()
        self.manifest = {"schema": 1, "study": STUDY, "status": "partial", "parameters": [],
                         "objectives": {}, "private_inputs": None,
                         "policy": "All raw arrays local only; derivative and input archives are separate, not a privacy guarantee."}
        diagnostic.save(output / "capture.json", self.manifest)

    def __call__(self, stage, values, parameters):
        if self.manifest["parameters"] and parameters != self.manifest["parameters"]:
            raise ValueError("Parameter metadata changed during capture")
        self.manifest["parameters"] = parameters
        if stage == "inputs":
            if self.manifest["private_inputs"] is not None or set(values) != set(INPUT_KEYS):
                raise ValueError("Unexpected or repeated private input capture")
            # Explicitly exclude even the local path from the derivative receipt.
            self.manifest["private_inputs"] = atomic_arrays(self.private_output / "inputs.npz", values)
            diagnostic.save(self.private_output / "receipt.json", self.manifest["private_inputs"])
        else:
            expected = {"logit_actual", "logit_reference", *(f"parameter_{i:03d}" for i in range(len(parameters)))}
            if stage not in VECTORS or stage in self.manifest["objectives"] or set(values) != expected:
                raise ValueError("Unexpected or repeated objective capture")
            path = self.output / "arrays" / f"{stage}.npz"
            self.manifest["objectives"][stage] = {"path": f"arrays/{stage}.npz", **atomic_arrays(path, values)}
        self.manifest["last_stage"] = stage
        complete = self.manifest["private_inputs"] is not None and set(self.manifest["objectives"]) == set(VECTORS)
        self.manifest["status"] = "complete" if complete else "partial"
        diagnostic.save(self.output / "capture.json", self.manifest)
        self.check_budget()


def frozen_binding(args):
    protocol = diagnostic._read(args.protocol)
    if (protocol["study"] != STUDY or protocol["status"] != "frozen_before_execution"
            or protocol["probe"] != {"state": "initial", "seed": 20261009, "train_id": "56e190bce3433e1400422fcb", "count": 1}
            or protocol["correctness"] != {"gradient_atol": 1e-5, "gradient_rtol": 1e-4}
            or protocol["budget"] != {"wall_seconds": 120, "max_rss_bytes": 16 * 2**30,
                                      "max_output_bytes": 128 * 2**20, "paid_resources": False}
            or protocol["scope"] != {"optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
                                      "quality_or_performance_claims": False, "full_192_probe_study": False}
            or protocol["replay_sources"] != ["scripts/verify_gradient_capture.py", "scripts/replay_private_gradient_capture.py",
                                               "scripts/verify_gradient_diagnostics.py"]):
        raise ValueError("Unexpected frozen capture protocol")
    source_protocol = ROOT / "configs/ce-kl-gradient-v1.json"
    if diagnostic.sha(source_protocol) != protocol["source_protocol_sha256"]:
        raise ValueError("Original diagnostic protocol changed")
    legacy = diagnostic._read(source_protocol)
    if (protocol["runtime"] != legacy["runtime"]
            or legacy["selected_ids"][0] != protocol["probe"]["train_id"]
            or legacy["seeds"][0] != protocol["probe"]["seed"]
            or any(legacy["correctness"][key] != value for key, value in protocol["correctness"].items())):
        raise ValueError("Capture differs from original first probe or numerical gate")
    sources = sorted((ROOT / "qa_lab").glob("*.py")) + [source_protocol,
        ROOT / "configs/ce-kl-gradient-capture-v1.json", args.protocol.resolve()]
    sources = list(dict.fromkeys(sources))
    sources += [ROOT / name for name in protocol["replay_sources"]]
    source_sha = {}
    for path in sources:
        name = path.relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT, stderr=subprocess.PIPE)
        if committed != path.read_bytes():
            raise ValueError("Capture source/protocol is not identical to committed HEAD")
        source_sha[name] = diagnostic.sha(path)
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(committed)
    for name, digest in protocol["preserved_failure_sha256"].items():
        if diagnostic.sha(ROOT / name) != digest:
            raise ValueError("Preserved v1 failure changed")
    return protocol, legacy, {"protocol_sha256": diagnostic.sha(args.protocol),
        "protocol_path": args.protocol.resolve().relative_to(ROOT).as_posix(),
        "protocol_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_committed_utc": subprocess.check_output(["git", "show", "-s", "--format=%cI", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": source_sha}


def _check_budget(started, protocol, args, run):
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(value if platform.system() == "Darwin" else value * 1024)
    run["peak_process_rss_bytes"] = max(run.get("peak_process_rss_bytes", 0), rss)
    size = sum(path.stat().st_size for folder in (args.output, args.private_output)
               for path in folder.rglob("*") if path.is_file())
    run["combined_output_bytes_last_check"] = size
    if (time.perf_counter() - started > protocol["budget"]["wall_seconds"]
            or run["peak_process_rss_bytes"] > protocol["budget"]["max_rss_bytes"]
            or size > protocol["budget"]["max_output_bytes"]):
        raise RuntimeError("Frozen capture resource budget exceeded")


def _final_integrity(args, protocol, legacy, run, cache, endpoints):
    for name, expected in run["source_sha256"].items():
        if diagnostic.sha(ROOT / name) != expected:
            raise ValueError("Capture source changed during execution")
    for name, expected in protocol["preserved_failure_sha256"].items():
        if diagnostic.sha(ROOT / name) != expected:
            raise ValueError("Preserved failure changed during execution")
    inputs = {"student_config": args.student_config, "cache_manifest": args.cache / "manifest.json",
              "artifact_manifest": args.artifact / "manifest.json", "artifact_train": args.artifact / "train.jsonl",
              "source_train": args.data_dir / "train.jsonl"}
    for key, path in inputs.items():
        if diagnostic.sha(path) != legacy["hashes"][key]:
            raise ValueError("Frozen input changed during capture")
    # Recheck every selected cache file and all endpoint files, including on a
    # numerical failure. This verifies the original identity set without using
    # endpoint models or running the other probes.
    records = {row["id"]: row for row in cache["records"]}
    for key in legacy["selected_ids"]:
        meta = records[key]
        if diagnostic.sha(args.cache / meta["path"]) != meta["sha256"]:
            raise ValueError("Selected cache record changed during capture")
    for key, expected in legacy["hashes"]["endpoint_training"].items():
        folder = args.training_root / key.replace(":", "-")
        if diagnostic.sha(folder / "training.json") != expected:
            raise ValueError("Endpoint receipt changed during capture")
        for name, identity in endpoints[key]["adapter_files"].items():
            if diagnostic._hash_file(folder / name) != identity:
                raise ValueError("Endpoint adapter changed during capture")
    current = diagnostic.verify_local_model(diagnostic._read(args.student_config), args.cache_dir)
    if current.receipt != cache["model_identities"]["student"]:
        raise ValueError("Student model files changed during capture")


def run_capture(args):
    if not __debug__:
        raise RuntimeError("Optimized Python execution is forbidden")
    archive, private = args.output.resolve(), args.private_output.resolve()
    work = (ROOT / "work").resolve()
    if (private == work or work not in private.parents or archive == work or work not in archive.parents
            or archive == private or archive in private.parents or private in archive.parents):
        raise ValueError("Both raw outputs must be independent ignored repository work directories")
    if args.output.exists() or args.private_output.exists():
        raise FileExistsError("Capture output directories must both be new")
    args.output.mkdir(parents=True, exist_ok=False)
    args.private_output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    run = {"schema": 1, "study": STUDY, "status": "running", "phase": "protocol",
           "started_utc": datetime.now(timezone.utc).isoformat(), "capture_complete": False,
           "numerical_gate_passed": None, "diagnostic_complete": False,
           "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
           "probe_count": 0, "quality_or_performance_conclusion": False,
           "budget_scope": "Wall/RSS/combined derivative and input archive disk checked at stage boundaries, including finalization; not a hard interruption of blocking calls."}
    state, probe, model, observer = {}, {}, None, None
    protocol = legacy = cache = endpoints = None
    failure = None
    diagnostic.save(args.output / "run.json", run)
    try:
        protocol, legacy, binding = frozen_binding(args)
        run.update(binding, protocol=protocol)
        check = lambda: _check_budget(started, protocol, args, run)
        check()
        run["phase"] = "identity_cache_preflight"
        config, cache, rows, source_rows, endpoints, snapshot = diagnostic.preflight(args, legacy)
        run.update(model_identities=cache["model_identities"], input_sha256=legacy["hashes"])
        check()
        import torch
        from peft import LoraConfig, get_peft_model
        torch.set_num_threads(legacy["runtime"]["cpu_threads"])
        run["packages"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")}
        run["python_version"] = platform.python_version()
        seed, key = protocol["probe"]["seed"], protocol["probe"]["train_id"]
        random.seed(seed)
        torch.manual_seed(seed)
        run["phase"] = "load_initial_state"
        diagnostic.save(args.output / "run.json", run)
        tokenizer, base = diagnostic.load_verified_model(config, snapshot)
        if diagnostic.tokenizer_fingerprint(tokenizer) != cache["student_tokenizer_sha256"]:
            raise ValueError("Tokenizer differs from bound cache")
        meta = next(row for row in cache["records"] if row["id"] == key)
        if diagnostic.sha(args.cache / meta["path"]) != meta["sha256"]:
            raise ValueError("First TRAIN cache file changed")
        saved = diagnostic._torch_load(args.cache / meta["path"])
        diagnostic.validate_cached_record(saved, meta, rows[key], tokenizer, config,
            legacy["training"]["max_tokens"], base.get_output_embeddings().weight.shape[0])
        cfg = legacy["training"]
        model = get_peft_model(base, LoraConfig(revision=config["revision"], r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
        initial = diagnostic.trainable_state_sha256(model)
        if initial != endpoints[f"gold:{seed}"]["initial_trainable_state_sha256"]:
            raise ValueError("Initial adapter state differs from frozen training")
        model.eval()
        model.config.use_cache = False
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Expected CPU float32 parameters")
        params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if len(params) != 96 or sum(p.numel() for _, p in params) != 540672:
            raise ValueError("Trainable parameter coverage differs")
        state = {"state": "initial", "seed": seed, "initial_trainable_state_sha256": initial,
                 "parameter_blocks": 96, "trainable_parameters": 540672,
                 "parameters_before": diagnostic.parameter_state(model)}
        diagnostic.save(args.output / "state.json", state)
        check()
        ids, labels = saved["input_ids"].long().unsqueeze(0), saved["labels"].long().unsqueeze(0)
        probe = {"schema": 1, "state": "initial", "seed": seed, "train_id": key, "status": "running",
                 "cache_record_sha256": meta["sha256"], "is_impossible": source_rows[key]["is_impossible"],
                 "token_classes": diagnostic.token_classes(labels[:, 1:], tokenizer)}
        diagnostic.save(args.output / "probe.json", probe)
        run["phase"], run["probe_count"] = "gradient_capture", 1
        diagnostic.save(args.output / "run.json", run)
        observer = CaptureObserver(args.output, args.private_output, check)
        values, _ = diagnostic.inspect_probe(model, ids, labels, saved["teacher_log_probs"],
            {"atol": protocol["correctness"]["gradient_atol"], "rtol": protocol["correctness"]["gradient_rtol"]},
            observer=observer)
        probe.update(values, status="complete" if values["correctness_passed"] else "numerical_gate_failed")
        diagnostic.save(args.output / "probe.json", probe)
        run["numerical_gate_passed"] = values["correctness_passed"]
        run["capture_complete"] = observer.manifest["status"] == "complete"
        if not run["capture_complete"]:
            raise ValueError("Incomplete derivative capture")
    except BaseException as error:
        failure = error
        run.update(error_type=type(error).__name__, error="Capture stopped; see phase and preserved partial artifacts")
        if probe:
            probe.update(status="failed", error_type=type(error).__name__)
            diagnostic.save(args.output / "probe.json", probe)
    finally:
        # A failed numerical gate must not skip either parameter or source audit.
        for phase, action in (
            ("final_parameter_audit", lambda: _finish_state(args, model, state)),
            ("final_identity_audit", lambda: _final_integrity(args, protocol, legacy, run, cache, endpoints)),
        ):
            if (phase == "final_parameter_audit" and not state) or (phase == "final_identity_audit" and cache is None):
                run[phase] = "not_available"
                continue
            try:
                action()
                run[phase] = "passed"
            except BaseException as error:
                run[phase] = "failed"
                run[phase + "_error_type"] = type(error).__name__
                failure = failure or error
        if observer is not None:
            run["capture_complete"] = observer.manifest["status"] == "complete"
            run["captured_objectives"] = list(observer.manifest["objectives"])
        run["status"] = "complete" if failure is None else "failed"
        run["phase"] = "complete" if failure is None else run["phase"]
        run["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run["wall_seconds"] = time.perf_counter() - started
        run["artifacts"] = diagnostic._artifact_manifest(args.output)
        diagnostic.save(args.output / "run.json", run)
        if protocol is not None:
            try:
                _check_budget(started, protocol, args, run)
                run["final_budget_check"] = "passed"
            except BaseException as error:
                failure = failure or error
                run.update(status="failed", final_budget_check="failed", budget_error_type=type(error).__name__)
            run["wall_seconds"] = time.perf_counter() - started
            diagnostic.save(args.output / "run.json", run)
    if failure is not None:
        raise failure
    return run


def _finish_state(args, model, state):
    state["parameters_after"] = diagnostic.parameter_state(model)
    state["parameters_unchanged"] = state["parameters_before"] == state["parameters_after"]
    state["grad_buffers_empty"] = all(p.grad is None for _, p in model.named_parameters())
    diagnostic.save(args.output / "state.json", state)
    if not state["parameters_unchanged"] or not state["grad_buffers_empty"]:
        raise ValueError("Capture changed model state or gradient buffers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "cache", "training-root", "artifact", "data-dir", "student-config", "cache-dir", "output", "private-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run_capture(parser.parse_args())


if __name__ == "__main__":
    main()
