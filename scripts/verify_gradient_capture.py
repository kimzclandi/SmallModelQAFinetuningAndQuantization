"""NumPy-only full-array replay of a local, single-probe forensic capture.

A complete capture may contain failed numerical gates; it never completes the
original 192-probe study. Neither autograd nor private inputs are rerun here.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

_spec = importlib.util.spec_from_file_location("_capture_receipt_math", Path(__file__).with_name("verify_gradient_diagnostics.py"))
receipt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(receipt)
need, read, digest = receipt.need, receipt.read, receipt.digest
VECTORS, COMPONENTS = receipt.VECTORS, receipt.COMPONENTS
STUDY = "ce-kl-gradient-capture-v1"
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}
LEGACY_PROTOCOL_SHA = "4786724fcc338a44269c9f7db628d8a4a57ea6944a040a2647860bd747eaaae0"
PROBE_ID = "56e190bce3433e1400422fcb"
META_KEYS = {"name", "layer", "module", "matrix", "shape", "numel"}
PRIVATE_KEYS = {"selected_logits", "targets", "teacher_log_probs", "input_ids", "labels", "supervised_positions"}


def comparison(actual, reference):
    """Original elementwise gate, independent from receipt arithmetic tolerance."""
    import numpy as np
    need(actual.shape == reference.shape and actual.size > 0, "Comparison shape differs")
    need(bool(np.isfinite(actual).all() and np.isfinite(reference).all()), "Nonfinite comparison")
    a, b = actual.astype(np.float64).reshape(-1), reference.astype(np.float64).reshape(-1)
    delta = a - b
    bad = np.flatnonzero(np.abs(delta) > TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(b))
    norm = float(np.sqrt(np.sum(b * b, dtype=np.float64)))
    error = float(np.sqrt(np.sum(delta * delta, dtype=np.float64)))
    return {"allclose": not bool(len(bad)), "failed_elements": int(len(bad)),
            "first_failure_flat": int(bad[0]) if len(bad) else None,
            "max_abs_error": float(np.abs(delta).max()), "l2_error": error,
            "relative_l2_error": error / norm if norm else None, "reference_norm": norm,
            "reconstructed_norm": float(np.sqrt(np.sum(a * a, dtype=np.float64))), **TOLERANCE,
            "comparison": "abs(actual-reference)<=atol+rtol*abs(reference), evaluated in float64"}


def array_metadata(array):
    import numpy as np
    return {"dtype": str(array.dtype), "shape": list(array.shape), "numel": int(array.size),
            "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            "finite": bool(np.isfinite(array).all())}


def load_objective(root, key, descriptor, parameters, logit_shape):
    import numpy as np
    need(set(descriptor) == {"path", "sha256", "bytes", "arrays"}, "Objective metadata keys differ")
    name = f"arrays/{key}.npz"
    need(descriptor["path"] == name, "Objective path differs or escapes archive")
    path = root / name
    need(path.is_file() and not path.is_symlink(), "Missing/unsafe objective array")
    need(digest(path) == receipt._sha(descriptor["sha256"]), "Objective file digest differs")
    need(type(descriptor["bytes"]) is int and descriptor["bytes"] == path.stat().st_size, "Objective byte count differs")
    shapes = {"logit_actual": ("float32", logit_shape), "logit_reference": ("float64", logit_shape)}
    shapes.update({f"parameter_{i:03d}": ("float32", p["shape"]) for i, p in enumerate(parameters)})
    need(set(descriptor["arrays"]) == set(shapes), "Array metadata coverage differs")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        need(len(members) == len(set(members)) and set(members) == {n + ".npy" for n in shapes},
             "NPZ members missing, extra or duplicated")
        need(sum(x.file_size for x in archive.infolist()) <= 128 * 2**20, "Uncompressed array budget exceeded")
    with np.load(path, allow_pickle=False) as archive:
        need(len(archive.files) == len(shapes) and set(archive.files) == set(shapes), "Array keys differ")
        arrays = {n: archive[n] for n in shapes}
    for name, (dtype, shape) in shapes.items():
        value = arrays[name]
        need(value.dtype == np.dtype(dtype) and list(value.shape) == shape, "Array dtype/shape differs: " + name)
        need(bool(np.isfinite(value).all()), "Nonfinite captured array: " + name)
        need(descriptor["arrays"][name] == array_metadata(value), "Array metadata differs: " + name)
    return arrays


def replay_arrays(root, capture, probe, *, expected_layers=24, vocabulary=151936):
    """Dimension overrides are for synthetic tests; verify() fixes real Qwen dimensions."""
    import numpy as np
    need(capture.get("schema") == 1 and capture.get("study") == STUDY and capture.get("status") == "complete", "Incomplete/wrong capture")
    parameters = probe["parameters"]
    need(capture["parameters"] == [{k: p[k] for k in META_KEYS} for p in parameters], "Capture parameter metadata differs")
    passed = receipt.check_probe(probe, TOLERANCE, expected_layers=expected_layers, vocabulary=vocabulary)
    need(set(capture["objectives"]) == set(VECTORS), "Objective coverage differs")
    actual = {p.relative_to(root).as_posix() for p in (root / "arrays").rglob("*") if p.is_file()}
    need(actual == {f"arrays/{k}.npz" for k in VECTORS}, "Extra/missing array file")
    arrays = {k: load_objective(root, k, capture["objectives"][k], parameters, [probe["supervised_tokens"], vocabulary]) for k in VECTORS}
    logit_failures, parameter_failures, f64_checks = [], [], []
    for key in VECTORS:
        check = comparison(arrays[key]["logit_actual"], arrays[key]["logit_reference"])
        receipt.compare_tree(probe["logit_gradient_checks"][key], check, "Logit " + key)
        if not check["allclose"]:
            flat = check["first_failure_flat"]
            logit_failures.append({"objective": key, **check, "first_failure_coordinate": [flat // vocabulary, flat % vocabulary]})
    values_replayed = 0
    for index, parameter in enumerate(parameters):
        values = {k: arrays[k][f"parameter_{index:03d}"] for k in VECTORS}
        for key, value in values.items():
            need(parameter["gradient_sha256"][key] == array_metadata(value)["sha256"], "Parameter gradient digest differs")
            need(parameter["zero_gradient"][key] == (np.count_nonzero(value) == 0), "Zero-gradient flag differs")
            values_replayed += int(value.size)
        flat = [values[k].astype(np.float64).reshape(-1) for k in COMPONENTS]
        gram = [[float(np.sum(a * b, dtype=np.float64)) for b in flat] for a in flat]
        receipt.compare_tree(parameter["gram"], gram, "Parameter Gram")
        for suffix, key in (("full", "full_kl"), ("gated", "gated_kl")):
            combined = values["combined_" + suffix]
            check = comparison(values["ce"] + values[key], combined)
            receipt.compare_tree(parameter["reconstruction"][suffix], check, "Parameter reconstruction")
            double = comparison(values["ce"].astype(np.float64) + values[key].astype(np.float64), combined)
            f64_checks.append({"parameter": parameter["name"], "component": suffix, **double})
            if not check["allclose"]:
                parameter_failures.append({"parameter": parameter["name"], "component": suffix, **check,
                                           "float64_add_failed_elements": double["failed_elements"]})
    return {"numerical_gate_passed": passed, "parameter_blocks_replayed": len(parameters),
            "parameter_gradient_values_replayed": values_replayed,
            "legacy_scalar_loss_checks_replayed": False,
            "logit_gradient_values_replayed": len(VECTORS) * probe["supervised_tokens"] * vocabulary,
            "logit_failures": logit_failures, "parameter_failures": parameter_failures,
            "float64_add_diagnostics": f64_checks}


def check_files(root, run):
    need(root.is_dir() and not root.is_symlink(), "Archive must be a real directory")
    actual = {}
    for path in root.rglob("*"):
        need(not path.is_symlink(), "Archive symlinks forbidden")
        if path.is_file() and path != root / "archive-manifest.json":
            actual[path.relative_to(root).as_posix()] = digest(path)
    need(run.get("artifacts") == {k: v for k, v in actual.items() if k != "run.json"}, "Run artifact hash map differs")
    if (root / "archive-manifest.json").exists():
        need(receipt.check_archive(root) == actual, "Archive manifest differs")
    return actual


def check_provenance(root, run, files, repository=None):
    need(re.fullmatch(r"[0-9a-f]{40}", run.get("protocol_commit", "")), "Invalid protocol commit")
    committed = receipt._time(run["protocol_committed_utc"])
    need(committed <= receipt._time(run["started_utc"]) <= receipt._time(run["finished_utc"]), "Protocol was not committed before execution")
    sources = run["source_sha256"]
    required = {"qa_lab/gradient_capture.py", "qa_lab/gradient_diagnostics.py", "qa_lab/confirmation_training.py",
                "qa_lab/logits_distillation.py", "qa_lab/model_identity.py", "qa_lab/train_artifact.py", "qa_lab/common.py",
                "configs/ce-kl-gradient-v1.json", "configs/ce-kl-gradient-capture-v1.json",
                "scripts/verify_gradient_capture.py", "scripts/replay_private_gradient_capture.py", "scripts/verify_gradient_diagnostics.py"}
    need(required <= set(sources), "Missing execution/verification source")
    need({p.removeprefix("source/") for p in files if p.startswith("source/")} == set(sources), "Source coverage differs")
    for name, sha in sources.items():
        receipt._relative(name)
        need(files["source/" + name] == receipt._sha(sha), "Frozen source hash differs")
    path = root / "source/configs/ce-kl-gradient-capture-v1.json"
    need(digest(path) == run["protocol_sha256"], "Capture protocol digest differs")
    protocol = read(path)
    need(protocol == run["protocol"] and protocol["study"] == STUDY and protocol["status"] == "frozen_before_execution", "Capture protocol identity differs")
    need(protocol["probe"] == {"state": "initial", "seed": 20261009, "train_id": PROBE_ID, "count": 1}, "Probe plan differs")
    need(protocol["correctness"] == {"gradient_atol": 1e-5, "gradient_rtol": 1e-4}, "Numerical tolerance changed")
    need(protocol["scope"] == {"optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
                               "quality_or_performance_claims": False, "full_192_probe_study": False}, "Scope changed")
    need(protocol["budget"] == {"wall_seconds": 120, "max_rss_bytes": 16 * 2**30,
                                "max_output_bytes": 128 * 2**20, "paid_resources": False}, "Budget changed")
    old = root / "source/configs/ce-kl-gradient-v1.json"
    need(protocol["source_protocol_sha256"] == digest(old) == LEGACY_PROTOCOL_SHA, "Original frozen protocol changed")
    legacy = read(old)
    need(run["input_sha256"] == legacy["hashes"], "Input binding differs")
    if repository is not None:
        repository = Path(repository)
        for name, sha in sources.items():
            data = subprocess.check_output(["git", "-C", str(repository), "show", run["protocol_commit"] + ":" + name])
            need(hashlib.sha256(data).hexdigest() == sha, "Source differs from pre-execution Git commit")
        stamp = subprocess.check_output(["git", "-C", str(repository), "show", "-s", "--format=%cI", run["protocol_commit"]], text=True).strip()
        need(receipt._time(stamp) == committed, "Git commit time differs")
        for name, sha in protocol["preserved_failure_sha256"].items():
            receipt._relative(name)
            need(digest(repository / name) == receipt._sha(sha), "Preserved historical failure changed")
    return protocol, legacy


def check_inputs(root, run, legacy, probe, state):
    inputs = {"student_config": "student_config.json", "artifact_manifest": "artifact_manifest.json",
              "artifact_train": "artifact_train.jsonl", "source_train": "source_train.jsonl"}
    for key, name in inputs.items():
        need(digest(root / "inputs" / name) == legacy["hashes"][key], "Frozen input differs: " + key)
    source, artifact = receipt.rows(root / "inputs/source_train.jsonl"), receipt.rows(root / "inputs/artifact_train.jsonl")
    a, b = [r["id"] for r in source], [r["id"] for r in artifact]
    need(len(a) == len(b) == len(set(a)) == len(set(b)) == 242 and set(a) == set(b), "TRAIN cohort coverage differs")
    need(PROBE_ID in a and probe["is_impossible"] is next(r["is_impossible"] for r in source if r["id"] == PROBE_ID), "Probe stratum differs")
    cached = read(root / "inputs/cache.receipt.json")
    need(cached["original_manifest_sha256"] == legacy["hashes"]["cache_manifest"] and cached["redactions"] == ["artifact_path"], "Cache receipt binding differs")
    cache = cached["redacted_manifest"]
    need("artifact_path" not in cache and cache["status"] == "complete" and cache["temperature"] == 2.
         and cache["model_identities"] == run["model_identities"]
         and cache["student_model_config"] == read(root / "inputs/student_config.json"), "Cache/model identity differs")
    records = {r["id"]: r for r in cache["records"]}
    need(len(records) == len(cache["records"]) == 242 and set(records) == set(a)
         and sum(r["supervised_tokens"] for r in records.values()) == 1289
         and all(r["vocabulary"] == 151936 for r in records.values()), "Cache coverage differs")
    row = records[PROBE_ID]
    need(probe["cache_record_sha256"] == row["sha256"] and probe["input_tokens"] == row["sequence_tokens"]
         and probe["supervised_tokens"] == row["supervised_tokens"], "Probe cache identity differs")
    endpoints = {}
    for key, sha in legacy["hashes"]["endpoint_training"].items():
        path = root / "inputs" / ("endpoint-" + key.replace(":", "-") + ".json")
        need(digest(path) == sha, "Endpoint training identity differs")
        value = read(path); endpoints[key] = value
        arm, seed = key.split(":")
        need(value["status"] == "complete" and value["arm"] == arm and value["seed"] == int(seed)
             and value["steps_completed"] == 242 and value["cache_manifest_sha256"] == legacy["hashes"]["cache_manifest"]
             and value["model_identities"] == run["model_identities"], "Endpoint receipt differs")
    need(state["initial_trainable_state_sha256"] == endpoints["gold:20261009"]["initial_trainable_state_sha256"]
         == state["parameters_before"]["adapter"]["sha256"], "Initial adapter identity differs")
    return {"inputs/" + name for name in inputs.values()} | {"inputs/cache.receipt.json"} | {
        "inputs/endpoint-" + key.replace(":", "-") + ".json" for key in endpoints}


def verify(root, repository=None):
    root = Path(root)
    need(not root.is_symlink(), "Archive root symlink forbidden")
    root = root.resolve()
    run = read(root / "run.json")
    need(run.get("schema") == 1 and run.get("study") == STUDY and run.get("status") == "complete"
         and run.get("capture_complete") is True and run.get("phase") == "complete", "Incomplete or failed execution")
    need(run.get("diagnostic_complete") is False and run.get("quality_or_performance_conclusion") is False, "Capture cannot become complete study or quality/performance claim")
    need(type(run.get("probe_count")) is int and run["probe_count"] == 1, "Unexpected probe count")
    for key in ("optimizer_steps", "free_generation", "heldout_predictions"):
        need(type(run.get(key)) is int and run[key] == 0, "Unexpected update or inference")
    for key in ("final_parameter_audit", "final_identity_audit", "final_budget_check"):
        need(run.get(key) == "passed", "Missing final audit: " + key)
    files = check_files(root, run)
    protocol, legacy = check_provenance(root, run, files, repository)
    probe, state, capture = (read(root / name) for name in ("probe.json", "state.json", "capture.json"))
    need(probe["schema"] == 1 and probe["state"] == state["state"] == "initial"
         and probe["seed"] == state["seed"] == 20261009 and probe["train_id"] == PROBE_ID, "Probe/state identity differs")
    need(state["parameters_unchanged"] is True and state["grad_buffers_empty"] is True
         and state["parameters_before"] == state["parameters_after"], "Parameter update or incomplete state audit")
    need(state["parameter_blocks"] == 96 and state["trainable_parameters"] == 540672
         and set(state["parameters_before"]) == {"adapter", "base"}, "Parameter state coverage differs")
    for key, count in (("adapter", 540672), ("base", 494032768)):
        value = state["parameters_before"][key]
        need(value["numel"] == count, "Parameter count differs"); receipt._sha(value["sha256"])
    input_files = check_inputs(root, run, legacy, probe, state)
    expected = {"run.json", "probe.json", "state.json", "capture.json"} | input_files
    expected |= {"source/" + name for name in run["source_sha256"]} | {f"arrays/{key}.npz" for key in VECTORS}
    need(set(files) == expected, "Unexpected archive file")
    need(run["captured_objectives"] == list(VECTORS) and capture["last_stage"] == VECTORS[-1], "Objective order differs")
    private = capture["private_inputs"]
    need(set(private) == {"sha256", "bytes", "arrays"} and set(private["arrays"]) == PRIVATE_KEYS, "Private receipt coverage differs or exposes path")
    receipt._sha(private["sha256"])
    need(type(private["bytes"]) is int and private["bytes"] > 0, "Invalid private byte count")
    for value in private["arrays"].values():
        need(set(value) == {"dtype", "shape", "numel", "sha256", "finite"} and value["finite"] is True, "Invalid private metadata")
        receipt._sha(value["sha256"])
        need(isinstance(value["shape"], list) and all(type(n) is int and n > 0 for n in value["shape"])
             and type(value["numel"]) is int and value["numel"] == math.prod(value["shape"]), "Private shape differs")
    expected_private = {
        "selected_logits": ("float32", [probe["supervised_tokens"], 151936]),
        "teacher_log_probs": ("float16", [probe["supervised_tokens"], 151936]),
        "targets": ("int64", [probe["supervised_tokens"]]),
        "input_ids": ("int64", [1, probe["input_tokens"]]),
        "labels": ("int64", [1, probe["input_tokens"]]),
        "supervised_positions": ("int64", [probe["supervised_tokens"], 2]),
    }
    for name, (dtype, shape) in expected_private.items():
        need(private["arrays"][name]["dtype"] == dtype and private["arrays"][name]["shape"] == shape,
             "Private input dtype/shape contract differs: " + name)
    for p in probe["parameters"]:
        shape = [8, 896] if p["matrix"] == "A" else ([896, 8] if p["module"] == "q_proj" else [128, 8])
        need(p["shape"] == shape, "Real Qwen LoRA shape differs")
    need(sum(p["numel"] for p in probe["parameters"]) == 540672, "Gradient coverage differs")
    replay = replay_arrays(root, capture, probe)
    passed = replay["numerical_gate_passed"]
    need(type(run["numerical_gate_passed"]) is bool and run["numerical_gate_passed"] == passed
         and probe["status"] == ("complete" if passed else "numerical_gate_failed"), "Numerical failure incorrectly upgraded or status differs")
    for key, limit in (("wall_seconds", "wall_seconds"), ("peak_process_rss_bytes", "max_rss_bytes"), ("combined_output_bytes_last_check", "max_output_bytes")):
        need(0 <= receipt._number(run[key]) <= protocol["budget"][limit], "Resource budget exceeded")
    need(sum((root / name).stat().st_size for name in files) + private["bytes"] <= protocol["budget"]["max_output_bytes"], "Capture disk budget exceeded")
    return {"status": "verified", "study": STUDY, "capture_complete": True, "diagnostic_complete": False,
            "protocol_commit": run["protocol_commit"], "protocol_sha256": run["protocol_sha256"],
            "git_source_provenance_checked": repository is not None, **replay,
            "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0, "quality_or_performance_conclusion": False,
            "limits": ["Replays locally saved derivatives, not autograd or private-input formulas; real arrays are not public CI assets.",
                       "Float64-add diagnostics isolate final addition rounding only and never replace the original float32-add gate.",
                       "Hashes/timestamps are receipts, not authenticity proof; optional repository checking compares pre-execution Git objects.",
                       "Model/cache bytes, parameter immutability and legacy scalar loss checks remain receipts in this replay.",
                       "One TRAIN probe does not complete the 192-probe study or prove historical root cause, quality or speed gains."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root, args.repository)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        need(not args.output.exists() and not args.output.resolve().is_relative_to(args.root.resolve()), "Output must be new and outside archive")
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
