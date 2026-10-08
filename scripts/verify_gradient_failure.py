"""Verify the frozen v1 early-stop evidence, never upgrade it to a passed study.

The only public raw block passed its reconstruction check. Failed parameter and
logit arrays were not archived and cannot be independently replayed offline.
No model, optimizer, generation, heldout labels, or autograd is executed here.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_gradient_receipt_math", Path(__file__).with_name("verify_gradient_diagnostics.py"))
math_verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(math_verify)
need, read, digest = math_verify.need, math_verify.read, math_verify.digest
FREEZE = "e63d128bbfec22377f16db96249d610b9eb55471"
PROTOCOL_SHA = "4786724fcc338a44269c9f7db628d8a4a57ea6944a040a2647860bd747eaaae0"
SOURCE_MAP_SHA = "953b29cf9196eb74b624c9169cd18d71ce819b259afe0b637cd866c71dddc610"
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def failure_details(probe):
    """Classify recorded failures; do not claim unarchived vector reproduction."""
    need(math_verify.check_probe(probe, TOLERANCE) is False, "Expected a failed numerical probe")
    failures = []
    for parameter in probe["parameters"]:
        for component, check in parameter["reconstruction"].items():
            if not check["allclose"]:
                failures.append({"layer": parameter["layer"], "module": parameter["module"],
                                 "matrix": parameter["matrix"], "component": component,
                                 "failed_elements": check["failed_elements"],
                                 "first_failure_flat": check["first_failure_flat"],
                                 "max_abs_error": check["max_abs_error"]})
    expected = {(0, "v_proj", "B", "full", 6, 72), (0, "v_proj", "B", "gated", 6, 72),
                (6, "v_proj", "B", "full", 2, 508), (6, "v_proj", "B", "gated", 2, 508)}
    need({tuple(row[key] for key in ("layer", "module", "matrix", "component", "failed_elements", "first_failure_flat"))
          for row in failures} == expected and len(failures) == 4, "Recorded parameter failure set changed")
    need(all(item["allclose"] for item in probe["legacy_loss_checks"].values()), "Historical loss agreement changed")
    logit = probe["logit_gradient_checks"]
    need(logit["ce"]["allclose"], "Recorded CE logit check changed")
    failed_keys = {key for key, check in logit.items() if not check["allclose"]}
    need(failed_keys == {"full_kl", "gated_kl", "combined_full", "combined_gated"}, "Recorded logit failure set changed")
    need(all(logit[key]["failed_elements"] == 1 and logit[key]["first_failure_flat"] == 304070
             for key in failed_keys), "Recorded logit failure coordinates changed")
    # Identical gates here, not four distinct failure sites or a gate improvement.
    need(probe["teacher_top1_correct"] == probe["supervised_tokens"] == 3,
         "First probe teacher agreement changed")
    need(probe["losses"]["full_kl"] == probe["losses"]["gated_kl"]
         and probe["losses"]["combined_full"] == probe["losses"]["combined_gated"], "All-token gated objectives differ")
    for parameter in probe["parameters"]:
        need(parameter["gradient_sha256"]["full_kl"] == parameter["gradient_sha256"]["gated_kl"]
             and parameter["gradient_sha256"]["combined_full"] == parameter["gradient_sha256"]["combined_gated"],
             "All-token gated gradient identities differ")
    return {"recorded_parameter_failures": failures,
            "recorded_logit_failed_objectives": sorted(failed_keys),
            "recorded_logit_first_failure": {"flat": 304070, "supervised_position_zero_based": 2,
                                              "vocabulary_index": 198},
            "failure_vectors_independently_replayed": False}


def verify(root):
    root = Path(root)
    need(not root.is_symlink(), "Archive root symlinks forbidden")
    root = root.resolve()
    files = math_verify.check_archive(root)
    run = read(root / "run.json")
    need(run.get("schema") == 1 and run.get("study") == "ce-kl-gradient-v1"
         and run.get("status") == "failed" and run.get("diagnostic_complete") is False,
         "Expected preserved failed diagnostic")
    need(run.get("probes_completed") == 0 and run.get("states_completed") == 0
         and run.get("active_probe") == 0 and run.get("active_state") == "initial:20261009",
         "Early-stop coverage changed")
    for key in ("probes_completed", "states_completed", "active_probe"):
        math_verify._integer(run[key], key)
    need(run.get("phase") == "gradient_probe" and run.get("error_type") == "ValueError",
         "Failure phase/type changed")
    need(run["protocol_commit"] == FREEZE and run["protocol_sha256"] == PROTOCOL_SHA,
         "Frozen v1 execution identity changed")
    sources = run["source_sha256"]
    need(hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == SOURCE_MAP_SHA,
         "Pinned execution source map changed")
    for name, sha in sources.items():
        need(files["source/" + math_verify._relative(name)] == sha, "Frozen source content differs")
    protocol_file = root / "source/configs/ce-kl-gradient-v1.json"
    need(digest(protocol_file) == PROTOCOL_SHA, "Frozen protocol bytes changed")
    protocol = read(protocol_file)
    need(run["protocol"] == protocol and run["input_sha256"] == protocol["hashes"], "Embedded protocol/input binding differs")
    need(run["artifacts"] == {key: value for key, value in files.items() if key != "run.json"},
         "Execution and public archive manifests differ")
    for key in ("optimizer_steps", "free_generation", "heldout_predictions"):
        need(type(run[key]) is int and run[key] == 0, "Unexpected update/generation/heldout work")
    need(run["probe_files"] == ["probes/000.json"] and run["state_files"] == ["states/initial-20261009.json"],
         "Unexpected probe/state files after early stop")
    inputs = {"student_config": "student_config.json", "artifact_manifest": "artifact_manifest.json",
              "artifact_train": "artifact_train.jsonl", "source_train": "source_train.jsonl"}
    allowed = {"run.json", "probes/000.json", "states/initial-20261009.json", "arrays/initial-20261009.npz"}
    allowed |= {"source/" + name for name in sources}
    allowed |= {"inputs/" + name for name in inputs.values()} | {"inputs/cache.receipt.json"}
    allowed |= {f"inputs/endpoint-{arm}-{seed}.json" for arm in math_verify.STATES[1:] for seed in math_verify.SEEDS}
    need(set(files) == allowed, "Unexpected result file/summary in failed archive")
    for key, name in inputs.items():
        need(digest(root / "inputs" / name) == protocol["hashes"][key], "Frozen input changed")
    source = math_verify.rows(root / "inputs/source_train.jsonl")
    artifact = math_verify.rows(root / "inputs/artifact_train.jsonl")
    source_by_id = {row["id"]: row for row in source}
    need(len(source) == len(artifact) == len(source_by_id) == 242
         and {row["id"] for row in artifact} == set(source_by_id), "TRAIN coverage differs")
    selected = []
    for flag in (False, True):
        selected += sorted((row["id"] for row in source if row["is_impossible"] is flag),
                           key=lambda key: hashlib.sha256(("ce-kl-gradient-v1|" + key).encode()).hexdigest())[:8]
    need(selected == protocol["selected_ids"] and len({source_by_id[key]["context_id"] for key in selected}) == 12,
         "Predeclared selection changed")
    cache_receipt = read(root / "inputs/cache.receipt.json")
    need(cache_receipt["original_manifest_sha256"] == protocol["hashes"]["cache_manifest"]
         and cache_receipt["redactions"] == ["artifact_path"], "Cache binding changed")
    cache = cache_receipt["redacted_manifest"]
    need("artifact_path" not in cache and cache["status"] == "complete" and cache["temperature"] == 2.
         and cache["model_identities"] == run["model_identities"]
         and cache["student_model_config"] == read(root / "inputs/student_config.json"), "Cache/model identity differs")
    records = {row["id"]: row for row in cache["records"]}
    need(len(records) == len(cache["records"]) == 242 and set(records) == set(source_by_id)
         and sum(row["supervised_tokens"] for row in records.values()) == 1289
         and all(row["vocabulary"] == 151936 for row in records.values()), "Cache coverage changed")
    endpoints = {}
    for key, expected in protocol["hashes"]["endpoint_training"].items():
        path = root / "inputs" / ("endpoint-" + key.replace(":", "-") + ".json")
        need(digest(path) == expected, "Frozen endpoint receipt changed")
        receipt = read(path); endpoints[key] = receipt
        arm, seed = key.split(":")
        need(receipt["status"] == "complete" and receipt["arm"] == arm and receipt["seed"] == int(seed)
             and receipt["steps_completed"] == 242 and receipt["model_identities"] == run["model_identities"]
             and receipt["cache_manifest_sha256"] == protocol["hashes"]["cache_manifest"], "Endpoint identity differs")
    state = read(root / "states/initial-20261009.json")
    need(state["status"] == "failed" and state["probes_completed"] == 0 and state["state"] == "initial"
         and state["seed"] == 20261009 and state["error_type"] == "ValueError", "State failure changed")
    need(state["parameters_unchanged"] is True and state["parameters_before"] == state["parameters_after"],
         "Parameters changed or audit missing")
    before = state["parameters_before"]
    need(before["adapter"]["numel"] == state["trainable_parameters"] == 540672
         and state["parameter_blocks"] == 96 and before["base"]["numel"] == 494032768,
         "Parameter coverage changed")
    need(before["adapter"]["sha256"] == state["initial_trainable_state_sha256"]
         == endpoints["gold:20261009"]["initial_trainable_state_sha256"], "Initial parameter identity differs")
    for part in before.values(): math_verify._sha(part["sha256"])
    start, finish = math_verify._time(run["started_utc"]), math_verify._time(run["finished_utc"])
    need(start <= math_verify._time(state["started_utc"]) <= math_verify._time(state["finished_utc"]) <= finish,
         "State/run timestamps differ")
    probe = read(root / "probes/000.json")
    need(probe["schema"] == 1 and probe["status"] == "failed" and probe["correctness_passed"] is False
         and probe["index"] == 0 and probe["seed"] == 20261009 and probe["state"] == "initial"
         and probe["train_id"] == selected[0] and probe["is_impossible"] is False
         and probe["role"] == "primary_initial" and probe["phase"] == "serialize_gradient_receipt"
         and probe["error_code"] == "diagnostic_probe_serialize_gradient_receipt", "Failed probe identity changed")
    record = records[probe["train_id"]]
    need(probe["cache_record_sha256"] == record["sha256"] and probe["input_tokens"] == record["sequence_tokens"]
         and probe["supervised_tokens"] == record["supervised_tokens"], "Probe cache binding changed")
    details = failure_details(probe)
    need(sum(p["numel"] for p in probe["parameters"]) == 540672, "Parameter size changed")
    for parameter in probe["parameters"]:
        shape = [8, 896] if parameter["matrix"] == "A" else ([896, 8] if parameter["module"] == "q_proj" else [128, 8])
        need(parameter["shape"] == shape, "Qwen parameter shape changed")
    raw = probe["raw_artifact"]; name = "arrays/initial-20261009.npz"
    parameter = next(p for p in probe["parameters"] if (p["layer"], p["module"], p["matrix"]) == (0, "q_proj", "B"))
    need(raw["path"] == name and raw["sha256"] == files[name] and raw["keys"] == list(math_verify.VECTORS)
         and raw["parameter"] == probe["public_parameter"] == parameter["name"], "Public block identity differs")
    need(all(item["allclose"] for item in parameter["reconstruction"].values()), "Public qB block is no longer a passing block")
    values = math_verify.replay_public_block(root / name, parameter, TOLERANCE)
    for key, limit in (("wall_seconds", "wall_seconds"), ("peak_process_rss_bytes", "max_rss_bytes"),
                       ("output_bytes_last_check", "max_output_bytes")):
        need(0 <= math_verify._number(run[key], key) <= protocol["budget"][limit], "Recorded resource budget changed")
    need(sum((root / name).stat().st_size for name in files) <= protocol["budget"]["max_output_bytes"], "Archive disk budget exceeded")
    return {"status": "verified_failure_evidence", "study": "ce-kl-gradient-v1", "protocol_commit": FREEZE,
            "protocol_sha256": PROTOCOL_SHA, "diagnostic_complete": False, "numerical_gate_passed": False,
            "successful_probes": 0, "failed_probes": 1, "completed_states": 0, "failed_states": 1,
            "final_input_cache_adapter_integrity_recheck_executed": False,
            "hypotheses": {key: None for key in protocol["hypotheses"]}, **details,
            "public_passing_block": {"layer": 0, "module": "q_proj", "matrix": "B",
                                     "independent_replay": "passed", "float32_values": values},
            "limits": ["Failed v_proj and logit arrays were not archived; their failure counts/coordinates are recorded evidence, not independent numerical failure replay.",
                       "The only raw q_proj block passed. Its replay cannot validate the unarchived failed blocks.",
                       "Recorded model/cache/adapter hashes and before/after identities do not independently replay weights, autograd or parameter immutability.",
                       "The failed execution stopped before final_integrity; its final source/input/cache/adapter recheck did not execute.",
                       "Early stop answers neither predeclared gradient hypothesis and proves no quality, training or model failure, improvement, or performance result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/ce-kl-gradient-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); result = verify(args.root)
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        need(not args.output.exists() and not args.output.resolve().is_relative_to(args.root.resolve()),
             "Output must be new and outside frozen archive")
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
