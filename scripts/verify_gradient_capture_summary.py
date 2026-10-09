"""Check published scalar receipts; never claim replay of unpublished tensors."""
import argparse
import hashlib
import json
from pathlib import Path

FILES = {"execution-01-rejected.json", "execution-02-summary.json", "numpy-replay-summary.json",
         "same-input-replay-summary.json", "local-artifact-manifest.json"}
COMMIT = "d925dcc3e55c55b939e101dde0fd83c3de0b3171"
PROTOCOL_SHA = "45d43a1b96b63feb1741e496037d436ff3445f2d41f5e643968fcace15f625f9"
OBJECTIVES = {"ce", "full_kl", "gated_kl", "combined_full", "combined_gated"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root = Path(root)
    need(not any(p.is_symlink() for p in root.rglob("*")), "Symlink forbidden")
    need({p.name for p in root.iterdir()} == FILES | {"manifest.json"}, "Only reviewed scalar JSON files allowed")
    read = lambda name: json.loads((root / name).read_text())
    manifest = read("manifest.json")
    need(set(manifest["files"]) == FILES, "Receipt coverage differs")
    records = {name: read(name) for name in FILES}
    for name in FILES:
        need(hashlib.sha256((root / name).read_bytes()).hexdigest() == manifest["files"][name], "Receipt hash differs")
    for value in [manifest, *records.values()]:
        need(value["real_tensor_values_included"] is False and value["ci_replays_real_tensor_values"] is False,
             "Unpublished tensor replay cannot be claimed")
    blocked = records["execution-01-rejected.json"]
    need(blocked["phase"] == "protocol" and blocked["status"] == "failed"
         and blocked["probe_count"] == blocked["model_loads"] == 0, "Preserve pre-model failure")
    run, replay, local = (records[n] for n in ("execution-02-summary.json", "numpy-replay-summary.json", "same-input-replay-summary.json"))
    for value in (run, replay, local):
        need(value["protocol_commit"] == COMMIT, "Frozen source commit changed")
    for value in (run, replay):
        need(value["protocol_sha256"] == PROTOCOL_SHA and value["capture_complete"] is True
             and value["numerical_gate_passed"] is False and value["diagnostic_complete"] is False,
             "Capture success must not upgrade numerical failure")
        need(value["optimizer_steps"] == value["free_generation"] == value["heldout_predictions"] == 0
             and value["quality_or_performance_conclusion"] is False, "Scope changed")
    need(run["probe_count"] == run["model_loads"] == 1 and run["status"] == "complete", "Unexpected model/probe count")
    need(all(run[key] == "passed" for key in ("final_parameter_audit", "final_identity_audit", "final_budget_check")), "Final audit missing")
    need(run["historical_recorded_summary_exact_match"] and all(v is True for v in run["historical_recorded_summary_exact_match"].values()), "Historical receipt mismatch")
    need(replay["parameter_blocks_replayed"] == 96 and replay["parameter_gradient_values_replayed"] == 2703360
         and replay["logit_gradient_values_replayed"] == 2279040
         and replay["legacy_scalar_loss_checks_replayed"] is False, "Replay scope differs")
    need({x["objective"] for x in replay["logit_failures"]} == OBJECTIVES - {"ce"}
         and len(replay["logit_failures"]) == 4, "Logit failure coverage differs")
    for failure in replay["logit_failures"]:
        need(failure["allclose"] is False and failure["failed_elements"] == 1
             and failure["first_failure_coordinate"] == [2, 198]
             and failure["atol"] == 1e-5 and failure["rtol"] == 1e-4, "Original logit gate changed")
    pairs = {(x["parameter"].split(".layers.")[1].split(".")[0], x["component"], x["failed_elements"], x["float64_add_failed_elements"])
             for x in replay["parameter_failures"]}
    need(pairs == {(layer, component, count, count) for layer, count in (("0", 6), ("6", 2)) for component in ("full", "gated")}
         and len(replay["parameter_failures"]) == 4, "Parameter failure summary changed")
    need(local["capture_run_sha256"] == run["run_sha256"] and local["shape"] == [3, 151936]
         and local["tolerance"] == {"atol": 1e-5, "rtol": 1e-4}, "Same-input identity differs")
    for key in ("fp32_replay_bitwise_matches_capture", "fp64_formula_bitwise_matches_capture"):
        need(set(local[key]) == OBJECTIVES and all(x is True for x in local[key].values()), "Same-input mismatch")
    need(set(local["fp64_autograd_vs_formula"]) == OBJECTIVES
         and all(v["allclose"] is True and v["failed_elements"] == 0 for v in local["fp64_autograd_vs_formula"].values()), "FP64 diagnostic differs")
    return {"status": "verified_scalar_receipt_consistency", "real_tensor_replay_in_ci": False,
            "local_capture_complete": True, "original_numerical_gate_passed": False,
            "limits": "Public CI checks saved scalar consistency, hashes and synthetic tests. Real-array replay occurred locally; these receipts are not independent public recomputation of unseen tensors."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/ce-kl-gradient-capture-v1")
    print(json.dumps(verify(parser.parse_args().root), indent=2))
