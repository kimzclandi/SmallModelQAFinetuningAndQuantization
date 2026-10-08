"""Synthetic full-derivative corruption checks; no model, Torch or heldout data."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_gradient_capture", REPO / "scripts/verify_gradient_capture.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def save_objective(root, key, arrays):
    path = root / "arrays" / (key + ".npz")
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **arrays)
    return {"path": "arrays/" + path.name, "sha256": verify.digest(path), "bytes": path.stat().st_size,
            "arrays": {name: verify.array_metadata(value) for name, value in arrays.items()}}


def fixture(root):
    names = sorted(f"base_model.model.model.layers.0.self_attn.{module}.lora_{matrix}.default.weight"
                   for module in ("q_proj", "v_proj") for matrix in ("A", "B"))
    arrays = {key: {"logit_actual": np.zeros((2, 7), np.float32), "logit_reference": np.zeros((2, 7), np.float64)} for key in verify.VECTORS}
    # Exactly one analytic failure at (1, 3), independent of parameter failures.
    arrays["full_kl"]["logit_actual"][1, 3] = .001
    parameters = []
    for i, name in enumerate(names):
        module, matrix = ("q_proj" if "q_proj" in name else "v_proj"), ("A" if "lora_A" in name else "B")
        ce = np.zeros((2, 2), np.float32) if matrix == "A" else np.array([[1., 2.], [0., -1.]], np.float32)
        full = -ce
        gated = ce * np.float32(.5)
        values = {"ce": ce, "full_kl": full, "gated_kl": gated,
                  "combined_full": ce + full, "combined_gated": ce + gated}
        if module == "q_proj" and matrix == "B":
            values["combined_full"][0, 1] = .001
        for key, value in values.items(): arrays[key][f"parameter_{i:03d}"] = value
        flat = [values[k].astype(np.float64).reshape(-1) for k in verify.COMPONENTS]
        parameters.append({"name": name, "layer": 0, "module": module, "matrix": matrix, "shape": [2, 2], "numel": 4,
            "gram": [[float(np.sum(a * b)) for b in flat] for a in flat],
            "gradient_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in values.items()},
            "zero_gradient": {k: not bool(np.count_nonzero(v)) for k, v in values.items()},
            "reconstruction": {"full": verify.comparison(ce + full, values["combined_full"]),
                               "gated": verify.comparison(ce + gated, values["combined_gated"])}})
    zero = np.zeros(1, np.float32)
    probe = {"parameters": parameters, "groups": verify.receipt.grouped_geometry(parameters), "supervised_tokens": 2,
             "input_tokens": 8, "teacher_top1_correct": 1,
             "losses": {"ce": 1., "full_kl": 2., "gated_kl": 1., "combined_full": 3., "combined_gated": 2.},
             "logit_gradient_checks": {k: verify.comparison(a["logit_actual"], a["logit_reference"]) for k, a in arrays.items()},
             "legacy_loss_checks": {k: verify.comparison(zero, zero) for k in ("full", "gated")}, "correctness_passed": False}
    capture = {"schema": 1, "study": verify.STUDY, "status": "complete",
               "parameters": [{k: p[k] for k in verify.META_KEYS} for p in parameters],
               "objectives": {k: save_objective(root, k, a) for k, a in arrays.items()}}
    return capture, probe, arrays


def replay(root, capture, probe):
    return verify.replay_arrays(root, capture, probe, expected_layers=1, vocabulary=7)


def test_replay_recomputes_failed_logit_parameter_and_zero_blocks(tmp_path):
    capture, probe, _ = fixture(tmp_path)
    result = replay(tmp_path, capture, probe)
    assert result["numerical_gate_passed"] is False
    assert result["parameter_blocks_replayed"] == 4
    assert result["parameter_gradient_values_replayed"] == 80
    assert result["logit_gradient_values_replayed"] == 70
    assert len(result["logit_failures"]) == len(result["parameter_failures"]) == 1
    assert result["logit_failures"][0]["first_failure_coordinate"] == [1, 3]
    assert result["parameter_failures"][0]["failed_elements"] == 1
    assert result["parameter_failures"][0]["first_failure_flat"] == 1
    assert len(result["float64_add_diagnostics"]) == 8
    assert probe["groups"]["matrix:A"]["pairs"]["full_kl"]["cosine"] is None
    assert probe["groups"]["matrix:B"]["pairs"]["full_kl"]["negative_dot"] is True


def test_float64_add_is_separate_not_new_gate():
    # A low-precision sum can pass although the exact sum of saved inputs fails.
    a = np.array([1e8], np.float32); b = np.array([1.], np.float32)
    ref = a + b
    low, high = verify.comparison(ref, ref), verify.comparison(a.astype(np.float64) + b.astype(np.float64), ref)
    assert low["max_abs_error"] == 0
    assert high["max_abs_error"] == 1
    assert verify.TOLERANCE == {"atol": 1e-5, "rtol": 1e-4}


@pytest.mark.parametrize("kind", ["extra", "missing", "dtype", "nan", "inf", "shape", "metadata"])
def test_array_contract_corruption_even_with_updated_file_hash(tmp_path, kind):
    capture, probe, arrays = fixture(tmp_path)
    values = arrays["ce"]
    if kind == "extra": values["extra"] = np.zeros(1, np.float32)
    elif kind == "missing": del values["parameter_003"]
    elif kind == "dtype": values["parameter_000"] = values["parameter_000"].astype(np.float64)
    elif kind == "nan": values["logit_actual"][0, 0] = np.nan
    elif kind == "inf": values["logit_reference"][0, 0] = np.inf
    elif kind == "shape": values["parameter_001"] = values["parameter_001"].reshape(4)
    capture["objectives"]["ce"] = save_objective(tmp_path, "ce", values)
    if kind == "metadata": capture["objectives"]["ce"]["arrays"]["parameter_000"]["numel"] = 99
    with pytest.raises(ValueError): replay(tmp_path, capture, probe)


@pytest.mark.parametrize("kind", ["sha", "path", "parameter", "flag", "threshold", "gram", "zero", "coordinate", "size"])
def test_forged_receipts_rejected(tmp_path, kind):
    capture, probe, _ = fixture(tmp_path)
    if kind == "sha": capture["objectives"]["ce"]["sha256"] = "0" * 64
    elif kind == "path": capture["objectives"]["ce"]["path"] = "../ce.npz"
    elif kind == "parameter": capture["parameters"][0]["name"] += "changed"
    elif kind == "flag": probe["correctness_passed"] = True
    elif kind == "threshold": probe["logit_gradient_checks"]["full_kl"]["atol"] = 1.
    elif kind == "gram": probe["parameters"][1]["gram"][0][0] += .01
    elif kind == "zero": probe["parameters"][0]["zero_gradient"]["combined_full"] = False
    elif kind == "coordinate": probe["logit_gradient_checks"]["full_kl"]["first_failure_flat"] = 9
    elif kind == "size": capture["objectives"]["ce"]["bytes"] += 1
    with pytest.raises(ValueError): replay(tmp_path, capture, probe)


def test_duplicate_npz_member_rejected_even_with_matching_file_hash(tmp_path):
    capture, probe, _ = fixture(tmp_path)
    path = tmp_path / "arrays/ce.npz"
    with zipfile.ZipFile(path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("logit_actual.npy", archive.read("logit_actual.npy"))
    capture["objectives"]["ce"].update(sha256=verify.digest(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="duplicated"): replay(tmp_path, capture, probe)


def test_missing_objective_extra_file_and_symlink_are_rejected(tmp_path):
    capture, probe, _ = fixture(tmp_path)
    del capture["objectives"]["ce"]
    with pytest.raises(ValueError, match="coverage"): replay(tmp_path, capture, probe)
    capture, probe, _ = fixture(tmp_path)
    (tmp_path / "arrays/extra.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="Extra"): replay(tmp_path, capture, probe)
    (tmp_path / "arrays/extra.npz").unlink()
    path = tmp_path / "arrays/ce.npz"
    target = tmp_path / "copy.npz"; path.rename(target); path.symlink_to(target)
    with pytest.raises(ValueError, match="unsafe"): replay(tmp_path, capture, probe)


@pytest.mark.parametrize("update", [{"status": "failed"}, {"diagnostic_complete": True},
                                     {"optimizer_steps": 1}, {"heldout_predictions": 1}, {"probe_count": 192},
                                     {"numerical_gate_passed": True, "capture_complete": False}])
def test_failed_or_partial_execution_never_becomes_complete_study(tmp_path, update):
    run = {"schema": 1, "study": verify.STUDY, "status": "complete", "capture_complete": True,
           "phase": "complete", "diagnostic_complete": False, "quality_or_performance_conclusion": False,
           "probe_count": 1, "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0}
    run.update(update); write(tmp_path / "run.json", run)
    with pytest.raises(ValueError): verify.verify(tmp_path)


def provenance_fixture(root):
    legacy = REPO / "configs/ce-kl-gradient-v1.json"
    target = root / "source/configs/ce-kl-gradient-v1.json"
    target.parent.mkdir(parents=True); shutil.copyfile(legacy, target)
    required = ["qa_lab/gradient_capture.py", "qa_lab/gradient_diagnostics.py", "qa_lab/confirmation_training.py",
                "qa_lab/logits_distillation.py", "qa_lab/model_identity.py", "qa_lab/train_artifact.py", "qa_lab/common.py",
                "scripts/verify_gradient_capture.py", "scripts/replay_private_gradient_capture.py", "scripts/verify_gradient_diagnostics.py"]
    for name in required:
        path = root / "source" / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# Synthetic provenance source\n")
    protocol = {"study": verify.STUDY, "status": "frozen_before_execution",
                "probe": {"state": "initial", "seed": 20261009, "train_id": verify.PROBE_ID, "count": 1},
                "correctness": {"gradient_atol": 1e-5, "gradient_rtol": 1e-4},
                "scope": {"optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0, "quality_or_performance_claims": False, "full_192_probe_study": False},
                "budget": {"wall_seconds": 120, "max_rss_bytes": 16 * 2**30, "max_output_bytes": 128 * 2**20, "paid_resources": False},
                "source_protocol_sha256": verify.LEGACY_PROTOCOL_SHA}
    write(root / "source/configs/ce-kl-gradient-capture-v1.json", protocol)
    files = {p.relative_to(root).as_posix(): verify.digest(p) for p in root.rglob("*") if p.is_file()}
    run = {"protocol_commit": "1" * 40, "protocol_committed_utc": "2026-10-09T00:00:00+00:00",
           "started_utc": "2026-10-09T00:01:00+00:00", "finished_utc": "2026-10-09T00:01:03+00:00",
           "source_sha256": {k.removeprefix("source/"): v for k, v in files.items()},
           "protocol_sha256": verify.digest(root / "source/configs/ce-kl-gradient-capture-v1.json"),
           "protocol": protocol, "input_sha256": json.loads(legacy.read_text())["hashes"]}
    return run, files


def test_precommitted_protocol_identity_is_checked_without_claiming_authenticity(tmp_path):
    run, files = provenance_fixture(tmp_path)
    protocol, legacy = verify.check_provenance(tmp_path, run, files)
    assert protocol["correctness"] == {"gradient_atol": 1e-5, "gradient_rtol": 1e-4}
    assert legacy["selected_ids"][0] == verify.PROBE_ID
    run["protocol_committed_utc"] = "2026-10-09T00:02:00+00:00"
    with pytest.raises(ValueError, match="before execution"): verify.check_provenance(tmp_path, run, files)


def test_protocol_tolerance_mutation_rejected_with_fresh_matching_hashes(tmp_path):
    run, files = provenance_fixture(tmp_path)
    run["protocol"]["correctness"]["gradient_atol"] = .1
    path = tmp_path / "source/configs/ce-kl-gradient-capture-v1.json"
    write(path, run["protocol"])
    run["protocol_sha256"] = files["source/configs/ce-kl-gradient-capture-v1.json"] = verify.digest(path)
    run["source_sha256"]["configs/ce-kl-gradient-capture-v1.json"] = verify.digest(path)
    with pytest.raises(ValueError, match="tolerance"): verify.check_provenance(tmp_path, run, files)


def test_file_inventory_rejects_missing_hash_extra_file_and_nested_symlink(tmp_path):
    write(tmp_path / "run.json", {})
    write(tmp_path / "data.json", {"synthetic": True})
    run = {"artifacts": {"data.json": verify.digest(tmp_path / "data.json")}}
    verify.check_files(tmp_path, run)
    write(tmp_path / "extra.json", {})
    with pytest.raises(ValueError, match="hash map"): verify.check_files(tmp_path, run)
    (tmp_path / "extra.json").unlink()
    (tmp_path / "shortcut").symlink_to(tmp_path / "data.json")
    with pytest.raises(ValueError, match="symlinks"): verify.check_files(tmp_path, run)


def private_metadata_fixture():
    dimensions = {"selected_logits": ("float32", [3, 151936]),
                  "teacher_log_probs": ("float16", [3, 151936]),
                  "targets": ("int64", [3]), "input_ids": ("int64", [1, 225]),
                  "labels": ("int64", [1, 225]), "supervised_positions": ("int64", [3, 2])}
    return {"sha256": "1" * 64, "bytes": 12345, "arrays": {
        name: {"dtype": dtype, "shape": shape, "numel": int(np.prod(shape)),
               "sha256": "2" * 64, "finite": True} for name, (dtype, shape) in dimensions.items()}}


def test_private_receipt_requires_exact_token_logit_contract():
    verify.check_private_metadata(private_metadata_fixture(), {"supervised_tokens": 3, "input_tokens": 225})


@pytest.mark.parametrize("key,field,value", [
    ("selected_logits", "dtype", "float64"), ("teacher_log_probs", "dtype", "float32"),
    ("targets", "dtype", "float32"), ("input_ids", "shape", [225]),
    ("labels", "shape", [1, 224]), ("supervised_positions", "shape", [3, 1]),
    ("selected_logits", "shape", [3, 151935]), ("targets", "shape", [4]),
])
def test_private_receipt_corruption_with_consistent_numel_rejected(key, field, value):
    meta = private_metadata_fixture()
    meta["arrays"][key][field] = value
    if field == "shape": meta["arrays"][key]["numel"] = int(np.prod(value))
    with pytest.raises(ValueError, match="contract"):
        verify.check_private_metadata(meta, {"supervised_tokens": 3, "input_tokens": 225})
