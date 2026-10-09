"""Public failed-run replay and corruption tests; no real model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "reports/ce-kl-gradient-v1"
spec = importlib.util.spec_from_file_location("verify_gradient_failure", REPO / "scripts/verify_gradient_failure.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def relock(root):
    """Repair outer receipts after deliberate semantic corruption."""
    files = {key: value for key, value in fingerprint(root).items() if key != "archive-manifest.json"}
    run = verify.read(root / "run.json")
    run["artifacts"] = {key: value for key, value in files.items() if key != "run.json"}
    write(root / "run.json", run)
    files["run.json"] = verify.digest(root / "run.json")
    write(root / "archive-manifest.json", {"schema": 1, "files": files})


def test_actual_failure_archive_replay_and_nonmutation():
    before = fingerprint(ARCHIVE)
    result = verify.verify(ARCHIVE)
    assert result["status"] == "verified_failure_evidence"
    assert result["diagnostic_complete"] is False
    assert result["numerical_gate_passed"] is False
    assert result["successful_probes"] == 0 and result["failed_probes"] == 1
    assert result["failure_vectors_independently_replayed"] is False
    assert result["final_input_cache_adapter_integrity_recheck_executed"] is False
    assert result["public_passing_block"]["float32_values"] == 35840
    assert result["public_passing_block"]["independent_replay"] == "passed"
    assert set(result["hypotheses"].values()) == {None}
    assert len(result["recorded_parameter_failures"]) == 4
    assert result["recorded_logit_first_failure"] == {"flat": 304070, "supervised_position_zero_based": 2, "vocabulary_index": 198}
    assert fingerprint(ARCHIVE) == before


@pytest.fixture
def archive_copy(tmp_path):
    target = tmp_path / "archive"
    shutil.copytree(ARCHIVE, target)
    return target


@pytest.mark.parametrize("mutation", ["run_complete", "counter", "summary", "gate", "logit", "parameter",
                                       "source", "protocol", "raw", "parameter_mutation", "initial", "extra", "raw_gram"])
def test_rehashed_corruption_rejected(archive_copy, mutation):
    root = archive_copy
    path = root / "run.json"
    if mutation in ("gate", "logit", "parameter", "raw_gram"):
        path = root / "probes/000.json"
    if mutation in ("parameter_mutation", "initial"):
        path = root / "states/initial-20261009.json"
    value = verify.read(path)
    if mutation == "run_complete": value["status"] = "complete"
    if mutation == "counter": value["probes_completed"] = 1
    if mutation == "summary": write(root / "summary.json", {"diagnostic_complete": True})
    if mutation == "gate": value["correctness_passed"] = True
    if mutation == "logit": value["logit_gradient_checks"]["full_kl"]["first_failure_flat"] += 1
    if mutation == "parameter": value["parameters"][0]["reconstruction"]["full"]["atol"] = .001
    if mutation == "source": value["source_sha256"] = {}
    if mutation == "protocol": value["protocol"]["correctness"]["gradient_atol"] = .001
    if mutation == "parameter_mutation": value["parameters_after"]["adapter"]["sha256"] = "1" * 64
    if mutation == "initial": value["initial_trainable_state_sha256"] = "1" * 64
    if mutation == "extra": write(root / "unexpected.json", {"claim": "passed"})
    if mutation == "raw_gram":
        parameter = next(p for p in value["parameters"] if p["name"] == value["public_parameter"])
        parameter["gram"][0][0] *= 2
    if mutation == "raw":
        raw_path = root / "arrays/initial-20261009.npz"
        with np.load(raw_path, allow_pickle=False) as raw:
            arrays = {key: raw[key] for key in raw.files}
        arrays["ce"][0, 0] += 1.
        np.savez_compressed(raw_path, **arrays)
    write(path, value); relock(root)
    with pytest.raises(ValueError):
        verify.verify(root)


def test_frozen_source_bytes_cannot_be_rebound(archive_copy):
    path = archive_copy / "source/qa_lab/gradient_diagnostics.py"
    path.write_text(path.read_text() + "\n# changed\n")
    run = verify.read(archive_copy / "run.json")
    run["source_sha256"]["qa_lab/gradient_diagnostics.py"] = verify.digest(path)
    write(archive_copy / "run.json", run); relock(archive_copy)
    with pytest.raises(ValueError, match="source map"):
        verify.verify(archive_copy)


def test_original_complete_study_verifier_still_rejects_failed_run():
    with pytest.raises(ValueError, match="Failed/partial"):
        verify.math_verify.verify(ARCHIVE)


def test_cli_and_frozen_archive_output_guard(tmp_path):
    output = tmp_path / "verification.json"
    result = subprocess.run([sys.executable, str(REPO / "scripts/verify_gradient_failure.py"),
                             "--root", str(ARCHIVE), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["numerical_gate_passed"] is False
    before = fingerprint(ARCHIVE)
    result = subprocess.run([sys.executable, str(REPO / "scripts/verify_gradient_failure.py"),
                             "--root", str(ARCHIVE), "--output", str(ARCHIVE / "forbidden.json")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert fingerprint(ARCHIVE) == before
