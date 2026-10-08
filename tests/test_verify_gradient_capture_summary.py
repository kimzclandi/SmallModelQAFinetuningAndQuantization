import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture_summary", ROOT / "scripts/verify_gradient_capture_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_published_receipt_does_not_claim_public_real_array_replay():
    result = summary.verify(ROOT / "reports/ce-kl-gradient-capture-v1")
    assert result["real_tensor_replay_in_ci"] is False
    assert result["original_numerical_gate_passed"] is False


@pytest.mark.parametrize("kind", ["gate", "privacy", "model_count", "tolerance", "raw_file"])
def test_invalid_claims_are_rejected_even_after_manifest_rehash(tmp_path, kind):
    root = tmp_path / "report"
    shutil.copytree(ROOT / "reports/ce-kl-gradient-capture-v1", root)
    name = "execution-02-summary.json"
    value = json.loads((root / name).read_text())
    if kind == "gate": value["numerical_gate_passed"] = True
    elif kind == "privacy": value["ci_replays_real_tensor_values"] = True
    elif kind == "model_count": value["model_loads"] = 2
    elif kind == "raw_file": (root / "raw.npz").write_bytes(b"unreviewed")
    else:
        name = "same-input-replay-summary.json"
        value = json.loads((root / name).read_text()); value["tolerance"]["atol"] = 1.
    (root / name).write_text(json.dumps(value))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): summary.verify(root)
