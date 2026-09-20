"""Detect modified rerun evidence, including metrics with an updated file manifest."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qa_lab.common import sha
from verify_retraining import ROOT, check


def copy_evidence(tmp_path):
    target = tmp_path / 'rerun'
    shutil.copytree(ROOT / 'reports/entrypoint-retrain-20260920', target)
    return target


def test_modified_prediction_is_rejected(tmp_path):
    root = copy_evidence(tmp_path)
    prediction = next((root / 'supervised-07/run').glob('*/dev-predictions.jsonl'))
    prediction.write_text(prediction.read_text() + '{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        check(root)


def test_changed_summary_is_rejected_even_with_updated_manifest(tmp_path):
    root = copy_evidence(tmp_path)
    summary_path = root / 'supervised-07/run/summary.json'
    summary = json.loads(summary_path.read_text())
    summary[next(iter(summary))]['strict_em'] = 1.0
    summary_path.write_text(json.dumps(summary))
    manifest_path = root / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][str(summary_path.relative_to(root))] = sha(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Metrics mismatch'):
        check(root)
