"""Recompute the saved public-entrypoint rerun; no model execution or weights."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qa_lab.common import sha, read_jsonl
from verify_quality_release import verify_manifest
from verify_supervised_quality import verify


def check(root):
    verify_manifest(root)
    receipt = json.loads((root / 'receipt.json').read_text())
    if receipt['status'] != 'complete' or receipt['stage'] != 'train':
        raise ValueError('Incomplete training entrypoint')
    original = ROOT / 'reports/quality-study-20260920'
    if receipt['release_manifest_sha256'] != sha(original / 'release-manifest.json'):
        raise ValueError('Wrong source release')
    cfg = json.loads((root / 'supervised-07/protocol.json').read_text())
    reference = json.loads((original / 'supervised-07/protocol.json').read_text())
    if cfg['reproduces_protocol_sha256'] != sha(original / 'supervised-07/protocol.json'):
        raise ValueError('Wrong source protocol')
    for key, value in reference.items():
        if key not in ('created_utc', 'code_hashes') and cfg.get(key) != value:
            raise ValueError('Changed experiment configuration: ' + key)
    actual = verify(root, require_adapters=False)
    recorded = json.loads((root / 'verification.json').read_text())
    # Local completion checked weights; the public package deliberately excludes them.
    if recorded['adapter_weights_checked'] is not True:
        raise ValueError('Missing original local adapter check')
    recorded['adapter_weights_checked'] = False
    if actual != recorded:
        raise ValueError('Saved rerun verification mismatch')
    comparisons = {}
    summary = json.loads((root / 'supervised-07/run/summary.json').read_text())
    for key, metric in summary.items():
        before = original / 'supervised-07/run' / key
        after = root / 'supervised-07/run' / key
        a = read_jsonl(before / 'dev-predictions.jsonl')
        b = read_jsonl(after / 'dev-predictions.jsonl')
        if [r['id'] for r in a] != [r['id'] for r in b]:
            raise ValueError('Comparison sample alignment')
        comparisons[key] = dict(original=json.loads((before / 'metrics.json').read_text()),
                                rerun=metric, n=len(a), prediction_text_matches=sum(
                                    x['prediction'] == y['prediction'] for x, y in zip(a, b)))
    if comparisons != json.loads((root / 'comparison.json').read_text()):
        raise ValueError('Comparison mismatch')
    return actual


if __name__ == '__main__':
    print(json.dumps(check(ROOT / 'reports/entrypoint-retrain-20260920'), indent=2))
