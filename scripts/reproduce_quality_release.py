"""Small actual CPU smoke or full fixed MPS training from verified published inputs."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'reports/quality-study-20260920'
sys.path.insert(0, str(ROOT))
from qa_lab.common import read_jsonl, write_json, sha
from verify_quality_release import verify_manifest


def smoke(cfg, release, output):
    from qa_lab.inference import load, generate, QAInput
    import torch
    torch.set_num_threads(4)
    config = dict(cfg['student'], device='cpu')
    tok, model = load(config)
    rows = read_jsonl(release / 'fresh-05/dev.jsonl')[:2]
    with (output / 'predictions.jsonl').open('x') as stream:
        for row in rows:
            prediction = dict(id=row['id'], **generate(
                QAInput(row['context'], row['question']), config, tok, model))
            stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            stream.flush()
    return dict(n=len(rows), config=config,
                scope='Actual CPU inference only; not quality validation or training')


def retrain(cfg, release, output):
    from supervised_quality_control import run as train
    from verify_supervised_quality import verify
    dest = output / 'supervised-07'
    dest.mkdir()
    fresh = output / 'fresh-05'
    fresh.mkdir()
    (fresh / 'train.jsonl').write_bytes((release / 'fresh-05/train.jsonl').read_bytes())
    shutil.copytree(release / 'fresh-05/teacher-train', fresh / 'teacher-train')
    for name in [g + '.jsonl' for g in cfg['groups']] + ['dev.jsonl']:
        (dest / name).write_bytes((release / 'supervised-07' / name).read_bytes())
    cfg.update(created_utc=datetime.now(timezone.utc).isoformat(),
               reproduces_protocol_sha256=sha(release / 'supervised-07/protocol.json'),
               code_hashes={n: sha(ROOT / 'scripts' / n) for n in cfg['code_hashes']})
    write_json(dest / 'protocol.json', cfg)
    write_json(dest / 'manifest.json', {p.name: sha(p) for p in sorted(dest.iterdir()) if p.is_file()})
    train(output)
    write_json(output / 'verification.json', verify(output))
    return dict(scope='Fresh fixed-protocol training and saved-result verification')


def run(stage, output):
    if stage not in ('smoke', 'train'):
        raise ValueError('Unknown stage: ' + stage)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if any(output.is_relative_to((ROOT / name).resolve()) for name in ('reports', 'data', 'configs')):
        raise ValueError('Output must not be inside frozen reports/data/configs; use a new work/ directory')
    # Verify before model imports, output creation or expensive computation.
    release_hash = verify_manifest(RELEASE)
    cfg = json.loads((RELEASE / 'supervised-07/protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', stage=stage, release_manifest_sha256=release_hash,
                   runner_sha256=sha(Path(__file__)), started_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'receipt.json', receipt)
    try:
        receipt.update((smoke if stage == 'smoke' else retrain)(cfg, RELEASE, output))
        receipt['status'] = 'complete'
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'train'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.output)
