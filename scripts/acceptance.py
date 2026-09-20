"""One offline acceptance entry point shared by local checks and CI.

Requires Python 3.12 and pytest only; never loads models or downloads data.
New outputs must live below work/. Frozen trees are fingerprinted before/after.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ('reports', 'data', 'configs')
            for p in sorted((ROOT / name).rglob('*')) if p.is_file()}


def main():
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
        raise SystemExit('Assertions must be enabled; remove -O and PYTHONOPTIMIZE.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'work') or output == ROOT / 'work':
        parser.error('Output must be a new subdirectory of repository work/.')
    output.mkdir(parents=True, exist_ok=False)
    before = fingerprint()
    env = dict(os.environ, PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1',
               HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    checks = [('tests', ['-m', 'pytest', '-q'])]
    checks += [(name, [f'scripts/verify_{name}.py']) for name in
               ('artifacts', 'closure', 'teacher_study', 'coverage', 'quantization', 'chinese')]
    checks += [('quality_release', ['scripts/verify_quality_release.py']),
               ('retraining', ['scripts/verify_retraining.py']),
               ('external_drcd', ['scripts/verify_external_drcd.py'])]
    checks += [('gold_pilot', ['scripts/compare_gold_pilot.py', '--output', str(output/'gold.json')]),
               ('teacher_audit', ['-m', 'qa_lab.teacher_audit', '--raw',
                'data/teacher-received-v1/user-message.txt', '--output', str(output/'teacher-audit')])]
    result = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'executable': sys.executable,
              'scope': 'Saved evidence recomputation and tests only; no model inference.', 'checks': []}
    try:
        for name, argv in checks:
            command = [sys.executable, *argv]
            with (output/f'{name}.log').open('w') as log:
                run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            result['checks'].append({'name': name, 'command': command, 'returncode': run.returncode})
            print(f'{name}: {"PASS" if run.returncode == 0 else "FAIL"}', flush=True)
    finally:
        result['frozen_unchanged'] = before == fingerprint()
        result['passed'] = len(result['checks']) == len(checks) and all(
            c['returncode'] == 0 for c in result['checks']) and result['frozen_unchanged']
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        (output/'frozen-sha256.json').write_text(json.dumps(before, indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
