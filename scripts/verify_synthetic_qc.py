"""Read-only replay of QC evidence; no model inference, training or semantic review."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa_lab.common import read_jsonl, sha
from qa_lab.quality_control import RULE_VERSION, audit

EVIDENCE_FILES = {'raw-candidates.jsonl', 'sources.jsonl', 'split-index.jsonl',
                  'records.jsonl', 'leakage.jsonl', 'accept.jsonl', 'reject.jsonl',
                  'review.jsonl', 'summary.json', 'run.json'}
CODE_FILES = {'scripts/synthetic_qc.py', 'qa_lab/quality_control.py',
              'qa_lab/data.py', 'qa_lab/chinese.py'}
HISTORICAL_ENTRYPOINT = ROOT / 'docs/maintenance/2026-09-22-detail/baseline/scripts/synthetic_qc.py.txt'
HISTORICAL_ENTRYPOINT_SHA = '6d7fac7658326c153e3c79b4e875c111586ce31c053107bfcd50d12fb22d0217'



def verify_run(folder):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('QC evidence folder must be a regular directory')
    entries = list(folder.iterdir())
    if ({p.name for p in entries} != EVIDENCE_FILES | {'manifest.json'} or
            any(p.is_symlink() or not p.is_file() for p in entries)):
        raise ValueError('QC evidence file set/hash mismatch: regular files required')
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not isinstance(manifest, dict) or set(manifest) != EVIDENCE_FILES:
        raise ValueError('QC evidence file set/hash mismatch: manifest contract')
    actual = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if actual != manifest:
        raise ValueError('QC evidence file set/hash mismatch')
    receipt = json.loads((folder / 'run.json').read_text())
    if (not isinstance(receipt, dict) or receipt.get('rule_version') != RULE_VERSION or
            not isinstance(receipt.get('code_sha256'), dict) or set(receipt['code_sha256']) != CODE_FILES or
            not isinstance(receipt.get('source_files'), dict)):
        raise ValueError('QC receipt binding contract mismatch')
    for name, digest in receipt['source_files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('QC source changed: ' + name)
    for name, digest in receipt['code_sha256'].items():
        source = ROOT / name
        if name == 'scripts/synthetic_qc.py' and digest == HISTORICAL_ENTRYPOINT_SHA:
            source = HISTORICAL_ENTRYPOINT
        if source.is_symlink() or not source.is_file() or sha(source) != digest:
            raise ValueError('QC implementation changed: ' + name)
    origin = receipt.get('input_origin')
    legacy = receipt['code_sha256']['scripts/synthetic_qc.py'] == HISTORICAL_ENTRYPOINT_SHA
    if origin is None and legacy and 'input_origin' not in receipt:
        pass  # Legacy receipts predate this field; do not infer a provenance claim.
    elif origin == {'kind': 'custom_unverified'}:
        pass
    elif (isinstance(origin, dict) and set(origin) == {'kind', 'name'} and
          origin['kind'] == 'historical_teacher_preset' and isinstance(origin['name'], str) and
          origin['name'] in {'english-original', 'english-prompted', 'chinese'}):
        from scripts.synthetic_qc import preset
        sources, candidates, index, language, paths = preset(origin['name'])
        if receipt['label_permission'] == 'none':
            sources = [{k:v for k,v in r.items() if k not in {'answers', 'is_impossible'}} for r in sources]
        expected_files = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        if (receipt['language'] != language or receipt['source_files'] != expected_files or
                read_jsonl(folder / 'sources.jsonl') != sources or
                read_jsonl(folder / 'raw-candidates.jsonl') != candidates or
                read_jsonl(folder / 'split-index.jsonl') != index):
            raise ValueError('QC preset origin does not match input and source bindings')
    else:
        raise ValueError('QC input origin contract mismatch')
    result = audit(read_jsonl(folder / 'sources.jsonl'), read_jsonl(folder / 'raw-candidates.jsonl'),
                   read_jsonl(folder / 'split-index.jsonl'), language=receipt['language'],
                   label_permission=receipt['label_permission'])
    for name in ('records', 'leakage'):
        if read_jsonl(folder / (name + '.jsonl')) != result[name]:
            raise ValueError('QC replay mismatch: ' + name)
    if json.loads((folder / 'summary.json').read_text()) != result['summary']:
        raise ValueError('QC summary mismatch')
    for decision in ('accept', 'reject', 'review'):
        if read_jsonl(folder / (decision + '.jsonl')) != [
                r for r in result['records'] if r['decision'] == decision]:
            raise ValueError('QC decision partition mismatch')
    return result['summary']


def verify(root):
    from scripts.synthetic_qc import preset
    expected = {f'{name}-{permission}' for name in
                ('english-original', 'english-prompted', 'chinese')
                for permission in ('none', 'train_reference')}
    if {p.name for p in root.iterdir() if p.is_dir()} != expected:
        raise ValueError('Expected all three presets in both permission modes')
    for name in ('english-original', 'english-prompted', 'chinese'):
        sources, candidates, index, lang, _ = preset(name)
        for permission in ('none', 'train_reference'):
            folder = root / f'{name}-{permission}'
            rows = sources if permission == 'train_reference' else [
                {k: v for k, v in r.items() if k not in {'answers', 'is_impossible'}} for r in sources]
            for filename, values in [('sources', rows), ('raw-candidates', candidates), ('split-index', index)]:
                if read_jsonl(folder / (filename + '.jsonl')) != values:
                    raise ValueError('QC snapshot is not the original preset input: ' + filename)
            receipt = json.loads((folder / 'run.json').read_text())
            if receipt['label_permission'] != permission or receipt['language'] != lang:
                raise ValueError('QC preset permission/language mismatch')
    results = {p.name: verify_run(p) for p in sorted(root.iterdir()) if p.is_dir()}
    accepted = read_jsonl(root / 'chinese-train_reference/accept.jsonl')
    old = read_jsonl(ROOT / 'reports/quality-study-20260920/supervised-07/oracle_selected.jsonl')
    current = {r['sample_id']: (r['source']['context'], r['source']['question'], r['output_response'])
               for r in accepted}
    historical = {r['id']: (r['context'], r['question'], r['target']) for r in old}
    if current != historical or len(accepted) != len(old):
        raise ValueError('Historical selected training group is not equivalent')
    return dict(status='pass', runs=results, historical_chinese_targets_equal=True,
                note='Saved-record replay, not new generation, training or semantic accuracy.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT / 'docs/synthetic-qc-20260922/evidence')
    args = p.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
