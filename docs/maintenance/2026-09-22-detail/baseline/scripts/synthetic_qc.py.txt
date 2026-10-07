"""Offline extractive candidate QC. Never generates responses or runs training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa_lab.common import read_jsonl, write_json, write_jsonl, sha
from qa_lab.quality_control import INDEX_KEYS, RULE_VERSION, audit
from scripts.report_output import create_output_directory


def preset(name):
    """Project split inputs immediately; evaluation labels never enter QC."""
    paths = []

    def read(path, jsonl=False):
        path = ROOT / path
        paths.append(path)
        return read_jsonl(path) if jsonl else json.loads(path.read_text())

    if name == 'chinese':
        base = Path('reports/quality-study-20260920/fresh-05')
        sources = read(base / 'train.jsonl', True)
        cache = base / 'teacher-train'
        manifest = read(cache / 'manifest.json')
        config = manifest['config']
        compact = lambda x: hashlib.sha256(json.dumps(
            x, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if manifest['input_sha256'] != compact(sources):
            raise ValueError('Cache source binding mismatch')
        predictions = []
        for row in sources:
            key = hashlib.sha256(row['id'].encode()).hexdigest() + '.json'
            record = read(cache / 'records' / key)
            p = record['prediction']
            if (record['input_sha256'] != compact(row) or record['config_sha256'] != compact(manifest)
                    or p['id'] != row['id'] or record['prediction_sha256'] != compact(p)):
                raise ValueError('Cache record binding mismatch')
            predictions.append(p)
        lang, eval_splits, run_id = 'zh', ('dev', 'holdout'), str(cache)
    else:
        base = Path('data/complexity-v1')
        sources = read(base / 'train.jsonl', True)
        folder = Path('reports/closure-v1/teacher-fp32' if name == 'english-original'
                      else 'reports/teacher-study-v2/teacher')
        predictions = read(folder / 'predictions.jsonl', True)
        config = read(folder / 'run.json')['config']
        lang, eval_splits, run_id = 'en', ('dev', 'test'), str(folder)
    index = [{k: r[k] for k in sorted(INDEX_KEYS)} for r in sources]
    for split in eval_splits:
        # Existing files co-locate labels and inputs. Only these five allowed
        # fields survive projection; no answer or answerability lookup here.
        index.extend({k: r[k] for k in sorted(INDEX_KEYS)}
                     for r in read(base / (split + '.jsonl'), True))
    ids = {p['id'] for p in predictions}
    sources = [r for r in sources if r['id'] in ids]
    provenance = dict(model_id=config.get('model_id', 'unknown'),
                      revision=config.get('revision', 'unknown'),
                      prompt_version=config.get('prompt_version', 'unknown'),
                      generation_parameters=config, run_id=run_id)
    predictions = [dict(p, provenance=provenance) for p in predictions]
    return sources, predictions, index, lang, paths


def run(sources, candidates, index, language, permission, output, paths=()):
    result = audit(sources, candidates, index, language=language, label_permission=permission)
    output = create_output_directory(output)
    write_jsonl(output / 'raw-candidates.jsonl', candidates)
    write_jsonl(output / 'sources.jsonl', sources)
    write_jsonl(output / 'split-index.jsonl', index)
    write_jsonl(output / 'records.jsonl', result['records'])
    write_jsonl(output / 'leakage.jsonl', result['leakage'])
    for decision in ('accept', 'reject', 'review'):
        write_jsonl(output / (decision + '.jsonl'),
                    [r for r in result['records'] if r['decision'] == decision])
    write_json(output / 'summary.json', result['summary'])
    write_json(output / 'run.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        rule_version=RULE_VERSION, language=language, label_permission=permission,
        source_files={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                      for p in paths},
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                     [Path(__file__), ROOT / 'qa_lab/quality_control.py',
                      ROOT / 'qa_lab/data.py', ROOT / 'qa_lab/chinese.py']},
        scope='Real saved teacher candidates; new rule execution, no new generation/training.',
        evaluation_access='Input-only projection for split isolation; no evaluation labels in QC.'))
    write_json(output / 'manifest.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    return result['summary']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=['english-original', 'english-prompted', 'chinese'])
    p.add_argument('--sources', type=Path)
    p.add_argument('--candidates', type=Path)
    p.add_argument('--split-index', type=Path)
    p.add_argument('--language', choices=['en', 'zh'])
    p.add_argument('--label-permission', choices=['none', 'train_reference'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.preset:
        if any((a.sources, a.candidates, a.split_index, a.language)):
            p.error('Do not combine preset with custom inputs')
        sources, candidates, index, lang, paths = preset(a.preset)
        if a.label_permission == 'none':
            sources = [{k: v for k, v in r.items() if k not in {'answers', 'is_impossible'}}
                       for r in sources]
    else:
        if not all((a.sources, a.candidates, a.split_index, a.language)):
            p.error('Custom inputs require sources, candidates, split-index and language')
        paths = [a.sources.resolve(), a.candidates.resolve(), a.split_index.resolve()]
        sources, candidates, index = [read_jsonl(v) for v in paths]
        lang = a.language
    print(json.dumps(run(sources, candidates, index, lang, a.label_permission, a.output, paths),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
