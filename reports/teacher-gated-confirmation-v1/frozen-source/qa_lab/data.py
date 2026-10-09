"""Group related paragraphs/questions BEFORE splitting. Never inspect model results."""
import argparse
import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from .common import digest, sha, write_json, write_jsonl


def norm(s):
    return ' '.join(re.findall(r'\w+', s.lower()))


def jaccard(a, b):
    a, b = set(norm(a).split()), set(norm(b).split())
    return len(a & b) / len(a | b) if a | b else 1.0


def related(a, b, cfg):
    return (a['context'] == b['context']
            or jaccard(a['context'], b['context']) >= cfg['context_jaccard']
            or jaccard(a['question'], b['question']) >= cfg['question_jaccard']
            or SequenceMatcher(None, norm(a['question']), norm(b['question']), autojunk=False).ratio() >= cfg['question_sequence_ratio'])


def prepare(raw, cfg):
    article = next(a for a in raw['data'] if a['title'] == cfg['title'])
    rows, seen, removed = [], set(), []
    for p in article['paragraphs']:
        for q in p['qas']:
            key = (norm(p['context']), norm(q['question']))
            if key in seen:
                removed.append(q['id'])
                continue
            seen.add(key)
            answers = list(dict.fromkeys(a['text'] for a in q['answers']))
            assert bool(answers) != q['is_impossible']
            for ans in q['answers']:
                assert p['context'][ans['answer_start']:ans['answer_start'] + len(ans['text'])] == ans['text']
            rows.append(dict(id=q['id'], source_title=article['title'], context=p['context'],
                             context_id=digest(norm(p['context']))[:16], question=q['question'],
                             answers=answers, is_impossible=q['is_impossible']))
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = []
    for i in range(len(rows)):
        for j in range(i):
            if related(rows[i], rows[j], cfg):
                parent[root(i)] = root(j)
                if rows[i]['context_id'] != rows[j]['context_id']:
                    edges.append([rows[i]['id'], rows[j]['id']])
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    families = sorted(groups.values(), key=lambda g: digest(str(cfg['seed']) + min(r['id'] for r in g)))
    train_end = int(len(families) * cfg['train_fraction'])
    dev_end = train_end + int(len(families) * cfg['dev_fraction'])
    for k, group in enumerate(families):
        split = 'train' if k < train_end else 'dev' if k < dev_end else 'test'
        family_id = digest('|'.join(sorted(r['id'] for r in group)))[:16]
        for r in group:
            r.update(split=split, family_id=family_id)
    return sorted(rows, key=lambda r: r['id']), dict(exact_duplicates_removed=removed, cross_context_edges=edges,
                                                     family_count=len(families), source_paragraph_count=len(article['paragraphs']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('work/squad-dev-v2.0.json'))
    p.add_argument('--output', type=Path, default=Path('data/complexity-v1'))
    args = p.parse_args()
    cfg = json.loads(Path('configs/data.json').read_text())
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a frozen dataset; use a new output directory.')
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if not args.raw.exists():
        urllib.request.urlretrieve(cfg['url'], args.raw)
    assert sha(args.raw) == cfg['sha256'], 'Source changed; do not silently reuse dataset version.'
    rows, audit = prepare(json.loads(args.raw.read_text()), cfg)
    args.output.mkdir(parents=True)
    for split in ['train', 'dev', 'test']:
        write_jsonl(args.output / f'{split}.jsonl', [r for r in rows if r['split'] == split])
    write_jsonl(args.output / 'samples.jsonl', [r for r in rows if r['split'] == 'train'][:4])
    write_json(args.output / 'manifest.json', dict(config=cfg, audit=audit,
        note='Local splits of public SQuAD dev; NOT official SQuAD test. Same-article paragraph-family isolation.',
        counts={s: dict(n=sum(r['split']==s for r in rows),
                        unanswerable=sum(r['split']==s and r['is_impossible'] for r in rows),
                        families=len({r['family_id'] for r in rows if r['split']==s})) for s in ['train','dev','test']},
        files={p.name: sha(p) for p in sorted(args.output.glob('*.jsonl'))}))
    print(json.dumps(json.loads((args.output/'manifest.json').read_text())['counts'], indent=2))

if __name__ == '__main__':
    main()
