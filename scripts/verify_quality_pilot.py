"""Read-only recomputation of a completed quality pilot, including sample pairing."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha
from qa_lab.chinese import evaluate_zh

def verify(root):
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'data_manifest.json').read_text())
    for name,h in manifest['files'].items():
        if sha(root/name)!=h:raise ValueError('Input hash mismatch '+name)
    run=root/'run';receipt=json.loads((run/'run.json').read_text())
    if receipt['status']!='complete':raise ValueError('Run not complete')
    if receipt['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Protocol mismatch')
    splits={s:read_jsonl(root/(s+'.jsonl')) for s in ['train','dev','holdout']}
    seen_ids=set();seen_families=set()
    for name,rows in splits.items():
        ids={r['id'] for r in rows};families={r['family_id'] for r in rows}
        if len(ids)!=len(rows) or ids&seen_ids or families&seen_families:raise ValueError('Split overlap')
        if any(r['split']!=name for r in rows):raise ValueError('Split label mismatch')
        seen_ids|=ids;seen_families|=families
    train=splits['train'];teacher=read_jsonl(run/'teacher-train.jsonl');by={p['id']:p for p in teacher}
    if len(by)!=len(teacher) or set(by)!={r['id'] for r in train}:raise ValueError('Teacher coverage')
    kept=[r for r in train if by[r['id']]['stop_reason']=='eos' and by[r['id']]['prediction'].strip() and by[r['id']]['prediction'].strip()!='NO_ANSWER' and by[r['id']]['prediction'].strip() in r['context']]
    random_rows=random.Random(protocol['seed']).sample(train,len(kept))
    expected={'raw':train,'filtered':kept,'random':random_rows,'gold':train}
    group_manifest=json.loads((run/'groups_manifest.json').read_text())
    summary=json.loads((run/'summary.json').read_text());result={}
    for group in ['baseline']+list(expected):
        if group!='baseline':
            path=run/(group+'-training.jsonl');rows=read_jsonl(path)
            if sha(path)!=group_manifest[group]['sha256'] or len(rows)!=group_manifest[group]['n']:raise ValueError('Training hash/count')
            if [r['id'] for r in rows]!=[r['id'] for r in expected[group]]:raise ValueError('Wrong subset')
            for r,source in zip(rows,expected[group]):
                target=source['answers'][0] if group=='gold' else by[r['id']]['prediction'].strip()
                if r['target']!=target or r['context']!=source['context'] or r['question']!=source['question']:raise ValueError('Wrong training content')
            meta=json.loads((run/group/'training.json').read_text());steps=json.loads((run/group/'steps.json').read_text())
            if meta['status']!='complete' or len(steps)!=protocol['steps'] or [s['step'] for s in steps]!=list(range(1,protocol['steps']+1)):raise ValueError('Incomplete training')
            if any(s['id'] not in {r['id'] for r in rows} for s in steps):raise ValueError('Unexpected training ID')
            if sha(run/group/'adapter_model.safetensors')!=meta['adapter_sha256']:raise ValueError('Adapter changed')
            for key,source in [('total_input_tokens','input_tokens'),('total_supervised_tokens','supervised_tokens')]:
                if sum(s[source] for s in steps)!=meta[key]:raise ValueError('Training token denominator')
        predictions=read_jsonl(run/'baseline-dev.jsonl' if group=='baseline' else run/group/'dev-predictions.jsonl')
        metrics,_=evaluate_zh(splits['dev'],predictions)
        if metrics!=summary[group]:raise ValueError('Metric mismatch '+group)
        result[group]=metrics
    return dict(status='pass',metrics=result,train_n=len(train),filtered_n=len(kept),holdout_n=len(splits['holdout']),note='All generated evaluation IDs are development IDs; no holdout inference in this script. Accuracy is exploratory, not independent test evidence.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),ensure_ascii=False,indent=2))
