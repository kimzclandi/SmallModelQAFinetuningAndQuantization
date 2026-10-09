"""Audit received text without silently editing labels or admitting it to training."""
import argparse
import json
from pathlib import Path
from .common import read_jsonl,sha,write_json,write_jsonl
from .teacher_io import training_rows,validate
from .metrics import evaluate


def parse_received(body):
    """Only repair the exact JSON string NO\\_ANSWER, never arbitrary malformed JSON."""
    try:
        return json.loads(body),0
    except json.JSONDecodeError:
        token=r'"NO\_ANSWER"'
        n=body.count(token)
        if not n:
            raise
        return json.loads(body.replace(token,'"NO_ANSWER"')),n


def audit(folder,pack_dir,response):
    manifest=json.loads((pack_dir/'manifest.json').read_text())
    rows={r['id']:r for r in training_rows(folder)}
    if sha(folder/'train.jsonl')!=manifest['source_sha256']:
        raise ValueError('Training source mismatch')
    for name,h in manifest['files'].items():
        if sha(pack_dir/name)!=h:
            raise ValueError('Request pack modified')
    request=json.loads((pack_dir/'request.json').read_text())['examples']
    if [r['id'] for r in request]!=manifest['ids']:
        raise ValueError('Request ID mismatch')
    for r in request:
        if r['id'] not in rows or r!={k:rows[r['id']][k] for k in ['id','context','question']}:
            raise ValueError('Non-training or modified example')
    validate(response,request)
    selected=[rows[r['id']] for r in request]
    preds=[{'id':r['id'],'prediction':r['answer']} for r in response['answers']]
    metrics,scored=evaluate(selected,preds)
    return metrics,scored,preds


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',required=True,type=Path,help='Exact received message: first line model label, remainder JSON')
    p.add_argument('--pack',type=Path,default=Path('data/teacher-pilot-v1'))
    p.add_argument('--data-dir',type=Path,default=Path('data/complexity-v1'))
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    _,body=a.raw.read_text().split('\n',1)
    response,repairs=parse_received(body)
    metrics,scored,preds=audit(a.data_dir,a.pack,response)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'metrics.json',metrics)
    write_jsonl(a.output/'scored.jsonl',scored)
    write_jsonl(a.output/'predictions.jsonl',preds)
    write_json(a.output/'audit.json',{'raw_sha256':sha(a.raw),'repaired_exact_strings':repairs,
        'teacher_identity':'User-reported; not independently verified',
        'meaning':'Agreement with gold on selected TRAINING examples; not teacher test performance or student improvement',
        'training_admission':False})

if __name__=='__main__':main()
