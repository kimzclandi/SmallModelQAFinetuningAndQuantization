"""Reference-supervised quality control, not a label-free selection method."""
import argparse,json,sys,random,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from resumable_teacher import checked_record,record_path,digest
from semantic_quality_pilot import train_group

def groups(rows,preds,seed):
    ids=[r['id'] for r in rows];pi=[p['id'] for p in preds]
    if len(set(ids))!=len(ids) or len(set(pi))!=len(pi) or set(ids)!=set(pi):raise ValueError('Input coverage')
    by={p['id']:p for p in preds}
    pool=[r for r in rows if by[r['id']]['stop_reason']=='eos' and by[r['id']]['prediction'].strip() not in ('','NO_ANSWER') and by[r['id']]['prediction'].strip() in r['context']]
    clean=[r for r in pool if by[r['id']]['prediction'].strip() in [a.strip() for a in r['answers']]]
    if len(clean)<8:raise ValueError('Insufficient clean reference matches')
    control=random.Random(seed).sample(pool,len(clean))
    return {name:[dict(r,target=(r['answers'][0] if name=='random_gold' else by[r['id']]['prediction'].strip())) for r in subset] for name,subset in [('oracle_selected',clean),('random_teacher',control),('random_gold',control)]}

def prepare(root):
    src=root/'fresh-05';out=root/'supervised-07';out.mkdir(exist_ok=False)
    rows=read_jsonl(src/'train.jsonl');cache=src/'teacher-train';binding=digest(json.loads((cache/'manifest.json').read_text()))
    preds=[checked_record(record_path(cache,r['id']),r,binding) for r in rows];g=groups(rows,preds,20260920)
    for name,data in g.items():write_jsonl(out/(name+'.jsonl'),data)
    (out/'dev.jsonl').write_bytes((src/'dev.jsonl').read_bytes())
    base=json.loads((root/'semantic-02/protocol.json').read_text());cfg={k:base[k] for k in ['seeds','steps','learning_rate','lora_r','lora_alpha','target_modules']}
    cfg.update(student=json.loads((src/'data_protocol.json').read_text())['student'],created_utc=datetime.now(timezone.utc).isoformat(),groups=list(g),subset_seed=20260920,max_wall_seconds=1800,
       source_hashes={str(p.relative_to(root)):sha(p) for p in [src/'train.jsonl',cache/'manifest.json',cache/'complete.json']},
       code_hashes={n:sha(Path(__file__).with_name(n)) for n in ['supervised_quality_control.py','semantic_quality_pilot.py','quality_pilot.py']},
       contrasts={'primary':'random_gold minus random_teacher: identical examples/order, only reference targets substituted; token lengths may differ','secondary':'oracle_selected minus random_teacher: equal sample count, selection and example difficulty confounded'},
       budget='3 groups x 3 seeds x 64 updates; equal updates, not equal tokens; fixed subset across seeds.',
       interpretation='Exploratory reused development data; no holdout access, no label-free claim, no independent validation. Do not select seed, adjust parameters, or adopt model from this experiment.',
       selection='EOS non-refusal extractive pool; oracle exact match against any training reference; random_gold first reference; all choices frozen before training.')
    write_json(out/'protocol.json',cfg)
    write_json(out/'group-audit.json',{n:dict(n=len(data),reference_match=sum(r['target'] in r['answers'] for r in data)) for n,data in g.items()})
    write_json(out/'manifest.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})

def run(root):
    import torch
    out=root/'supervised-07'
    for n,h in json.loads((out/'manifest.json').read_text()).items():
        if sha(out/n)!=h:raise ValueError('Frozen input changed')
    cfg=json.loads((out/'protocol.json').read_text())
    for n,h in cfg['code_hashes'].items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('Source changed')
    dest=out/'run';dest.mkdir(exist_ok=False);dev=read_jsonl(out/'dev.jsonl');torch.set_num_threads(8);start=time.monotonic();meta={'status':'running'};summary={}
    write_json(dest/'receipt.json',meta)
    try:
        for seed in cfg['seeds']:
            for name in cfg['groups']:
                summary[f'{name}-{seed}']=train_group(read_jsonl(out/(name+'.jsonl')),dev,cfg,seed,dest/f'{name}-{seed}',start+cfg['max_wall_seconds'])
                write_json(dest/'summary.json',summary)
        meta['status']='complete'
    except BaseException as e:meta.update(status='failed',error=repr(e));raise
    finally:
        meta.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'receipt.json',meta)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('root',type=Path);a=p.parse_args();(prepare if a.stage=='prepare' else run)(a.root)
