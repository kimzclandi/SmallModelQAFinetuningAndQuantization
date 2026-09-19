"""Recompute controlled-training data, updates, artifacts and paired dev contrasts."""
import argparse,json,math,random,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha
from qa_lab.chinese import evaluate_zh
from supervised_quality_control import groups
from resumable_teacher import checked_record,record_path,digest

def verify(root):
    out=root/'supervised-07';cfg=json.loads((out/'protocol.json').read_text())
    for n,h in json.loads((out/'manifest.json').read_text()).items():
        if sha(out/n)!=h:raise ValueError('Frozen input changed')
    for n,h in cfg['source_hashes'].items():
        if sha(root/n)!=h:raise ValueError('Upstream source changed')
    cache=root/'fresh-05/teacher-train';rows=read_jsonl(root/'fresh-05/train.jsonl');binding=digest(json.loads((cache/'manifest.json').read_text()))
    expected=groups(rows,[checked_record(record_path(cache,r['id']),r,binding) for r in rows],cfg['subset_seed'])
    run=out/'run';receipt=json.loads((run/'receipt.json').read_text())
    if receipt['status']!='complete':raise ValueError('Incomplete run')
    dev=read_jsonl(out/'dev.jsonl');summary=json.loads((run/'summary.json').read_text());scores={};tokens={}
    if len({r['family_id'] for r in dev})!=len(dev):raise ValueError('Article bootstrap needs one row per article')
    for group,rows in expected.items():
        if rows!=read_jsonl(out/(group+'.jsonl')):raise ValueError('Group mismatch')
        for seed in cfg['seeds']:
            key=f'{group}-{seed}';folder=run/key;meta=json.loads((folder/'training.json').read_text());steps=json.loads((folder/'steps.json').read_text())
            if meta['status']!='complete' or meta['seed']!=seed or meta['steps']!=cfg['steps'] or len(steps)!=cfg['steps']:raise ValueError('Incomplete updates')
            order=list(rows);random.Random(seed).shuffle(order)
            for i,s in enumerate(steps):
                if s['step']!=i+1 or s['id']!=order[i%len(order)]['id']:raise ValueError('Training order')
                if not math.isfinite(s['loss']) or not math.isfinite(s['grad_norm']):raise ValueError('Nonfinite training')
            if sha(folder/'adapter_model.safetensors')!=meta['adapter_sha256']:raise ValueError('Adapter changed')
            for total,field in [('total_input_tokens','input_tokens'),('total_supervised_tokens','supervised_tokens')]:
                if meta[total]!=sum(s[field] for s in steps):raise ValueError('Token accounting')
            metrics,items=evaluate_zh(dev,read_jsonl(folder/'dev-predictions.jsonl'))
            if metrics!=summary[key] or metrics!=json.loads((folder/'metrics.json').read_text()) or items!=read_jsonl(folder/'scored.jsonl'):raise ValueError('Metrics mismatch')
            scores[key]=items;tokens[key]={k:meta[k] for k in ['total_input_tokens','total_supervised_tokens','elapsed_seconds']}
    if set(scores)!=set(summary):raise ValueError('Unexpected model summary')
    aggregates={g:{m:statistics.mean(summary[f'{g}-{s}'][m] for s in cfg['seeds']) for m in ['strict_em','char_lcs_f1']} for g in cfg['groups']};paired={}
    for group in ['random_gold','oracle_selected']:
        for metric in ['strict_em','char_lcs_f1']:
            values=[statistics.mean(scores[f'{group}-{s}'][i][metric]-scores[f'random_teacher-{s}'][i][metric] for s in cfg['seeds']) for i in range(len(dev))]
            rng=random.Random(20260920);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
            paired[group+':'+metric]=dict(delta=statistics.mean(values),percentile_95=[samples[249],samples[9749]],scope='Exploratory article bootstrap conditional on fixed subsets/seeds; no multiplicity correction or independent-test claim')
    return dict(status='pass',runs=len(scores),updates=len(scores)*cfg['steps'],dev_n=len(dev),groups=aggregates,paired=paired,token_totals=tokens)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
