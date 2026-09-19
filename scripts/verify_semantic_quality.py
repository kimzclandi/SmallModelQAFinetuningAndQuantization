"""Read-only verification and paired development analysis for semantic quality pilot."""
import argparse,json,random,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha
from qa_lab.chinese import evaluate_zh

def verify(root):
    for name,h in json.loads((root/'manifest.json').read_text()).items():
        if sha(root/name)!=h:raise ValueError('Input hash mismatch '+name)
    cfg=json.loads((root/'protocol.json').read_text());run=root/'run';meta=json.loads((run/'run.json').read_text())
    if meta['status']!='complete':raise ValueError('Not a completed training run: '+meta['status'])
    if meta['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Protocol changed')
    train=read_jsonl(root/'train.jsonl');dev=read_jsonl(root/'dev.jsonl');teacher=read_jsonl(root/'teacher-train.jsonl');judges=read_jsonl(run/'judgments.jsonl')
    def unique(rows):
        d={r['id']:r for r in rows}
        if len(d)!=len(rows):raise ValueError('Duplicate IDs')
        return d
    tr=unique(train);by=unique(teacher);jb=unique(judges)
    if set(tr)!=set(by):raise ValueError('Teacher ID coverage')
    pool=[r for r in train if by[r['id']]['stop_reason']=='eos' and by[r['id']]['prediction'].strip() not in ('','NO_ANSWER') and by[r['id']]['prediction'].strip() in r['context']]
    if set(jb)!={r['id'] for r in pool}:raise ValueError('Judge ID coverage')
    kept=[r for r in pool if jb[r['id']]['prediction'].strip()=='SUPPORTED' and jb[r['id']]['stop_reason']=='eos']
    control=random.Random(cfg['subset_seed']).sample(pool,len(kept))
    controls=read_jsonl(run/'judge-controls.jsonl')
    if len(controls)!=8 or sum(r['prediction'].strip()==r['expected'] and r['stop_reason']=='eos' for r in controls)<6:raise ValueError('Control gate failed')
    if len(kept)<8 or len(pool)-len(kept)<4:raise ValueError('Retention gate failed')
    expected={'semantic':kept,'random_structural':control,'selected_gold':kept};gm=json.loads((run/'groups_manifest.json').read_text());summary=json.loads((run/'summary.json').read_text());scores={};token_totals={}
    for group,source in expected.items():
        path=run/(group+'-training.jsonl');rows=read_jsonl(path)
        if sha(path)!=gm[group] or [r['id'] for r in rows]!=[r['id'] for r in source]:raise ValueError('Selected group changed')
        for r,o in zip(rows,source):
            target=o['answers'][0] if group=='selected_gold' else by[o['id']]['prediction'].strip()
            if r!=dict(o,target=target):raise ValueError('Training input or target changed')
        for seed in cfg['seeds']:
            key=f'{group}-{seed}';folder=run/key
            if not folder.exists():
                if group=='selected_gold' and all(by[r['id']]['prediction'].strip()==r['answers'][0] for r in kept):
                    folder=run/f'semantic-{seed}'
                else:raise ValueError('Missing trained group '+key)
            receipt=json.loads((folder/'training.json').read_text());steps=json.loads((folder/'steps.json').read_text())
            if receipt['status']!='complete' or receipt['seed']!=seed or receipt['steps']!=cfg['steps'] or [x['step'] for x in steps]!=list(range(1,cfg['steps']+1)):raise ValueError('Training steps/seed')
            order=list(rows);random.Random(seed).shuffle(order)
            if [x['id'] for x in steps]!=[order[i%len(order)]['id'] for i in range(cfg['steps'])]:raise ValueError('Training order mismatch')
            if sha(folder/'adapter_model.safetensors')!=receipt['adapter_sha256']:raise ValueError('Adapter hash')
            for total,field in [('total_input_tokens','input_tokens'),('total_supervised_tokens','supervised_tokens')]:
                if receipt[total]!=sum(s[field] for s in steps):raise ValueError('Token total')
            predictions=read_jsonl(folder/'dev-predictions.jsonl');m,s=evaluate_zh(dev,predictions)
            if m!=summary[key] or m!=json.loads((folder/'metrics.json').read_text()):raise ValueError('Metric mismatch')
            if read_jsonl(folder/'scored.jsonl')!=s:raise ValueError('Per-example scores changed')
            scores[key]=s;token_totals[key]={k:receipt[k] for k in ['total_input_tokens','total_supervised_tokens','elapsed_seconds']}
    # Uncertainty conditional on the fixed seeds/subsets. Resample articles (one dev question/article).
    if len({r['family_id'] for r in dev})!=len(dev):raise ValueError('Bootstrap assumes one question/article')
    aggregates={g:{metric:statistics.mean(summary[f'{g}-{seed}'][metric] for seed in cfg['seeds']) for metric in ['strict_em','char_lcs_f1','format_valid']} for g in expected}
    paired={}
    for other in ['random_structural','selected_gold']:
        for metric in ['strict_em','char_lcs_f1']:
            values=[statistics.mean(scores[f'semantic-{seed}'][i][metric]-scores[f'{other}-{seed}'][i][metric] for seed in cfg['seeds']) for i in range(len(dev))]
            rng=random.Random(20260920);samples=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
            paired[other+':'+metric]=dict(delta=statistics.mean(values),bootstrap_percentile_95=[samples[249],samples[9749]],n_articles=len(dev),replicates=10000,note='Exploratory, conditional on fixed seeds and subsets; does not include seed-selection or subset-selection uncertainty, no multiplicity correction')
    return dict(status='pass',groups=aggregates,paired=paired,token_totals=token_totals,training_runs=len(scores),dev_n=len(dev),holdout_inference=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),ensure_ascii=False,indent=2))
