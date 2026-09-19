"""One preregistered evaluation of all supervised-control adapters and baseline."""
import argparse,gc,json,random,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from qa_lab.chinese import evaluate_zh
from qa_lab.inference import load
from quality_pilot import predict

def prepare(root):
    parent=root/'supervised-07';out=root/'holdout-08';out.mkdir(exist_ok=False)
    cfg=json.loads((parent/'protocol.json').read_text());models={'baseline':None};hashes={}
    if json.loads((parent/'run/receipt.json').read_text())['status']!='complete':raise ValueError('Incomplete training')
    for group in cfg['groups']:
        for seed in cfg['seeds']:
            key=f'{group}-{seed}';folder=parent/'run'/key;models[key]=str(folder.relative_to(root));hashes[key]={n:sha(folder/n) for n in ['adapter_model.safetensors','adapter_config.json']}
    source=root/'fresh-05/holdout.jsonl';original=json.loads((root/'fresh-05/data_manifest.json').read_text())['holdout.jsonl']
    if sha(source)!=original:raise ValueError('Holdout changed')
    (out/'holdout.jsonl').write_bytes(source.read_bytes())
    write_json(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),student=cfg['student'],models=models,model_hashes=hashes,seeds=cfg['seeds'],groups=cfg['groups'],holdout_sha256=original,max_wall_seconds=1800,
        primary='random_gold minus random_teacher mean strict EM across all fixed seeds; paired article-bootstrap95 lower bound >0 is research evidence only',
        secondary='F1, oracle_selected and baseline contrasts descriptive; no multiplicity correction',
        policy='One access; no seed selection, no extra training, no prompt/budget changes after results. Holdout becomes consumed. Same public corpus, not external source or pretraining-unseen.',source_sha256=sha(Path(__file__))))
    write_json(out/'manifest.json',{n:sha(out/n) for n in ['protocol.json','holdout.jsonl']})

def check(out):
    for n,h in json.loads((out/'manifest.json').read_text()).items():
        if sha(out/n)!=h:raise ValueError('Frozen evaluation changed')
    return json.loads((out/'protocol.json').read_text())

def analyze(scores,cfg):
    n=len(scores['baseline']);paired={}
    for group,other in [('random_gold','random_teacher'),('oracle_selected','random_teacher'),('random_gold','baseline')]:
        for metric in ['strict_em','char_lcs_f1']:
            v=[statistics.mean(scores[f'{group}-{s}'][i][metric]-(scores['baseline'][i][metric] if other=='baseline' else scores[f'{other}-{s}'][i][metric]) for s in cfg['seeds']) for i in range(n)]
            rng=random.Random(20260920);b=sorted(statistics.mean(rng.choices(v,k=n)) for _ in range(10000))
            paired[f'{group}-{other}:{metric}']=dict(delta=statistics.mean(v),percentile_95=[b[249],b[9749]],n_articles=n)
    return paired

def run(root):
    import torch
    from peft import PeftModel
    out=root/'holdout-08';cfg=check(out)
    if sha(Path(__file__))!=cfg['source_sha256']:raise ValueError('Runner changed')
    dest=out/'run';dest.mkdir(exist_ok=False);rows=read_jsonl(out/'holdout.jsonl');torch.set_num_threads(8);start=time.monotonic();summary={};scores={};receipt=dict(status='running',holdout_consumed=True,started_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'access.json',receipt)
    if len(rows)!=96 or len({r['family_id'] for r in rows})!=96:raise ValueError('Expected96 independent article IDs')
    try:
        for key,adapter in cfg['models'].items():
            if adapter:
                for n,h in cfg['model_hashes'][key].items():
                    if sha(root/adapter/n)!=h:raise ValueError('Adapter changed')
            tok,model=load(cfg['student'])
            if adapter:model=PeftModel.from_pretrained(model,root/adapter).eval()
            folder=dest/key;folder.mkdir();p=predict(rows,cfg['student'],tok,model,folder/'predictions.jsonl',start+cfg['max_wall_seconds'])
            m,s=evaluate_zh(rows,p);summary[key]=m;scores[key]=s;write_json(folder/'metrics.json',m);write_jsonl(folder/'scored.jsonl',s);write_json(dest/'summary.json',summary)
            del model;gc.collect();torch.mps.empty_cache()
        paired=analyze(scores,cfg);write_json(dest/'paired.json',paired);write_json(dest/'decision.json',dict(primary_positive=paired['random_gold-random_teacher:strict_em']['percentile_95'][0]>0,scope='Fixed models on one same-source holdout; not deployment approval',holdout_consumed=True));receipt['status']='complete'
    except BaseException as e:receipt.update(status='failed',error=repr(e));raise
    finally:receipt.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'access.json',receipt)

def verify(root):
    out=root/'holdout-08';cfg=check(out);run=out/'run';rows=read_jsonl(out/'holdout.jsonl');scores={};summary=json.loads((run/'summary.json').read_text())
    if json.loads((run/'access.json').read_text())['status']!='complete':raise ValueError('Incomplete evaluation')
    if set(summary)!=set(cfg['models']):raise ValueError('Model coverage')
    for key in cfg['models']:
        m,s=evaluate_zh(rows,read_jsonl(run/key/'predictions.jsonl'))
        if m!=summary[key] or m!=json.loads((run/key/'metrics.json').read_text()) or s!=read_jsonl(run/key/'scored.jsonl'):raise ValueError('Metric mismatch')
        scores[key]=s
    paired=analyze(scores,cfg)
    if paired!=json.loads((run/'paired.json').read_text()):raise ValueError('Paired mismatch')
    if json.loads((run/'decision.json').read_text())['primary_positive']!=(paired['random_gold-random_teacher:strict_em']['percentile_95'][0]>0):raise ValueError('Decision mismatch')
    return dict(status='pass',n=len(rows),model_runs=len(scores),paired=paired,groups={g:{m:statistics.mean(summary[f'{g}-{s}'][m] for s in cfg['seeds']) for m in ['strict_em','char_lcs_f1']} for g in cfg['groups']},baseline=summary['baseline'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','verify']);p.add_argument('root',type=Path);a=p.parse_args()
    if a.stage=='verify':print(json.dumps(verify(a.root),indent=2))
    else:(prepare if a.stage=='prepare' else run)(a.root)
