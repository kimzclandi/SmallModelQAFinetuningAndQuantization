"""One frozen holdout evaluation of previously trained adapters, without tuning."""
import argparse,gc,json,random,statistics,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from qa_lab.chinese import evaluate_zh
from qa_lab.inference import load
from quality_pilot import predict

def paired_analysis(scores,seeds):
    n=len(scores['baseline'])
    out={}
    for other in ['random_structural','selected_gold','baseline']:
        for metric in ['strict_em','char_lcs_f1']:
            values=[statistics.mean(scores[f'semantic-{s}'][i][metric]-(scores['baseline'][i][metric] if other=='baseline' else scores[f'{other}-{s}'][i][metric]) for s in seeds) for i in range(n)]
            rng=random.Random(20260920);bs=sorted(statistics.mean(rng.choices(values,k=n)) for _ in range(10000))
            out[other+':'+metric]=dict(delta=statistics.mean(values),percentile_95=[bs[249],bs[9749]],n_articles=n,replicates=10000,scope='conditional on three fixed seeds/subsets, exploratory small-sample bootstrap; secondary comparisons unadjusted')
    return out

def check_protocol(root):
    cfg=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest.items():
        if sha(root/name)!=h:raise ValueError('Evaluation protocol/input changed')
    return cfg

def prepare(a):
    parent=a.parent;cfg=json.loads((parent/'protocol.json').read_text());m=json.loads((parent/'manifest.json').read_text())
    if sha(parent/'holdout.jsonl')!=m['holdout.jsonl']:raise ValueError('Holdout changed')
    if json.loads((parent/'run/run.json').read_text())['status']!='complete':raise ValueError('Training incomplete')
    # All model identities and comparisons are registered before any holdout prediction.
    models={'baseline':None};hashes={}
    for group in cfg['groups']:
        for seed in cfg['seeds']:
            folder=parent/'run'/f'{group}-{seed}';receipt=json.loads((folder/'training.json').read_text())
            if sha(folder/'adapter_model.safetensors')!=receipt['adapter_sha256']:raise ValueError('Adapter changed')
            models[f'{group}-{seed}']=str(folder.resolve());hashes[f'{group}-{seed}']={n:sha(folder/n) for n in ['adapter_model.safetensors','adapter_config.json','training.json']}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'holdout.jsonl').write_bytes((parent/'holdout.jsonl').read_bytes())
    student=dict(cfg['student'],prompt_version='zh-extract-or-abstain-v1',warmup='none; no warmup in this evaluation')
    write_json(a.out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),parent=str(parent.resolve()),parent_protocol_sha256=sha(parent/'protocol.json'),holdout_original_sha256=m['holdout.jsonl'],student=student,seeds=cfg['seeds'],models=models,model_hashes=hashes,primary='Mean strict EM difference semantic minus equal-count random_structural across fixed seeds',research_gate='Evidence of positive primary difference only if paired article-bootstrap 95% lower bound >0; not a deployment gate',secondary='Character LCS F1, semantic versus selected_gold and baseline are descriptive',selection='No seed selection; all three frozen seeds; no additional training, prompt changes or holdout-driven tuning',max_wall_seconds=900,source_limit='Same public CMRC source previously indexed; independent of current train/dev by article and registered heuristic, not new source or pretraining-unseen'))
    write_json(a.out/'manifest.json',{n:sha(a.out/n) for n in ['protocol.json','holdout.jsonl']})

def run(a):
    import torch
    from peft import PeftModel
    cfg=check_protocol(a.out);dest=a.out/'run';dest.mkdir(exist_ok=False)
    rows=read_jsonl(a.out/'holdout.jsonl')
    if len(rows)!=24 or len({r['family_id'] for r in rows})!=24:raise ValueError('Expected 24 distinct articles')
    torch.set_num_threads(8);start=time.monotonic();deadline=start+cfg['max_wall_seconds'];summary={};scores={}
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(a.out/'protocol.json'),script_sha256=sha(Path(__file__)),role='first registered holdout access; test now consumed for this protocol')
    write_json(dest/'access.json',meta)
    try:
        for key,adapter in cfg['models'].items():
            if adapter:
                for name,h in cfg['model_hashes'][key].items():
                    if sha(Path(adapter)/name)!=h:raise ValueError('Frozen model changed')
            tok,model=load(cfg['student'])
            if adapter:model=PeftModel.from_pretrained(model,adapter).eval()
            folder=dest/key;folder.mkdir(exist_ok=False)
            pred=predict(rows,cfg['student'],tok,model,folder/'predictions.jsonl',deadline)
            metric,scored=evaluate_zh(rows,pred);summary[key]=metric;scores[key]=scored
            write_json(folder/'metrics.json',metric);write_jsonl(folder/'scored.jsonl',scored);write_json(dest/'summary.json',summary)
            print('completed',key,flush=True)
            del model;gc.collect();torch.mps.empty_cache()
        comparisons=paired_analysis(scores,cfg['seeds']);write_json(dest/'paired.json',comparisons)
        write_json(dest/'decision.json',dict(primary_positive=comparisons['random_structural:strict_em']['percentile_95'][0]>0,scope='Research evidence gate only; not deployment approval',holdout_consumed=True))
        meta['status']='complete'
    except BaseException as e:meta.update(status='failed',error=repr(e));raise
    finally:
        meta.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'access.json',meta)

def verify(root):
    cfg=check_protocol(root);run=root/'run';receipt=json.loads((run/'access.json').read_text())
    if receipt['status']!='complete' or receipt['protocol_sha256']!=sha(root/'protocol.json'):raise ValueError('Incomplete or changed run')
    rows=read_jsonl(root/'holdout.jsonl');summary=json.loads((run/'summary.json').read_text());scores={}
    if set(summary)!=set(cfg['models']):raise ValueError('Missing model result')
    for key,adapter in cfg['models'].items():
        if adapter:
            for name,h in cfg['model_hashes'][key].items():
                if sha(Path(adapter)/name)!=h:raise ValueError('Model changed')
        folder=run/key;m,s=evaluate_zh(rows,read_jsonl(folder/'predictions.jsonl'))
        if m!=summary[key] or m!=json.loads((folder/'metrics.json').read_text()) or s!=read_jsonl(folder/'scored.jsonl'):raise ValueError('Metrics changed')
        scores[key]=s
    paired=paired_analysis(scores,cfg['seeds'])
    if paired!=json.loads((run/'paired.json').read_text()):raise ValueError('Paired results changed')
    decision=json.loads((run/'decision.json').read_text())
    if decision['primary_positive']!=(paired['random_structural:strict_em']['percentile_95'][0]>0):raise ValueError('Decision mismatch')
    return dict(status='pass',n=24,model_runs=len(scores),predictions=24*len(scores),paired=paired,decision=decision)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','verify']);p.add_argument('--parent',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.stage=='verify':print(json.dumps(verify(a.out),ensure_ascii=False,indent=2))
    else:(prepare if a.stage=='prepare' else run)(a)
