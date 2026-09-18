"""Bounded, paired three-seed study. Run only TRAIN teacher generation and DEV inference."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from qa_lab.common import read_jsonl, sha, write_json
from qa_lab.closure_data import pilot_rows, freeze_artifact, build_distilled
from qa_lab.metrics import evaluate
from qa_lab.train_artifact import verified_rows

ROOT=Path('reports/teacher-study-v2')
CONFIG=Path('configs/teacher-study-v2')
DATA=Path('data/complexity-v1')
SELECTION=Path('data/teacher-pilot-v1/manifest.json')


def check_registration():
    reg=json.loads((ROOT/'preregistration.json').read_text())
    for p,h in reg['files'].items():
        if sha(p)!=h: raise ValueError(f'Preregistration changed: {p}')
    return json.loads((CONFIG/'protocol.json').read_text())


def gold_artifact(out):
    rows=pilot_rows(DATA,SELECTION)
    targets=[dict(id=r['id'],context=r['context'],question=r['question'],split='train',
                  target='NO_ANSWER' if r['is_impossible'] else r['answers'][0]) for r in rows]
    freeze_artifact(out,targets,dict(method='gold_sft',source_sha256=sha(DATA/'train.jsonl'),
        pilot_selection_sha256=sha(SELECTION),policy='Identical 24 pilot IDs; first reference answer or exact NO_ANSWER. No sample addition.'))


def execute(args,log):
    command=[sys.executable,*args]
    with (ROOT/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),argv=['.venv/bin/python',*args],log=str(log)))+'\n')
    print('RUN', ' '.join(args),flush=True)
    env=dict(os.environ,HF_HOME=str(Path('.cache/huggingface').resolve()),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    with log.open('x') as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)


def run():
    protocol=check_registration()
    if (ROOT/'source').exists(): raise ValueError('Run already started; preserve outputs and investigate, do not overwrite.')
    (ROOT/'source').mkdir()
    shutil.copytree('qa_lab',ROOT/'source/qa_lab',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree('scripts',ROOT/'source/scripts',ignore=shutil.ignore_patterns('__pycache__'))
    execute(['-m','qa_lab.closure_data','teacher','--teacher-config',str(CONFIG/'teacher.json'),
             '--output',str(ROOT/'teacher')],ROOT/'teacher.log')
    gold_artifact(Path('data/teacher-study-v2-gold'))
    build_distilled(DATA,SELECTION,ROOT/'teacher',Path('data/teacher-study-v2-prompted'),CONFIG/'teacher.json')
    artifacts={'gold':Path('data/teacher-study-v2-gold'),'original_teacher':Path('data/distilled-local-v1'),
               'prompted_teacher':Path('data/teacher-study-v2-prompted')}
    expected=json.loads(SELECTION.read_text())['ids']
    for a in artifacts.values():
        rows,_=verified_rows(a,DATA)
        assert [r['id'] for r in rows]==expected
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            label=f'{arm}-{seed}'
            adapter=Path('checkpoints')/f'teacher-study-v2-{label}'
            execute(['-m','qa_lab.train_artifact','--artifact',str(artifacts[arm]),'--config',str(CONFIG/f'train-{seed}.json'),
                     '--output',str(adapter)],ROOT/f'{label}.train.log')
            report=ROOT/label;report.mkdir()
            for name in ['training.json','steps.json']:shutil.copyfile(adapter/name,report/name)
            write_json(report/'adapter-manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in adapter.iterdir() if p.suffix in ['.safetensors','.json']})
            execute(['-m','qa_lab.inference','--adapter',str(adapter),'--splits','dev','--output',str(report/'dev')],ROOT/f'{label}.dev.log')
    analyze()


def paired(scored_a,scored_b):
    a={s['id']:s['em'] for s in scored_a};b={s['id']:s['em'] for s in scored_b}
    if set(a)!=set(b):raise ValueError('Paired IDs must match')
    return dict(fixes=sorted(i for i in a if a[i]==0 and b[i]==1),regressions=sorted(i for i in a if a[i]==1 and b[i]==0),
                delta_em=sum(b[i]-a[i] for i in a)/len(a))


def aggregate(values):
    return dict(n=len(values),mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values),maximum=max(values))


def analyze():
    protocol=check_registration();rows=read_jsonl(DATA/'dev.jsonl')
    result=dict(experiment=protocol['experiment'],evaluation_split='dev',n=len(rows),test_evaluated=False,
                teacher={},runs={},aggregate={},paired={},passing_arms=[],limitations=protocol['limits'])
    for label,folder in [('original',Path('reports/closure-v1/teacher-fp32')),('prompted',ROOT/'teacher')]:
        result['teacher'][label]=evaluate(pilot_rows(DATA,SELECTION),read_jsonl(folder/'predictions.jsonl'))[0]
    all_scored={}
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            label=f'{arm}-{seed}';folder=ROOT/label
            metrics,scored=evaluate(rows,read_jsonl(folder/'dev/dev.predictions.jsonl'))
            training=json.loads((folder/'training.json').read_text())
            assert training['status']=='complete' and len(training['steps'])==48
            passed=(metrics['overall']['em']>protocol['gate']['all_seeds_overall_em_strictly_above'] and
                    metrics['answerable']['em']>=protocol['gate']['all_seeds_answerable_em_at_least'])
            result['runs'][label]=dict(metrics=metrics,seed=seed,arm=arm,gate_pass=passed,
                supervised_tokens=training['total_supervised_tokens'],loss_first=training['steps'][0]['loss'],loss_last=training['steps'][-1]['loss'])
            all_scored[label]=scored
        for a,b in [('gold','original_teacher'),('gold','prompted_teacher'),('original_teacher','prompted_teacher')]:
            result['paired'][f'{a}_to_{b}-{seed}']=paired(all_scored[f'{a}-{seed}'],all_scored[f'{b}-{seed}'])
    for arm in protocol['arms']:
        runs=[result['runs'][f'{arm}-{s}'] for s in protocol['seeds']]
        result['aggregate'][arm]={f'{split}_{m}':aggregate([r['metrics'][split][m] for r in runs]) for split in ['overall','answerable','unanswerable'] for m in ['em','f1']}
        if all(r['gate_pass'] for r in runs):result['passing_arms'].append(arm)
    result['decision']='eligible_for_external_validation' if result['passing_arms'] else 'no_method_passed_all_seeds'
    path=ROOT/'summary.json'
    if path.exists():assert json.loads(path.read_text())==result,'Stored summary differs'
    else:write_json(path,result)
    print(json.dumps({k:result[k] for k in ['teacher','aggregate','passing_arms','decision']},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','analyze']);a=p.parse_args()
    run() if a.mode=='run' else analyze()
