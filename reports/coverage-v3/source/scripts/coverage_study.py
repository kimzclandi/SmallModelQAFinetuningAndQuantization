"""Fixed coverage study, with train/dev phase separated from cross-article evaluation."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from difflib import SequenceMatcher
import json,os,shutil,subprocess,sys
from pathlib import Path
from qa_lab.common import sha,digest,read_jsonl,write_json,write_jsonl
from qa_lab.data import norm,jaccard,related
from qa_lab.closure_data import freeze_artifact
from qa_lab.teacher_io import training_rows
from qa_lab.train_artifact import verified_rows
from qa_lab.metrics import evaluate
from scripts.teacher_study import paired,aggregate
R=Path('reports/coverage-v3');C=Path('configs/coverage-v3');D=Path('data/complexity-v1');H=Path('data/cross-article-v3')


def protocol():
    for p,h in json.loads((R/'preregistration.json').read_text())['files'].items():assert sha(p)==h
    return json.loads((C/'protocol.json').read_text())


def question_related(a,b,cfg):
    return jaccard(a['question'],b['question'])>=cfg['question_jaccard'] or SequenceMatcher(None,norm(a['question']),norm(b['question']),autojunk=False).ratio()>=cfg['question_sequence_ratio']


def select_holdout(raw,old,policy,cfg):
    selected=[];excluded=[];seen=set();used=set()
    for title in policy['holdout_titles']:
        article=next(a for a in raw['data'] if a['title']==title)
        pool=[]
        for p in article['paragraphs']:
            for q in p['qas']:
                answers=list(dict.fromkeys(a['text'] for a in q['answers']))
                assert bool(answers)!=q['is_impossible']
                for a in q['answers']:assert p['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text']
                pool.append(dict(id=q['id'],context=p['context'],question=q['question'],answers=answers,is_impossible=q['is_impossible'],source_title=title,context_id=digest(norm(p['context']))[:16],family_id=digest(title+':'+norm(p['context']))[:16],split='test'))
        for impossible in [False,True]:
            count=0
            for row in sorted((r for r in pool if r['is_impossible']==impossible),key=lambda r:digest('cross-article-v3:'+r['id'])):
                key=(row['context_id'],impossible)
                if key in used:continue
                same=next((r for r in old if related(row,r,cfg)),None)
                if same is not None:excluded.append({'id':row['id'],'reason':'related_to_old_corpus','other_id':same['id']});continue
                same=next((r for r in selected if question_related(row,r,cfg) or (row['context_id']!=r['context_id'] and jaccard(row['context'],r['context'])>=cfg['context_jaccard'])),None)
                if same is not None:excluded.append({'id':row['id'],'reason':'related_within_holdout','other_id':same['id']});continue
                if row['id'] in seen:raise ValueError('Duplicate raw ID')
                selected.append(row);seen.add(row['id']);used.add(key);count+=1
                if count==policy['per_article_per_class']:break
            if count!=policy['per_article_per_class']:raise ValueError(f'Insufficient isolated rows: {title}, impossible={impossible}, n={count}')
    return selected,excluded


def prepare(rawpath):
    p=protocol();cfg=json.loads(Path('configs/data.json').read_text());assert sha(rawpath)==cfg['sha256']
    old=sum([read_jsonl(D/f'{s}.jsonl') for s in ['train','dev','test']],[])
    rows,excluded=select_holdout(json.loads(rawpath.read_text()),old,p,cfg)
    H.mkdir(exist_ok=False);write_jsonl(H/'test.jsonl',rows)
    write_json(H/'manifest.json',dict(source_url=cfg['url'],source_sha256=sha(rawpath),license='CC BY-SA 4.0',policy=p['holdout_policy'],titles=p['holdout_titles'],n=len(rows),excluded=excluded,near_duplicate_thresholds=cfg,old_split_hashes={s:sha(D/f'{s}.jsonl') for s in ['train','dev','test']},files={'test.jsonl':sha(H/'test.jsonl')}))
    gold=training_rows(D)
    freeze_artifact(Path('data/coverage-v3-gold242'),[dict(id=r['id'],context=r['context'],question=r['question'],split='train',target='NO_ANSWER' if r['is_impossible'] else r['answers'][0]) for r in gold],dict(method='gold_sft',source_sha256=sha(D/'train.jsonl'),policy='All 242 original frozen TRAIN rows; no additions from dev/test or external articles.'))
    print('Prepared',len(gold),'training rows and',len(rows),'cross-article rows; excluded',len(excluded))


def execute(args,log):
    with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':['.venv/bin/python',*args],'log':str(log)})+'\n')
    print('RUN',' '.join(args),flush=True)
    with log.open('x') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,HF_HOME=str(Path('.cache/huggingface').resolve()),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1'))


def train():
    p=protocol();shutil.copytree('qa_lab',R/'source/qa_lab',ignore=shutil.ignore_patterns('__pycache__'))
    (R/'source/scripts').mkdir();shutil.copyfile(__file__,R/'source/scripts/coverage_study.py')
    for seed in p['seeds']:
        for arm in p['arms']:
            artifact=Path('data/teacher-study-v2-gold' if arm=='gold24' else 'data/coverage-v3-gold242')
            verified_rows(artifact,D)
            name=f'{arm}-{seed}';adapter=Path('checkpoints')/f'coverage-v3-{name}';report=R/name
            execute(['-m','qa_lab.train_artifact','--artifact',str(artifact),'--config',str(C/f'train-{seed}.json'),'--output',str(adapter)],R/f'{name}.train.log')
            report.mkdir()
            for f in ['training.json','steps.json']:shutil.copyfile(adapter/f,report/f)
            write_json(report/'adapter-manifest.json',{f.name:sha(f) for f in adapter.iterdir() if f.is_file()})
            execute(['-m','qa_lab.inference','--adapter',str(adapter),'--splits','dev','--output',str(report/'dev')],R/f'{name}.dev.log')
    metrics={}
    for seed in p['seeds']:
        for arm in p['arms']:
            name=f'{arm}-{seed}';metrics[name]=evaluate(read_jsonl(D/'dev.jsonl'),read_jsonl(R/name/'dev/dev.predictions.jsonl'))[0]
    passing=[arm for arm in p['arms'] if all(metrics[f'{arm}-{seed}']['overall']['em']>p['dev_gate']['overall_gt'] and metrics[f'{arm}-{seed}']['answerable']['em']>=p['dev_gate']['answerable_ge'] for seed in p['seeds'])]
    write_json(R/'selection.json',dict(phase='dev_only_before_cross_article_inference',metrics=metrics,passing_arms=passing,decision='eligible_for_external_validation' if passing else 'no_arm_passed_all_seeds',holdout_sha256=sha(H/'test.jsonl'),adapter_manifests={f'{a}-{s}':sha(R/f'{a}-{s}'/'adapter-manifest.json') for s in p['seeds'] for a in p['arms']}))


def external():
    p=protocol();freeze=json.loads((R/'freeze.json').read_text())
    assert sha(R/'selection.json')==freeze['selection_sha256'] and sha(H/'test.jsonl')==freeze['holdout_sha256']
    assert subprocess.check_output(['git','show',freeze['commit']+':'+str(R/'selection.json')])==(R/'selection.json').read_bytes()
    names=['baseline']+[f'{a}-{s}' for s in p['seeds'] for a in p['arms']]
    for name in names:
        args=['-m','qa_lab.inference','--data-dir',str(H),'--splits','test','--output',str(R/name/'external')]
        if name!='baseline':args+=['--adapter',str(Path('checkpoints')/f'coverage-v3-{name}')]
        execute(args,R/f'{name}.external.log')
    summarize()


def summarize():
    p=protocol();selection=json.loads((R/'selection.json').read_text());rows=read_jsonl(H/'test.jsonl')
    names=['baseline']+[f'{a}-{s}' for s in p['seeds'] for a in p['arms']]
    metrics={};scored={};article={}
    for name in names:
        pred=read_jsonl(R/name/'external/test.predictions.jsonl');metrics[name],scored[name]=evaluate(rows,pred)
        article[name]={}
        for title in p['holdout_titles']:
            sub=[r for r in rows if r['source_title']==title];ids={r['id'] for r in sub}
            article[name][title]=evaluate(sub,[r for r in pred if r['id'] in ids])[0]
    result=dict(dev=selection,external=metrics,by_article=article,aggregate={},paired={})
    for split,values in [('dev',selection['metrics']),('external',metrics)]:
        result['aggregate'][split]={a:{k:aggregate([values[f'{a}-{s}'][k]['em'] for s in p['seeds']]) for k in ['overall','answerable','unanswerable']} for a in p['arms']}
    for seed in p['seeds']:result['paired'][str(seed)]=paired(scored[f'gold24-{seed}'],scored[f'gold242-{seed}'])
    out=R/'summary.json'
    if out.exists():assert json.loads(out.read_text())==result
    else:write_json(out,result)
    print(json.dumps({'aggregate':result['aggregate'],'decision':selection['decision']},indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train','external','summarize']);ap.add_argument('--raw',type=Path,default=Path('../../work/squad-dev-v2.0.json'));a=ap.parse_args()
    if a.mode=='prepare':prepare(a.raw)
    elif a.mode=='train':train()
    elif a.mode=='external':external()
    else:summarize()
