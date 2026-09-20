"""Frozen DRCD transfer evaluation; preparation never reads model predictions."""
import argparse, gc, json, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl, write_json, write_jsonl, sha, digest
from qa_lab.chinese import normalize_zh, evaluate_zh
from qa_lab.inference import QAInput, messages, load
from quality_pilot import predict
from final_quality_holdout import analyze

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REV = 'b944790de5af02c5fbb7cd9cb1473d27d169eebf'

def prepare(a):
    from opencc import OpenCC
    from transformers import AutoTokenizer
    cc = OpenCC('t2s')
    norm = lambda text: normalize_zh(cc.convert(text))
    grams = lambda text: set(text[i:i+5] for i in range(max(1, len(text)-4)))
    cfg = json.loads((ROOT/'reports/quality-study-20260920/supervised-07/protocol.json').read_text())
    tok = AutoTokenizer.from_pretrained(cfg['student']['model_id'], revision=cfg['student']['revision'], local_files_only=True)
    old_contexts=set();old_questions=set();old_titles=set();prior_hashes={}
    for name,path in [('cmrc_train',a.cmrc_train),('cmrc_dev',a.cmrc_dev)]:
        prior_hashes[name]=sha(path)
        for article in json.loads(path.read_text())['data']:
            old_titles.add(norm(article['title']))
            for paragraph in article['paragraphs']:
                old_contexts.add(norm(paragraph['context']))
                old_questions.update(norm(q['question']) for q in paragraph['qas'])
    # Include all saved project inputs, including prior synthetic/English studies.
    for folder in ('data','reports'):
        for path in sorted((ROOT/folder).rglob('*.jsonl')):
            rows=read_jsonl(path)
            records=[r for r in rows if isinstance(r,dict) and 'context' in r and 'question' in r]
            if records:
                prior_hashes[str(path.relative_to(ROOT))]=sha(path)
                for r in records:
                    old_contexts.add(norm(r['context']));old_questions.add(norm(r['question']))
                    if r.get('source_title'):old_titles.add(norm(r['source_title']))
    old_sets=[grams(s) for s in sorted(old_contexts)];index=defaultdict(list)
    for i,g in enumerate(old_sets):
        for token in g:index[token].append(i)
    def overlaps(g):
        counts=Counter(i for token in g for i in index.get(token,()))
        return any(n/(len(g)+len(old_sets[i])-n)>=.7 for i,n in counts.items())
    raw=json.loads(a.source.read_text());pool=[];seen=set();excluded=[]
    for article in raw['data']:
        for paragraph in article['paragraphs']:
            for q in paragraph['qas']:
                if q['id'] in seen:raise ValueError('Duplicate source id')
                seen.add(q['id'])
                row=dict(id='drcd:'+q['id'],source_id=q['id'],source_title=article['title'],context=paragraph['context'],question=q['question'],answers=list(dict.fromkeys(x['text'] for x in q['answers'])),is_impossible=False,family_id=digest(norm(article['title'])))
                if not q['answers'] or not all(x['text'].strip() and paragraph['context'][x['answer_start']:x['answer_start']+len(x['text'])]==x['text'] for x in q['answers']):
                    excluded.append(dict(id=row['id'],reason='invalid_reference'));continue
                pool.append(row)
    selected=[];titles=set();questions=set();selected_grams=[]
    for row in sorted(pool,key=lambda r:digest('external-drcd-20260920:'+r['id'])):
        title=norm(row['source_title']);q=norm(row['question']);g=grams(norm(row['context']));reason=None
        if title in old_titles:reason='prior_title'
        elif title in titles:reason='selected_title'
        elif q in old_questions or q in questions:reason='exact_normalized_question'
        elif overlaps(g) or any(len(g&h)/len(g|h)>=.7 for h in selected_grams):reason='context_jaccard_0.7'
        else:
            n=len(tok.apply_chat_template(messages(QAInput(row['context'],row['question']),cfg['student']),tokenize=True,add_generation_prompt=True))
            if n>cfg['student']['max_input_tokens']:reason='prompt_over_768'
            elif any(len(tok.encode(x,add_special_tokens=False))>47 for x in row['answers']):reason='reference_over_47_tokens'
        if reason:excluded.append(dict(id=row['id'],reason=reason));continue
        row['prompt_tokens']=n;selected.append(row);titles.add(title);questions.add(q);selected_grams.append(g)
        if len(selected)==96:break
    if len(selected)!=96:raise ValueError(f'Only {len(selected)} eligible articles')
    models={'baseline':None};hashes={}
    for group in cfg['groups']:
        for seed in cfg['seeds']:
            key=f'{group}-{seed}';folder=a.adapters/key;models[key]=key
            hashes[key]={n:sha(folder/n) for n in ['adapter_model.safetensors','adapter_config.json']}
    a.out.mkdir(parents=True,exist_ok=False)
    write_jsonl(a.out/'eval.jsonl',selected)
    write_json(a.out/'selection.json',dict(raw_questions=len(seen),selected_n=96,old_contexts=len(old_contexts),old_questions=len(old_questions),source_sha256=sha(a.source),source_revision=SOURCE_REV,source_url=f'https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/{SOURCE_REV}/DRCD_dev.json',prior_hashes=prior_hashes,exclusions=excluded,exclusion_counts=dict(Counter(x['reason'] for x in excluded)),scope='Exclusions include invalid references and traversed candidates until 96 selected; title/exact-question and 5gram Jaccard heuristic after OpenCC t2s; no semantic or pretraining contamination guarantee.'))
    write_json(a.out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),student=cfg['student'],models=models,model_hashes=hashes,seeds=cfg['seeds'],groups=cfg['groups'],n=96,max_wall_seconds=1800,source_sha256=sha(Path(__file__)),primary='random_gold minus random_teacher mean strict EM; paired article bootstrap 10000 draws seed20260920, lower95>0. Other contrasts descriptive.',policy='External DRCD annotation dataset, shared Wikipedia source and Traditional Chinese shift; all answerable. Fixed models, prompt, budgets and all seeds; no tuning, retraining or model selection. Consumed on first prediction.'))
    write_json(a.out/'manifest.json',{n:sha(a.out/n) for n in ['eval.jsonl','selection.json','protocol.json']})
    print('Frozen96 external articles',flush=True)

def check(out):
    manifest=json.loads((out/'manifest.json').read_text())
    if not isinstance(manifest,dict) or set(manifest)!={'eval.jsonl','selection.json','protocol.json'}:raise ValueError('Frozen external manifest coverage')
    for name,h in manifest.items():
        if sha(out/name)!=h:raise ValueError('Frozen external input changed')
    cfg=json.loads((out/'protocol.json').read_text());rows=read_jsonl(out/'eval.jsonl')
    if len(rows)!=cfg['n'] or len({r['family_id'] for r in rows})!=len(rows):raise ValueError('Article coverage')
    if any(not isinstance(r.get('id'),str) or not r['id'] for r in rows) or len({r['id'] for r in rows})!=len(rows):raise ValueError('External sample ID coverage')
    return cfg,rows

def run(a):
    import torch
    from peft import PeftModel
    cfg,rows=check(a.out)
    if sha(Path(__file__))!=cfg['source_sha256']:raise ValueError('Runner changed')
    for key,files in cfg['model_hashes'].items():
        for name,h in files.items():
            if sha(a.adapters/key/name)!=h:raise ValueError('Adapter changed')
    dest=a.out/'run';dest.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(8)
    receipt=dict(status='running',consumed=True,started_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'receipt.json',receipt);summary={};scores={}
    try:
        for key,adapter in cfg['models'].items():
            tok,model=load(cfg['student'])
            if adapter:model=PeftModel.from_pretrained(model,a.adapters/adapter).eval()
            folder=dest/key;folder.mkdir();preds=predict(rows,cfg['student'],tok,model,folder/'predictions.jsonl',start+cfg['max_wall_seconds'])
            m,s=evaluate_zh(rows,preds);summary[key]=m;scores[key]=s;write_json(folder/'metrics.json',m);write_jsonl(folder/'scored.jsonl',s);write_json(dest/'summary.json',summary)
            del model;gc.collect();torch.mps.empty_cache()
        write_json(dest/'paired.json',analyze(scores,cfg));receipt['status']='complete'
    except BaseException as e:receipt.update(status='failed',error=repr(e));raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'receipt.json',receipt)

def verify(out):
    cfg,rows=check(out);dest=out/'run';scores={};summary=json.loads((dest/'summary.json').read_text())
    if json.loads((dest/'receipt.json').read_text())['status']!='complete':raise ValueError('Incomplete external run')
    if set(summary)!=set(cfg['models']):raise ValueError('Model coverage')
    for key in cfg['models']:
        m,s=evaluate_zh(rows,read_jsonl(dest/key/'predictions.jsonl'))
        if m!=summary[key] or m!=json.loads((dest/key/'metrics.json').read_text()) or s!=read_jsonl(dest/key/'scored.jsonl'):raise ValueError('Metric mismatch')
        scores[key]=s
    paired=analyze(scores,cfg)
    if paired!=json.loads((dest/'paired.json').read_text()):raise ValueError('Paired mismatch')
    return dict(status='pass',n=len(rows),model_runs=len(scores),paired=paired,summary=summary,primary_positive=paired['random_gold-random_teacher:strict_em']['percentile_95'][0]>0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','verify']);p.add_argument('--out',type=Path,required=True)
    for n in ['source','cmrc-train','cmrc-dev','adapters']:p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    if a.stage=='verify':print(json.dumps(verify(a.out),indent=2))
    else:(prepare if a.stage=='prepare' else run)(a)
