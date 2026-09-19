"""Bounded answerable-only Chinese response-distillation pilot; never evaluates holdout."""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
from datetime import datetime, timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl, write_json, write_jsonl, sha
from qa_lab.chinese import normalize_zh, near, evaluate_zh
from qa_lab.inference import QAInput, messages, load, generate, sync
from qa_lab.train_artifact import target_tokens
SEED=20260920

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,x):write_json(p,x)
def accept(row,p):
    s=p['prediction'].strip()
    return p['stop_reason']=='eos' and bool(s) and s!='NO_ANSWER' and s in row['context']
def prepare(a):
    a.out.mkdir(parents=True,exist_ok=False)
    base=json.loads((a.repo/'configs/baseline.json').read_text())
    base.update(system_prompt='请仅根据给定文段回答问题。如果文段包含答案，只输出能回答问题的最短连续原文片段，不要改写，不要解释。如果文段不包含答案，只输出 NO_ANSWER。',max_input_tokens=768,max_new_tokens=48,seed=SEED)
    teacher=dict(base,model_id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(base['model_id'],revision=base['revision'],local_files_only=True)
    raw=json.loads(a.source.read_text());oldraw=json.loads(a.olddev.read_text())
    old=[];blocked_titles=set()
    for article in oldraw['data']:
        blocked_titles.add(normalize_zh(article['title']))
        for p in article['paragraphs']:
            for q in p['qas']:old.append(dict(context=p['context'],question=q['question']))
    oldids=set()
    for name in ['dev.jsonl','holdout.jsonl']:
        oldids.update(r['id'] for r in read_jsonl(a.engine/name))
    pool=[]
    for article in raw['data']:
        title=normalize_zh(article['title'])
        if title in blocked_titles:continue
        if any(q['id'] in oldids for p in article['paragraphs'] for q in p['qas']):continue
        for p in article['paragraphs']:
            for q in p['qas']:
                if not q['answers'] or not all(x['text'].strip() and p['context'][x['answer_start']:x['answer_start']+len(x['text'])]==x['text'] for x in q['answers']):continue
                r=dict(id=q['id'],context=p['context'],question=q['question'],answers=list(dict.fromkeys(x['text'] for x in q['answers'])),is_impossible=False,source_title=article['title'],family_id=digest(title))
                n=len(tok.apply_chat_template(messages(QAInput(r['context'],r['question']),base),tokenize=True,add_generation_prompt=True))
                if n+max(len(tok.encode(x)) for x in r['answers'])+8<=768:pool.append(r)
    selected=[];titles=set();excluded=[]
    for r in sorted(pool,key=lambda r:digest(str(SEED)+r['id'])):
        if r['family_id'] in titles:continue
        if any(near(r,o) for o in selected+old):excluded.append(r['id']);continue
        titles.add(r['family_id']);selected.append(r)
        if len(selected)==112:break
    if len(selected)!=112:raise ValueError('Insufficient isolated samples')
    for start,end,split in [(0,64,'train'),(64,88,'dev'),(88,112,'holdout')]:
        rows=[dict(r,split=split) for r in selected[start:end]];write_jsonl(a.out/(split+'.jsonl'),rows)
    dump(a.out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),status='frozen_pilot_before_generation',student=base,teacher=teacher,seed=SEED,steps=64,learning_rate=1e-4,lora_r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],training_precision='float32',groups=['raw','filtered','random','gold'],filter='EOS and nonempty exact context substring, excluding NO_ANSWER; no gold access',random_control='same count as filtered, random from all train IDs; distributions not matched',budget='64 optimizer steps per group; token budget NOT matched; record actual tokens',limits='single seed, answerable-only, 24 development questions, not adoption evidence; no test evaluation or quantization',source_reuse='CMRC train was previously indexed in Chinese Evidence Data Engine. Historical evaluation article titles and all official CMRC dev near-duplicates excluded. Not a new source, not pretraining-unseen.',max_wall_seconds_per_stage=1800))
    dump(a.out/'data_manifest.json',dict(source_sha256=sha(a.source),official_dev_excluded_sha256=sha(a.olddev),historical_queries={n:sha(a.engine/n) for n in ['dev.jsonl','holdout.jsonl']},files={n:sha(a.out/n) for n in ['train.jsonl','dev.jsonl','holdout.jsonl','protocol.json']},excluded_near_ids=excluded,eligible_pool=len(pool)))
    print('Prepared 64 train / 24 dev / 24 sealed holdout',flush=True)

def verify(out):
    m=json.loads((out/'data_manifest.json').read_text())
    for n,h in m['files'].items():
        if sha(out/n)!=h:raise ValueError('Frozen input changed: '+n)
    return json.loads((out/'protocol.json').read_text())
def predict(rows,cfg,tok,model,path,deadline):
    if path.exists():raise FileExistsError(path)
    out=[]
    with path.open('x') as f:
        for i,r in enumerate(rows):
            if time.monotonic()>deadline:raise TimeoutError('stage wall budget reached')
            p=dict(id=r['id'],**generate(QAInput(r['context'],r['question']),cfg,tok,model))
            f.write(json.dumps(p,ensure_ascii=False)+'\n');f.flush();out.append(p)
            print(path.name,i+1,len(rows),p['stop_reason'],flush=True)
    return out

def run(a):
    import torch,gc,importlib.metadata
    from peft import LoraConfig,get_peft_model
    cfg=verify(a.out);torch.set_num_threads(8)
    train=read_jsonl(a.out/'train.jsonl');dev=read_jsonl(a.out/'dev.jsonl')
    stage=a.out/'run';stage.mkdir(exist_ok=False)
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(a.out/'protocol.json'),script_sha256=sha(Path(__file__)),packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},device='mps')
    dump(stage/'run.json',meta);started=time.monotonic();deadline=started+cfg['max_wall_seconds_per_stage']
    try:
        tok,teacher=load(cfg['teacher'])
        tp=predict(train,cfg['teacher'],tok,teacher,stage/'teacher-train.jsonl',deadline)
        del teacher;gc.collect();torch.mps.empty_cache()
        by={p['id']:p for p in tp};kept=[r for r in train if accept(r,by[r['id']])]
        if not kept:raise ValueError('No teacher output passes fixed filter')
        rng=random.Random(SEED);random_rows=rng.sample(train,len(kept))
        audits=[dict(id=r['id'],accepted=accept(r,by[r['id']]),teacher_prediction=by[r['id']]['prediction'],gold_match=by[r['id']]['prediction'].strip() in r['answers']) for r in train]
        write_jsonl(stage/'quality-audit.jsonl',audits)
        groups={name:[dict(r,target=r['answers'][0] if name=='gold' else by[r['id']]['prediction'].strip()) for r in rows] for name,rows in [('raw',train),('filtered',kept),('random',random_rows),('gold',train)]}
        # No generation failures or empty targets may silently disappear from raw/random.
        if any(not r['target'] for rows in groups.values() for r in rows):raise ValueError('Empty teacher target: no silent row drop')
        for name,rows in groups.items():write_jsonl(stage/(name+'-training.jsonl'),rows)
        dump(stage/'groups_manifest.json',{name:dict(n=len(rows),sha256=sha(stage/(name+'-training.jsonl'))) for name,rows in groups.items()})
        tok,model=load(cfg['student'])
        baseline=predict(dev,cfg['student'],tok,model,stage/'baseline-dev.jsonl',deadline)
        summary={'baseline':evaluate_zh(dev,baseline)[0]}
        del model;gc.collect();torch.mps.empty_cache()
        for name,rows in groups.items():
            random.seed(SEED);torch.manual_seed(SEED);torch.mps.manual_seed(SEED)
            tok,model=load(cfg['student']);model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=cfg['target_modules'],task_type='CAUSAL_LM'))
            model.train();model.config.use_cache=False
            optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
            order=list(rows);random.shuffle(order);logs=[];t0=time.monotonic()
            folder=stage/name;folder.mkdir()
            for step in range(cfg['steps']):
                if time.monotonic()>deadline:raise TimeoutError('stage wall budget reached')
                r=order[step%len(order)];ids,labels=target_tokens(r,tok,cfg['student'],768)
                x=torch.tensor([ids],device='mps');y=torch.tensor([labels],device='mps')
                optim.zero_grad(set_to_none=True);loss=model(input_ids=x,labels=y,attention_mask=torch.ones_like(x)).loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
                if not __import__('math').isfinite(grad):raise ValueError('Nonfinite gradient')
                optim.step();sync('mps')
                logs.append(dict(step=step+1,id=r['id'],loss=float(loss.detach().cpu()),grad_norm=grad,input_tokens=len(ids),supervised_tokens=sum(v!=-100 for v in labels)))
                dump(folder/'steps.json',logs)
                print(name,'step',step+1,'loss',logs[-1]['loss'],flush=True)
            model.save_pretrained(folder);model.eval();model.config.use_cache=True
            dump(folder/'training.json',dict(status='complete',steps=64,seconds=time.monotonic()-t0,seed=SEED,total_input_tokens=sum(v['input_tokens'] for v in logs),total_supervised_tokens=sum(v['supervised_tokens'] for v in logs),unique_training_ids=len({v['id'] for v in logs}),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),adapter_sha256=sha(folder/'adapter_model.safetensors')))
            pred=predict(dev,cfg['student'],tok,model,folder/'dev-predictions.jsonl',deadline)
            metrics,scored=evaluate_zh(dev,pred);dump(folder/'metrics.json',metrics);write_jsonl(folder/'scored.jsonl',scored);summary[name]=metrics
            dump(stage/'summary.json',summary)
            del model,optim,loss,x,y;gc.collect();torch.mps.empty_cache()
        meta['status']='complete'
    except BaseException as e:meta.update(status='failed',error=repr(e));raise
    finally:
        meta.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat());dump(stage/'run.json',meta)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--out',type=Path,required=True);p.add_argument('--repo',type=Path);p.add_argument('--source',type=Path);p.add_argument('--olddev',type=Path);p.add_argument('--engine',type=Path);a=p.parse_args()
    (prepare if a.stage=='prepare' else run)(a)
