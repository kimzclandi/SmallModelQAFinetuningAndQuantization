"""Train from audited local-teacher or gold-repair artifacts; unchanged student hyperparameters."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import time
from .common import read_jsonl,sha,write_json,source_hashes
from .teacher_io import training_rows
from .inference import load,messages,QAInput,sync,hardware_chip


def verified_rows(artifact,source):
    m=json.loads((artifact/'manifest.json').read_text())
    if sha(artifact/'train.jsonl')!=m['train_sha256'] or sha(source/'train.jsonl')!=m['source_sha256']:
        raise ValueError('Artifact or source hash mismatch')
    rows=read_jsonl(artifact/'train.jsonl');gold={r['id']:r for r in training_rows(source)}
    if len(rows)!=m['n'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate/count mismatch')
    for r in rows:
        if r['id'] not in gold or r['split']!='train' or any(r[k]!=gold[r['id']][k] for k in ['context','question']):
            raise ValueError('Training-only input boundary violated')
        if not isinstance(r['target'],str) or not r['target']:raise ValueError('Empty target')
    if m['method']=='gold_data_repair':
        for r in rows:
            g=gold[r['id']];target='NO_ANSWER' if g['is_impossible'] else g['answers'][0]
            if r['target']!=target:raise ValueError('Gold target changed')
    elif m['method']=='response_distillation':
        report=Path(m['teacher_report'])
        if sha(report/'manifest.json')!=m['teacher_manifest_sha256']:raise ValueError('Teacher manifest changed')
        for name,h in json.loads((report/'manifest.json').read_text()).items():
            if sha(report/name)!=h:raise ValueError('Teacher output changed')
        tr=json.loads((report/'run.json').read_text())
        if tr['method']!='local_public_teacher' or tr['status']!='complete' or tr['config']!=m['teacher_config']:
            raise ValueError('Non-local or incomplete teacher')
        pm={p['id']:p for p in read_jsonl(report/'predictions.jsonl')}
        if set(pm)!={r['id'] for r in rows}:raise ValueError('Teacher ID mismatch')
        for r in rows:
            if r['target']!=pm[r['id']]['prediction'].strip() or pm[r['id']]['stop_reason']!='eos':raise ValueError('Teacher target altered')
    else:raise ValueError('Unsupported artifact method')
    return rows,m


def target_tokens(row,tok,base,max_tokens):
    msg=messages(QAInput(row['context'],row['question']),base)
    prefix=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True)
    full=tok.apply_chat_template(msg+[{'role':'assistant','content':row['target']}],tokenize=True)
    if full[:len(prefix)]!=prefix or len(full)>max_tokens:raise ValueError('Prefix mismatch or truncation required')
    return full,[-100]*len(prefix)+full[len(prefix):]


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--config',default='configs/gold-sft-pilot-v1.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows,manifest=verified_rows(a.artifact,Path('data/complexity-v1'))
    cfg=json.loads(Path(a.config).read_text());base=json.loads(Path(cfg['base_config']).read_text())
    a.output.mkdir(parents=True,exist_ok=False)
    import torch,importlib.metadata
    from peft import LoraConfig,get_peft_model
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(8)
    meta=dict(status='running',method=manifest['method'],config=cfg,base=base,source_sha256=source_hashes(),
              artifact_sha256=sha(a.artifact/'manifest.json'),artifact_path=str(a.artifact),
              selected_ids=[r['id'] for r in rows],hardware=hardware_chip(),
              started_utc=datetime.now(timezone.utc).isoformat(),
              packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft']},
              note='Same LoRA/optimizer/steps/seed as gold24. Target token count can differ; only response-level CE, no logits matching.')
    write_json(a.output/'training.json',meta);log=[];t0=time.perf_counter()
    try:
        tok,model=load(base)
        model=get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],target_modules=cfg['target_modules'],task_type='CAUSAL_LM'))
        model.train();model.config.use_cache=False
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
        random.shuffle(rows)
        for step in range(cfg['steps']):
            row=rows[step%len(rows)];ids,labels=target_tokens(row,tok,base,cfg['max_tokens'])
            x=torch.tensor([ids],device=base['device']);y=torch.tensor([labels],device=base['device'])
            optimizer.zero_grad(set_to_none=True)
            loss=model(input_ids=x,labels=y,attention_mask=torch.ones_like(x)).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite loss')
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));optimizer.step();sync(base['device'])
            log.append(dict(step=step+1,id=row['id'],loss=float(loss.detach().cpu()),grad_norm=grad,input_tokens=len(ids),supervised_tokens=sum(t!=-100 for t in labels)))
            write_json(a.output/'steps.json',log);print(log[-1],flush=True)
        model.save_pretrained(a.output)
        meta.update(status='complete',trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    except BaseException as e:
        meta.update(status='failed',error=repr(e));raise
    finally:
        meta.update(steps=log,total_supervised_tokens=sum(s['supervised_tokens'] for s in log),elapsed_seconds=time.perf_counter()-t0,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.output/'training.json',meta)

if __name__=='__main__':main()
