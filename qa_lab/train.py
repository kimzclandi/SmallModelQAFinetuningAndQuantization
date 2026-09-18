"""Train-only supervised LoRA entrypoint. This is NOT teacher distillation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from .common import read_jsonl, sha, source_hashes, write_json
from .inference import QAInput, load, messages, sync


def supervised_tokens(row,tokenizer,cfg,max_tokens):
    msgs=messages(QAInput(row['context'],row['question']),cfg)
    prefix=tokenizer.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
    answer='NO_ANSWER' if row['is_impossible'] else row['answers'][0]
    full=tokenizer.apply_chat_template(msgs+[{'role':'assistant','content':answer}],tokenize=True)
    if full[:len(prefix)]!=prefix:
        raise ValueError('Chat-template prefix mismatch: cannot safely mask prompt loss.')
    if len(full)>max_tokens:
        raise ValueError('Training truncation forbidden.')
    return full,[-100]*len(prefix)+full[len(prefix):]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train-smoke.json')
    p.add_argument('--data',default='data/complexity-v1/train.jsonl',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'])
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    base=json.loads(Path(cfg['base_config']).read_text())
    if a.device:
        base['device']=a.device
    rows=read_jsonl(a.data)
    manifest_path=a.data.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if a.data.name!='train.jsonl' or sha(a.data)!=manifest['files']['train.jsonl'] or any(r['split']!='train' for r in rows):
        raise ValueError('Training requires the frozen train file; dev/test input forbidden.')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from peft import LoraConfig,get_peft_model
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(8)
    tok,model=load(base)
    model=get_peft_model(model,LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],
          lora_dropout=cfg['lora_dropout'],target_modules=cfg['target_modules'],task_type='CAUSAL_LM'))
    model.train()
    model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    random.shuffle(rows)
    log=[]
    t0=time.perf_counter()
    for step in range(cfg['steps']):
        r=rows[step%len(rows)]
        ids,labels=supervised_tokens(r,tok,base,cfg['max_tokens'])
        # Shape [1,L]; prompt labels=-100, answer + end-of-turn contribute causal LM loss.
        x=torch.tensor([ids],device=base['device'])
        y=torch.tensor([labels],device=base['device'])
        optimizer.zero_grad(set_to_none=True)
        loss=model(input_ids=x,labels=y,attention_mask=torch.ones_like(x)).loss
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss')
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        optimizer.step()
        sync(base['device'])
        log.append(dict(step=step+1,id=r['id'],loss=float(loss.detach().cpu()),grad_norm=grad,
                        input_tokens=len(ids),supervised_tokens=sum(t!=-100 for t in labels)))
        write_json(a.output/'steps.json',log)
        print(log[-1],flush=True)
    model.save_pretrained(a.output)
    write_json(a.output/'training.json',dict(status='complete',method='gold-label supervised LoRA',
        config=cfg,base=base,source_sha256=source_hashes(),data_sha256=sha(a.data),steps=log,
        elapsed_seconds=time.perf_counter()-t0,finished_utc=datetime.now(timezone.utc).isoformat(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        warning='Execution smoke only. No teacher data, no quality improvement claimed.'))

if __name__=='__main__':
    main()
