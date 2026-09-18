"""Offline, batch-one greedy decoding with KV caching and synchronized timing."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import subprocess
import threading
import time
from .common import digest, read_jsonl, sha, source_hashes, write_json, write_jsonl
from .metrics import evaluate


@dataclass(frozen=True)
class QAInput:
    context: str
    question: str


def messages(item, cfg):
    return [{'role':'system','content':cfg['system_prompt']},
            {'role':'user','content':f'Passage:\n{item.context}\n\nQuestion: {item.question}'}]


def load(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if cfg['device']=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process. Use --device cpu explicitly; no silent fallback.')
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(cfg['model_id'],revision=cfg['revision'],
             local_files_only=True, torch_dtype=getattr(torch,cfg['dtype']),
             attn_implementation=cfg['attention_implementation']).to(cfg['device']).eval()
    return tokenizer, model


def sync(device):
    if device=='mps':
        import torch
        torch.mps.synchronize()


class MemoryMonitor:
    """RSS sampled at 10 ms; MPS allocator/driver sampled at token boundaries (not exact peaks)."""
    def __init__(self,device):
        import psutil
        self.process=psutil.Process()
        self.device=device
        self.rss=0
        self.mps_alloc=0
        self.mps_driver=0
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.poll,daemon=True)
    def poll(self):
        while not self.stop.is_set():
            self.rss=max(self.rss,self.process.memory_info().rss)
            self.stop.wait(0.01)
    def sample(self):
        if self.device=='mps':
            import torch
            self.mps_alloc=max(self.mps_alloc,torch.mps.current_allocated_memory())
            self.mps_driver=max(self.mps_driver,torch.mps.driver_allocated_memory())
    def values(self):
        return dict(rss_peak_sampled_bytes=self.rss,mps_alloc_peak_sampled_bytes=self.mps_alloc,
                    mps_driver_peak_sampled_bytes=self.mps_driver,
                    note='Process RSS and MPS memory overlap in unified memory; DO NOT add them. Sampled maxima, not exact hardware peaks.')


def generate(item,cfg,tokenizer,model,monitor=None):
    import torch
    started=time.perf_counter()
    prompt=tokenizer.apply_chat_template(messages(item,cfg),tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors='pt').to(cfg['device'])
    n=inputs['input_ids'].shape[1]
    if n>cfg['max_input_tokens']:
        raise ValueError(f'Input has {n} tokens; truncation is forbidden.')
    ids=inputs['input_ids']
    mask=inputs['attention_mask']
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    past=None
    output=[]
    times=[]
    sync(cfg['device'])
    t0=time.perf_counter()
    with torch.inference_mode():
        for _ in range(cfg['max_new_tokens']):
            result=model(input_ids=ids,attention_mask=mask,past_key_values=past,use_cache=True)
            nxt=result.logits[:,-1,:].argmax(-1,keepdim=True)
            sync(cfg['device'])
            times.append(time.perf_counter())
            token=int(nxt.item())
            output.append(token)
            past=result.past_key_values
            if monitor:
                monitor.sample()
            if token in eos:
                break
            ids=nxt
            mask=torch.cat([mask,torch.ones((1,1),device=mask.device,dtype=mask.dtype)],dim=1)
    elapsed=times[-1]-t0
    decode_seconds=times[-1]-times[0]
    text=tokenizer.decode(output,skip_special_tokens=True)
    return dict(prediction=text, input_tokens=n, generated_tokens=len(output),
                token_ids=output, prompt_sha256=digest(prompt),
                ttft_seconds=times[0]-t0, generation_seconds=elapsed,
                decode_seconds=decode_seconds,
                decode_tokens_per_second=(len(output)-1)/decode_seconds if len(output)>1 else None,
                end_to_end_seconds=time.perf_counter()-started,
                stop_reason='eos' if output[-1] in eos else 'max_new_tokens')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/baseline.json')
    parser.add_argument('--data-dir',default='data/complexity-v1',type=Path)
    parser.add_argument('--splits',nargs='+',choices=['dev','test'],default=['dev','test'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','mps'])
    parser.add_argument('--limit',type=int)
    parser.add_argument('--adapter',type=Path)
    a=parser.parse_args()
    if a.limit is not None and a.limit < 1:
        parser.error('--limit must be positive')
    cfg=json.loads(Path(a.config).read_text())
    if a.device:
        cfg['device']=a.device
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(8)
    monitor=MemoryMonitor(cfg['device'])
    monitor.thread.start()
    start=time.perf_counter()
    provenance=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),config=cfg,
        command=sys.argv,python=platform.python_version(),os=platform.platform(),machine=platform.machine(),
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','peft','psutil']},
        source_sha256=source_hashes(),data_sha256={s:sha(a.data_dir/f'{s}.jsonl') for s in a.splits},
        limit=a.limit, adapter=None,
        hardware=dict(chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip() if sys.platform=='darwin' else platform.processor(),
                      unified_memory_bytes=__import__('psutil').virtual_memory().total,
                      cpu_count=os.cpu_count()),
        timing='Warm batch=1, greedy, KV cache on, eager attention. TTFT starts after tokenization/device transfer and includes synchronized prefill. Decode rate excludes first token, includes EOS; no network/queue time.')
    write_json(a.output/'run.json',provenance)
    try:
        tokenizer,model=load(cfg)
        if a.adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,str(a.adapter),local_files_only=True).eval()
            provenance['adapter']={p.name:sha(p) for p in a.adapter.iterdir() if p.is_file()}
        provenance['load_seconds']=time.perf_counter()-start
        provenance['parameter_count']=sum(p.numel() for p in model.parameters())
        provenance['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
        for _ in range(2):
            generate(QAInput('A queue follows first in, first out.','What order does a queue follow?'),cfg,tokenizer,model,monitor)
        for split in a.splits:
            rows=read_jsonl(a.data_dir/f'{split}.jsonl')
            if a.limit:
                rows=rows[:a.limit]
            predictions=[]
            path=a.output/f'{split}.predictions.jsonl'
            with path.open('x') as f:
                for i,row in enumerate(rows):
                    result=generate(QAInput(context=row['context'],question=row['question']),cfg,tokenizer,model,monitor)
                    result.update(id=row['id'],split=split)
                    predictions.append(result)
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    f.flush()
                    print(f'{split}: {i+1}/{len(rows)} tokens={result["generated_tokens"]}',flush=True)
            metrics,scored=evaluate(rows,predictions)
            dec=sum(p['decode_seconds'] for p in predictions)
            metrics['performance']=dict(n=len(rows),
                mean_ttft_seconds=statistics.mean(p['ttft_seconds'] for p in predictions),
                median_ttft_seconds=statistics.median(p['ttft_seconds'] for p in predictions),
                mean_generation_seconds=statistics.mean(p['generation_seconds'] for p in predictions),
                decode_tokens_per_second=sum(p['generated_tokens']-1 for p in predictions)/dec if dec else None,
                total_generated_tokens=sum(p['generated_tokens'] for p in predictions),
                capped_outputs=sum(p['stop_reason']=='max_new_tokens' for p in predictions))
            write_json(a.output/f'{split}.metrics.json',metrics)
            write_jsonl(a.output/f'{split}.scored.jsonl',scored)
            rowmap={r['id']:r for r in rows}
            predmap={p['id']:p for p in predictions}
            write_jsonl(a.output/f'{split}.failures.jsonl',
                [dict(**s,question=rowmap[s['id']]['question'],answers=rowmap[s['id']]['answers'],
                      prediction=predmap[s['id']]['prediction']) for s in scored if s['em']<1])
        provenance['status']='complete'
    except BaseException as e:
        provenance.update(status='failed',error=repr(e))
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        provenance.update(memory=monitor.values(),wall_seconds=time.perf_counter()-start,
                          finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.output/'run.json',provenance)
        write_json(a.output/'manifest.json',{p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':
    main()
