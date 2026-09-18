"""Apple MLX weight-only comparison. Same timed greedy decoder for FP16 and affine Q4."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time
from .common import read_jsonl,write_json,write_jsonl,sha,digest,source_hashes
from .metrics import evaluate
from .inference import QAInput,messages,MemoryMonitor,hardware_chip


def generate(item,cfg,tok,model,fixed_tokens=None):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    prompt=tok.apply_chat_template(messages(item,cfg),tokenize=False,add_generation_prompt=True)
    ids=tok.encode(prompt,add_special_tokens=False)
    if len(ids)>cfg['max_input_tokens']:raise ValueError('Truncation forbidden')
    cache=make_prompt_cache(model)  # unquantized floating KVCache, independent for each sample
    x=mx.array([ids]);mx.eval(x);mx.synchronize()
    count=fixed_tokens or cfg['max_new_tokens'];out=[];times=[]
    t0=time.perf_counter()
    for _ in range(count):
        logits=model(x,cache=cache)
        nxt=mx.argmax(logits[:,-1,:],axis=-1)
        mx.eval(nxt);mx.synchronize();times.append(time.perf_counter())
        token=int(nxt.item());out.append(token)
        if fixed_tokens is None and token in tok.eos_token_ids:break
        x=nxt.reshape(1,1)
    seconds=times[-1]-t0;decode=times[-1]-times[0]
    return dict(prediction=tok.decode(out,skip_special_tokens=True),token_ids=out,input_tokens=len(ids),
        generated_tokens=len(out),prompt_sha256=digest(prompt),input_token_ids_sha256=digest(json.dumps(ids)),
        ttft_seconds=times[0]-t0,generation_seconds=seconds,decode_seconds=decode,
        decode_tokens_per_second=(len(out)-1)/decode if len(out)>1 else None,
        stop_reason='fixed_length' if fixed_tokens else ('eos' if out[-1] in tok.eos_token_ids else 'max_new_tokens'))


def model_files(path):
    return {p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(path.iterdir()) if p.is_file()}


def convert_models(source,out,quantize):
    from mlx_lm.convert import convert
    if not source.is_dir():raise ValueError('Local source required; no implicit network download')
    files=model_files(source)
    convert(str(source),mlx_path=str(out),quantize=quantize,q_group_size=64,q_bits=4,dtype='float16')
    write_json(out/'conversion-provenance.json',dict(source_files=files,output_files=model_files(out),
        quantize=quantize,weight_bits=4 if quantize else 16,group_size=64 if quantize else None,
        kv_cache_quantization=False,source_student_revision='7ae557604adf67be50417f59c2c2f167def9a775',
        packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers']},
        created_utc=datetime.now(timezone.utc).isoformat()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['convert','quality','benchmark','parity'])
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quantize',action='store_true');p.add_argument('--splits',nargs='+',choices=['dev','test'],default=['dev','test'])
    a=p.parse_args()
    if a.mode=='convert':return convert_models(a.model,a.output,a.quantize)
    if not a.model.is_dir():raise ValueError('Local model required')
    a.output.mkdir(parents=True,exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    cfg=json.loads(Path('configs/baseline.json').read_text());cfg.update(device='mlx-metal',dtype='float16',attention_implementation='mlx native')
    files=model_files(a.model)
    model_cfg=json.loads((a.model/'config.json').read_text())
    monitor=MemoryMonitor('mlx');monitor.thread.start();t0=time.perf_counter()
    run=dict(status='running',mode=a.mode,started_utc=datetime.now(timezone.utc).isoformat(),config=cfg,
        model_path=str(a.model),model_files=files,weight_bytes=sum(v['bytes'] for k,v in files.items() if k.endswith('.safetensors')),
        quantization=model_cfg.get('quantization'),kv_quantized=False,source_sha256=source_hashes(),
        hardware=hardware_chip(),os=platform.platform(),python=platform.python_version(),
        packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','psutil']},
        timing='Synchronized serial batch1. TTFT excludes encoding; decode rate excludes first token. No cross-framework speed attribution.',
        protocol_sha256=sha('configs/closure-v1.json'))
    write_json(a.output/'run.json',run)
    try:
        mx.random.seed(cfg['seed']);model,tok=load(str(a.model));mx.eval(model.parameters());mx.synchronize()
        run['load_seconds']=time.perf_counter()-t0
        run['model_loaded_active_bytes']=mx.get_active_memory()
        for _ in range(2):generate(QAInput('A queue follows first in, first out.','What order does a queue follow?'),cfg,tok,model)
        if a.mode=='parity':
            from mlx_lm.generate import generate_step
            rows=read_jsonl('data/complexity-v1/train.jsonl')[:2];checked=[]
            for row in rows:
                item=QAInput(row['context'],row['question']);ours=generate(item,cfg,tok,model)
                prompt=tok.apply_chat_template(messages(item,cfg),tokenize=False,add_generation_prompt=True)
                expected=[]
                for token,_ in generate_step(mx.array(tok.encode(prompt,add_special_tokens=False)),model,max_tokens=cfg['max_new_tokens'],kv_bits=None):
                    expected.append(int(token))
                    if int(token) in tok.eos_token_ids:break
                if ours['token_ids']!=expected:raise AssertionError('Timed decoder differs from MLX-LM greedy')
                checked.append(dict(id=row['id'],matching_tokens=len(expected)))
            write_json(a.output/'parity.json',checked)
        elif a.mode=='quality':
            run['data_sha256']={s:sha(f'data/complexity-v1/{s}.jsonl') for s in a.splits}
            for split in a.splits:
                rows=read_jsonl(f'data/complexity-v1/{split}.jsonl');pred=[]
                with (a.output/f'{split}.predictions.jsonl').open('x') as f:
                    for n,r in enumerate(rows):
                        q=generate(QAInput(r['context'],r['question']),cfg,tok,model);q.update(id=r['id'],split=split)
                        pred.append(q);f.write(json.dumps(q)+'\n');f.flush();print(f'{split} {n+1}/{len(rows)}',flush=True)
                m,sc=evaluate(rows,pred);write_json(a.output/f'{split}.metrics.json',m);write_jsonl(a.output/f'{split}.scored.jsonl',sc)
        else:
            req=json.loads(Path('data/teacher-pilot-v1/request.json').read_text())['examples'][:8]
            run['request_sha256']=sha('data/teacher-pilot-v1/request.json');run['forced_output_tokens']=32
            pred=[]
            for r in req:
                q=generate(QAInput(r['context'],r['question']),cfg,tok,model,fixed_tokens=32);q['id']=r['id'];pred.append(q)
            write_jsonl(a.output/'timings.jsonl',pred)
            write_json(a.output/'performance.json',dict(n=len(pred),mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in pred),
                decode_tokens_per_second=sum(r['generated_tokens']-1 for r in pred)/sum(r['decode_seconds'] for r in pred),
                mean_generation_seconds=statistics.mean(r['generation_seconds'] for r in pred),generated_tokens=sum(r['generated_tokens'] for r in pred)))
        run['status']='complete'
    except BaseException as e:
        run.update(status='failed',error=repr(e));raise
    finally:
        monitor.stop.set();monitor.thread.join()
        run.update(memory=dict(rss_peak_sampled_bytes=monitor.rss,mlx_peak_active_bytes=mx.get_peak_memory(),mlx_active_bytes=mx.get_active_memory(),mlx_cache_bytes=mx.get_cache_memory(),note='MLX allocator peak excludes cached allocations; RSS overlaps unified memory. Do not add. Process includes model load and warmup.'),elapsed_seconds=time.perf_counter()-t0,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(a.output/'run.json',run);write_json(a.output/'manifest.json',{f.name:sha(f) for f in a.output.iterdir() if f.is_file() and f.name!='manifest.json'})

if __name__=='__main__':main()
