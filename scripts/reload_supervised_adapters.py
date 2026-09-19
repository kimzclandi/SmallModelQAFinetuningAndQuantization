"""Reload every saved adapter and compare the first two dev generations exactly."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha,write_json
from qa_lab.inference import load,generate,QAInput

def run(root):
    import torch
    from peft import PeftModel
    out=root/'supervised-07';cfg=json.loads((out/'protocol.json').read_text());dest=out/'reload';dest.mkdir(exist_ok=False);torch.set_num_threads(8);checks=[];start=time.monotonic()
    rows=read_jsonl(out/'dev.jsonl')[:2]
    for seed in cfg['seeds']:
        for group in cfg['groups']:
            key=f'{group}-{seed}';folder=out/'run'/key;meta=json.loads((folder/'training.json').read_text())
            if sha(folder/'adapter_model.safetensors')!=meta['adapter_sha256']:raise ValueError('Adapter changed')
            tok,base=load(cfg['student']);model=PeftModel.from_pretrained(base,folder).eval();old=read_jsonl(folder/'dev-predictions.jsonl')[:2]
            for row,expected in zip(rows,old):
                result=generate(QAInput(row['context'],row['question']),cfg['student'],tok,model)
                equal=row['id']==expected['id'] and all(result[k]==expected[k] for k in ['token_ids','prediction','stop_reason'])
                checks.append(dict(model=key,id=row['id'],equal=equal));write_json(dest/'checks.json',checks)
                if not equal:raise ValueError('Reload mismatch')
            del model,base;gc.collect();torch.mps.empty_cache()
    write_json(dest/'receipt.json',dict(status='pass',checks=len(checks),elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
