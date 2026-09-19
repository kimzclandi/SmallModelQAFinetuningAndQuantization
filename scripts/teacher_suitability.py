"""Paired frozen teacher/student screening on new development data only."""
import argparse,json,sys,time,gc
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from qa_lab.inference import load
from qa_lab.chinese import evaluate_zh
from quality_pilot import predict

def main(a):
 import torch
 for n,h in json.loads((a.data/'data_manifest.json').read_text()).items():
  if sha(a.data/n)!=h:raise ValueError('Changed data')
 a.out.mkdir(exist_ok=False);cfg=json.loads((a.data/'data_protocol.json').read_text())['student'];teacher=dict(cfg,model_id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),dev_sha256=sha(a.data/'dev.jsonl'),student=cfg,teacher=teacher,scope='New dev only; 96 holdout remains sealed; paired teacher screening, no method selection on test')
 write_json(a.out/'protocol.json',protocol);rows=read_jsonl(a.data/'dev.jsonl');torch.set_num_threads(8);summary={};start=time.monotonic()
 for name,config in [('student',cfg),('teacher',teacher)]:
  tok,model=load(config);preds=predict(rows,config,tok,model,a.out/(name+'-predictions.jsonl'),start+900);metrics,scored=evaluate_zh(rows,preds);summary[name]=metrics;write_jsonl(a.out/(name+'-scored.jsonl'),scored);write_json(a.out/'summary.json',summary)
  del model;gc.collect();torch.mps.empty_cache()
 write_json(a.out/'receipt.json',dict(status='complete',elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(a.out/'protocol.json'),source_sha256=sha(Path(__file__))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
