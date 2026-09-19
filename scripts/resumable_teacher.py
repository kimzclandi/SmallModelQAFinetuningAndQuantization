"""Content-bound atomic teacher-output cache; retries may execute tasks more than once."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha
from qa_lab.inference import load,generate,QAInput

def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic(path,value):
 data=json.dumps(value,ensure_ascii=False,indent=2)+'\n';tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def record_path(root,id):return root/'records'/(hashlib.sha256(id.encode()).hexdigest()+'.json')

def checked_record(path,row,config_hash):
 value=json.loads(path.read_text());p=value['prediction']
 if value['input_sha256']!=digest(row) or value['config_sha256']!=config_hash or p['id']!=row['id'] or value['prediction_sha256']!=digest(p):raise ValueError('Cached record identity/hash mismatch')
 return p

def produce(rows,root,cfg,binding,predict_one,interrupt_after=0):
 ids=[r['id'] for r in rows]
 if len(set(ids))!=len(ids):raise ValueError('Duplicate input IDs')
 root.mkdir(parents=True,exist_ok=True);(root/'records').mkdir(exist_ok=True)
 expected=dict(input_sha256=digest(rows),config=cfg,binding=binding);path=root/'manifest.json'
 if path.exists():
  if json.loads(path.read_text())!=expected:raise ValueError('Input/config/code cache binding changed')
 else:atomic(path,expected)
 ch=digest(expected);out=[];reused=created=0
 for row in rows:
  p=record_path(root,row['id'])
  if p.exists():prediction=checked_record(p,row,ch);reused+=1
  else:
   prediction=predict_one(row)
   if prediction.get('id')!=row['id']:raise ValueError('Predictor returned wrong ID')
   value=dict(input_sha256=digest(row),config_sha256=ch,prediction=prediction,prediction_sha256=digest(prediction));atomic(p,value);created+=1
  out.append(prediction)
  print('progress',len(out),'reused',reused,'created',created,flush=True)
  if interrupt_after and created==interrupt_after:os._exit(73)
 result=dict(status='complete',n=len(out),manifest_sha256=sha(path),records_sha256={record_path(root,r['id']).name:sha(record_path(root,r['id'])) for r in rows})
 final=root/'complete.json'
 if final.exists():
  if json.loads(final.read_text())!=result:raise ValueError('Completed receipt changed')
 else:atomic(final,result)
 return dict(n=len(out),reused=reused,created=created)

def main(a):
 import torch
 from qa_lab import inference
 from importlib.metadata import version
 for n,h in json.loads((a.data/'data_manifest.json').read_text()).items():
  if sha(a.data/n)!=h:raise ValueError('Source changed')
 cfg=json.loads((a.data/'data_protocol.json').read_text())['student'];cfg.update(model_id='Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
 rows=read_jsonl(a.data/'train.jsonl');torch.set_num_threads(8);state={};started=time.monotonic()
 def predictor(row):
  if time.monotonic()-started>900:raise TimeoutError('Teacher generation budget')
  if not state:state['tok'],state['model']=load(cfg)
  return dict(id=row['id'],**generate(QAInput(row['context'],row['question']),cfg,state['tok'],state['model']))
 binding=dict(inference_source=sha(Path(inference.__file__)),runner_source=sha(Path(__file__)),versions={n:version(n) for n in ['torch','transformers']})
 result=produce(rows,a.out,cfg,binding,predictor,a.interrupt_after)
 receipt=dict(**result,elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
 print(json.dumps(receipt),flush=True)
 if a.receipt:
  with a.receipt.open('x') as f:json.dump(receipt,f,indent=2)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--interrupt-after',type=int,default=0);p.add_argument('--receipt',type=Path);main(p.parse_args())
