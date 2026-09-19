"""Calibrated forced-label likelihood margins; never interpreted as calibrated confidence."""
import argparse,json,sys,time,math
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from qa_lab.inference import load,messages
from judge_calibration import inputs

def summarize(rows,scores,threshold):
 by={p['id']:p for p in scores}
 if len(by)!=len(scores) or set(by)!={r['id'] for r in rows}:raise ValueError('Score ID coverage')
 pos=[by[r['id']]['margin']>=threshold for r in rows if r['expected']];neg=[by[r['id']]['margin']<threshold for r in rows if not r['expected']]
 tpr=sum(pos)/len(pos);tnr=sum(neg)/len(neg)
 return dict(tpr=tpr,tnr=tnr,balanced_accuracy=(tpr+tnr)/2,threshold=threshold)

def threshold(rows,scores):
 vals=sorted({p['margin'] for p in scores});ts=[vals[0]-1]+[(x+y)/2 for x,y in zip(vals,vals[1:])]+[vals[-1]+1]
 metrics=[summarize(rows,scores,t) for t in ts]
 return sorted(metrics,key=lambda m:(-min(m['tpr'],m['tnr']),-m['balanced_accuracy'],abs(m['threshold'])))[0]

def score(row,prompt,cfg,tok,model):
 import torch
 msgs=messages(inputs(row,prompt),cfg);prefix=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True)
 eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);sequences=[];lens=[]
 for label in [prompt['yes'],prompt['no']]:
  full=tok.apply_chat_template(msgs+[{'role':'assistant','content':label}],tokenize=True)
  if full[:len(prefix)]!=prefix:raise ValueError('Template prefix mismatch')
  cont=[]
  for token in full[len(prefix):]:
   if token in eos:break
   cont.append(token)
  if not cont:raise ValueError('Empty forced label')
  sequences.append(prefix+cont);lens.append(len(cont))
 n=max(map(len,sequences));ids=torch.tensor([s+[tok.pad_token_id]*(n-len(s)) for s in sequences],device='mps');mask=torch.tensor([[1]*len(s)+[0]*(n-len(s)) for s in sequences],device='mps')
 with torch.inference_mode():
  out=model(input_ids=ids,attention_mask=mask,use_cache=False);logps=[]
  for i,l in enumerate(lens):
   logits=out.logits[i,len(prefix)-1:len(prefix)+l-1].float();targets=ids[i,len(prefix):len(prefix)+l];logps.append(float(logits.log_softmax(-1).gather(-1,targets[:,None]).sum().cpu()))
 if not all(math.isfinite(x) for x in logps):raise ValueError('Nonfinite likelihood')
 return dict(id=row['id'],yes_logprob=logps[0],no_logprob=logps[1],margin=logps[0]-logps[1],label_tokens=lens)

def main(a):
 import torch
 a.out.mkdir(parents=True,exist_ok=False)
 parent=json.loads((a.calibration/'protocol.json').read_text());cfg=parent['teacher']
 protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_sha256=sha(a.calibration/'protocol.json'),rule='Select threshold maximizing min(TPR,TNR), then balanced accuracy, then smallest absolute threshold. Eligible TPR/TNR >=0.875. Select eligible prompt with best balanced accuracy, tie by name. Freeze before held-out synthetic templates.',validation_gate='TPR and TNR >=0.875',scope='Likelihood margin over forced labels, not calibrated probability; threshold fitting uses calibration only. Synthetic validation is not real QA evidence.',source_hashes={n:sha(a.calibration/(n+'.jsonl')) for n in ['calibration','validation']})
 write_json(a.out/'protocol.json',protocol);torch.set_num_threads(8);tok,model=load(cfg);rows=read_jsonl(a.calibration/'calibration.jsonl');results={};start=time.monotonic()
 for name,prompt in parent['prompts'].items():
  config=dict(cfg,system_prompt=prompt['system']);scores=[]
  for r in rows:
   if time.monotonic()-start>900:raise TimeoutError('Score budget')
   scores.append(score(r,prompt,config,tok,model));write_jsonl(a.out/(name+'-calibration.jsonl'),scores)
  results[name]=threshold(rows,scores);print(name,results[name],flush=True)
 write_json(a.out/'calibration-summary.json',results)
 eligible=[n for n,m in results.items() if m['tpr']>=.875 and m['tnr']>=.875]
 if not eligible:write_json(a.out/'decision.json',dict(proceed=False,reason='No margin classifier passes calibration gate',elapsed_seconds=time.monotonic()-start));return
 chosen=sorted(eligible,key=lambda n:(-results[n]['balanced_accuracy'],n))[0];t=results[chosen]['threshold']
 write_json(a.out/'selected-before-validation.json',dict(prompt=chosen,threshold=t,utc=datetime.now(timezone.utc).isoformat()))
 prompt=parent['prompts'][chosen];config=dict(cfg,system_prompt=prompt['system']);rows=read_jsonl(a.calibration/'validation.jsonl');scores=[]
 for r in rows:scores.append(score(r,prompt,config,tok,model));write_jsonl(a.out/'selected-validation.jsonl',scores)
 result=summarize(rows,scores,t);write_json(a.out/'decision.json',dict(proceed=result['tpr']>=.875 and result['tnr']>=.875,prompt=chosen,threshold=t,validation=result,elapsed_seconds=time.monotonic()-start))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
