"""Recompute screening decisions, source identities and recovery evidence without inference."""
import argparse,json,sys,random,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha,write_json,write_jsonl
from qa_lab.chinese import evaluate_zh
from resumable_teacher import checked_record,record_path,digest

def classify(rows,predictions,yes,no):
 by={p['id']:p for p in predictions}
 if len(by)!=len(predictions) or set(by)!={r['id'] for r in rows}:raise ValueError('Judgment coverage')
 pos=neg=invalid=0
 for r in rows:
  p=by[r['id']];s=p['prediction'].strip();valid=p['stop_reason']=='eos' and s in [yes,no];invalid+=not valid
  pos+=bool(r['expected'] and valid and s==yes);neg+=bool(not r['expected'] and valid and s==no)
 tpr=pos/sum(r['expected'] for r in rows);tnr=neg/sum(not r['expected'] for r in rows)
 return dict(n=len(rows),tpr=tpr,tnr=tnr,balanced_accuracy=(tpr+tnr)/2,invalid=invalid)

def margin_metric(rows,scores,t):
 by={p['id']:p for p in scores}
 if len(by)!=len(scores) or set(by)!={r['id'] for r in rows}:raise ValueError('Score coverage')
 for p in scores:
  if abs((p['yes_logprob']-p['no_logprob'])-p['margin'])>1e-12:raise ValueError('Margin mismatch')
 positive=[by[r['id']]['margin']>=t for r in rows if r['expected']];negative=[by[r['id']]['margin']<t for r in rows if not r['expected']]
 a=sum(positive)/len(positive);b=sum(negative)/len(negative)
 return dict(tpr=a,tnr=b,balanced_accuracy=(a+b)/2,threshold=t)

def verify(root):
 cal=root/'calibration-04';score=root/'score-calibration-04';fresh=root/'fresh-05'
 for folder,name in [(cal,'manifest.json'),(fresh,'data_manifest.json')]:
  for n,h in json.loads((folder/name).read_text()).items():
   if sha(folder/n)!=h:raise ValueError('Frozen data changed '+n)
 cfg=json.loads((cal/'protocol.json').read_text());cr=read_jsonl(cal/'calibration.jsonl');vr=read_jsonl(cal/'validation.jsonl');greedy={}
 for name,prompt in cfg['prompts'].items():greedy[name]=classify(cr,read_jsonl(cal/'run'/(name+'-calibration.jsonl')),prompt['yes'],prompt['no'])
 if greedy!=json.loads((cal/'run/calibration-summary.json').read_text()):raise ValueError('Greedy metrics mismatch')
 scored={}
 for name in cfg['prompts']:
  values=read_jsonl(score/(name+'-calibration.jsonl'));v=sorted({p['margin'] for p in values});thresholds=[v[0]-1]+[(a+b)/2 for a,b in zip(v,v[1:])]+[v[-1]+1]
  scored[name]=min([margin_metric(cr,values,t) for t in thresholds],key=lambda m:(-min(m['tpr'],m['tnr']),-m['balanced_accuracy'],abs(m['threshold'])))
 if scored!=json.loads((score/'calibration-summary.json').read_text()):raise ValueError('Threshold recomputation mismatch')
 eligible=[n for n,m in scored.items() if m['tpr']>=.875 and m['tnr']>=.875];chosen=sorted(eligible,key=lambda n:(-scored[n]['balanced_accuracy'],n))[0]
 selected=json.loads((score/'selected-before-validation.json').read_text())
 if selected['prompt']!=chosen or selected['threshold']!=scored[chosen]['threshold']:raise ValueError('Selection was not calibrated policy')
 validation=margin_metric(vr,read_jsonl(score/'selected-validation.jsonl'),selected['threshold']);decision=json.loads((score/'decision.json').read_text())
 if validation!=decision['validation'] or decision['proceed']!=(validation['tpr']>=.875 and validation['tnr']>=.875):raise ValueError('Validation decision mismatch')
 dev=read_jsonl(fresh/'dev.jsonl');suit=fresh/'suitability';summary={};per={}
 for name in ['student','teacher']:
  summary[name],per[name]=evaluate_zh(dev,read_jsonl(suit/(name+'-predictions.jsonl')))
  if per[name]!=read_jsonl(suit/(name+'-scored.jsonl')):raise ValueError('Per-item score mismatch')
 if summary!=json.loads((suit/'summary.json').read_text()):raise ValueError('Teacher suitability mismatch')
 differences=[t['strict_em']-s['strict_em'] for t,s in zip(per['teacher'],per['student'])];rng=random.Random(20260920);bs=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(10000))
 train=read_jsonl(fresh/'train.jsonl');cache=fresh/'teacher-train';manifest=json.loads((cache/'manifest.json').read_text());ch=digest(manifest)
 if manifest['input_sha256']!=digest(train):raise ValueError('Teacher input binding')
 complete=json.loads((cache/'complete.json').read_text());preds=[checked_record(record_path(cache,r['id']),r,ch) for r in train]
 if len({p['id'] for p in preds})!=192 or complete['n']!=192 or sha(cache/'manifest.json')!=complete['manifest_sha256']:raise ValueError('Teacher output denominator')
 for name,h in complete['records_sha256'].items():
  if sha(cache/'records'/name)!=h:raise ValueError('Teacher record changed')
 old=json.loads((fresh/'interrupted-record-hashes.json').read_text())
 if len(old)!=8 or any(sha(cache/'records'/n)!=h for n,h in old.items()):raise ValueError('Interrupted committed records changed')
 gate=decision['proceed'] and summary['teacher']['strict_em']>summary['student']['strict_em'] and summary['teacher']['char_lcs_f1']>=summary['student']['char_lcs_f1']
 tm,_=evaluate_zh(train,preds)
 return dict(status='pass',greedy=greedy,forced_label_calibration=scored,forced_label_validation=validation,suitability=summary,teacher_student_em_difference=dict(delta=statistics.mean(differences),percentile_95=[bs[249],bs[9749]],teacher_only_correct=sum(t['strict_em']>s['strict_em'] for t,s in zip(per['teacher'],per['student'])),student_only_correct=sum(t['strict_em']<s['strict_em'] for t,s in zip(per['teacher'],per['student']))),teacher_train=tm,training_gate_passed=gate,holdout_n=96,holdout_inference=False,recovered_records=8,total_teacher_records=192)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),ensure_ascii=False,indent=2))
