"""Read-only independent arithmetic for a saved 3B screen, no model imports required."""
import argparse,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,sha
from qa_lab.chinese import evaluate_zh

def rates(rows,scores,t):
    by={s['id']:s for s in scores}
    if len(by)!=len(scores) or set(by)!={r['id'] for r in rows}:raise ValueError('Score coverage')
    for s in scores:
        if not all(math.isfinite(s[k]) for k in ['yes_logprob','no_logprob','margin']):raise ValueError('Nonfinite score')
        if abs(s['yes_logprob']-s['no_logprob']-s['margin'])>1e-12:raise ValueError('Margin inconsistent')
    p=[by[r['id']]['margin']>=t for r in rows if r['expected']]
    n=[by[r['id']]['margin']<t for r in rows if not r['expected']]
    a,b=sum(p)/len(p),sum(n)/len(n)
    return dict(tpr=a,tnr=b,balanced_accuracy=(a+b)/2,threshold=t)

def verify(root):
    folder=root/'teacher-3b-06';run=folder/'run';protocol=json.loads((folder/'protocol.json').read_text())
    for n,h in protocol['hashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    cal=read_jsonl(folder/'calibration.jsonl');scores=read_jsonl(run/'calibration-scores.jsonl')
    values=sorted({s['margin'] for s in scores});ts=[values[0]-1]+[(a+b)/2 for a,b in zip(values,values[1:])]+[values[-1]+1]
    cm=min([rates(cal,scores,t) for t in ts],key=lambda m:(-min(m['tpr'],m['tnr']),-m['balanced_accuracy'],abs(m['threshold'])))
    frozen=json.loads((run/'threshold-before-validation.json').read_text())
    if any(frozen[k]!=v for k,v in cm.items()):raise ValueError('Threshold selection changed')
    val=read_jsonl(folder/'validation.jsonl');vs=read_jsonl(run/'validation-scores.jsonl');vm=rates(val,vs,cm['threshold']);families={}
    for f in sorted({r['family'] for r in val}):
        subset=[r for r in val if r['family']==f];ids={r['id'] for r in subset};families[f]=rates(subset,[s for s in vs if s['id'] in ids],cm['threshold'])
    if dict(calibration=cm,validation=vm,families=families)!=json.loads((run/'judge-summary.json').read_text()):raise ValueError('Judge summary mismatch')
    metrics,items=evaluate_zh(read_jsonl(root/'fresh-05/dev.jsonl'),read_jsonl(run/'dev-predictions.jsonl'))
    if metrics!=json.loads((run/'dev-metrics.json').read_text()) or items!=read_jsonl(run/'dev-scored.jsonl'):raise ValueError('QA scores mismatch')
    judge=all(m['tpr']>=.875 and m['tnr']>=.875 for m in [cm,vm]) and all(m['tpr']>=.75 and m['tnr']>=.75 for m in families.values())
    teacher=metrics['strict_em']>.375 and metrics['char_lcs_f1']>=.708195752464856
    decision=json.loads((run/'decision.json').read_text())
    if decision['proceed']!=(judge and teacher) or decision['judge_passed']!=judge or decision['teacher_passed']!=teacher:raise ValueError('Gate mismatch')
    return dict(status='pass',n_validation=len(val),n_dev=len(items),judge_passed=judge,teacher_passed=teacher,proceed=judge and teacher,metrics=metrics)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
