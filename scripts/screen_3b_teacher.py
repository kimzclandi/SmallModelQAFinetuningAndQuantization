"""Frozen larger-teacher screen; synthetic validation is a challenge set, not natural QA."""
import argparse,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha
from qa_lab.inference import load
from qa_lab.chinese import evaluate_zh
from judge_scores import score,threshold,summarize
from quality_pilot import predict

def challenges():
    rows=[]
    for i in range(8):
        a=f'甲{i}号';b=f'乙{i}号';c=f'丙{i}号'
        cases=[
          ('comparison',f'{a}比{b}高，{b}比{c}高。', '谁最高？',a,c),
          ('transfer',f'钥匙原来由{a}保管，后来{a}将钥匙交给{b}，此后没有转交。','现在钥匙由谁保管？',b,a),
          ('exception',f'名单中有{a}、{b}。除{b}外，名单中的人都参加了会议。','名单中谁没有参加会议？',b,a),
          ('conditional',f'规则规定：若合格，则发给{a}；若不合格，则发给{b}。本件产品不合格。','按照规则应发给谁？',b,a),
          ('correction',f'初稿将负责人写成{a}，正式更正为{b}，以更正内容为准。','更正后的负责人是谁？',b,a),
          ('role',f'{a}向{b}借了一本书，书属于{b}，{a}只借阅三天。','这本书属于谁？',b,a)]
        for family,context,question,yes,no in cases:
            for expected,candidate in [(True,yes),(False,no)]:
                rows.append(dict(id=f'3b-{family}-{i}-{int(expected)}',family=family,context=context,question=question,candidate=candidate,expected=expected))
    return rows

def prepare(root):
    out=root/'teacher-3b-06';parent=json.loads((root/'calibration-04/protocol.json').read_text())
    revision=json.loads((out/'download-plan.json').read_text())['revision']
    cfg=dict(parent['teacher'],model_id='Qwen/Qwen2.5-3B-Instruct',revision=revision,warmup='none')
    data=read_jsonl(root/'calibration-04/calibration.jsonl');valid=challenges()
    if (out/'protocol.json').exists():raise FileExistsError('Frozen protocol exists')
    write_jsonl(out/'calibration.jsonl',data);write_jsonl(out/'validation.jsonl',valid)
    write_json(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),teacher=cfg,prompt=parent['prompts']['chinese_short_span'],
        hashes={str(p.relative_to(root)):sha(p) for p in [out/'calibration.jsonl',out/'validation.jsonl',root/'fresh-05/dev.jsonl']},
        selection='One fixed Chinese short-span prompt; threshold learned only on the existing calibration64. No prompt selection on new validation.',
        gate='Calibration and validation TPR/TNR >= .875, each validation family TPR/TNR >= .75; new teacher dev strict EM > existing 1.5B .375 and F1 >= .708195752464856. All required.',
        max_wall_seconds=1800,scope='New hand-constructed challenge instances informed by prior failure; not independent natural-data validation. Existing dev reused for resource allocation, never independent final evaluation. Holdout96 untouched.',source_sha256=sha(Path(__file__))))

def run(root):
    import torch
    out=root/'teacher-3b-06';cfgp=json.loads((out/'protocol.json').read_text())
    for n,h in cfgp['hashes'].items():
        if sha(root/n)!=h:raise ValueError('Frozen input mismatch')
    if sha(Path(__file__))!=cfgp['source_sha256']:raise ValueError('Runner changed')
    dest=out/'run';dest.mkdir(exist_ok=False);start=time.monotonic();deadline=start+cfgp['max_wall_seconds']
    receipt=dict(status='running',protocol_sha256=sha(out/'protocol.json'));write_json(dest/'receipt.json',receipt)
    try:
        torch.set_num_threads(8);cfg=cfgp['teacher'];tok,model=load(cfg);prompt=cfgp['prompt'];jc=dict(cfg,system_prompt=prompt['system'])
        def scores(rows,name):
            results=[]
            for row in rows:
                if time.monotonic()>deadline:raise TimeoutError('Screen budget')
                results.append(score(row,prompt,jc,tok,model));write_jsonl(dest/name,results)
            return results
        cal=read_jsonl(out/'calibration.jsonl');cs=scores(cal,'calibration-scores.jsonl');cm=threshold(cal,cs)
        write_json(dest/'threshold-before-validation.json',dict(**cm,utc=datetime.now(timezone.utc).isoformat()))
        val=read_jsonl(out/'validation.jsonl');vs=scores(val,'validation-scores.jsonl');vm=summarize(val,vs,cm['threshold']);families={}
        for f in sorted({r['family'] for r in val}):
            subset=[r for r in val if r['family']==f];ids={r['id'] for r in subset};families[f]=summarize(subset,[p for p in vs if p['id'] in ids],cm['threshold'])
        write_json(dest/'judge-summary.json',dict(calibration=cm,validation=vm,families=families));print('judge',vm,families,flush=True)
        qc=json.loads((root/'fresh-05/data_protocol.json').read_text())['student'];qc.update(model_id=cfg['model_id'],revision=cfg['revision'])
        dev=read_jsonl(root/'fresh-05/dev.jsonl');preds=predict(dev,qc,tok,model,dest/'dev-predictions.jsonl',deadline);metrics,items=evaluate_zh(dev,preds)
        write_json(dest/'dev-metrics.json',metrics);write_jsonl(dest/'dev-scored.jsonl',items)
        judge=all(m['tpr']>=.875 and m['tnr']>=.875 for m in [cm,vm]) and all(m['tpr']>=.75 and m['tnr']>=.75 for m in families.values())
        teacher=metrics['strict_em']>.375 and metrics['char_lcs_f1']>=.708195752464856
        write_json(dest/'decision.json',dict(proceed=judge and teacher,judge_passed=judge,teacher_passed=teacher,teacher_metrics=metrics,holdout_inference=False))
        receipt['status']='complete';print('decision',judge,teacher,metrics,flush=True)
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat());write_json(dest/'receipt.json',receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('root',type=Path);a=p.parse_args();(prepare if a.stage=='prepare' else run)(a.root)
