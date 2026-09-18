"""Recompute matched dev comparison and apply the preregistered gate. No test inference."""
import argparse
import json
from pathlib import Path
from qa_lab.common import read_jsonl,sha,write_json
from qa_lab.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
root=Path('reports/gold-sft-pilot-v1'); before=Path('reports/baseline-v1');after=root/'eval-dev'
rows=read_jsonl('data/complexity-v1/dev.jsonl')
bm,bs=evaluate(rows,read_jsonl(before/'dev.predictions.jsonl'))
am,ass=evaluate(rows,read_jsonl(after/'dev.predictions.jsonl'))
b_run=json.loads((before/'run.json').read_text());a_run=json.loads((after/'run.json').read_text())
assert b_run['config']==a_run['config']
assert b_run['data_sha256']['dev']==a_run['data_sha256']['dev']
train=json.loads((root/'training.json').read_text());reg=json.loads((root/'preregistration.json').read_text())
assert reg['config_sha256']==sha('configs/gold-sft-pilot-v1.json')
assert reg['selection_sha256']==train['selection_sha256']
assert train['method']=='gold-label supervised LoRA' and len(train['steps'])==48
assert set(train['selected_ids'])==set(json.loads(Path('data/teacher-pilot-v1/manifest.json').read_text())['ids'])
assert {r['split'] for r in read_jsonl(after/'dev.predictions.jsonl')}=={'dev'}
for base,run in [(before,b_run),(after,a_run)]:
 for name,h in json.loads((base/'manifest.json').read_text()).items():assert sha(base/name)==h
for name,h in train['source_sha256'].items():assert sha(root/'source'/name)==h
b={s['id']:s for s in bs}; aa={s['id']:s for s in ass}
fixed=[i for i in b if b[i]['em']==0 and aa[i]['em']==1]
regressed=[i for i in b if b[i]['em']==1 and aa[i]['em']==0]
accepted=am['overall']['em']>19/74 and am['answerable']['em']>=19/33
result={'baseline':bm,'gold_sft':am,'fixed_ids':fixed,'regressed_ids':regressed,
        'fixed_count':len(fixed),'regressed_count':len(regressed),'gate':'PASS' if accepted else 'REJECT',
        'gate_reason':'Requires higher overall EM AND no answerable EM regression versus baseline',
        'teacher_training_performed':False,'new_test_inference_performed':False,
        'preregistration_sha256':sha(root/'preregistration.json')}
write_json(a.output,result)
print(json.dumps({'fixed':len(fixed),'regressed':len(regressed),'gate':result['gate']},indent=2))
