import json
import shutil
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from external_drcd import ROOT, check
from qa_lab.common import sha

def test_external_inputs_reject_modification(tmp_path):
    root=ROOT/'reports/external-drcd-20260920'
    for n in ('manifest.json','protocol.json','eval.jsonl'):
        shutil.copyfile(root/n,tmp_path/n)
    shutil.copyfile(root/'selection.json',tmp_path/'selection.json')
    with (tmp_path/'eval.jsonl').open('a') as f:f.write('{}\n')
    with pytest.raises(ValueError,match='input changed'):check(tmp_path)

def test_external_articles_cannot_be_counted_twice(tmp_path):
    root=ROOT/'reports/external-drcd-20260920'
    for n in ('manifest.json','protocol.json','eval.jsonl','selection.json'):
        shutil.copyfile(root/n,tmp_path/n)
    path=tmp_path/'eval.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[1]['family_id']=rows[0]['family_id'];path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    path=tmp_path/'manifest.json';m=json.loads(path.read_text());m['eval.jsonl']=sha(tmp_path/'eval.jsonl');path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Article coverage'):check(tmp_path)

@pytest.mark.parametrize('change',['empty_manifest','missing_entry','extra_entry','duplicate_id'])
def test_external_input_contract_cannot_be_bypassed(tmp_path,change):
    root=ROOT/'reports/external-drcd-20260920'
    for n in ('manifest.json','protocol.json','eval.jsonl','selection.json'):shutil.copyfile(root/n,tmp_path/n)
    path=tmp_path/'manifest.json';m=json.loads(path.read_text())
    if change=='empty_manifest':m={}
    if change=='missing_entry':del m['eval.jsonl']
    if change=='extra_entry':
        (tmp_path/'extra.json').write_text('{}');m['extra.json']=sha(tmp_path/'extra.json')
    if change=='duplicate_id':
        p=tmp_path/'eval.jsonl';rr=[json.loads(s) for s in p.read_text().splitlines()];rr[1]['id']=rr[0]['id'];p.write_text(''.join(json.dumps(r)+'\n' for r in rr));m['eval.jsonl']=sha(p)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):check(tmp_path)
