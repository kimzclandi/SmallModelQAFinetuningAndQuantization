from pathlib import Path
import pytest
from qa_lab.common import read_jsonl,sha,write_json,write_jsonl
from qa_lab.train_artifact import verified_rows
from scripts.teacher_study import gold_artifact,paired,aggregate


def test_gold_control_same_pilot_and_rejects_relabel(tmp_path):
    out=tmp_path/'gold';gold_artifact(out)
    rows,manifest=verified_rows(out,Path('data/complexity-v1'))
    assert manifest['method']=='gold_sft' and len(rows)==24
    assert sum(r['target']=='NO_ANSWER' for r in rows)==12
    rows[0]['target']='invented'
    write_jsonl(out/'train.jsonl',rows)
    manifest['train_sha256']=sha(out/'train.jsonl');write_json(out/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Gold target changed'):
        verified_rows(out,Path('data/complexity-v1'))


def test_paired_reports_regressions_even_when_net_zero():
    before=[{'id':'a','em':1},{'id':'b','em':0}]
    after=[{'id':'b','em':1},{'id':'a','em':0}]
    result=paired(before,after)
    assert result=={'fixes':['b'],'regressions':['a'],'delta_em':0}
    with pytest.raises(ValueError):paired(before,after[:1])


def test_three_seed_sample_std_not_standard_error():
    result=aggregate([0.25,0.5,0.75])
    assert result['mean']==0.5 and result['sample_std']==0.25 and result['n']==3
