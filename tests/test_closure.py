import json
from pathlib import Path
import pytest
from qa_lab.closure_data import build_repair
from qa_lab.train_artifact import verified_rows
from qa_lab.common import sha,write_json


def test_repair_train_only_and_no_relabel(tmp_path):
    out=tmp_path/'repair';build_repair(Path('data/complexity-v1'),Path('data/teacher-pilot-v1/manifest.json'),out)
    rows,m=verified_rows(out,Path('data/complexity-v1'))
    assert len(rows)==48 and len(m['added_ids'])==24
    assert sum(r['target']=='NO_ANSWER' for r in rows)==12
    assert all(r['split']=='train' for r in rows)


@pytest.mark.parametrize('mutation',['target','id','context'])
def test_training_artifact_rejects_forged_rows(tmp_path,mutation):
    out=tmp_path/'repair';build_repair(Path('data/complexity-v1'),Path('data/teacher-pilot-v1/manifest.json'),out)
    rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
    rows[0][mutation]='not-in-source'
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    m=json.loads((out/'manifest.json').read_text());m['train_sha256']=sha(out/'train.jsonl');write_json(out/'manifest.json',m)
    with pytest.raises(ValueError):verified_rows(out,Path('data/complexity-v1'))
