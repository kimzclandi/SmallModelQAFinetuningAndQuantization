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


def test_training_rejects_duplicate_teacher_predictions(tmp_path):
    import shutil
    from qa_lab.common import read_jsonl, write_jsonl
    artifact = tmp_path / 'distilled'
    teacher = tmp_path / 'teacher'
    shutil.copytree('data/distilled-local-v1', artifact)
    shutil.copytree('reports/closure-v1/teacher-fp32', teacher)
    predictions = read_jsonl(teacher / 'predictions.jsonl')
    write_jsonl(teacher / 'predictions.jsonl', predictions + [predictions[0]])
    manifest = json.loads((teacher / 'manifest.json').read_text())
    manifest['predictions.jsonl'] = sha(teacher / 'predictions.jsonl')
    write_json(teacher / 'manifest.json', manifest)
    m = json.loads((artifact / 'manifest.json').read_text())
    m.update(teacher_report=str(teacher), teacher_manifest_sha256=sha(teacher / 'manifest.json'))
    write_json(artifact / 'manifest.json', m)
    with pytest.raises(ValueError, match='Teacher.*(coverage|ID)'):
        verified_rows(artifact, Path('data/complexity-v1'))
