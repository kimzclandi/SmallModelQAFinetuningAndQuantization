import json
from pathlib import Path
import pytest
from qa_lab.teacher_io import select,training_rows,validate,pack,import_candidates
from qa_lab.common import sha


def test_pack_is_balanced_train_only_and_no_labels(tmp_path):
    rows=training_rows(Path('data/complexity-v1'))
    chosen=select(rows)
    assert len(chosen)==24 and sum(r['is_impossible'] for r in chosen)==12
    assert all(r['split']=='train' for r in chosen)
    out=tmp_path/'pack';pack(Path('data/complexity-v1'),out,24)
    req=json.loads((out/'request.json').read_text())['examples']
    assert all(set(r)=={'id','context','question'} for r in req)
    assert [r['id'] for r in req]==[r['id'] for r in chosen]


@pytest.mark.parametrize('response',[
 {'answers':[{'id':'test-id','answer':'x'}]},
 {'answers':[{'id':'a','answer':'x'},{'id':'a','answer':'x'}]},
 {'answers':[]},
 {'answers':[{'id':'a','answer':'invented'}]},
 {'answers':[{'id':'a','answer':''}]},
 {'answers':[{'id':'a','answer':'No_answer'}]},
])
def test_invalid_teacher_responses_rejected(response):
    with pytest.raises(ValueError):validate(response,[{'id':'a','context':'x','question':'?'}])


def test_import_preserves_raw_and_does_not_claim_teacher_run(tmp_path):
    folder=Path('data/complexity-v1');out=tmp_path/'pack';pack(folder,out,2)
    ids=json.loads((out/'manifest.json').read_text())['ids']
    raw=tmp_path/'synthetic-unit-test.json'
    raw.write_text(json.dumps({'answers':[{'id':i,'answer':'NO_ANSWER'} for i in ids]}))
    dest=tmp_path/'imported'
    import_candidates(folder,out,raw,dest,'SYNTHETIC UNIT TEST ONLY','2026-09-18')
    assert sha(raw)==sha(dest/'original-response.json')
    assert json.loads((dest/'provenance.json').read_text())['teacher_snapshot'] is None
    with pytest.raises(FileExistsError):
        import_candidates(folder,out,raw,dest,'SYNTHETIC UNIT TEST ONLY','2026-09-18')


def test_transport_repair_is_narrow():
    from qa_lab.teacher_audit import parse_received
    parsed,n=parse_received(r'{"answers":[{"id":"a","answer":"NO\_ANSWER"}]}')
    assert n==1 and parsed['answers'][0]['answer']=='NO_ANSWER'
    parsed,n=parse_received('{"answers":[]}')
    assert n==0
    with pytest.raises(json.JSONDecodeError):
        parse_received(r'{"answers":[{"id":"a","answer":"bad\_text"}]}')


def test_received_teacher_audit_reproduces():
    from qa_lab.teacher_audit import parse_received,audit
    raw=Path('data/teacher-received-v1/user-message.txt').read_text().split('\n',1)[1]
    response,n=parse_received(raw)
    metrics,scored,preds=audit(Path('data/complexity-v1'),Path('data/teacher-pilot-v1'),response)
    assert n==13
    assert metrics==json.loads(Path('reports/teacher-audit-v1/metrics.json').read_text())
    assert sum(s['em'] for s in scored)==17


@pytest.mark.parametrize('case',['wrong_hash','test_id','duplicate_id'])
def test_gold_selection_rejects_invalid_source_before_model_loading(tmp_path,monkeypatch,case):
    from qa_lab.train import main
    import sys
    manifest=json.loads(Path('data/teacher-pilot-v1/manifest.json').read_text())
    if case=='wrong_hash':manifest['source_sha256']='invalid'
    elif case=='test_id':manifest['ids']=['not-a-training-id']
    else:manifest['ids']=[manifest['ids'][0]]*2
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys,'argv',['train','--selection',str(selection),'--output',str(tmp_path/'out')])
    with pytest.raises(ValueError):main()
    assert not (tmp_path/'out').exists()
