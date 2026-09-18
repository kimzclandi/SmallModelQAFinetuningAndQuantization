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
