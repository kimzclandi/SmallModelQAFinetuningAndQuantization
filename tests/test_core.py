import json
from pathlib import Path
import pytest
from qa_lab.data import prepare, related
from qa_lab.common import read_jsonl, sha
from qa_lab.inference import QAInput, messages
from qa_lab.metrics import evaluate, pair_score, score


def row(**kw):
    x=dict(id='1',context='A queue uses first in first out.',question='What order?',
           answers=['first in first out'],is_impossible=False,family_id='f')
    x.update(kw)
    return x


def test_normalization_and_multiset_f1():
    assert pair_score('The Cat!','cat')==(1,1)
    assert pair_score('x x y','x y y')[1]==pytest.approx(2/3)


def test_multiple_gold():
    assert score(row(answers=['FIFO','first in first out']),dict(prediction='FIFO'))['em']==1


def test_empty_is_not_correct_abstention():
    assert score(row(is_impossible=True,answers=[]),dict(prediction=''))['em']==0
    assert score(row(is_impossible=True,answers=[]),dict(prediction='NO_ANSWER'))['em']==1
    assert score(row(is_impossible=True,answers=[]),dict(prediction='NO_ANSWER.'))['em']==0


def test_format_separate_from_quality():
    s=score(row(),dict(prediction='The first in first out.'))
    assert s['em']==1 and not s['format_valid']


@pytest.mark.parametrize('preds', [[],[{'id':'2','prediction':'x'}],
    [{'id':'1','prediction':'x'},{'id':'1','prediction':'x'}]])
def test_coverage(preds):
    with pytest.raises(ValueError):
        evaluate([row()],preds)


def test_prompt_does_not_accept_labels():
    cfg=json.loads(Path('configs/baseline.json').read_text())
    item=QAInput('passage','question')
    assert 'SECRET_GOLD' not in str(messages(item,cfg))
    with pytest.raises(TypeError):
        QAInput('passage','question',answers=['SECRET_GOLD'])


def test_frozen_data_and_splits():
    folder=Path('data/complexity-v1')
    manifest=json.loads((folder/'manifest.json').read_text())
    for file,expected in manifest['files'].items():
        assert sha(folder/file)==expected
    rows=sum([read_jsonl(folder/f'{s}.jsonl') for s in ['train','dev','test']],[])
    assert len(rows)==len({r['id'] for r in rows})
    for i,a in enumerate(rows):
        for b in rows[:i]:
            if a['split']!=b['split']:
                assert a['family_id']!=b['family_id']
                assert not related(a,b,manifest['config'])


def test_transitive_question_family_and_dedupe():
    cfg=json.loads(Path('configs/data.json').read_text())
    paragraphs=[]
    for i,context in enumerate(['red green blue','dog cat mouse','iron gold copper']):
        paragraphs.append(dict(context=context,qas=[dict(id=str(i),question='Which item is present?',
             answers=[],is_impossible=True)]))
    paragraphs.append(paragraphs[0])
    rows,audit=prepare({'data':[{'title':cfg['title'],'paragraphs':paragraphs}]},cfg)
    assert len(rows)==3 and audit['exact_duplicates_removed']==['0']
    assert len({r['family_id'] for r in rows})==1
    assert len({r['split'] for r in rows})==1


def test_denied_hardware_query_does_not_block_inference(monkeypatch):
    import qa_lab.inference as inf
    def denied(*args,**kwargs):
        raise inf.subprocess.CalledProcessError(1,'sysctl')
    monkeypatch.setattr(inf.sys,'platform','darwin')
    monkeypatch.setattr(inf.subprocess,'check_output',denied)
    assert inf.hardware_chip()=='unknown (hardware query denied)'
