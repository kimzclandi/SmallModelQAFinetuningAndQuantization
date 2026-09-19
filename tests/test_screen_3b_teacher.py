import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from screen_3b_teacher import challenges
from judge_calibration import inputs


def test_challenges_balanced_paired_and_labels_excluded_from_model_input():
    rows=challenges()
    assert len(rows)==len({r['id'] for r in rows})==96
    counts=Counter((r['family'],r['expected']) for r in rows)
    assert len(counts)==12 and set(counts.values())=={8}
    for r in rows:
        assert r['candidate'] in r['context']
        p=inputs(r,{'format':'plain'})
        assert 'expected' not in p.question and 'family' not in p.question
    for pos,neg in zip(rows[::2],rows[1::2]):
        assert pos['expected'] and not neg['expected']
        assert pos['context']==neg['context'] and pos['question']==neg['question']
        assert pos['candidate']!=neg['candidate']


def test_readonly_rates_reject_corrupt_or_missing_scores():
    import pytest
    from verify_3b_screen import rates
    rows=[{'id':'a','expected':True},{'id':'b','expected':False}]
    scores=[dict(id='a',yes_logprob=-1,no_logprob=-2,margin=1),dict(id='b',yes_logprob=-2,no_logprob=-1,margin=-1)]
    assert rates(rows,scores,0)['balanced_accuracy']==1
    with pytest.raises(ValueError,match='coverage'):rates(rows,scores[:1],0)
    scores[0]['margin']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):rates(rows,scores,0)
