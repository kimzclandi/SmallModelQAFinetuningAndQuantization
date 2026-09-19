import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from supervised_quality_control import groups

def fixture():
    rows=[dict(id=str(i),context='甲乙',question='谁',answers=['甲'],is_impossible=False) for i in range(20)]
    preds=[dict(id=str(i),prediction='甲' if i<10 else '乙',stop_reason='eos') for i in range(20)]
    return rows,preds

def test_same_ids_and_targets_isolate_label_replacement():
    rows,preds=fixture();g=groups(rows,preds,20260920)
    assert {len(v) for v in g.values()}=={10}
    assert [r['id'] for r in g['random_teacher']]==[r['id'] for r in g['random_gold']]
    assert all(r['target']=='甲' for r in g['oracle_selected']+g['random_gold'])
    assert any(r['target']=='乙' for r in g['random_teacher'])
    assert g==groups(rows,list(reversed(preds)),20260920)

def test_incomplete_predictions_and_non_eos_matches_not_accepted():
    rows,preds=fixture()
    with pytest.raises(ValueError,match='coverage'):groups(rows,preds[:-1],1)
    for p in preds[:4]:p['stop_reason']='max_new_tokens'
    with pytest.raises(ValueError,match='Insufficient'):groups(rows,preds,1)
