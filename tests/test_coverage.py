import copy
import pytest
from scripts.coverage_study import select_holdout


def inputs():
    raw={'data':[{'title':'A','paragraphs':[{'context':'Zinc conducts electricity.','qas':[
      {'id':'x','question':'Which material conducts electricity?','answers':[{'text':'Zinc','answer_start':0}],'is_impossible':False},
      {'id':'y','question':'When was the substance discovered?','answers':[],'is_impossible':True}]}]}]}
    return raw,{'holdout_titles':['A'],'per_article_per_class':1},{'context_jaccard':.7,'question_jaccard':.8,'question_sequence_ratio':.9}


def test_external_balanced_and_old_context_excluded():
    raw,p,c=inputs();rows,_=select_holdout(raw,[],p,c)
    assert len(rows)==2 and sum(r['is_impossible'] for r in rows)==1
    assert {r['split'] for r in rows}=={'test'}
    assert len({r['family_id'] for r in rows})==1
    with pytest.raises(ValueError,match='Insufficient isolated'):
        select_holdout(raw,[rows[0]],p,c)


def test_external_rejects_bad_gold_offset():
    raw,p,c=inputs();raw['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']=1
    with pytest.raises(AssertionError):select_holdout(raw,[],p,c)
