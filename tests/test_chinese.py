import pytest
from qa_lab.chinese import normalize_zh,lcs_length,evaluate_zh,select


def row():return {'id':'a','context':'首都是北京。','question':'首都在哪里？','answers':['北京'],'is_impossible':False}


def test_chinese_lcs_respects_order_and_unicode_punctuation():
 assert normalize_zh(' “北京”， A！ ')=='北京a'
 assert lcs_length('甲乙','乙甲')==1
 assert lcs_length('甲甲乙','甲乙')==2


def test_strict_and_normalized_scores_are_distinct():
 m,_=evaluate_zh([row()],[{'id':'a','prediction':'“北京”'}])
 assert m['strict_em']==0 and m['normalized_em']==1 and m['char_lcs_f1']==1 and m['format_valid']==0


@pytest.mark.parametrize('prediction',['','。','NO_ANSWER'])
def test_empty_punctuation_and_abstention_never_correct(prediction):
 m,_=evaluate_zh([row()],[{'id':'a','prediction':prediction}])
 assert m['strict_em']==m['normalized_em']==m['char_lcs_f1']==0


def test_coverage_and_multi_reference():
 r=row();r['answers'].append('北京。')
 m,_=evaluate_zh([r],[{'id':'a','prediction':'北京。'}]);assert m['strict_em']==1
 with pytest.raises(ValueError):evaluate_zh([r],[{'id':'a','prediction':'北京'},{'id':'a','prediction':'北京'}])
 with pytest.raises(ValueError):evaluate_zh([r],[])


def test_selector_excludes_prior_content_and_overlength():
 raw={'data':[{'title':'城市','paragraphs':[{'context':'首都是北京。','qas':[{'id':'a','question':'首都在哪里？','answers':[{'text':'北京','answer_start':3}]}]}]}]}
 selected,_=select(raw,[],lambda r:10,n=1);assert selected[0]['id']=='a'
 with pytest.raises(ValueError):select(raw,[row()],lambda r:10,n=1)
 with pytest.raises(ValueError):select(raw,[],lambda r:2049,n=1)
