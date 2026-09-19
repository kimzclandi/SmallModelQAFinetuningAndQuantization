import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from judge_calibration import synthetic,inputs,PROMPTS,metrics
from judge_scores import threshold,summarize

def test_synthetic_coverage_and_template_isolation():
 c=synthetic('calibration');v=synthetic('validation')
 assert len(c)==64 and len(v)==64 and sum(r['expected'] for r in c)==32
 assert len({r['id'] for r in c+v})==128
 assert not {r['template'] for r in c}&{r['template'] for r in v}
 for r in c+v:assert r['candidate'] in r['context']

def test_expected_label_is_never_in_prompt():
 r=dict(context='上下文',question='问题',candidate='候选',expected='SECRET_LABEL',split='SECRET_SPLIT')
 for prompt in PROMPTS.values():
  x=inputs(r,prompt);assert 'SECRET' not in x.context+x.question

def test_invalid_output_is_not_classified_as_correct_negative():
 rows=[{'id':'p','expected':True},{'id':'n','expected':False}]
 preds=[dict(id='p',prediction='YES',stop_reason='eos'),dict(id='n',prediction='maybe',stop_reason='eos')]
 m=metrics(rows,preds,PROMPTS['english_short_span']);assert m['tpr']==1 and m['tnr']==0 and m['invalid']==1

def test_score_threshold_separates_known_margins():
 rows=[dict(id=str(i),expected=i>=2) for i in range(4)]
 scores=[dict(id=str(i),margin=m) for i,m in enumerate([-2,-1,1,2])]
 r=threshold(rows,scores);assert r['tpr']==r['tnr']==1
 with pytest.raises(ValueError,match='coverage'):summarize(rows,scores+[scores[0]],0)

def test_constant_score_cannot_pass_both_class_gates():
 rows=[dict(id='p',expected=True),dict(id='n',expected=False)]
 r=threshold(rows,[dict(id='p',margin=1),dict(id='n',margin=1)])
 assert min(r['tpr'],r['tnr'])==0
