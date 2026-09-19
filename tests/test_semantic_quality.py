import importlib.util,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from semantic_quality_pilot import judge_input,supported,build_groups

def test_no_gold_in_judge_input():
 r={'context':'甲在东城，乙在西城。','question':'甲在哪里？','answers':['DO_NOT_TRANSMIT'],'split':'holdout'}
 v=judge_input(r,'东城')
 assert 'DO_NOT_TRANSMIT' not in v.context+v.question
 assert 'holdout' not in v.context+v.question

@pytest.mark.parametrize('s,stop',[('UNSUPPORTED','eos'),('SUPPORTED because','eos'),('SUPPORTED','max_new_tokens'),('','eos')])
def test_invalid_judge_not_accepted(s,stop):assert not supported({'prediction':s,'stop_reason':stop})

def test_group_selection_and_same_id_gold():
 rows=[{'id':str(i),'context':'甲乙','question':'?','answers':['乙']} for i in range(4)]
 pred=[{'id':str(i),'prediction':'甲','stop_reason':'eos'} for i in range(4)]
 judges=[{'id':str(i),'prediction':'SUPPORTED' if i<2 else 'UNSUPPORTED','stop_reason':'eos'} for i in range(4)]
 groups=build_groups(rows,pred,judges,1)
 assert [r['id'] for r in groups['semantic']]==['0','1']
 assert len(groups['random_structural'])==2
 assert [r['id'] for r in groups['selected_gold']]==['0','1']
 assert groups['semantic'][0]['target']=='甲' and groups['selected_gold'][0]['target']=='乙'
 with pytest.raises(ValueError,match='Duplicate'):build_groups(rows,pred,judges+[judges[0]],1)
 with pytest.raises(ValueError,match='coverage'):build_groups(rows,pred,judges[:-1],1)

def test_verifier_rejects_partial_run(tmp_path):
 import json
 from qa_lab.common import sha
 from verify_semantic_quality import verify
 (tmp_path/'protocol.json').write_text('{}')
 (tmp_path/'manifest.json').write_text(json.dumps({'protocol.json':sha(tmp_path/'protocol.json')}))
 (tmp_path/'run').mkdir();(tmp_path/'run/run.json').write_text('{"status":"running"}')
 with pytest.raises(ValueError,match='Not a completed'):verify(tmp_path)
 (tmp_path/'protocol.json').write_text('{"changed":true}')
 with pytest.raises(ValueError,match='Input hash'):verify(tmp_path)
