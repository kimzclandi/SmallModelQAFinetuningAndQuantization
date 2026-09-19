import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from resumable_teacher import produce,record_path

def test_cache_hits_do_not_execute_and_config_change_fails(tmp_path):
 rows=[{'id':'a','context':'x','question':'q'}];calls=[]
 def f(r):calls.append(r['id']);return {'id':r['id'],'prediction':'x'}
 assert produce(rows,tmp_path,{'temperature':0},{'code':'v1'},f)['created']==1
 assert produce(rows,tmp_path,{'temperature':0},{'code':'v1'},f)['reused']==1
 assert calls==['a']
 with pytest.raises(ValueError,match='binding'):produce(rows,tmp_path,{'temperature':1},{'code':'v1'},f)

def test_corrupted_prediction_fails(tmp_path):
 rows=[{'id':'a'}];produce(rows,tmp_path,{}, {},lambda r:{'id':'a','prediction':'x'})
 p=record_path(tmp_path,'a');v=json.loads(p.read_text());v['prediction']['prediction']='changed';p.write_text(json.dumps(v))
 with pytest.raises(ValueError,match='mismatch'):produce(rows,tmp_path,{}, {},lambda r:pytest.fail('should not regenerate'))

def test_midrun_failure_resumes_completed_records(tmp_path):
 rows=[{'id':'a'},{'id':'b'}]
 def fail(r):
  if r['id']=='b':raise RuntimeError('injected')
  return {'id':'a','prediction':'x'}
 with pytest.raises(RuntimeError):produce(rows,tmp_path,{}, {},fail)
 calls=[]
 def recover(r):calls.append(r['id']);return {'id':r['id'],'prediction':'y'}
 result=produce(rows,tmp_path,{}, {},recover);assert calls==['b'] and result['reused']==result['created']==1

def test_duplicate_and_wrong_ids_fail(tmp_path):
 with pytest.raises(ValueError,match='Duplicate'):produce([{'id':'a'},{'id':'a'}],tmp_path,{}, {},lambda r:None)
 with pytest.raises(ValueError,match='wrong ID'):produce([{'id':'a'}],tmp_path,{}, {},lambda r:{'id':'b'})
