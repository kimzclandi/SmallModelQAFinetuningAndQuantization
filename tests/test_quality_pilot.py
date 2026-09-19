import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('quality_pilot',Path(__file__).parents[1]/'scripts/quality_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_filter_does_not_read_gold():
    row={'context':'北京是中国的首都。'}
    assert m.accept(row,{'prediction':'北京','stop_reason':'eos'})
    assert m.accept(row,{'prediction':'中国','stop_reason':'eos'})  # valid span is not semantic correctness

@pytest.mark.parametrize('text,stop',[('','eos'),('NO_ANSWER','eos'),('上海','eos'),('北京','max_new_tokens')])
def test_filter_rejects_bad_structure(text,stop):
    assert not m.accept({'context':'北京是中国的首都。'},{'prediction':text,'stop_reason':stop})

def test_frozen_inputs_checked(tmp_path):
    m.dump(tmp_path/'protocol.json',{'steps':64})
    m.dump(tmp_path/'data_manifest.json',{'files':{'protocol.json':m.sha(tmp_path/'protocol.json')}})
    assert m.verify(tmp_path)['steps']==64
    m.dump(tmp_path/'protocol.json',{'steps':65})
    with pytest.raises(ValueError,match='Frozen input'):m.verify(tmp_path)
