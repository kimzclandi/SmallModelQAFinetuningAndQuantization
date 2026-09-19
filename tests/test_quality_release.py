import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from verify_quality_release import verify
from reproduce_quality_release import run

def test_release_missing_or_extra_files_fail_before_recomputation(tmp_path):
    (tmp_path/'release-manifest.json').write_text(json.dumps({'files':{'missing.json':'abc'}}))
    with pytest.raises(ValueError,match='coverage'):verify(tmp_path)
    (tmp_path/'release-manifest.json').write_text(json.dumps({'files':{}}))
    (tmp_path/'unexpected.json').write_text('{}')
    with pytest.raises(ValueError,match='coverage'):verify(tmp_path)

def test_public_reproduction_refuses_existing_output_without_loading_model(tmp_path):
    with pytest.raises(FileExistsError):run('smoke',tmp_path)

def test_retrain_wrapper_copies_frozen_inputs_to_new_protocol(tmp_path,monkeypatch):
    import supervised_quality_control,verify_supervised_quality
    from qa_lab.common import sha
    called=[]
    def train(root):
        folder=root/'supervised-07';cfg=json.loads((folder/'protocol.json').read_text())
        assert cfg['steps']==64 and len(cfg['groups'])==3 and len(cfg['seeds'])==3
        for n,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/n)==h
        for n,h in cfg['source_hashes'].items():assert sha(root/n)==h
        assert not (root/'fresh-05/holdout.jsonl').exists()
        called.append('train')
    monkeypatch.setattr(supervised_quality_control,'run',train)
    monkeypatch.setattr(verify_supervised_quality,'verify',lambda root:{'status':'test-double'})
    output=tmp_path/'new';run('train',output)
    assert called==['train'] and (output/'verification.json').exists()
