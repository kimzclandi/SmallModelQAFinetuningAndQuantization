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

def test_corrupt_input_is_rejected_before_model_or_output(tmp_path,monkeypatch):
    import reproduce_quality_release as module
    from qa_lab.common import sha
    release=tmp_path/'release';release.mkdir();source=release/'sample.json'
    source.write_text('{}');manifest={'files':{'sample.json':sha(source)}}
    (release/'release-manifest.json').write_text(json.dumps(manifest));source.write_text('changed')
    monkeypatch.setattr(module,'RELEASE',release)
    monkeypatch.setattr(module,'smoke',lambda *args:pytest.fail('Model must not run'))
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='hash mismatch'):module.run('smoke',output)
    assert not output.exists()


def test_frozen_output_and_symlink_alias_are_rejected(tmp_path,monkeypatch):
    import reproduce_quality_release as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    for name in ('reports','data','configs'):
        (tmp_path/name).mkdir()
        with pytest.raises(ValueError,match='frozen'):module.run('smoke',tmp_path/name/'new')
    alias=tmp_path/'alias';alias.symlink_to(tmp_path/'reports',target_is_directory=True)
    with pytest.raises(ValueError,match='frozen'):module.run('train',alias/'new')
    assert not (tmp_path/'reports/new').exists()


def test_partial_inference_and_failure_receipt_survive_exception(tmp_path,monkeypatch):
    import types
    import reproduce_quality_release as module
    from qa_lab import inference
    calls=[]
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(set_num_threads=lambda n:None))
    monkeypatch.setattr(inference,'load',lambda cfg:(None,None))
    def generate(*args):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('injected second-generation failure')
        return {'prediction':'saved first answer','stop_reason':'eos'}
    monkeypatch.setattr(inference,'generate',generate)
    output=tmp_path/'out'
    with pytest.raises(RuntimeError,match='injected'):module.run('smoke',output)
    rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['prediction']=='saved first answer'
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['error_type']=='RuntimeError'
    with pytest.raises(FileExistsError):module.run('smoke',output)
