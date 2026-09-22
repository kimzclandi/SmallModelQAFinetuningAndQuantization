"""Evidence structure and provenance must not promote fixtures into model runs."""
import json
import pytest
from qa_lab.common import sha
from qa_lab.quality_control import INDEX_KEYS
from scripts import synthetic_qc, report_output
from scripts.verify_synthetic_qc import verify_run

@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(report_output, 'ROOT', tmp_path)
    row=dict(id='fixture',split='train',family_id='f',context='Alice met Bob.',
             question='Who met Bob?',answers=['Alice'],is_impossible=False)
    output=tmp_path/'work/run'
    synthetic_qc.run([row],[dict(id='fixture',prediction='Alice',stop_reason='eos')],
                     [{k:row[k] for k in INDEX_KEYS}], 'en','train_reference',output)
    return output

def test_custom_run_does_not_claim_real_teacher(saved):
    receipt=json.loads((saved/'run.json').read_text())
    assert receipt.get('input_origin') == {'kind':'custom_unverified'}
    assert 'Real saved teacher' not in receipt['scope']

@pytest.mark.parametrize('mutation',['symlink','extra_directory','extra_file','missing_code_binding'])
def test_evidence_contract_rejects_structural_mutations(saved,mutation):
    manifest=json.loads((saved/'manifest.json').read_text())
    if mutation=='symlink':
        f=saved/'records.jsonl';outside=saved.parent/'outside.jsonl'
        outside.write_bytes(f.read_bytes());f.unlink();f.symlink_to(outside)
    elif mutation=='extra_directory':
        (saved/'untracked-directory').mkdir()
    elif mutation=='extra_file':
        (saved/'untracked.txt').write_text('not part of the QC format')
        manifest['untracked.txt']=sha(saved/'untracked.txt')
    else:
        receipt=json.loads((saved/'run.json').read_text());receipt['code_sha256']={}
        (saved/'run.json').write_text(json.dumps(receipt));manifest['run.json']=sha(saved/'run.json')
    (saved/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):verify_run(saved)


def test_custom_input_cannot_claim_preset_origin(tmp_path,monkeypatch):
    monkeypatch.setattr(report_output,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Preset origin'):
        synthetic_qc.run([],[],[],'en','train_reference',tmp_path/'work/forged',
                         preset_name='english-original')
    assert not (tmp_path/'work/forged').exists()
