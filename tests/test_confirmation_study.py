import json
from pathlib import Path

import pytest

from qa_lab.confirmation_study import checked_inputs, digest, lock_predictions, validate_bindings, verify_paired_training, write


def test_prediction_worker_rejects_label_fields(tmp_path):
    path = tmp_path/'inputs.jsonl'
    row = dict(id='x', context='a', question='q', answers=['a'])
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='label'):
        checked_inputs(path, digest(path))


def test_prediction_inputs_bind_exact_bytes_and_unique_ids(tmp_path):
    path = tmp_path/'inputs.jsonl'
    row = dict(id='x', context='a', question='q')
    path.write_text(json.dumps(row)+'\n')
    expected = digest(path)
    assert checked_inputs(path, expected) == [row]
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='hash'):
        checked_inputs(path, expected)
    with pytest.raises(ValueError, match='duplicate'):
        checked_inputs(path, digest(path))


def test_source_binding_rejects_mutation_and_escape(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('v1')
    expected = digest(path)
    validate_bindings({'source.py': expected}, tmp_path)
    path.write_text('v2')
    with pytest.raises(ValueError, match='changed'):
        validate_bindings({'source.py': expected}, tmp_path)
    with pytest.raises(ValueError, match='changed'):
        validate_bindings({'../escape': expected}, tmp_path)


def test_lock_requires_all_completed_predictions_and_exact_coverage(tmp_path):
    cohort = tmp_path/'cohort'
    cohort.mkdir()
    inputs = cohort/'inputs.jsonl'
    inputs.write_text(json.dumps(dict(id='x', context='a', question='q'))+'\n')
    # Invalid JSON proves lock construction hashes labels but never parses them.
    (cohort/'sealed-labels.jsonl').write_text('SEALED: not parsed here')
    write(tmp_path/'execution-lock.json', {'adapters': {'gold:1': {'files': {'a': 'hash'}}}, 'inputs_sha256': digest(inputs)})
    folder = tmp_path/'predictions'/'gold-1'
    folder.mkdir(parents=True)
    pred = folder/'predictions.jsonl'
    pred.write_text(json.dumps(dict(id='x', prediction='a'))+'\n')
    receipt = dict(status='running', execution_lock_sha256=digest(tmp_path/'execution-lock.json'),
                   inputs_sha256=digest(inputs), predictions_sha256=digest(pred), key='gold:1', adapter_files={'a':'hash'}, n=1)
    write(folder/'run.json', receipt)
    with pytest.raises(ValueError, match='Incomplete'):
        lock_predictions(tmp_path, cohort, ['gold:1'])
    receipt['status'] = 'complete'
    write(folder/'run.json', receipt)
    result = lock_predictions(tmp_path, cohort, ['gold:1'])
    assert result['status'] == 'all_predictions_locked_before_scoring'
    receipt['key'] = 'full_kd:1'
    write(folder/'run.json', receipt)
    with pytest.raises(ValueError, match='Incomplete'):
        lock_predictions(tmp_path, cohort, ['gold:1'])
    receipt['key'] = 'gold:1'
    receipt['adapter_files'] = {'a':'swapped'}
    write(folder/'run.json', receipt)
    with pytest.raises(ValueError, match='Incomplete'):
        lock_predictions(tmp_path, cohort, ['gold:1'])
    receipt['adapter_files'] = {'a':'hash'}
    pred.write_text(json.dumps(dict(id='wrong', prediction='a'))+'\n')
    receipt['predictions_sha256'] = digest(pred)
    write(folder/'run.json', receipt)
    with pytest.raises(ValueError, match='coverage'):
        lock_predictions(tmp_path, cohort, ['gold:1'])
    with pytest.raises(ValueError, match='all planned'):
        lock_predictions(tmp_path, cohort, ['gold:1','full_kd:1'])


def test_nonfinite_receipt_is_not_written(tmp_path):
    with pytest.raises(ValueError):
        write(tmp_path/'bad.json', {'x': float('nan')})
    assert not (tmp_path/'bad.json').exists()


def test_matched_training_requires_same_initial_state_and_order():
    row = dict(initial_trainable_state_sha256='init', training_id_order_sha256='order',
               cache_manifest_sha256='cache', steps_completed=242, total_supervised_tokens=1289)
    receipts = {f'{arm}:1': dict(row) for arm in ('gold','full_kd','gated_kd')}
    protocol = {'seeds':[1], 'arms':['gold','full_kd','gated_kd'], 'training':{'steps':242}}
    verify_paired_training(receipts, protocol)
    receipts['gated_kd:1']['initial_trainable_state_sha256'] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        verify_paired_training(receipts, protocol)
