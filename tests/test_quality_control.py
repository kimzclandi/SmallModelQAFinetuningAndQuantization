import copy
import json
import pytest
from qa_lab.quality_control import INDEX_KEYS, audit, leakage_audit
from scripts import synthetic_qc


def row(**kw):
    return dict(dict(id='a', split='train', family_id='one', context='Alice met Bob.',
                     question='Who met Bob?', answers=['Alice'], is_impossible=False), **kw)


def index(rows):
    return [{k: r[k] for k in INDEX_KEYS} for r in rows]


def check(prediction, source=None, **kw):
    source = source or row()
    candidate = dict(id='a', prediction=prediction, stop_reason='eos')
    return audit([source], [candidate], index([source]), language='en',
                 label_permission='train_reference', **kw)['records'][0]


def test_exact_agreement_is_not_semantic_judging():
    r = check('Alice')
    assert r['decision'] == 'accept'
    assert r['checks']['reference_match'] is True
    assert r['checks']['semantic_correctness'] == 'not_assessed'


@pytest.mark.parametrize('prediction', ['', None, 12, [], ' Alice', 'Alice\n',
                                        'Answer: Alice', 'alice', '```Alice```'])
def test_invalid_response_never_accepted(prediction):
    r = check(prediction)
    assert r['decision'] == 'reject'
    assert r['raw_response'] == r['output_response'] == prediction
    assert r['transformations'] == []


def test_wrong_span_and_unlisted_valid_alias_need_review():
    assert check('Bob')['decision'] == 'review'
    r = check('Alice met Bob.')
    assert r['checks']['span_locatable'] is True
    assert 'reference_mismatch_not_semantic_proof' in r['review_reasons']


def test_refusal_both_directions_and_legal_sentinel():
    assert check('NO_ANSWER')['decision'] == 'reject'
    unanswerable = row(answers=[], is_impossible=True)
    assert check('NO_ANSWER', unanswerable)['decision'] == 'accept'
    assert check('Alice', unanswerable)['decision'] == 'reject'


@pytest.mark.parametrize('changes', [dict(answers=[]), dict(answers=['Elsewhere']),
                                     dict(answers='Alice'), dict(is_impossible='false')])
def test_broken_reference_is_not_gold(changes):
    assert check('Alice', row(**changes))['decision'] == 'review'


def test_no_label_permission_is_enforced():
    r = row()
    with pytest.raises(ValueError, match='Remove labels'):
        audit([r], [], index([r]), language='en')
    unlabeled = {k: v for k, v in r.items() if k not in {'answers', 'is_impossible'}}
    out = audit([unlabeled], [dict(id='a', prediction='Alice', stop_reason='eos')],
                index([r]), language='en')
    assert out['records'][0]['decision'] == 'review'
    assert out['records'][0]['checks']['reference_match'] is None
    assert out['summary']['slices']['unknown']['n'] == 1


@pytest.mark.parametrize('split', ['dev', 'test', 'holdout'])
def test_eval_sources_and_eval_index_labels_are_forbidden(split):
    r = row(split=split)
    with pytest.raises(ValueError, match='training IDs'):
        audit([r], [], index([r]), language='en')
    with pytest.raises(ValueError, match='label-free'):
        leakage_audit([r], 'en')


@pytest.mark.parametrize('field', ['id', 'family_id', 'context'])
def test_cross_split_exact_leakage_blocks_acceptance(field):
    r = row()
    other = row(id='other', family_id='other', split='holdout',
                context='Zebras drink water.', question='Where?')
    other[field] = r[field]
    out = audit([r], [dict(id='a', prediction='Alice', stop_reason='eos')],
                index([r, other]), language='en', label_permission='train_reference')
    assert out['records'][0]['decision'] == 'reject'
    assert out['leakage'][0]['kind'] == 'exact'


def test_near_duplicate_is_flagged_not_auto_deleted():
    r = row()
    other = row(id='b', family_id='two', split='dev', context='Alice met Bob!')
    out = audit([r], [dict(id='a', prediction='Alice', stop_reason='eos')],
                index([r, other]), language='en', label_permission='train_reference')
    assert out['records'][0]['decision'] == 'review'
    assert out['records'][0]['reject_reasons'] == []


def test_chinese_question_near_match_is_a_review_flag():
    sources = [row(context='甲城位于北方，乙城位于南方。', question='哪座城位于北方？', answers=['甲城'])]
    other = row(id='b', family_id='two', split='holdout',
                context='丁村建于清朝。', question='哪座城位于北方？')
    out = audit(sources, [dict(id='a', prediction='甲城', stop_reason='eos')],
                index(sources + [other]), language='zh', label_permission='train_reference')
    assert out['records'][0]['decision'] == 'review'
    assert out['leakage'][0]['kind'] == 'near'


def test_duplicates_unknown_missing_and_shared_answers():
    sources = [row(), row(id='b', question='Name Alice?', family_id='two')]
    p = dict(id='a', prediction='Alice', stop_reason='eos')
    out = audit(sources, [p, p, dict(p, id='alien'), None], index(sources),
                language='en', label_permission='train_reference')
    assert all(r['decision'] == 'reject' for r in out['records'])
    assert out['summary']['missing_candidate_ids'] == ['b']
    out = audit(sources, [p, dict(p, id='b')], index(sources), language='en',
                label_permission='train_reference')
    assert out['summary']['decisions'] == {'accept': 2}
    sources[1]['question'] = sources[0]['question']
    out = audit(sources, [p, dict(p, id='b')], index(sources), language='en',
                label_permission='train_reference')
    assert out['summary']['decisions'] == {'review': 2}


def test_incomplete_generation_and_unknown_metadata():
    r = row()
    for stop, expected in [(None, 'review'), ('max_new_tokens', 'reject')]:
        out = audit([r], [dict(id='a', prediction='Alice', stop_reason=stop)],
                    index([r]), language='en', label_permission='train_reference')
        assert out['records'][0]['decision'] == expected
        assert out['records'][0]['provenance']['revision'] == 'unknown'


def test_output_protection_and_reproducible_sidecars(tmp_path, monkeypatch):
    from scripts import report_output
    monkeypatch.setattr(report_output, 'ROOT', tmp_path)
    r = row()
    candidates = [dict(id='a', prediction='Alice', stop_reason='eos')]
    before = copy.deepcopy(candidates)
    args = ([r], candidates, index([r]), 'en', 'train_reference')
    out = tmp_path / 'work' / 'run'
    synthetic_qc.run(*args, out)
    assert candidates == before
    assert json.loads((out / 'summary.json').read_text())['decisions'] == {'accept': 1}
    from scripts.verify_synthetic_qc import verify_run
    assert verify_run(out)['n'] == 1
    (out / 'accept.jsonl').write_text('')
    with pytest.raises(ValueError, match='file set/hash'):
        verify_run(out)
    # Even a self-consistent rewritten manifest cannot hide a wrong partition.
    from qa_lab.common import sha
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['accept.jsonl'] = sha(out / 'accept.jsonl')
    (out / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='partition mismatch'):
        verify_run(out)
    with pytest.raises(FileExistsError):
        synthetic_qc.run(*args, out)
    with pytest.raises(ValueError, match='work/'):
        synthetic_qc.run(*args, tmp_path / 'reports' / 'bad')
    (tmp_path / 'work' / 'link').symlink_to(tmp_path / 'reports', target_is_directory=True)
    with pytest.raises(ValueError, match='work/'):
        synthetic_qc.run(*args, tmp_path / 'work' / 'link' / 'bad')
