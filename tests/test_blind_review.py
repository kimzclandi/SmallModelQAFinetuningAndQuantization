"""Synthetic annotation fixtures; these are not human semantic judgments."""
import copy
import pytest
from qa_lab.blind_review import prepare, analyze

def population():
    return [('batch',dict(record_id=str(i),decision=d,slice='answerable',source={'context':'Alice met Bob.','question':'Who met Bob?','answers':['Alice']},raw_response='Alice')) for i,d in enumerate(['accept']*5+['review']*3+['reject']*2)]

def complete(template):
    a=copy.deepcopy(template);a['origin']='synthetic_fixture'
    for r in a['annotations']:r.update(judgment='correct',error='none',note='Synthetic test only')
    return a

def test_blinding_determinism_and_population_coverage():
    b=prepare(population(),2,17)
    assert b==prepare(population()[::-1],2,17)
    assert len(b['items'])==6
    assert all(set(r)=={'blind_id','context','question','response'} for r in b['items'])
    assert sorted(s['population_n'] for s in b['strata'])==[2,3,5]
    assert len({r['blind_id'] for r in b['items']})==6

def test_pending_has_no_semantic_accuracy():
    b=prepare(population(),2,17);r=analyze(b,[b['template']])
    assert r['status']=='pending' and r['reviewers'][0]['strata'] is None

@pytest.mark.parametrize('fault',['missing','duplicate','unknown','inconsistent'])
def test_bad_annotation_coverage_or_labels_rejected(fault):
    b=prepare(population(),2,17);a=complete(b['template'])
    if fault=='missing':a['annotations'].pop()
    if fault=='duplicate':a['annotations'][-1]=a['annotations'][0]
    if fault=='unknown':a['annotations'][0]['blind_id']='unknown'
    if fault=='inconsistent':a['annotations'][0]['error']='wrong_answer'
    with pytest.raises(ValueError):analyze(b,[a])

def test_fixture_results_not_promoted_to_human_measurement():
    b=prepare(population(),2,17);a=complete(b['template']);r=analyze(b,[a])
    assert r['status']=='synthetic_test_only'
    assert r['reviewers'][0]['strata'][0]['correct_n']==2

def test_disagreement_keeps_uncertain_and_no_consensus_score():
    b=prepare(population(),2,17);a=complete(b['template']);c=complete(b['template']);c['reviewer_alias']='reviewer-b'
    c['annotations'][0].update(judgment='uncertain',error='uncertain',note='ambiguous')
    r=analyze(b,[a,c]);assert r['agreement']['disagreements']==1
    assert len(r['adjudication_queue'])==1 and 'consensus_accuracy' not in r

def test_packet_tamper_and_duplicate_reviewers_rejected():
    b=prepare(population(),2,17);a=complete(b['template'])
    with pytest.raises(ValueError):analyze(b,[a,a])
    b['items'][0]['response']='edited'
    with pytest.raises(ValueError):analyze(b,[a])

def test_partial_human_annotation_has_no_proportions():
    b=prepare(population(),2,17);a=complete(b['template']);a['origin']='human_attested'
    a['annotations'][0].update(judgment=None,error=None,note='')
    r=analyze(b,[a]);assert r['status']=='pending'
    assert r['reviewers'][0]['strata'] is None

def test_mixed_origin_cannot_be_human_estimate():
    b=prepare(population(),2,17);a=complete(b['template']);c=complete(b['template'])
    c.update(reviewer_alias='reviewer-b',origin='human_attested')
    assert analyze(b,[a,c])['status']=='mixed_origins_not_human_estimate'
