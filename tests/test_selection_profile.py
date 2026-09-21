"""Synthetic coverage cases; length bands are proxies, never difficulty labels."""
import pytest
from qa_lab.selection_profile import profile

def record(i, decision, answerability='answerable', context='abc', question='q', family='f'):
    return dict(sample_id=i, decision=decision, slice=answerability,
                source=dict(id=i, context=context, question=question, family_id=family))

def test_same_acceptance_can_hide_answerability_shift():
    before=[record('a','accept'),record('b','review'),record('c','accept','unanswerable'),record('d','review','unanswerable')]
    after=[dict(r,decision='accept' if r['slice']=='unanswerable' else 'review') for r in before]
    a,b=profile(before),profile(after)
    assert a['accepted_candidates']==b['accepted_candidates']==2
    assert a['answerable_fraction_accepted']==.5
    assert b['answerable_fraction_accepted']==0
    assert b['slices']['answerable']['accept_rate']==0

def test_candidate_count_is_not_question_coverage():
    p=profile([record('a','accept'),record('b','accept')])
    assert p['candidate_count']==2
    assert p['input_coverage']['unique_questions']==1
    assert p['input_coverage']['unique_ids']==2

def test_zero_acceptance_and_empty_have_no_fabricated_rate():
    assert profile([])['accept_rate'] is None
    assert profile([record('a','review','unknown')])['answerable_fraction_accepted'] is None

def test_invalid_decisions_fail_closed():
    with pytest.raises(ValueError):profile([record('a','accepted')])

def test_rejected_unknown_source_is_counted_but_not_covered():
    p=profile([dict(sample_id=None,source=None,decision='reject',slice='unknown')])
    assert p['candidate_count']==1 and p['input_coverage']['unique_questions']==0
