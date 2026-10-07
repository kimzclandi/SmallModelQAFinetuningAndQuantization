import copy
import math
import pytest
from scripts.diagnose_logits_v2 import aggregate, distribution_rows, loss_diagnostic, paired


def test_paired_rejects_duplicate_and_mismatched_ids():
    a = dict(id='a', em=1., is_impossible=False, family_id='f', category='correct')
    b = dict(a, em=0., category='over_abstention')
    result = paired([a],[b])
    assert result['fixes'] == ['a'] and result['regressions'] == []
    for rows in ([a,a],[dict(a,id='b')]):
        with pytest.raises(ValueError): paired(rows,[b])
    with pytest.raises(ValueError): paired([a],[dict(b,family_id='other')])


def test_loss_scaling_and_incomplete_trace():
    steps = [dict(step=i+1,id=str(i),hard_ce=2.,soft_kl=.5,loss=2.,grad_norm=1.) for i in range(242)]
    assert loss_diagnostic(steps)['kl_share_of_summed_scalar_loss'] == .5
    with pytest.raises(ValueError): loss_diagnostic(steps[:-1])
    steps[0]['loss'] = 1.25  # Missing T-squared scaling must fail.
    with pytest.raises(ValueError): loss_diagnostic(steps)


def test_aggregate_rejects_fabricated_agreement_and_duplicate_position():
    row = dict(id='a',prediction_position=1,target_id=0,teacher_top1_id=0,
               top1_matches_gold=True,gold_probability=.5,entropy_nats=math.log(2),
               raw_probability_mass=1.,vocabulary=2)
    assert aggregate([row])['top1_gold_agreement'] == 1.
    with pytest.raises(ValueError): aggregate([row,row])
    with pytest.raises(ValueError): aggregate([dict(row,top1_matches_gold=False)])
    with pytest.raises(ValueError): aggregate([dict(row,gold_probability=float('nan'))])


def test_uniform_distribution_causal_shift_and_normalization():
    torch = pytest.importorskip('torch')
    record = dict(input_ids=torch.tensor([0,1,0]),labels=torch.tensor([-100,-100,0]),
                  teacher_log_probs=torch.tensor([[math.log(.5)]*2],dtype=torch.float16))
    rows = distribution_rows(record,'a')
    assert rows[0]['prediction_position'] == 1
    assert rows[0]['gold_probability'] == pytest.approx(.5)
    assert rows[0]['entropy_nats'] == pytest.approx(math.log(2))
    bad = copy.deepcopy(record); bad['labels'][2]=1
    with pytest.raises(ValueError): distribution_rows(bad,'a')
    bad = copy.deepcopy(record); bad['teacher_log_probs'][0,0]=float('nan')
    with pytest.raises(ValueError): distribution_rows(bad,'a')
