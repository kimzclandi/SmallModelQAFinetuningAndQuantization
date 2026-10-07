import pytest

from qa_lab.logits_distillation import sequence_config, validate_objective, validate_target_method


@pytest.mark.parametrize("objective", [
    {"temperature": 0, "hard_ce_weight": 1, "soft_kl_weight": 1},
    {"temperature": 2, "hard_ce_weight": -1, "soft_kl_weight": 1},
    {"temperature": 2, "hard_ce_weight": 0, "soft_kl_weight": 0},
])
def test_invalid_objective_rejected(objective):
    with pytest.raises(ValueError):
        validate_objective(objective)


def test_sequence_binding_excludes_runtime_device_but_not_prompt():
    base = {"model_id": "student", "revision": "abc", "system_prompt": "answer", "prompt_version": "v1",
            "device": "mps", "dtype": "float32"}
    cpu = dict(base, device="cpu")
    changed = dict(base, system_prompt="different")
    assert sequence_config(base) == sequence_config(cpu)
    assert sequence_config(base) != sequence_config(changed)


def test_target_method_distinguishes_teacher_response_and_gold_reference():
    assert validate_target_method("response_distillation") == "teacher_response"
    assert validate_target_method("gold_sft") == "gold_reference"
    with pytest.raises(ValueError, match="requires one of"):
        validate_target_method("gold_data_repair")


def test_identical_teacher_student_distribution_has_zero_kl():
    torch = pytest.importorskip("torch")
    from qa_lab.logits_distillation import distillation_losses

    logits = torch.tensor([[[9.0, -9.0, 0.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]])
    labels = torch.tensor([[-100, 2, 0]])
    selected = logits[:, :-1, :][labels[:, 1:] != -100]
    teacher = torch.log_softmax(selected / 2.0, dim=-1)
    total, hard, kl, count = distillation_losses(
        logits, labels, teacher,
        {"temperature": 2.0, "hard_ce_weight": 0.5, "soft_kl_weight": 0.5})
    assert count == 2
    assert hard.item() > 0
    assert abs(kl.item()) < 1e-6
    assert total.item() == pytest.approx(0.5 * hard.item(), rel=1e-6)


def test_causal_shift_and_shape_mismatch_are_enforced():
    torch = pytest.importorskip("torch")
    from qa_lab.logits_distillation import distillation_losses, supervised_prediction_mask

    labels = torch.tensor([[-100, -100, 1, 2]])
    assert supervised_prediction_mask(labels).tolist() == [[False, True, True]]
    logits = torch.randn(1, 4, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        distillation_losses(
            logits, labels, torch.randn(1, 5),
            {"temperature": 2, "hard_ce_weight": 0, "soft_kl_weight": 1})
