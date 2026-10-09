"""Synthetic identical-input replay and corruption rejection; no model/cache."""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_private_gradient_capture.py"
spec = importlib.util.spec_from_file_location("private_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    z = torch.tensor([[1., -2., 4., .5], [.3, 1.2, -.2, 2.]], requires_grad=True)
    target = torch.tensor([2, 0])
    teacher = torch.tensor([[.2, .1, .6, .1], [.1, .5, .2, .2]]).log().half()
    reference = replay.formula(z, target, teacher)
    losses, _ = replay.objectives(z, target, teacher.float())
    public = {key: {"logit_actual": torch.autograd.grad(loss, z, retain_graph=True)[0].numpy(),
                    "logit_reference": reference[key].numpy()}
              for key, loss in losses.items()}
    return z.detach(), target, teacher, public


def test_same_input_replay_keeps_mixed_gate_and_nonunit_teacher_mass():
    z, target, teacher, public = fixture()
    before = (z.clone(), target.clone(), teacher.clone())
    result = replay.analyze(z, target, teacher, public)
    assert all(result["fp32_replay_bitwise_matches_capture"].values())
    assert all(value["allclose"] for value in result["fp64_autograd_vs_formula"].values())
    assert result["teacher_half_mass"] != [1., 1.]
    assert public["full_kl"]["logit_actual"][1].any()
    assert not public["gated_kl"]["logit_actual"][1].any()
    assert all(torch.equal(a, b) for a, b in zip(before, (z, target, teacher)))


@pytest.mark.parametrize("field", ["logit_actual", "logit_reference"])
def test_replay_rejects_capture_mismatch_before_interpretation(field):
    z, target, teacher, public = fixture()
    public["full_kl"][field][0, 0] += .01
    with pytest.raises(ValueError, match="differs from captured"):
        replay.analyze(z, target, teacher, public)


@pytest.mark.parametrize("bad", ["dtype", "nan", "teacher", "shape"])
def test_private_contract_rejects_invalid_tensor(bad):
    z, target, teacher, public = fixture()
    if bad == "dtype":
        z = z.double()
    elif bad == "nan":
        z[0, 0] = float("nan")
    elif bad == "teacher":
        teacher = teacher.float()
    else:
        target = target[:, None]
    with pytest.raises(ValueError, match="contract"):
        replay.analyze(z, target, teacher, public)


def test_replay_refuses_post_capture_code_mutation(tmp_path):
    names = ("scripts/replay_private_gradient_capture.py", "scripts/verify_gradient_capture.py",
             "scripts/verify_gradient_diagnostics.py")
    (tmp_path / "scripts").mkdir()
    for name in names:
        (tmp_path / name).write_text("frozen-source")
    run = {"source_sha256": {name: replay.digest(tmp_path / name) for name in names}}
    replay.verify_executing_sources(run, tmp_path)
    (tmp_path / names[0]).write_text("modified-source")
    with pytest.raises(ValueError, match="precommitted"):
        replay.verify_executing_sources(run, tmp_path)
