"""CPU tensor/fixture contracts; no pretrained weights or dataset labels."""
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F

from qa_lab import confirmation_training as training
from qa_lab.logits_distillation import distillation_losses


def tensors():
    logits = torch.tensor([[[9., 1., -2.], [.2, -.8, .6], [1.1, -.2, -.7], [8., -3., 1.]]], requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2]], dtype=torch.long)
    teacher = torch.log_softmax(torch.tensor([[0., 3., -1.], [2., 0., 1.]]) / 2, dim=-1).half()
    return logits, labels, teacher


def test_gate_denominator_and_causal_alignment_with_independent_formula():
    logits, labels, teacher = tensors()
    value = training.confirmation_losses(logits, labels, teacher, "gated_kd")
    t = teacher.float()
    lp = F.log_softmax(logits[0, 1:3] / 2, dim=-1)
    kl = (t.exp() * (t - lp)).sum(dim=-1)
    ce = (-F.log_softmax(logits[0, 1:3], dim=-1)[torch.arange(2), torch.tensor([1, 2])]).mean()
    assert value["teacher_top1_correct"] == value["kl_retained_tokens"] == 1
    assert value["supervised_tokens"] == 2
    assert torch.allclose(value["optimized_kl"], kl[0] / 2)
    assert torch.allclose(value["loss"], .5 * ce + 2 * kl[0] / 2)
    assert not torch.allclose(value["optimized_kl"], kl[0])
    value["loss"].backward()
    assert torch.equal(logits.grad[0, 0], torch.zeros(3))
    assert torch.equal(logits.grad[0, 3], torch.zeros(3))
    assert torch.isfinite(logits.grad).all()


def test_full_kd_matches_existing_v2_loss_and_gold_retains_full_ce_weight():
    logits, labels, teacher = tensors()
    old = distillation_losses(logits, labels, teacher, {"temperature": 2, "hard_ce_weight": .5, "soft_kl_weight": .5})
    full = training.confirmation_losses(logits, labels, teacher, "full_kd")
    assert torch.allclose(full["loss"], old[0], atol=1e-6, rtol=1e-6)
    assert full["kl_retained_tokens"] == full["supervised_tokens"] == 2
    gold = training.confirmation_losses(logits, labels, teacher, "gold")
    assert torch.equal(gold["loss"], gold["hard_ce"])
    assert gold["kl_retained_tokens"] == 0 and gold["optimized_kl"].item() == 0


def test_zero_retained_gate_is_finite_and_ties_use_first_index():
    logits, labels, _ = tensors()
    teacher = torch.log_softmax(torch.zeros(2, 3), dim=-1).half()
    result = training.confirmation_losses(logits, labels, teacher, "gated_kd")
    assert result["kl_retained_tokens"] == 0
    assert torch.equal(result["loss"], .5 * result["hard_ce"])
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    labels[0, 2] = 0
    assert training.confirmation_losses(logits, labels, teacher, "gated_kd")["kl_retained_tokens"] == 1


@pytest.mark.parametrize("bad", ["nan_student", "inf_teacher", "shape", "empty", "target", "dtype", "arm"])
def test_invalid_loss_inputs_fail_closed(bad):
    logits, labels, teacher = tensors()
    if bad == "nan_student":
        logits = logits.detach(); logits[0, 0, 0] = float("nan")
    elif bad == "inf_teacher":
        teacher[0, 0] = float("inf")
    elif bad == "shape":
        teacher = teacher[:, :2]
    elif bad == "empty":
        labels[:] = -100
    elif bad == "target":
        labels[0, 2] = 99
    elif bad == "dtype":
        teacher = teacher.float()
    with pytest.raises(ValueError):
        training.confirmation_losses(logits, labels, teacher, "unknown" if bad == "arm" else "gated_kd")


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    # A synthetic three-vocabulary tensor parameter, not a pretrained model.
    class TinyModel(torch.nn.Module):
        def __init__(self, fail_at=None):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(1, 4, 3))
            self.config = SimpleNamespace(use_cache=True)
            self.calls = 0
            self.fail_at = fail_at
        def get_output_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(3, 1))
        def forward(self, **kwargs):
            self.calls += 1
            result = self.logits
            if self.calls == self.fail_at:
                result = result * float("nan")
            return SimpleNamespace(logits=result)
        def save_pretrained(self, output, **kwargs):
            (output / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fixture/student", "revision": "pinned"}))
            (output / "adapter_model.safetensors").write_bytes(self.logits.detach().numpy().tobytes())
    cfg = dict(training.FIXED_TRAINING, rows=2, steps=2)
    protocol = {"training": cfg, "budget": {"training_run_seconds": 900, "max_sampled_rss_bytes": 2**50, "max_output_bytes": 2**30}}
    binding = {"protocol_sha256": "a" * 64, "protocol_commit": "b" * 40, "source_sha256": {"fixture": "c" * 64}}
    config = {"model_id": "fixture/student", "revision": "pinned", "device": "cpu", "dtype": "float32", "attention_implementation": "eager"}
    cache_dir = tmp_path / "cache"; cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("bound fixture")
    records = []
    for index in range(2):
        path = cache_dir / f"record{index}.pt"
        _, labels, teacher = tensors()
        torch.save({"input_ids": torch.tensor([0, 1, 1, 2]), "labels": labels[0], "teacher_log_probs": teacher}, path)
        records.append({"id": f"train-{index}", "path": path.name, "sha256": training._sha(path)})
    artifact = tmp_path / "artifact"; artifact.mkdir(); (artifact / "manifest.json").write_text("artifact")
    data = tmp_path / "source"; data.mkdir(); (data / "train.jsonl").write_text("TRAIN fixture")
    cache = {"records": records, "student_tokenizer_sha256": "tokenizer", "model_identities": {"student": {}, "teacher": {}}}
    args = SimpleNamespace(arm="gold", seed=20261009, protocol=tmp_path / "protocol.json",
        student_config=tmp_path / "student.json", cache=cache_dir, artifact=artifact,
        data_dir=data, cache_dir=tmp_path / "hub", output=tmp_path / "run")
    monkeypatch.setattr(training, "frozen_protocol", lambda *a: (copy.deepcopy(protocol), binding))
    monkeypatch.setattr(training, "_bound_preflight", lambda *a: (config, cache, {r["id"]: {} for r in records}, SimpleNamespace(receipt={})))
    monkeypatch.setattr(training, "tokenizer_fingerprint", lambda _: "tokenizer")
    monkeypatch.setattr(training, "validate_cached_record", Mock())
    loader = Mock(side_effect=lambda *a: (object(), TinyModel()))
    monkeypatch.setattr(training, "load_verified_model", loader)
    peft = ModuleType("peft")
    peft.LoraConfig = Mock(side_effect=lambda **kw: kw)
    peft.get_peft_model = Mock(side_effect=lambda model, _: model)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(training.importlib.metadata, "version", lambda _: "fixture")
    return args, protocol, loader, TinyModel, peft


def test_same_seed_three_arms_share_initial_bytes_order_and_bound_receipts(tiny_run):
    args, _, _, _, _ = tiny_run
    receipts = []
    for arm in training.ARMS:
        args.arm = arm
        args.output = args.output.parent / arm
        receipts.append(training.train(args))
        steps = json.loads((args.output / "steps.json").read_text())
        assert len(steps) == 2 and all(math_field in steps[0] for math_field in ("full_kl", "optimized_kl", "grad_norm_before_clip", "clip_scale"))
        assert receipts[-1]["status"] == "complete"
        assert set(receipts[-1]["adapter_files"]) == {"adapter_config.json", "adapter_model.safetensors"}
        assert receipts[-1]["legacy_teacher_unbound"] is False
        assert receipts[-1]["protocol_sha256"] == "a" * 64
        assert str(args.cache_dir) not in (args.output / "training.json").read_text()
    assert len({r["initial_trainable_state_sha256"] for r in receipts}) == 1
    assert len({r["training_id_order_sha256"] for r in receipts}) == 1
    assert receipts[0]["total_kl_retained_tokens"] == 0
    assert receipts[1]["total_kl_retained_tokens"] == 4
    assert receipts[2]["total_kl_retained_tokens"] == 2


def test_nonfinite_second_step_preserves_first_and_cannot_be_retried(tiny_run):
    args, _, loader, TinyModel, _ = tiny_run
    loader.side_effect = lambda *a: (object(), TinyModel(fail_at=2))
    with pytest.raises(ValueError):
        training.train(args)
    receipt = json.loads((args.output / "training.json").read_text())
    assert receipt["status"] == "failed" and receipt["phase"] == "training"
    assert receipt["steps_completed"] == 1 and receipt["error_type"] == "ValueError"
    assert len(json.loads((args.output / "steps.json").read_text())) == 1
    assert not (args.output / "adapter_model.safetensors").exists()
    before = (args.output / "training.json").read_bytes()
    with pytest.raises(FileExistsError):
        training.train(args)
    assert (args.output / "training.json").read_bytes() == before
    assert loader.call_count == 1


def test_preflight_failure_is_path_free_and_happens_before_any_load(tiny_run, monkeypatch):
    args, _, loader, _, peft = tiny_run
    monkeypatch.setattr(training, "_bound_preflight", Mock(side_effect=ValueError("private /Users/secret/weights")))
    with pytest.raises(ValueError):
        training.train(args)
    receipt = (args.output / "training.json").read_text()
    assert "secret" not in receipt and json.loads(receipt)["steps_completed"] == 0
    loader.assert_not_called(); peft.get_peft_model.assert_not_called()


def test_budget_failure_keeps_failed_receipt_and_no_load(tiny_run):
    args, protocol, loader, _, _ = tiny_run
    protocol["budget"]["max_sampled_rss_bytes"] = 1
    with pytest.raises(RuntimeError, match="RSS"):
        training.train(args)
    receipt = json.loads((args.output / "training.json").read_text())
    assert receipt["status"] == "failed" and receipt["max_observed_rss_bytes"] > 1
    loader.assert_not_called()


def test_json_nonfinite_rejection_preserves_previous_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    training._save(path, {"status": "running"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        training._save(path, {"loss": float("nan")})
    assert path.read_bytes() == before


def test_frozen_source_protocol_and_inputs_reject_changes(tmp_path, monkeypatch):
    root = tmp_path / "repo"; (root / "qa_lab").mkdir(parents=True)
    source = root / "qa_lab" / "confirmation_training.py"; source.write_text("fixture")
    data = root / "frozen.json"; data.write_text("fixed")
    protocol = {"study": "teacher-gated-confirmation-v1", "status": "frozen_before_execution",
                "arms": list(training.ARMS), "seeds": list(training.SEEDS),
                "training": training.FIXED_TRAINING, "frozen_inputs": {"frozen.json": training._sha(data)}}
    path = root / "protocol.json"; path.write_text(json.dumps(protocol))
    committed = {"protocol.json": path.read_bytes(), "qa_lab/confirmation_training.py": source.read_bytes()}
    monkeypatch.setattr(training, "ROOT", root)
    monkeypatch.setattr(training, "_git", lambda command, value: b"f" * 40 if command == "rev-parse" else committed[value.removeprefix("HEAD:")])
    actual, receipt = training.frozen_protocol(path, "gold", 20261009)
    assert actual == protocol and receipt["protocol_commit"] == "f" * 40
    source.write_text("changed")
    with pytest.raises(ValueError, match="source"):
        training.frozen_protocol(path, "gold", 20261009)
    source.write_bytes(committed["qa_lab/confirmation_training.py"])
    data.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        training.frozen_protocol(path, "gold", 20261009)
    with pytest.raises(ValueError, match="Arm or seed"):
        training.frozen_protocol(path, "gold", 20260918)


def test_legacy_cache_rejected_before_snapshot_load(tmp_path, monkeypatch):
    config = {"model_id": "fixture", "revision": "pinned", "system_prompt": "fixture", "prompt_version": "v1",
              "device": "cpu", "dtype": "float32", "attention_implementation": "eager"}
    prefix = "configs/teacher-gated-confirmation-v1/"
    student = tmp_path / (prefix + "student.json"); student.parent.mkdir(parents=True)
    student.write_text(json.dumps(config))
    teacher = student.with_name("teacher.json"); teacher.write_text(json.dumps(config))
    artifact = tmp_path / "artifact"; artifact.mkdir()
    source = tmp_path / "source"; source.mkdir()
    for file in (artifact / "manifest.json", artifact / "train.jsonl", source / "train.jsonl"):
        file.write_text("fixture")
    protocol = {"training": dict(training.FIXED_TRAINING, artifact="artifact", source="source"),
                "frozen_inputs": {prefix + "student.json": training._sha(student),
                                  "artifact/manifest.json": training._sha(artifact / "manifest.json"),
                                  "artifact/train.jsonl": training._sha(artifact / "train.jsonl"),
                                  "source/train.jsonl": training._sha(source / "train.jsonl")}}
    cache = {"status": "complete", "temperature": 2., "student_prompt_config": training.sequence_config(config),
             "student_model_config": config, "teacher_model_config": config}
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "_load_cache_manifest", lambda _: cache)
    verify = Mock(side_effect=AssertionError("must not reach snapshot"))
    monkeypatch.setattr(training, "verify_local_model", verify)
    args = SimpleNamespace(student_config=student, artifact=artifact, data_dir=source, cache=tmp_path, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Legacy cache"):
        training._bound_preflight(args, protocol)
    verify.assert_not_called()
