"""Synthetic CPU gradients and failure injection; no pretrained model runs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from qa_lab import gradient_diagnostics as diagnostic
from qa_lab.confirmation_training import confirmation_losses


@pytest.fixture
def tiny_model():
    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor([[.2, -.3, .7], [.5, .1, -.4]]))
            self.b = torch.nn.Parameter(torch.zeros(3, 2))
            self.register_buffer("base", torch.tensor([[1., .1, .2], [-.2, .7, .3], [.4, -.1, .2]]))
        def named_parameters(self, *args, **kwargs):
            prefix = "base_model.model.model.layers.0.self_attn.q_proj."
            return iter([(prefix + "lora_A.default.weight", self.a), (prefix + "lora_B.default.weight", self.b)])
        def forward(self, input_ids, **kwargs):
            features = torch.nn.functional.one_hot(input_ids, 3).float()
            return SimpleNamespace(logits=features @ (self.base + self.b @ self.a).T)
    return TinyLoRA()


def example():
    ids = torch.tensor([[0, 1, 1, 2]])
    labels = torch.tensor([[-100, -100, 1, 2]])
    teacher = torch.log_softmax(torch.tensor([[0., 3., -1.], [2., 0., 1.]]) / 2, dim=-1).half()
    return ids, labels, teacher


def test_synthetic_lora_components_reconstruct_both_objectives_without_updates(tiny_model):
    before = diagnostic.parameter_state(tiny_model)
    values, raw = diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, public_block=True)
    assert values["correctness_passed"]
    assert set(raw) == set(diagnostic.VECTORS)
    assert values["supervised_tokens"] == 2 and values["teacher_top1_correct"] == 1
    assert all(p.grad is None for _, p in tiny_model.named_parameters())
    assert diagnostic.parameter_state(tiny_model) == before
    a = next(row for row in values["parameters"] if row["matrix"] == "A")
    assert all(a["zero_gradient"].values())
    assert values["groups"]["matrix:A"]["pairs"]["full_kl"]["cosine"] is None
    assert values["groups"]["matrix:A"]["pairs"]["full_kl"]["cosine_undefined_reason"].startswith("zero_gradient:")
    b = next(row for row in values["parameters"] if row["matrix"] == "B")
    for key in diagnostic.VECTORS:
        assert b["gradient_sha256"][key] == hashlib.sha256(raw[key].tobytes()).hexdigest()
    assert all(x["allclose"] for x in values["logit_gradient_checks"].values())


def test_cached_mass_is_retained_in_independent_float64_logit_derivative():
    z = torch.tensor([[1., -.5]], requires_grad=True)
    target = torch.tensor([0])
    teacher = torch.log(torch.tensor([[.6, .4]]) * .997).half()
    loss, _ = diagnostic.weighted_objectives(z, target, teacher)
    actual = torch.autograd.grad(loss["full_kl"], z)[0]
    reference = diagnostic.analytic_logit_gradients(z, target, teacher)["full_kl"]
    assert torch.allclose(actual.double(), reference, atol=1e-7, rtol=1e-5)
    p = teacher.double().exp()
    incorrect_mass_one = torch.softmax(z.detach().double() / 2., dim=-1) - p
    assert float((actual.double() - incorrect_mass_one).abs().max()) > .001


def test_teacher_top1_gold_agreement_does_not_guarantee_gradient_alignment():
    # A one-parameter-family synthetic classifier: logits themselves are theta.
    # No real checkpoint result or model-quality inference is made by this test.
    z = torch.nn.Parameter(torch.tensor([[4., 0.]]))
    target = torch.tensor([0])
    teacher = torch.log(torch.tensor([[.6, .4]])).half()
    losses, gate = diagnostic.weighted_objectives(z, target, teacher)
    assert gate.tolist() == [True]
    ce = torch.autograd.grad(losses["ce"], z, retain_graph=True)[0]
    gated = torch.autograd.grad(losses["gated_kl"], z)[0]
    assert float((ce * gated).sum()) < 0
    assert ce[0, 0] < 0 and gated[0, 0] > 0


def test_objectives_preserve_frozen_temperature_denominator_and_gold_mask():
    ids, labels, teacher = example()
    logits = torch.tensor([[[9., -2., 1.], [.2, -.8, .6], [1.1, -.2, -.7], [7., 8., 9.]]], requires_grad=True)
    mask = labels[:, 1:] != -100
    selected, targets = logits[:, :-1][mask], labels[:, 1:][mask]
    values, gate = diagnostic.weighted_objectives(selected, targets, teacher)
    for arm, key in [("full_kd", "combined_full"), ("gated_kd", "combined_gated")]:
        old = confirmation_losses(logits, labels, teacher, arm)
        assert torch.allclose(old["loss"], values[key], atol=1e-7, rtol=1e-6)
    token_kl = torch.nn.functional.kl_div(torch.log_softmax(selected / 2, -1), teacher.float(), reduction="none", log_target=True).sum(-1)
    assert torch.allclose(values["gated_kl"], 2 * token_kl[0] / 2)
    assert not torch.allclose(values["gated_kl"], 2 * token_kl[0])
    gradient = torch.autograd.grad(values["combined_gated"], logits)[0]
    assert torch.count_nonzero(gradient[0, 0]) == torch.count_nonzero(gradient[0, -1]) == 0


@pytest.mark.parametrize("mutation", ["nan_logits", "nan_teacher", "shape", "target", "dtype"])
def test_loss_contract_rejects_invalid_inputs(mutation):
    z = torch.zeros(2, 3, requires_grad=True)
    target = torch.tensor([1, 2])
    teacher = torch.log_softmax(torch.zeros(2, 3), dim=-1).half()
    if mutation == "nan_logits":
        z = z.detach(); z[0, 0] = float("nan")
    elif mutation == "nan_teacher":
        teacher[0, 0] = float("nan")
    elif mutation == "shape":
        teacher = teacher[:1]
    elif mutation == "target":
        target[0] = -1
    else:
        teacher = teacher.float()
    with pytest.raises(ValueError):
        diagnostic.weighted_objectives(z, target, teacher)


def test_geometry_records_zero_vectors_and_uses_strict_unrounded_signs():
    zero = diagnostic.geometry([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    assert zero["pairs"]["full_kl"]["cosine"] is None
    assert zero["pairs"]["full_kl"]["norm_ratio_to_ce"] is None
    close = diagnostic.geometry([[1., -1e-20, 0.], [-1e-20, 1., 0.], [0., 0., 1.]])
    assert close["pairs"]["full_kl"]["negative_dot"]
    assert not close["pairs"]["full_kl"]["norm_larger_than_ce"]
    assert not close["pairs"]["gated_kl"]["negative_dot"]


def test_reconstruction_uses_explicit_reference_and_rejects_nonfinite():
    reference = torch.tensor([1., 0.])
    actual = torch.tensor([1.00001, 1e-4])
    result = diagnostic.gradient_check(actual, reference, atol=1e-5, rtol=1e-4)
    assert not result["allclose"] and result["failed_elements"] == 1
    assert result["first_failure_flat"] == 1
    with pytest.raises(ValueError):
        diagnostic.gradient_check(torch.tensor([float("inf")]), torch.ones(1), 1e-5, 1e-4)
    zero = diagnostic.gradient_check(torch.zeros(1), torch.zeros(1), 1e-5, 1e-4)
    assert zero["relative_l2_error"] is None and zero["allclose"]


def summary_rows():
    rows = []
    for seed in (20261009, 20261010, 20261011):
        for state in ("initial", "gold", "full_kd", "gated_kd"):
            for index in range(16):
                gram = [[1., .5, .5], [.5, 4., .5], [.5, .5, 1.]]
                rows.append({"seed": seed, "state": state, "train_id": str(index), "correctness_passed": True,
                             "groups": {"all": diagnostic.geometry(gram)}})
    return rows


def test_summary_strict_majority_counterexamples_and_partial_scope():
    rows = summary_rows()
    initial = [row for row in rows if row["state"] == "initial"]
    for row in initial[:24]:
        row["groups"]["all"] = diagnostic.geometry([[1., .5, .5], [.5, 1., .5], [.5, .5, 1.]])
    summary = diagnostic.summarize(rows)
    assert summary["primary_initial"]["full_norm_larger_count"] == 24
    assert summary["primary_initial"]["hypotheses"]["initial_full_kl_larger_than_ce_majority"] is False
    assert summary["primary_initial"]["hypotheses"]["initial_gate_eliminates_negative_dot"] is True
    initial[0]["groups"]["all"] = diagnostic.geometry([[1., 0., -1e-20], [0., 1., 0.], [-1e-20, 0., 1.]])
    assert diagnostic.summarize(rows)["primary_initial"]["hypotheses"]["initial_gate_eliminates_negative_dot"] is False
    initial[0]["groups"]["all"] = diagnostic.geometry([[1., .5, 0.], [.5, 1., 0.], [0., 0., 0.]])
    summary = diagnostic.summarize(rows)
    assert summary["primary_initial"]["gated_cosine_undefined_count"] == 1
    assert summary["primary_initial"]["hypotheses"]["initial_gate_eliminates_negative_dot"] is False
    assert all(value is None for value in diagnostic.summarize(rows[:4])["primary_initial"]["hypotheses"].values())


def test_parameter_bytes_and_json_fail_closed(tiny_model, tmp_path):
    before = diagnostic.parameter_state(tiny_model)
    with torch.no_grad():
        tiny_model.b[0, 0] = .1
    after = diagnostic.parameter_state(tiny_model)
    assert before["base"] == after["base"] and before["adapter"] != after["adapter"]
    path = tmp_path / "receipt.json"
    diagnostic.save(path, {"status": "running"})
    old = path.read_bytes()
    with pytest.raises(ValueError):
        diagnostic.save(path, {"value": float("nan")})
    assert path.read_bytes() == old


def test_supervised_token_classes_do_not_infer_answer_vs_template_suffix():
    labels = torch.tensor([[-100, 5, 9, 2]])
    tokenizer = SimpleNamespace(eos_token_id=2, all_special_ids=[2, 9])
    result = diagnostic.token_classes(labels, tokenizer)
    assert (result["eos_count"], result["other_special_count"], result["other_count"]) == (1, 1, 1)
    assert "not assigned" in result["scope"]


@pytest.mark.parametrize("fail_on_second_state", [False, True])
def test_runner_preserves_one_success_then_failed_probe_and_refuses_overwrite(tmp_path, monkeypatch, fail_on_second_state):
    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.items = []
            for layer in range(24):
                for module in ("q_proj", "v_proj"):
                    for matrix in ("A", "B"):
                        shape = (8, 896) if matrix == "A" else (896 if module == "q_proj" else 128, 8)
                        name = f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_{matrix}.default.weight"
                        self.items.append((name, torch.nn.Parameter(torch.zeros(shape))))
        def named_parameters(self): return iter(self.items)
        def parameters(self): return (p for _, p in self.items)
        def eval(self): return self
        def get_output_embeddings(self): return SimpleNamespace(weight=torch.zeros(3, 1))
    cache_dir = tmp_path / "cache"; cache_dir.mkdir()
    records = []
    for index in range(2):
        file = cache_dir / f"r{index}.pt"
        torch.save({"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([-100, 1, 2]),
                    "teacher_log_probs": torch.log_softmax(torch.zeros(2, 3), -1).half()}, file)
        records.append({"id": str(index), "path": file.name, "sha256": diagnostic.sha(file)})
    protocol = {"seeds": [1], "states": ["initial"], "selected_ids": ["0", "1"],
        "training": {"max_tokens": 2048, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0., "target_modules": ["q_proj", "v_proj"]},
        "runtime": {"cpu_threads": 1}, "correctness": {"gradient_atol": 1e-5, "gradient_rtol": 1e-4},
        "budget": {"wall_seconds": 1200, "max_rss_bytes": 2**50, "max_output_bytes": 2**30}, "hashes": {}}
    config = {"revision": "fixture"}
    cache = {"records": records, "model_identities": {}, "student_tokenizer_sha256": "fingerprint"}
    source = {key: {"is_impossible": False} for key in ["0", "1"]}
    endpoints = {"gold:1": {"initial_trainable_state_sha256": "initial"}}
    if fail_on_second_state:
        protocol["states"].append("full_kd")
        folder = tmp_path / "full_kd-1"
        folder.mkdir()
        files = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            path = folder / name
            path.write_bytes(b"synthetic fixture, not model weights")
            files[name] = diagnostic._hash_file(path)
        endpoints["full_kd:1"] = {"adapter_files": files}
    args = SimpleNamespace(output=tmp_path / "new-run", cache=cache_dir, training_root=tmp_path,
        protocol=tmp_path / "protocol.json", artifact=tmp_path, data_dir=tmp_path,
        student_config=tmp_path / "student.json", cache_dir=tmp_path)
    monkeypatch.setattr(diagnostic, "frozen_binding", lambda *a: (protocol, {"source_sha256": {}, "protocol_commit": "a" * 40, "protocol_sha256": "b" * 64}))
    monkeypatch.setattr(diagnostic, "preflight", lambda *a: (config, cache, source, source, endpoints, object()))
    monkeypatch.setattr(diagnostic, "load_verified_model", lambda *a: (SimpleNamespace(eos_token_id=2, all_special_ids=[2]), FakeModel()))
    monkeypatch.setattr(diagnostic, "tokenizer_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(diagnostic, "validate_cached_record", Mock())
    monkeypatch.setattr(diagnostic, "trainable_state_sha256", lambda _: "initial")
    monkeypatch.setattr(diagnostic.importlib.metadata, "version", lambda _: "fixture")
    peft = ModuleType("peft"); peft.LoraConfig = lambda **kwargs: kwargs
    peft.get_peft_model = lambda model, _: model
    peft.PeftModel = SimpleNamespace(from_pretrained=lambda model, *a, **k: model)
    monkeypatch.setitem(sys.modules, "peft", peft)
    successes = 2 if fail_on_second_state else 1
    call = Mock(side_effect=[({"correctness_passed": True}, {})] * successes
                + [ValueError("/Users/private/path must not leak")])
    monkeypatch.setattr(diagnostic, "inspect_probe", call)
    with pytest.raises(ValueError, match="private"):
        diagnostic.run_diagnostic(args)
    run = json.loads((args.output / "run.json").read_text())
    first = json.loads((args.output / "probes/000.json").read_text())
    failed_path = args.output / f"probes/{successes:03d}.json"
    second = json.loads(failed_path.read_text())
    assert run["status"] == "failed" and run["probes_completed"] == successes
    assert run["probe_files"] == [f"probes/{i:03d}.json" for i in range(successes + 1)]
    assert run["state_files"] == (["states/initial-1.json", "states/full_kd-1.json"]
                                  if fail_on_second_state else ["states/initial-1.json"])
    assert first["status"] == "complete" and second["status"] == "failed"
    assert second["phase"] == "autograd" and second["error_code"] == "diagnostic_probe_autograd"
    assert second["error_type"] == "ValueError"
    assert "private/path" not in failed_path.read_text()
    assert run["artifacts"][f"probes/{successes:03d}.json"] == diagnostic.sha(failed_path)
    before = (args.output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(args)
    assert (args.output / "run.json").read_bytes() == before
    assert call.call_count == successes + 1
