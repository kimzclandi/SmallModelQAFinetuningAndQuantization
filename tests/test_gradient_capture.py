"""Failure injection against write-ahead capture; no pretrained models."""
import copy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qa_lab import gradient_capture as capture
from qa_lab import gradient_diagnostics as diagnostic


def read(path):
    return json.loads(path.read_text())


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
    return (torch.tensor([[0, 1, 1, 2]]), torch.tensor([[-100, -100, 1, 2]]),
            torch.log_softmax(torch.tensor([[0., 3., -1.], [2., 0., 1.]]) / 2, -1).half())


def observer_at(tmp_path, check=lambda: None):
    output, private = tmp_path / "archive", tmp_path / "private"
    output.mkdir(); private.mkdir()
    return capture.CaptureObserver(output, private, check), output, private


def test_all_five_derivatives_capture_before_checks_without_updates(tiny_model, tmp_path):
    observer, output, private = observer_at(tmp_path)
    before = diagnostic.parameter_state(tiny_model)
    values, _ = diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    assert values["correctness_passed"]
    manifest = read(output / "capture.json")
    assert manifest["status"] == "complete" and set(manifest["objectives"]) == set(capture.VECTORS)
    assert diagnostic.parameter_state(tiny_model) == before
    assert all(p.grad is None for _, p in tiny_model.named_parameters())
    for key, receipt in manifest["objectives"].items():
        path = output / receipt["path"]
        assert diagnostic.sha(path) == receipt["sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"logit_actual", "logit_reference", "parameter_000", "parameter_001"}
            assert arrays["logit_actual"].dtype == np.float32
            assert arrays["logit_reference"].dtype == np.float64
            for array_key in arrays.files:
                assert receipt["arrays"][array_key] == capture._array_metadata(arrays[array_key])
            for index in range(2):
                assert hashlib.sha256(arrays[f"parameter_{index:03d}"].tobytes()).hexdigest() == values["parameters"][index]["gradient_sha256"][key]
    with np.load(private / "inputs.npz", allow_pickle=False) as inputs:
        assert set(inputs.files) == set(capture.INPUT_KEYS)
        assert inputs["teacher_log_probs"].dtype == np.float16
        np.testing.assert_array_equal(inputs["supervised_positions"], [[0, 1], [0, 2]])
    assert str(private) not in (output / "capture.json").read_text()
    assert not (output / "inputs.npz").exists()


def test_private_inputs_preserved_before_nonfinite_model_logit_error(tiny_model, tmp_path):
    observer, output, private = observer_at(tmp_path)
    tiny_model.base[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite model logits"):
        diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    manifest = read(output / "capture.json")
    assert manifest["status"] == "partial" and manifest["objectives"] == {}
    assert not manifest["private_inputs"]["arrays"]["selected_logits"]["finite"]
    assert (private / "inputs.npz").is_file()


def test_nonfinite_derivative_is_saved_before_gate_rejects_it(tiny_model, tmp_path, monkeypatch):
    observer, output, private = observer_at(tmp_path)
    original = torch.autograd.grad
    def corrupt(*args, **kwargs):
        result = list(original(*args, **kwargs))
        result[0] = result[0].clone()
        result[0][0, 0] = float("inf")
        return result
    monkeypatch.setattr(torch.autograd, "grad", corrupt)
    with pytest.raises(ValueError, match="Invalid gradient comparison"):
        diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    manifest = read(output / "capture.json")
    assert list(manifest["objectives"]) == ["ce"] and manifest["status"] == "partial"
    assert not manifest["objectives"]["ce"]["arrays"]["logit_actual"]["finite"]
    with np.load(output / "arrays/ce.npz") as arrays:
        assert np.isinf(arrays["logit_actual"][0, 0])


def test_autograd_exception_retains_prior_objective_and_original_parameters(tiny_model, tmp_path, monkeypatch):
    observer, output, private = observer_at(tmp_path)
    original = torch.autograd.grad
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interrupted VJP")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch.autograd, "grad", interrupt)
    before = diagnostic.parameter_state(tiny_model)
    with pytest.raises(RuntimeError, match="interrupted"):
        diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    assert list(read(output / "capture.json")["objectives"]) == ["ce"]
    assert diagnostic.parameter_state(tiny_model) == before


def test_budget_failure_occurs_after_atomic_stage_manifest(tiny_model, tmp_path):
    def fail():
        raise RuntimeError("stage budget")
    observer, output, private = observer_at(tmp_path, fail)
    with pytest.raises(RuntimeError, match="stage budget"):
        diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    assert read(output / "capture.json")["private_inputs"]["sha256"] == diagnostic.sha(private / "inputs.npz")
    assert not list(tmp_path.rglob("*.tmp"))


def test_observer_refuses_duplicate_stage_without_overwrite(tiny_model, tmp_path):
    observer, output, private = observer_at(tmp_path)
    diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    before = (output / "capture.json").read_bytes()
    with pytest.raises(ValueError, match="repeated private"):
        diagnostic.inspect_probe(tiny_model, *example(), {"atol": 1e-5, "rtol": 1e-4}, observer=observer)
    assert (output / "capture.json").read_bytes() == before


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
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
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    args = SimpleNamespace(output=tmp_path / "work/archive", private_output=tmp_path / "work/private",
        cache=tmp_path / "cache", training_root=tmp_path / "training", artifact=tmp_path,
        data_dir=tmp_path, student_config=tmp_path / "student.json", cache_dir=tmp_path,
        protocol=tmp_path / "protocol.json")
    args.cache.mkdir()
    saved = {"input_ids": torch.tensor([0, 1, 2]), "labels": torch.tensor([-100, 1, 2]),
             "teacher_log_probs": torch.log_softmax(torch.zeros(2, 3), -1).half()}
    torch.save(saved, args.cache / "record.pt")
    protocol = {"probe": {"state": "initial", "seed": 20261009, "train_id": "fixture", "count": 1},
        "correctness": {"gradient_atol": 1e-5, "gradient_rtol": 1e-4},
        "budget": {"wall_seconds": 120, "max_rss_bytes": 2**50, "max_output_bytes": 128 * 2**20}}
    legacy = {"runtime": {"cpu_threads": 1}, "training": {"max_tokens": 2048, "lora_r": 8,
        "lora_alpha": 16, "lora_dropout": 0., "target_modules": ["q_proj", "v_proj"]}, "hashes": {}}
    config = {"revision": "fixture"}
    cache = {"records": [{"id": "fixture", "path": "record.pt", "sha256": diagnostic.sha(args.cache / "record.pt")}],
             "model_identities": {}, "student_tokenizer_sha256": "fingerprint"}
    endpoints = {"gold:20261009": {"initial_trainable_state_sha256": "initial"}}
    source = {"fixture": {"is_impossible": False}}
    monkeypatch.setattr(capture, "frozen_binding", lambda _: (protocol, legacy,
        {"protocol_commit": "a"*40, "protocol_sha256": "b"*64, "source_sha256": {}}))
    monkeypatch.setattr(diagnostic, "preflight", lambda *a: (config, cache, source, source, endpoints, object()))
    monkeypatch.setattr(diagnostic, "load_verified_model", lambda *a: (SimpleNamespace(eos_token_id=2, all_special_ids=[2]), FakeModel()))
    monkeypatch.setattr(diagnostic, "tokenizer_fingerprint", lambda _: "fingerprint")
    monkeypatch.setattr(diagnostic, "validate_cached_record", Mock())
    monkeypatch.setattr(diagnostic, "trainable_state_sha256", lambda _: "initial")
    monkeypatch.setattr(capture.importlib.metadata, "version", lambda _: "fixture")
    final = Mock()
    monkeypatch.setattr(capture, "_final_integrity", final)
    peft = ModuleType("peft"); peft.LoraConfig = lambda **kwargs: kwargs
    peft.get_peft_model = lambda model, _: model
    monkeypatch.setitem(sys.modules, "peft", peft)
    return args, protocol, final


def fake_inspect(model, ids, labels, teacher, tolerance, observer):
    params = sorted(model.named_parameters())
    metadata = [diagnostic.parameter_metadata(name, value) for name, value in params]
    observer("inputs", {"selected_logits": torch.zeros(2, 3), "targets": labels[labels != -100],
        "teacher_log_probs": teacher, "input_ids": ids, "labels": labels,
        "supervised_positions": torch.tensor([[0, 0], [0, 1]])}, metadata)
    for key in capture.VECTORS:
        observer(key, {"logit_actual": torch.zeros(2, 3), "logit_reference": torch.zeros(2, 3, dtype=torch.float64),
            **{f"parameter_{i:03d}": torch.zeros_like(value) for i, (_, value) in enumerate(params)}}, metadata)
    return {"correctness_passed": False}, {}


def test_runner_completes_capture_of_failed_gate_and_runs_final_audits(fake_runner, monkeypatch):
    args, protocol, final = fake_runner
    monkeypatch.setattr(diagnostic, "inspect_probe", fake_inspect)
    result = capture.run_capture(args)
    assert result["status"] == "complete" and result["capture_complete"]
    assert result["numerical_gate_passed"] is False and result["diagnostic_complete"] is False
    assert result["probe_count"] == 1 and result["optimizer_steps"] == 0
    assert result["final_parameter_audit"] == result["final_identity_audit"] == "passed"
    assert result["final_budget_check"] == "passed" and final.call_count == 1
    state = read(args.output / "state.json")
    assert state["parameters_unchanged"] and state["grad_buffers_empty"]
    assert not any("inputs.npz" in name for name in result["artifacts"])
    before = (args.output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        capture.run_capture(args)
    assert (args.output / "run.json").read_bytes() == before


def test_runner_final_identity_failure_keeps_complete_arrays_but_fails_run(fake_runner, monkeypatch):
    args, protocol, final = fake_runner
    final.side_effect = ValueError("/private/example must not appear in receipts")
    monkeypatch.setattr(diagnostic, "inspect_probe", fake_inspect)
    with pytest.raises(ValueError, match="private"):
        capture.run_capture(args)
    run = read(args.output / "run.json")
    assert run["status"] == "failed" and run["capture_complete"]
    assert run["final_identity_audit"] == "failed" and run["final_parameter_audit"] == "passed"
    assert "/private/example" not in (args.output / "run.json").read_text()


def test_runner_exception_during_autograd_still_audits_state_and_identity(fake_runner, monkeypatch):
    args, protocol, final = fake_runner
    monkeypatch.setattr(diagnostic, "inspect_probe", Mock(side_effect=RuntimeError("synthetic VJP")))
    with pytest.raises(RuntimeError, match="VJP"):
        capture.run_capture(args)
    run = read(args.output / "run.json")
    assert run["status"] == "failed" and not run["capture_complete"]
    assert run["numerical_gate_passed"] is None and run["final_parameter_audit"] == "passed"
    assert final.call_count == 1


@pytest.mark.parametrize("kind", ["outside", "nested", "same", "root"])
def test_runner_refuses_unsafe_raw_output_locations_before_creation(fake_runner, kind):
    args, protocol, final = fake_runner
    if kind == "outside": args.output = args.output.parents[1] / "reports"
    elif kind == "nested": args.private_output = args.output / "private"
    elif kind == "same": args.private_output = args.output
    else: args.output = args.output.parent
    with pytest.raises(ValueError, match="independent ignored"):
        capture.run_capture(args)
    assert not args.output.exists()


def test_committed_binding_rejects_uncommitted_protocol_before_execution(tmp_path, monkeypatch):
    root = diagnostic.ROOT
    args = SimpleNamespace(protocol=root / "configs/ce-kl-gradient-capture-v1.json", output=tmp_path)
    monkeypatch.setattr(capture.subprocess, "check_output", lambda *a, **k: b"wrong committed bytes")
    with pytest.raises(ValueError, match="not identical to committed"):
        capture.frozen_binding(args)
    assert not (tmp_path / "source").exists()


def test_atomic_array_refuses_existing_file(tmp_path):
    path = tmp_path / "a.npz"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        capture.atomic_arrays(path, {"value": torch.zeros(1)})
    assert path.read_bytes() == b"existing"


def test_atomic_array_disk_failure_does_not_install_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "a.npz"
    def fail(*args, **kwargs):
        raise OSError("injected disk failure")
    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError, match="disk failure"):
        capture.atomic_arrays(path, {"value": torch.zeros(1)})
    assert not path.exists()
