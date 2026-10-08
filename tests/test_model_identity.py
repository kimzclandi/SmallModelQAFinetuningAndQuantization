"""No-model tests for checkpoint bytes, loader routing and legacy cache scope."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from qa_lab import model_identity as identity
from qa_lab import logits_distillation as logits


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    cache = tmp_path / "private-local-cache"
    registry = copy.deepcopy(identity.MODEL_IDENTITIES)
    configs, paths = {}, {}
    for (model_id, revision), entry in registry.items():
        role = entry["role"]
        config = {"model_id": model_id, "revision": revision, "device": "cpu",
                  "dtype": "float32", "attention_implementation": "eager"}
        snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        snapshot.mkdir(parents=True)
        metadata = {}
        for name in identity.REQUIRED_FILES:
            data = f"{role}:{name}:fixture".encode()
            (snapshot / name).write_bytes(data)
            metadata[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"revision": revision, "files": metadata} if role == "student" else metadata
        file = repository / entry["manifest"]
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(manifest))
        entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        configs[role], paths[role] = config, snapshot
    monkeypatch.setattr(identity, "ROOT", repository)
    monkeypatch.setattr(identity, "MODEL_IDENTITIES", registry)
    hub = ModuleType("huggingface_hub")

    def lookup(model_id, filename, revision, cache_dir):
        return str(Path(cache_dir) / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision / filename)

    hub.try_to_load_from_cache = Mock(side_effect=lookup)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return configs, paths, cache, registry


def test_both_manifest_formats_and_path_free_receipts(snapshots):
    configs, paths, cache, _ = snapshots
    for role in ("student", "teacher"):
        checked = identity.verify_local_model(configs[role], cache)
        assert checked.path == paths[role]
        assert checked.receipt == identity.expected_identity(configs[role])
        assert str(cache) not in json.dumps(checked.receipt)


@pytest.mark.parametrize("mutation", ["byte", "missing", "extra_weight", "extra_tokenizer", "extra_template", "directory"])
def test_snapshot_mutations_rejected(snapshots, mutation):
    configs, paths, cache, _ = snapshots
    snapshot = paths["student"]
    if mutation == "byte":
        path = snapshot / "model.safetensors"
        data = bytearray(path.read_bytes()); data[0] ^= 1; path.write_bytes(data)
    elif mutation == "missing":
        (snapshot / "vocab.json").unlink()
    elif mutation == "directory":
        (snapshot / "chat_templates").mkdir()
    else:
        name = {"extra_weight": "pytorch_model.bin", "extra_tokenizer": "added_tokens.json",
                "extra_template": "chat_template.jinja"}[mutation]
        (snapshot / name).write_text("unverified")
    with pytest.raises(ValueError):
        identity.verify_local_model(configs["student"], cache)


def test_unknown_revision_and_manifest_change_rejected_before_snapshot_lookup(snapshots):
    configs, _, cache, registry = snapshots
    hub = sys.modules["huggingface_hub"]
    bad = dict(configs["student"], revision="0" * 40)
    with pytest.raises(ValueError, match="No pinned"):
        identity.verify_local_model(bad, cache)
    entry = registry[(configs["student"]["model_id"], configs["student"]["revision"])]
    with (identity.ROOT / entry["manifest"]).open("a") as file:
        file.write(" ")
    with pytest.raises(ValueError, match="manifest changed"):
        identity.verify_local_model(configs["student"], cache)
    hub.try_to_load_from_cache.assert_not_called()


def test_readonly_cache_lookup_cannot_silently_change_revision(snapshots):
    configs, paths, cache, _ = snapshots
    lookup = sys.modules["huggingface_hub"].try_to_load_from_cache
    lookup.side_effect = None
    lookup.return_value = None
    with pytest.raises(ValueError, match="unavailable"):
        identity.verify_local_model(configs["student"], cache)
    lookup.return_value = str(paths["student"].parent / ("f" * 40) / "config.json")
    with pytest.raises(ValueError, match="pinned snapshot"):
        identity.verify_local_model(configs["student"], cache)


def test_normal_hf_blob_symlink_supported_but_external_escape_rejected(snapshots, tmp_path):
    configs, paths, cache, _ = snapshots
    path = paths["student"] / "model.safetensors"
    blob = paths["student"].parent.parent / "blobs" / "fixture"
    blob.parent.mkdir(); path.replace(blob)
    path.symlink_to(blob)
    identity.verify_local_model(configs["student"], cache)
    outside = tmp_path / "outside-model"
    outside.write_bytes(blob.read_bytes())
    path.unlink(); path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink escapes"):
        identity.verify_local_model(configs["student"], cache)


def test_weight_hash_is_streamed_without_read_bytes(snapshots, monkeypatch):
    configs, _, cache, _ = snapshots
    monkeypatch.setattr(identity, "CHUNK_BYTES", 3)
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("whole file read forbidden")))
    identity.verify_local_model(configs["student"], cache)


def bound_cache(configs):
    return {"student_model_config": configs["student"], "teacher_model_config": configs["teacher"],
            "model_identities": {role: identity.expected_identity(configs[role]) for role in configs}}


def test_cache_binding_and_explicit_legacy_scope(snapshots):
    configs, _, _, _ = snapshots
    cache = bound_cache(configs)
    assert identity.validate_cache_identities(cache) == "bound_cache_receipt"
    del cache["model_identities"]
    with pytest.raises(ValueError, match="Legacy cache"):
        identity.validate_cache_identities(cache)
    assert identity.validate_cache_identities(cache, allow_legacy=True) == "legacy_unbound"


@pytest.mark.parametrize("mutation", ["missing_role", "null", "teacher_hash", "revision"])
def test_legacy_override_never_bypasses_malformed_bound_receipts(snapshots, mutation):
    configs, _, _, _ = snapshots
    cache = bound_cache(configs)
    if mutation == "missing_role":
        del cache["model_identities"]["teacher"]
    elif mutation == "null":
        cache["model_identities"] = None
    elif mutation == "teacher_hash":
        cache["model_identities"]["teacher"]["files"]["model.safetensors"]["sha256"] = "0" * 64
    else:
        cache["teacher_model_config"] = dict(configs["teacher"], revision="0" * 40)
    with pytest.raises(ValueError):
        identity.validate_cache_identities(cache, allow_legacy=True)


def test_loaders_use_verified_directory_and_restore_portable_peft_metadata(snapshots, monkeypatch):
    configs, _, cache, _ = snapshots
    checked = identity.verify_local_model(configs["student"], cache)
    calls = []
    tokenizer = SimpleNamespace(name_or_path=str(checked.path), init_kwargs={"name_or_path": str(checked.path)})
    model = SimpleNamespace(name_or_path=str(checked.path), config=SimpleNamespace(_name_or_path=str(checked.path)))
    model.to = lambda device: model
    model.eval = lambda: model
    transformers = ModuleType("transformers")
    def tokenizer_load(path, **kwargs):
        calls.append(("tokenizer", path, kwargs)); return tokenizer
    def model_load(path, **kwargs):
        calls.append(("model", path, kwargs)); return model
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer_load)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=model_load)
    torch = ModuleType("torch"); torch.float32 = "fixture-float32"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    actual_tokenizer, actual_model = identity.load_verified_model(configs["student"], checked)
    assert actual_tokenizer is tokenizer and actual_model is model
    for _, path, kwargs in calls:
        assert path == str(checked.path)
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert "revision" not in kwargs
    assert calls[1][2]["use_safetensors"] is True
    # This is the exact field PEFT get_peft_model reads for its serialized base ID.
    portable = {"base_model_name_or_path": model.__dict__["name_or_path"],
                "_name_or_path": model.config._name_or_path, "tokenizer": tokenizer.init_kwargs}
    assert str(cache) not in json.dumps(portable)
    assert portable["base_model_name_or_path"] == configs["student"]["model_id"]


def config_file(tmp_path, name, data):
    file = tmp_path / name
    file.write_text(json.dumps(data))
    return file


def cache_args(tmp_path, configs, cache):
    return SimpleNamespace(objective=config_file(tmp_path, "objective.json", {
        "temperature": 2, "hard_ce_weight": .5, "soft_kl_weight": .5, "max_tokens": 16}),
        student_config=config_file(tmp_path, "student.json", configs["student"]),
        teacher_config=config_file(tmp_path, "teacher.json", configs["teacher"]),
        teacher_device=None, artifact=tmp_path, data_dir=tmp_path, limit=None,
        model_cache_dir=cache, output=tmp_path / "new-cache")


def test_teacher_verification_failure_precedes_every_tokenizer_and_model_load(snapshots, monkeypatch, tmp_path):
    configs, paths, cache, _ = snapshots
    args = cache_args(tmp_path, configs, cache)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(logits, "verified_rows", lambda *a: ([{"id": "train"}], {"method": "gold_sft"}))
    tokenizer = Mock(side_effect=AssertionError("must not load tokenizer"))
    model = Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(logits, "load_verified_tokenizer", tokenizer)
    monkeypatch.setattr(logits, "load_verified_model", model)
    (paths["teacher"] / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        logits.cache_teacher(args)
    tokenizer.assert_not_called(); model.assert_not_called()
    assert not args.output.exists()


def test_both_preflights_finish_before_first_cache_loader(snapshots, monkeypatch, tmp_path):
    configs, _, cache, _ = snapshots
    args = cache_args(tmp_path, configs, cache)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setattr(logits, "verified_rows", lambda *a: ([{"id": "train"}], {"method": "gold_sft"}))
    calls = []
    def preflight(config, folder):
        checked = identity.verify_local_model(config, folder)
        calls.append(checked.receipt["model_id"])
        return checked
    def first_loader(snapshot):
        assert calls == [configs["student"]["model_id"], configs["teacher"]["model_id"]]
        raise RuntimeError("stop before any model execution")
    monkeypatch.setattr(logits, "verify_local_model", preflight)
    monkeypatch.setattr(logits, "load_verified_tokenizer", first_loader)
    with pytest.raises(RuntimeError, match="stop before"):
        logits.cache_teacher(args)


@pytest.mark.parametrize("legacy_override", [False, True])
def test_train_never_loads_or_creates_optimizer_before_identity_gate(snapshots, monkeypatch, tmp_path, legacy_override):
    configs, paths, cache, _ = snapshots
    config = dict(configs["student"], system_prompt="fixture", prompt_version="v1")
    args = SimpleNamespace(objective=config_file(tmp_path, "objective.json", {
        "temperature": 2, "hard_ce_weight": .5, "soft_kl_weight": .5}),
        student_config=config_file(tmp_path, "student.json", config), student_device=None,
        train_config=config_file(tmp_path, "train.json", {"seed": 1}), cache=tmp_path,
        artifact=tmp_path, data_dir=tmp_path, smoke=False,
        allow_legacy_unbound_cache=legacy_override, model_cache_dir=cache, output=tmp_path / "train-output")
    manifest = {"status": "complete", "temperature": 2, "student_prompt_config": logits.sequence_config(config)}
    monkeypatch.setattr(logits, "_load_cache_manifest", lambda _: manifest)
    monkeypatch.setattr(logits, "validate_cache_lineage", lambda *a: {})
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    peft = ModuleType("peft"); peft.LoraConfig = Mock(); peft.get_peft_model = Mock()
    monkeypatch.setitem(sys.modules, "peft", peft)
    load = Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(logits, "load_verified_model", load)
    (paths["student"] / "model.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch" if legacy_override else "Legacy cache"):
        logits.train_student(args)
    load.assert_not_called(); peft.LoraConfig.assert_not_called()
    assert not args.output.exists()


def test_cli_hashes_only_and_refuses_existing_output(snapshots, monkeypatch, tmp_path, capsys):
    _, _, cache, _ = snapshots
    output = tmp_path / "new-output-folder" / "audit.json"
    monkeypatch.setattr(sys, "argv", ["model_identity", "--model-cache-dir", str(cache), "--output", str(output)])
    # The CLI never imports either framework, even when their imports are blocked.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    identity.main()
    result = json.loads(output.read_text())
    assert set(result["model_identities"]) == {"student", "teacher"}
    assert result["model_loading_performed"] is False
    assert result["training_performed"] is False
    assert str(cache) not in output.read_text()
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    original = output.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        identity.main()
    assert output.read_bytes() == original


def test_cli_failure_is_nonzero_and_preserves_completed_role_without_private_path(snapshots, monkeypatch, tmp_path, capsys):
    _, paths, cache, _ = snapshots
    (paths["teacher"] / "model.safetensors").write_text("changed")
    output = tmp_path / "failed-audit.json"
    monkeypatch.setattr(sys, "argv", ["model_identity", "--model-cache-dir", str(cache), "--output", str(output)])
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit) as stop:
        identity.main()
    assert stop.value.code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "failed" and result["failed_role"] == "teacher"
    assert result["error_type"] == "ValueError"
    assert set(result["model_identities"]) == {"student"}
    assert str(cache) not in output.read_text()
    assert result["model_loading_performed"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
