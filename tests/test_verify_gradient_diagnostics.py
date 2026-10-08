"""Independent synthetic evidence checks; never run a model or inspect holdout."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_gradient_diagnostics", REPO / "scripts/verify_gradient_diagnostics.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def manifest(root):
    write(root / "archive-manifest.json", {"schema": 1, "files": {
        path.relative_to(root).as_posix(): verify.digest(path)
        for path in root.rglob("*")
        if path.is_file() and path != root / "archive-manifest.json"
    }})


@pytest.mark.parametrize("payload", ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_strict_json_rejects_nonfinite_and_duplicate_keys(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        verify.read(path)


def test_geometry_uses_weighted_vectors_and_preserves_negative_dot():
    vectors = np.array([[1., 0., 1.], [-2., 0., 0.], [1., 0., -2.]])
    gram = verify.checked_gram((vectors @ vectors.T).tolist())
    output = verify.geometry(gram)
    assert output["full"]["dot"] == -2
    assert output["gated"]["dot"] == -1
    assert output["full"]["cosine"] == pytest.approx(-1 / np.sqrt(2))
    assert output["full"]["norm_ratio"] == pytest.approx(np.sqrt(2))


def test_zero_vector_is_undefined_not_positive_alignment():
    gram = verify.checked_gram([[0., 0., 0.], [0., 1., -1.], [0., -1., 1.]])
    output = verify.geometry(gram)
    assert output["full"]["cosine"] is None
    assert output["full"]["norm_ratio"] is None
    assert output["full"]["zero_ce"] is True
    assert output["full"]["zero_kl"] is False


@pytest.mark.parametrize("gram", [
    [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]],
    [[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[0., 1e-14, 0.], [1e-14, 1., 0.], [0., 0., 1.]],
    [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
])
def test_invalid_gram_rejected(gram):
    with pytest.raises(ValueError):
        verify.checked_gram(gram)


def test_archive_checks_complete_file_set_including_nested_manifest(tmp_path):
    write(tmp_path / "data.json", {"ok": True})
    manifest(tmp_path)
    verify.check_archive(tmp_path)
    write(tmp_path / "nested/archive-manifest.json", {})
    with pytest.raises(ValueError, match="file set"):
        verify.check_archive(tmp_path)


def test_archive_rejects_weights_even_when_hash_matches(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"not-real-weights")
    manifest(tmp_path)
    with pytest.raises(ValueError, match="weights"):
        verify.check_archive(tmp_path)


def test_archive_rejects_symlinks(tmp_path):
    write(tmp_path / "data.json", {"ok": True})
    manifest(tmp_path)
    (tmp_path / "alias.json").symlink_to("data.json")
    with pytest.raises(ValueError, match="symlinks"):
        verify.check_archive(tmp_path)


def test_arithmetic_tolerance_is_not_classification_tolerance():
    verify.close(1. + 1e-12, 1., "sum")
    gram = verify.checked_gram([[1., -1e-16, 0.], [-1e-16, 1., 0.], [0., 0., 1.]])
    assert verify.geometry(gram)["full"]["dot"] < 0


TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def comparison(actual, reference):
    a, b = actual.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    error = a - b
    failed = np.flatnonzero(abs(error) > 1e-5 + 1e-4 * abs(b))
    norm = float(np.linalg.norm(b)); residual = float(np.linalg.norm(error))
    return {"allclose": len(failed) == 0, "failed_elements": int(len(failed)),
            "first_failure_flat": int(failed[0]) if len(failed) else None,
            "max_abs_error": float(abs(error).max()), "l2_error": residual,
            "relative_l2_error": residual / norm if norm else None,
            "reference_norm": norm, "reconstructed_norm": float(np.linalg.norm(a)),
            **TOLERANCE,
            "comparison": "abs(actual-reference)<=atol+rtol*abs(reference), evaluated in float64"}


def fixture_parameter(layer=0, module="q_proj", matrix="B"):
    arrays = {"ce": np.array([[.25, .5]], dtype=np.float32),
              "full_kl": np.array([[-1., .25]], dtype=np.float32),
              "gated_kl": np.array([[0., 0.]], dtype=np.float32)}
    arrays["combined_full"] = arrays["ce"] + arrays["full_kl"]
    arrays["combined_gated"] = arrays["ce"] + arrays["gated_kl"]
    components = np.stack([arrays[key].flatten().astype(np.float64) for key in verify.COMPONENTS])
    parameter = {"name": f"base_model.model.model.layers.{layer}.self_attn.{module}.lora_{matrix}.default.weight",
                 "layer": layer, "module": module, "matrix": matrix, "shape": [1, 2], "numel": 2,
                 "gram": (components @ components.T).tolist(),
                 "gradient_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()},
                 "zero_gradient": {key: bool(np.count_nonzero(value) == 0) for key, value in arrays.items()},
                 "reconstruction": {suffix: comparison(arrays["ce"] + arrays[key], arrays["combined_" + suffix])
                                    for suffix, key in (("full", "full_kl"), ("gated", "gated_kl"))}}
    return parameter, arrays


def fixture_probe(index=0, seed=20261009, state="initial", train_id="fixture-0"):
    parameters = [fixture_parameter(layer, module, matrix)[0]
                  for layer in range(24) for module in ("q_proj", "v_proj") for matrix in ("A", "B")]
    parameters.sort(key=lambda p: p["name"])
    good = comparison(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float32))
    return {"index": index, "seed": seed, "state": state, "train_id": train_id,
            "parameters": parameters, "groups": verify.grouped_geometry(parameters),
            "losses": {"ce": .5, "full_kl": 1., "gated_kl": 0., "combined_full": 1.5, "combined_gated": .5},
            "input_tokens": 9, "supervised_tokens": 2, "teacher_top1_correct": 0,
            "logit_gradient_checks": {key: deepcopy(good) for key in verify.VECTORS},
            "legacy_loss_checks": {key: deepcopy(good) for key in ("full", "gated")},
            "correctness_passed": True, "public_parameter": None}


def test_independent_raw_block_replay(tmp_path):
    parameter, arrays = fixture_parameter()
    path = tmp_path / "gradients.npz"
    np.savez(path, **arrays)
    assert verify.replay_public_block(path, parameter, TOLERANCE) == 10


@pytest.mark.parametrize("mutation", ["bytes", "dtype", "nan", "extra", "shape", "digest", "gram", "reconstruction"])
def test_public_vector_semantic_corruption_is_rejected(tmp_path, mutation):
    parameter, arrays = fixture_parameter()
    if mutation == "bytes": arrays["ce"][0, 0] += .25
    if mutation == "dtype": arrays["ce"] = arrays["ce"].astype(np.float64)
    if mutation == "nan": arrays["ce"][0, 0] = np.nan
    if mutation == "extra": arrays["hidden"] = np.array([1.], dtype=np.float32)
    if mutation == "shape": arrays["ce"] = arrays["ce"].reshape(2, 1)
    if mutation == "digest": parameter["gradient_sha256"]["ce"] = "0" * 64
    if mutation == "gram": parameter["gram"][0][0] += 1.
    if mutation == "reconstruction": parameter["reconstruction"]["full"]["max_abs_error"] = .5
    path = tmp_path / "gradients.npz"; np.savez(path, **arrays)
    with pytest.raises(ValueError):
        verify.replay_public_block(path, parameter, TOLERANCE)


def test_all_parameter_receipt_replay_and_grouping():
    assert verify.check_probe(fixture_probe(), TOLERANCE)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "metadata", "group", "sign", "zero", "pass", "tolerance", "count", "objective"])
def test_probe_semantic_corruption_is_rejected(mutation):
    probe = fixture_probe()
    if mutation == "missing": probe["parameters"].pop()
    if mutation == "duplicate": probe["parameters"][1] = deepcopy(probe["parameters"][0])
    if mutation == "metadata": probe["parameters"][0]["layer"] = 8
    if mutation == "group": probe["groups"]["all"]["norms"]["ce"] += 1.
    if mutation == "sign": probe["groups"]["all"]["pairs"]["full_kl"]["negative_dot"] = False
    if mutation == "zero": probe["parameters"][0]["zero_gradient"]["gated_kl"] = False
    if mutation == "pass": probe["correctness_passed"] = False
    if mutation == "tolerance": probe["parameters"][0]["reconstruction"]["full"]["atol"] = .1
    if mutation == "count": probe["logit_gradient_checks"]["ce"]["failed_elements"] = 1
    if mutation == "objective": probe["losses"]["combined_full"] = 1.6
    with pytest.raises(ValueError):
        verify.check_probe(probe, TOLERANCE)


def test_summary_keeps_zero_gated_direction_undefined_and_secondary_separate():
    probe = fixture_probe()
    minimal = {key: probe[key] for key in ("groups", "correctness_passed")}
    probes = [{**minimal, "state": state, "seed": seed, "train_id": str(index)}
              for state in verify.STATES for seed in verify.SEEDS for index in range(16)]
    summary = verify.summarize(probes)
    assert summary["diagnostic_complete"] is True
    assert summary["primary_initial"]["probes"] == 48
    assert summary["primary_initial"]["distinct_train_ids"] == 16
    assert summary["primary_initial"]["hypotheses"] == {
        "initial_full_kl_larger_than_ce_majority": True,
        "initial_gate_eliminates_negative_dot": False}
    assert summary["primary_initial"]["gated_cosine_undefined_count"] == 48
    assert verify.summarize(probes[:-1])["primary_initial"]["hypotheses"] == {
        "initial_full_kl_larger_than_ce_majority": None,
        "initial_gate_eliminates_negative_dot": None}


def relock(root):
    files = {path.relative_to(root).as_posix(): verify.digest(path)
             for path in root.rglob("*") if path.is_file() and path.name != "archive-manifest.json"}
    run = verify.read(root / "run.json")
    run["artifacts"] = {name: sha for name, sha in files.items() if name != "run.json"}
    write(root / "run.json", run)
    manifest(root)


@pytest.fixture(scope="module")
def synthetic_archive(tmp_path_factory):
    """Full 192-receipt shape with fabricated TRAIN identifiers and gradients."""
    root = tmp_path_factory.mktemp("synthetic_gradients")
    protocol = verify.read(REPO / "configs/ce-kl-gradient-v1.json")
    source = [{"id": f"synthetic-{index}", "is_impossible": bool(index % 2),
               "context_id": f"context-{index // 2}"} for index in range(242)]
    selected = []
    for flag in (False, True):
        selected += sorted((row["id"] for row in source if row["is_impossible"] is flag),
                           key=lambda key: hashlib.sha256(("ce-kl-gradient-v1|" + key).encode()).hexdigest())[:8]
    protocol["selected_ids"] = selected
    input_dir = root / "inputs"; input_dir.mkdir()
    for name in ("source_train.jsonl", "artifact_train.jsonl"):
        (input_dir / name).write_text("".join(json.dumps(row) + "\n" for row in source))
    config = {"model_id": "synthetic", "revision": "f" * 40}
    write(input_dir / "student_config.json", config)
    write(input_dir / "artifact_manifest.json", {"synthetic": True})
    protocol["hashes"] = {key: verify.digest(input_dir / name) for key, name in {
        "source_train": "source_train.jsonl", "artifact_train": "artifact_train.jsonl",
        "artifact_manifest": "artifact_manifest.json", "student_config": "student_config.json"}.items()}
    protocol["hashes"]["cache_manifest"] = "1" * 64
    protocol["hashes"]["endpoint_training"] = {}
    records = [{"id": row["id"], "sha256": hashlib.sha256(row["id"].encode()).hexdigest(),
                "supervised_tokens": 6 if index < 79 else 5, "sequence_tokens": 99, "vocabulary": 151936}
               for index, row in enumerate(source)]
    identities = {"student": {"model_id": "synthetic", "revision": "f" * 40}, "teacher": {"model_id": "synthetic-teacher"}}
    cache = {"status": "complete", "temperature": 2., "model_identities": identities,
             "student_model_config": config, "records": records}
    write(input_dir / "cache.receipt.json", {"original_manifest_sha256": "1" * 64,
          "redactions": ["artifact_path"], "redacted_manifest": cache})
    adapters = {"adapter_config.json": {"sha256": "2" * 64, "bytes": 1},
                "adapter_model.safetensors": {"sha256": "3" * 64, "bytes": 1}}
    for seed in verify.SEEDS:
        for state in verify.STATES[1:]:
            path = input_dir / f"endpoint-{state}-{seed}.json"
            write(path, {"status": "complete", "arm": state, "seed": seed, "steps_completed": 242,
                  "cache_manifest_sha256": "1" * 64, "model_identities": identities,
                  "initial_trainable_state_sha256": "4" * 64, "adapter_files": adapters})
            protocol["hashes"]["endpoint_training"][f"{state}:{seed}"] = verify.digest(path)
    protocol_path = root / "source/configs/ce-kl-gradient-v1.json"; write(protocol_path, protocol)
    for name in ("gradient_diagnostics", "confirmation_training", "logits_distillation", "train_artifact", "model_identity", "common"):
        path = root / "source/qa_lab" / (name + ".py"); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic nonexecutable source receipt\n")
    source_hashes = {p.relative_to(root / "source").as_posix(): verify.digest(p)
                     for p in (root / "source").rglob("*") if p.is_file()}
    parameters, fixed_arrays = [], None
    for layer in range(24):
        for module in ("q_proj", "v_proj"):
            for matrix in ("A", "B"):
                parameter, tiny = fixture_parameter(layer, module, matrix)
                shape = [8, 896] if matrix == "A" else ([896, 8] if module == "q_proj" else [128, 8])
                arrays = {key: np.tile(value, (shape[0], shape[1] // 2)).astype(np.float32) for key, value in tiny.items()}
                components = [arrays[key].astype(np.float64).ravel() for key in verify.COMPONENTS]
                parameter.update(shape=shape, numel=int(np.prod(shape)),
                    gram=[[float(np.sum(left * right)) for right in components] for left in components],
                    gradient_sha256={key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()},
                    reconstruction={suffix: comparison(arrays["ce"] + arrays[key], arrays["combined_" + suffix])
                                    for suffix, key in (("full", "full_kl"), ("gated", "gated_kl"))})
                if (layer, module, matrix) == (0, "q_proj", "B"): fixed_arrays = arrays
                parameters.append(parameter)
    parameters.sort(key=lambda p: p["name"])
    template = fixture_probe(); template.update(parameters=parameters, groups=verify.grouped_geometry(parameters))
    probes, state_files = [], []
    records = {item["id"]: item for item in records}
    (root / "arrays").mkdir()
    first_parameter = next(p["name"] for p in parameters if (p["layer"], p["module"], p["matrix"]) == (0, "q_proj", "B"))
    for seed in verify.SEEDS:
        for state in verify.STATES:
            array_name = f"arrays/{state}-{seed}.npz"; np.savez_compressed(root / array_name, **fixed_arrays)
            parameter_state = {"adapter": {"sha256": "4" * 64, "numel": 540672}, "base": {"sha256": "5" * 64, "numel": 1000000}}
            receipt = {"status": "complete", "state": state, "seed": seed, "probes_completed": 16,
                       "parameter_blocks": 96, "trainable_parameters": 540672, "parameters_unchanged": True,
                       "parameters_before": parameter_state, "parameters_after": parameter_state,
                       "started_utc": "2026-10-09T00:00:01+00:00", "finished_utc": "2026-10-09T00:00:02+00:00"}
            if state == "initial": receipt["initial_trainable_state_sha256"] = "4" * 64
            else: receipt["adapter_files"] = adapters
            state_name = f"states/{state}-{seed}.json"; write(root / state_name, receipt); state_files.append(state_name)
            for selected_index, key in enumerate(selected):
                count = records[key]["supervised_tokens"]
                probe = {**deepcopy(template), "schema": 1, "status": "complete", "index": len(probes),
                         "state": state, "seed": seed, "train_id": key,
                         "is_impossible": next(r["is_impossible"] for r in source if r["id"] == key),
                         "role": "primary_initial" if state == "initial" else "secondary_endpoint",
                         "cache_record_sha256": records[key]["sha256"], "input_tokens": 99, "supervised_tokens": count,
                         "token_classes": {"eos_count": 1, "other_special_count": 0, "other_count": count - 1, "supervised_tokens": count},
                         "raw_artifact": None}
                if selected_index == 0:
                    probe["public_parameter"] = first_parameter
                    probe["raw_artifact"] = {"path": array_name, "sha256": verify.digest(root / array_name),
                                             "keys": list(verify.VECTORS), "parameter": first_parameter}
                write(root / f"probes/{len(probes):03d}.json", probe); probes.append(probe)
    write(root / "summary.json", verify.summarize(probes))
    write(root / "run.json", {"schema": 1, "study": "ce-kl-gradient-v1", "status": "complete", "diagnostic_complete": True,
          "phase": "complete", "probes_completed": 192, "states_completed": 12,
          "optimizer_steps": 0, "free_generation": 0, "heldout_predictions": 0,
          "started_utc": "2026-10-09T00:00:00+00:00", "finished_utc": "2026-10-09T00:00:03+00:00",
          "protocol_commit": "f" * 40, "source_sha256": source_hashes, "protocol_sha256": verify.digest(protocol_path),
          "protocol": protocol, "input_sha256": protocol["hashes"], "model_identities": identities,
          "state_files": state_files, "probe_files": [f"probes/{index:03d}.json" for index in range(192)],
          "wall_seconds": 3., "peak_process_rss_bytes": 1000000, "output_bytes_last_check": 1000000})
    relock(root)
    return root


def test_complete_synthetic_archive_and_nonmutation(synthetic_archive):
    before = {p.relative_to(synthetic_archive).as_posix(): verify.digest(p) for p in synthetic_archive.rglob("*") if p.is_file()}
    result = verify.verify(synthetic_archive)
    assert result["public_blocks_replayed"] == 12
    assert result["public_gradient_values_replayed"] == 12 * 896 * 8 * 5
    assert result["distinct_train_questions"] == 16
    assert result["summary"]["primary_initial"]["hypotheses"]["initial_gate_eliminates_negative_dot"] is False
    after = {p.relative_to(synthetic_archive).as_posix(): verify.digest(p) for p in synthetic_archive.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("mutation", ["scope", "source", "status", "coverage", "initial", "summary"])
def test_rehashed_semantic_archive_corruption_rejected(synthetic_archive, mutation):
    path = synthetic_archive / ("states/initial-20261009.json" if mutation == "initial" else "summary.json" if mutation == "summary" else "run.json")
    original = path.read_bytes()
    value = verify.read(path)
    if mutation == "scope": value["heldout_predictions"] = 1
    if mutation == "source": value["source_sha256"] = {}
    if mutation == "status": value["status"] = "failed"
    if mutation == "coverage": value["probe_files"].pop()
    if mutation == "initial": value["initial_trainable_state_sha256"] = "6" * 64
    if mutation == "summary": value["primary_initial"]["hypotheses"]["initial_gate_eliminates_negative_dot"] = True
    write(path, value); relock(synthetic_archive)
    try:
        with pytest.raises(ValueError):
            verify.verify(synthetic_archive)
    finally:
        path.write_bytes(original); relock(synthetic_archive)
