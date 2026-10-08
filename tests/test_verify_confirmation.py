"""Synthetic receipts/answers only; never read real confirmation heldout files."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_confirmation", REPO / "scripts/verify_confirmation.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
ARMS, SEEDS = verify.ARMS, verify.SEEDS
FREEZE = "dff77b115431bbf8e76113d7dd97712bd98ca1f3"
# CI may use a shallow checkout. Public source snapshots, when present, avoid
# depending on either Git history depth or future edits to live qa_lab modules.
ARCHIVED_SOURCE = REPO / "reports/teacher-gated-confirmation-v1/frozen-source"
SOURCE_ROOT = ARCHIVED_SOURCE if ARCHIVED_SOURCE.is_dir() else REPO


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in items))


def source(name):
    # Public frozen protocol/TRAIN identities only, never model/heldout data.
    return (SOURCE_ROOT / name).read_bytes()


def stamp(seconds):
    return (datetime(2026, 10, 9, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def manifest(root):
    write(root / "archive-manifest.json", {"schema": 1, "files": {
        p.relative_to(root).as_posix(): verify.digest(p)
        for p in sorted(root.rglob("*")) if p.is_file() and p != root / "archive-manifest.json"
    }})


def relock(root):
    """Rebind outer hashes after a semantic corruption, so guards must inspect it."""
    execution = verify.read(root / "execution-lock.json")
    study = verify.read(root / "study.json")
    for key in execution["adapters"]:
        folder = root / "training" / key.replace(":", "-")
        execution["adapters"][key]["training_sha256"] = verify.digest(folder / "training.json")
    write(root / "execution-lock.json", execution)
    lock = verify.read(root / "prediction-lock.json")
    lock["execution_lock_sha256"] = verify.digest(root / "execution-lock.json")
    for key in lock["files"]:
        folder = root / "predictions" / key.replace(":", "-")
        run = verify.read(folder / "run.json")
        run["execution_lock_sha256"] = lock["execution_lock_sha256"]
        run["predictions_sha256"] = verify.digest(folder / "predictions.jsonl")
        write(folder / "run.json", run)
        lock["files"][key] = {"run_sha256": verify.digest(folder / "run.json"), "predictions_sha256": run["predictions_sha256"]}
    write(root / "prediction-lock.json", lock)
    study["prediction_lock_sha256"] = verify.digest(root / "prediction-lock.json")
    study["scores_sha256"] = verify.digest(root / "scores.json")
    write(root / "study.json", study)
    manifest(root)


@pytest.fixture(scope="module")
def complete_archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_confirmation")
    frozen = root / "frozen-source"
    names = [p.relative_to(SOURCE_ROOT).as_posix() for p in (SOURCE_ROOT / "qa_lab").glob("*.py")]
    names += [p.relative_to(SOURCE_ROOT).as_posix() for p in (SOURCE_ROOT / "configs/teacher-gated-confirmation-v1").glob("*.json")]
    names += ["scripts/prepare_confirmation_data.py", "configs/confirmation-exclusions-v1.json"]
    names.sort()
    bindings = {}
    for name in names:
        target = frozen / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source(name)); bindings[name] = verify.digest(target)
    protocol = verify.read(frozen / verify.PROTOCOL_PATH)
    for name in protocol["frozen_inputs"]:
        target = frozen / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source(name))
    inputs, labels = [], []
    for context in range(128):
        for impossible in (False, True):
            row = {"id": f"synthetic-{context}-{int(impossible)}", "context_id": f"context-{context}",
                   "article_id": f"article-{context % 25}", "family_id": f"article-{context % 25}",
                   "context": f"Alice met Bob in context {context}.", "question": "Who?"}
            inputs.append(row)
            labels.append({k: v for k, v in row.items() if k != "question"} |
                          {"answers": [] if impossible else ["Alice"], "is_impossible": impossible})
    write_rows(root / "cohort/inputs.jsonl", inputs)
    write_rows(root / "cohort/sealed-labels.jsonl", labels)
    input_sha, label_sha = [verify.digest(root / "cohort" / name) for name in ("inputs.jsonl", "sealed-labels.jsonl")]
    exclusions = verify.read(frozen / protocol["data"]["exclusions"])
    write(root / "cohort/manifest.json", {
        "schema": "confirmation-data-v1", "status": "complete", "source_sha256": exclusions["source"]["sha256"],
        "exclusions_sha256": verify.digest(frozen / protocol["data"]["exclusions"]),
        "selector_sha256": verify.digest(frozen / protocol["data"]["sampler"]), "selection": exclusions["selection"],
        "inputs_sha256": input_sha, "labels_sha256": label_sha, "n_contexts": 128, "n_questions": 256,
        "answerable": 128, "unanswerable": 128,
        "article_context_counts": dict(verify.Counter({r["context_id"]: r["article_id"] for r in inputs}.values())),
    })
    training_artifact = frozen / protocol["training"]["artifact"]
    train_ids = [r["id"] for r in verify.rows(training_artifact / "train.jsonl")]
    records = [{"id": key, "path": "records/" + hashlib.sha256(key.encode()).hexdigest() + ".pt",
                "sha256": hashlib.sha256(key.encode()).hexdigest(), "sequence_tokens": 100,
                "supervised_tokens": 6 if i < 79 else 5, "vocabulary": 151936} for i, key in enumerate(train_ids)]
    configs, identities = {}, {}
    for role, path in (("student", "reports/model-artifacts.json"), ("teacher", "reports/closure-v1/teacher-model-artifacts.json")):
        config = verify.read(frozen / "configs/teacher-gated-confirmation-v1" / (role + ".json"))
        original = verify.read(frozen / path)
        configs[role] = config
        identities[role] = {"schema": 1, "model_id": config["model_id"], "revision": config["revision"],
                            "manifest": path, "manifest_sha256": verify.digest(frozen / path), "files": original.get("files", original)}
    cache = {"format": "answer-token-full-logprobs-v1", "status": "complete", "smoke_limit": None,
             "created_utc": stamp(1), "temperature": 2., "artifact_method": "gold_sft", "hard_target_source": "gold_reference",
             "records": records, "cached_row_count": 242, "artifact_row_count": 242,
             "artifact_manifest_sha256": verify.digest(training_artifact / "manifest.json"),
             "train_source_sha256": verify.digest(frozen / protocol["training"]["source"] / "train.jsonl"),
             "student_tokenizer_sha256": "a"*64, "teacher_tokenizer_sha256": "a"*64,
             "student_model_config": configs["student"], "teacher_model_config": configs["teacher"],
             "student_prompt_config": {k: configs["student"][k] for k in ("model_id", "revision", "system_prompt", "prompt_version")},
             "model_identities": identities}
    cache_sha = "b"*64
    write(root / "teacher-cache.receipt.json", {"original_manifest_sha256": cache_sha,
          "redacted_manifest": cache, "records_total_bytes": 1289*151936*2 + 242*100*8 + 500000})
    execution = {"created_utc": stamp(30), "status": "all_adapters_locked_before_predictions",
                 "adapters": {}, "inputs_sha256": input_sha, "labels_sha256": label_sha,
                 "frozen_files": bindings, "protocol_commit": FREEZE, "protocol_sha256": verify.PROTOCOL_SHA}
    lookup = {r["id"]: r for r in records}
    order = protocol["order"]["training"]
    for i, key in enumerate(order):
        arm, seed = key.split(":"); seed = int(seed)
        folder = root / "training" / key.replace(":", "-")
        write(folder / "adapter_config.json", {"base_model_name_or_path": configs["student"]["model_id"], "revision": configs["student"]["revision"]})
        adapter_files = {"adapter_config.json": {"sha256": verify.digest(folder / "adapter_config.json"), "bytes": (folder / "adapter_config.json").stat().st_size},
                         "adapter_model.safetensors": {"sha256": hashlib.sha256(key.encode()).hexdigest(), "bytes": 1024}}
        shuffled = train_ids.copy(); random.Random(seed).shuffle(shuffled)
        steps = []
        for index, item in enumerate(shuffled, 1):
            n = lookup[item]["supervised_tokens"]
            optimized = 0 if arm == "gold" else .1 if arm == "gated_kd" else .2
            steps.append({"id": item, "step": index, "input_tokens": 100, "supervised_tokens": n,
                "teacher_top1_correct": n-1, "kl_retained_tokens": 0 if arm == "gold" else n-1 if arm == "gated_kd" else n,
                "hard_ce": 1., "full_kl": .2, "optimized_kl": optimized,
                "loss": 1. if arm == "gold" else .5 + 2*optimized,
                "grad_norm_before_clip": .5, "clip_scale": 1., "gradient_clip_applied": False})
        write(folder / "steps.json", steps)
        training = {"status": "complete", "phase": "complete", "arm": arm, "seed": seed,
            "started_utc": stamp(2+i*3), "finished_utc": stamp(4+i*3), "protocol_commit": FREEZE, "protocol_sha256": verify.PROTOCOL_SHA,
            "source_sha256": {k: v for k, v in bindings.items() if k.startswith("qa_lab/") and k.endswith(".py")},
            "training_config": protocol["training"], "budget": protocol["budget"], "student_config": configs["student"],
            "model_identities": identities, "teacher_identity_binding": "bound_cache_receipt", "legacy_teacher_unbound": False,
            "cache_manifest_sha256": cache_sha, "verified_artifact_manifest_sha256": cache["artifact_manifest_sha256"],
            "verified_train_source_sha256": cache["train_source_sha256"], "adapter_files": adapter_files,
            "training_id_order": shuffled, "training_id_order_sha256": hashlib.sha256(json.dumps(shuffled, separators=(",", ":")).encode()).hexdigest(),
            "steps_completed": 242, "total_input_tokens": 24200, "total_supervised_tokens": 1289,
            "total_teacher_top1_correct": 1289-242, "total_kl_retained_tokens": sum(s["kl_retained_tokens"] for s in steps),
            "run_total_seconds": 2., "max_observed_rss_bytes": 1024,
            "initial_trainable_state_sha256": hashlib.sha256(str(seed).encode()).hexdigest()}
        write(folder / "training.json", training)
        execution["adapters"][key] = {"files": {k: v["sha256"] for k, v in adapter_files.items()}, "training_sha256": verify.digest(folder / "training.json")}
    write(root / "execution-lock.json", execution)
    lock = {"status": "all_predictions_locked_before_scoring", "created_utc": stamp(60),
            "execution_lock_sha256": verify.digest(root / "execution-lock.json"), "labels_sha256": label_sha, "files": {}}
    predictions = {arm: {} for arm in ARMS}
    for i, key in enumerate(order):
        arm, seed = key.split(":"); folder = root / "predictions" / key.replace(":", "-")
        items = [{"id": r["id"], "prediction": "Bob" if index % 7 == 0 else "NO_ANSWER" if r["is_impossible"] else "Alice", "generated_tokens": 1,
                  "token_ids": [1], "input_tokens": 10, "stop_reason": "eos", "prompt_sha256": "a"*64,
                  "ttft_seconds": .1, "generation_seconds": .1, "decode_seconds": 0., "end_to_end_seconds": .2,
                  "decode_tokens_per_second": None} for index, r in enumerate(labels)]
        predictions[arm][seed] = items
        write_rows(folder / "predictions.jsonl", items)
        run = {"status": "complete", "key": key, "n": 256, "execution_lock_sha256": lock["execution_lock_sha256"],
               "adapter_files": execution["adapters"][key]["files"], "inputs_sha256": input_sha,
               "model_identity": identities["student"], "predictions_sha256": verify.digest(folder / "predictions.jsonl"),
               "started_utc": stamp(31+i*3), "finished_utc": stamp(33+i*3), "max_input_tokens_observed": 10, "wall_seconds": 2.}
        write(folder / "run.json", run)
        lock["files"][key] = {"run_sha256": verify.digest(folder / "run.json"), "predictions_sha256": run["predictions_sha256"]}
    write(root / "prediction-lock.json", lock)
    write(root / "scores.json", verify.replay_scores(frozen, labels, predictions, protocol))
    names = ["teacher-cache"] + ["train-" + k.replace(":", "-") for k in order] + ["predict-" + k.replace(":", "-") for k in order]
    study = {"status": "complete", "started_utc": stamp(0), "finished_utc": stamp(61),
             "frozen_files": bindings, "protocol_commit": FREEZE, "protocol_sha256": verify.PROTOCOL_SHA,
             "cohort_manifest_sha256": verify.digest(root / "cohort/manifest.json"), "inputs_sha256": input_sha,
             "labels_sha256": label_sha, "stages": [{"name": name, "returncode": 0, "wall_seconds": 2., "rss_peak_sampled_bytes": 1024, "log_sha256": "d"*64} for name in names],
             "wall_seconds": 61., "prediction_lock_sha256": verify.digest(root / "prediction-lock.json"), "scores_sha256": verify.digest(root / "scores.json")}
    write(root / "study.json", study)
    write(root / "attempts/execution-01/study.json", {"status": "failed", "finished_utc": stamp(-10)})
    manifest(root)
    return root


@pytest.fixture
def archive(complete_archive, tmp_path):
    root = tmp_path / "archive"
    shutil.copytree(complete_archive, root)
    return root


def test_complete_synthetic_archive_replays_without_modifying_files(archive):
    before = verify.read(archive / "archive-manifest.json")
    result = verify.verify_archive(archive)
    assert result["status"] == "verified_public_receipts_and_score_replay"
    assert result["training_runs"] == result["prediction_runs"] == 9
    assert result["primary_accepted"] is False
    assert result["teacher_fp16_distribution_payload_bytes"] == 391691008
    assert verify.check_archive(archive) == before["files"]
    assert not list(archive.rglob("__pycache__"))


def test_published_confirmation_archive_replays_without_modification():
    # Required CI integration coverage: absence is an error, never a skip.
    root = REPO / "reports/teacher-gated-confirmation-v1"
    before = {p.relative_to(root).as_posix(): verify.digest(p) for p in root.rglob("*") if p.is_file()}
    result = verify.verify_archive(root)
    assert result["status"] == "verified_public_receipts_and_score_replay"
    assert result["training_runs"] == result["prediction_runs"] == 9
    assert result["questions_per_run"] == 256
    assert result["primary_accepted"] is False
    after = {p.relative_to(root).as_posix(): verify.digest(p) for p in root.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("mutation", ["unlisted", "missing", "corrupt", "nested_manifest", "weights", "symlink"])
def test_archive_file_set_and_hash_guards(archive, mutation):
    if mutation == "unlisted": (archive / "extra.txt").write_text("extra")
    elif mutation == "missing": (archive / "scores.json").unlink()
    elif mutation == "corrupt": (archive / "scores.json").write_text("{}")
    elif mutation == "nested_manifest":
        (archive / "extra").mkdir(); (archive / "extra/archive-manifest.json").write_text("{}")
    elif mutation == "weights":
        (archive / "private.safetensors").write_text("no weights"); manifest(archive)
    else: (archive / "link").symlink_to(archive / "scores.json")
    with pytest.raises(ValueError): verify.verify_archive(archive)


@pytest.mark.parametrize("mutation", ["prediction_key", "adapter_binding", "duplicate_prediction", "missing_arm", "same_seed_init", "training_order", "loss", "gate_count", "score", "timestamps", "cache_count", "cache_bytes", "source"])
def test_semantic_tamper_is_rejected_after_outer_hashes_are_updated(archive, mutation):
    key = "gated_kd-20261009"
    training = archive / "training" / key / "training.json"
    prediction = archive / "predictions" / key / "run.json"
    if mutation == "prediction_key":
        value = verify.read(prediction); value["key"] = "gold:20261009"; write(prediction, value)
    elif mutation == "adapter_binding":
        value = verify.read(prediction); value["adapter_files"]["adapter_model.safetensors"] = "0"*64; write(prediction, value)
    elif mutation == "duplicate_prediction":
        path = prediction.parent / "predictions.jsonl"; items = verify.rows(path); items[-1] = deepcopy(items[0]); write_rows(path, items)
    elif mutation == "missing_arm":
        value = verify.read(archive / "execution-lock.json"); value["adapters"].pop("full_kd:20261011"); write(archive / "execution-lock.json", value)
    elif mutation == "same_seed_init":
        value = verify.read(training); value["initial_trainable_state_sha256"] = "0"*64; write(training, value)
    elif mutation == "training_order":
        value = verify.read(training); value["training_id_order"].reverse(); write(training, value)
    elif mutation in {"loss", "gate_count"}:
        path = training.parent / "steps.json"; values = verify.read(path)
        if mutation == "loss": values[0]["loss"] += .1
        else: values[0]["kl_retained_tokens"] += 1
        write(path, values)
    elif mutation == "score":
        path = archive / "scores.json"; value = verify.read(path); value["primary"]["accepted"] = True; write(path, value)
    elif mutation == "timestamps":
        value = verify.read(prediction); value["started_utc"] = stamp(1); write(prediction, value)
    elif mutation.startswith("cache"):
        path = archive / "teacher-cache.receipt.json"; value = verify.read(path)
        if mutation == "cache_count": value["redacted_manifest"]["cached_row_count"] = 241
        else: value["records_total_bytes"] = 1
        write(path, value)
    else:
        path = archive / "frozen-source/qa_lab/confirmation_scoring.py"; path.write_text(path.read_text() + "\n# modified\n")
    relock(archive)
    with pytest.raises(ValueError): verify.verify_archive(archive)


def test_duplicate_or_nonfinite_json_is_rejected():
    for content in ('{"x":1,"x":2}', '{"loss":NaN}', '{"loss":Infinity}', '{"loss":1e999}'):
        with pytest.raises(ValueError): verify.loads(content)


def test_family_macro_only_has_narrow_nondecision_tolerance(archive):
    original = verify.read(archive / "scores.json")
    slightly_different = deepcopy(original)
    slightly_different["arms"]["gold"]["per_seed"][str(SEEDS[0])]["family_macro_em"] += 5e-16
    verify.compare_scores(original, slightly_different)
    too_different = deepcopy(original)
    too_different["arms"]["gold"]["per_seed"][str(SEEDS[0])]["family_macro_em"] += 2e-15
    with pytest.raises(ValueError, match="family macro"):
        verify.compare_scores(original, too_different)
    for path in ("overall_em", "primary_difference", "gate"):
        changed = deepcopy(original)
        if path == "overall_em": changed["arms"]["gold"]["per_seed"][str(SEEDS[0])]["overall"]["em"] += 5e-16
        elif path == "primary_difference": changed["primary"]["em_difference"]["mean"] += 5e-16
        else: changed["primary"]["accepted"] = not changed["primary"]["accepted"]
        with pytest.raises(ValueError, match="exactly"):
            verify.compare_scores(original, changed)


def test_frozen_replay_is_valid_across_python_hash_seeds(archive):
    program = (
        "import importlib.util,json,sys;from pathlib import Path;"
        "s=importlib.util.spec_from_file_location('v',sys.argv[1]);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);"
        "r=Path(sys.argv[2]);p=v.read(r/'frozen-source'/v.PROTOCOL_PATH);"
        "pred={arm:{str(seed):v.rows(r/'predictions'/f'{arm}-{seed}'/'predictions.jsonl') for seed in v.SEEDS} for arm in v.ARMS};"
        "a=v.replay_scores(r/'frozen-source',v.rows(r/'cohort/sealed-labels.jsonl'),pred,p);print(json.dumps(a))"
    )
    answers = []
    for seed in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", program, str(REPO / "scripts/verify_confirmation.py"), str(archive)],
                                env=dict(os.environ, PYTHONHASHSEED=seed), capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        answers.append(json.loads(result.stdout))
    verify.compare_scores(*answers)
    assert answers[0]["primary"] == answers[1]["primary"]
    assert not list(archive.rglob("__pycache__"))


def test_clip_boolean_matches_frozen_scale_not_nominal_grad_threshold(archive):
    # Float32 producer can clip by epsilon even when the nominal norm is < 1.
    path = archive / "training/gated_kd-20261009/steps.json"
    steps = verify.read(path)
    steps[0].update(grad_norm_before_clip=.9999999403953552,
                    clip_scale=.9999990463256836, gradient_clip_applied=True)
    write(path, steps); relock(archive)
    assert verify.verify_archive(archive)["training_runs"] == 9
    steps[0]["gradient_clip_applied"] = False
    write(path, steps); relock(archive)
    with pytest.raises(ValueError, match="clip arithmetic"):
        verify.verify_archive(archive)


def test_optimized_python_does_not_disable_checks(archive):
    (archive / "extra.txt").write_text("extra")
    result = subprocess.run([sys.executable, "-O", str(REPO / "scripts/verify_confirmation.py"), "--root", str(archive)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "file set differs" in result.stderr
