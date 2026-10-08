"""Offline receipt/score replay; no model, weights, torch, or GPU is loaded.

Hash consistency is not an authenticity signature. Training, prediction and
label-access order are checked as recorded, not independently re-executed.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import sys
import types


PROTOCOL_PATH = "configs/teacher-gated-confirmation-v1/protocol.json"
PROTOCOL_SHA = "d1673cf172d18cefc1f94cdc920c86af77d0ff789c0ace56ba77ca688cc8c508"
# These pure-Python replay modules were committed before study execution.
REPLAY_SHA = {
    "common": "1b2bb44e1a59fd07b7b53d6a4581b86ee61ffd297a711843dfd0fbe328d92418",
    "metrics": "65e3c472761fda70a0e90b9f929a5c85508242eec6fe1259464d05738404c005",
    "confirmation_scoring": "62a1137691230033394e580062616d9876afe64a285f632ef2e6130ff0a7d00d",
}
ARMS = ("gold", "full_kd", "gated_kd")
SEEDS = (20261009, 20261010, 20261011)
SHA = re.compile(r"[0-9a-f]{64}\Z")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Non-finite JSON number: " + value)


def _float(value):
    result = float(value)
    need(math.isfinite(result), "Non-finite JSON float")
    return result


def loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)


def read(path):
    return loads(Path(path).read_text())


def rows(path):
    return [loads(line) for line in Path(path).read_text().splitlines() if line]


def _relative(name):
    need(isinstance(name, str) and name and "\\" not in name, "Invalid archive path")
    path = PurePosixPath(name)
    need(not path.is_absolute() and ".." not in path.parts and path.as_posix() == name,
         "Archive path escapes or is not canonical")
    return name


def _sha(value):
    need(isinstance(value, str) and SHA.fullmatch(value), "Invalid SHA256")
    return value


def _number(value, name, minimum=None):
    need(type(value) in (int, float) and math.isfinite(value), "Invalid finite number: " + name)
    if minimum is not None:
        need(value >= minimum, "Negative/out-of-range number: " + name)
    return value


def _integer(value, name, minimum=0):
    need(type(value) is int and value >= minimum, "Invalid integer: " + name)
    return value


def _time(value):
    time = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(time.tzinfo is not None, "Timestamp lacks timezone")
    return time


def check_archive(root):
    need(root.is_dir() and not root.is_symlink(), "Archive root must be a real directory")
    manifest = read(root / "archive-manifest.json")
    need(manifest.get("schema") == 1 and isinstance(manifest.get("files"), dict), "Invalid archive manifest")
    expected = manifest["files"]
    need("archive-manifest.json" not in expected, "Root archive manifest cannot hash itself")
    actual = set()
    for path in root.rglob("*"):
        need(not path.is_symlink(), "Archive symlinks are forbidden")
        if path.is_file() and path != root / "archive-manifest.json":
            actual.add(path.relative_to(root).as_posix())
    need(actual == set(expected), "Archive file set differs from manifest (missing or extra file)")
    for name, expected_sha in expected.items():
        _relative(name)
        need(digest(root / name) == _sha(expected_sha), "Archive hash differs: " + name)
        need(Path(name).suffix not in {".safetensors", ".pt", ".bin"}, "Weights or full tensor cache must not be published")
    return expected


def replay_scores(frozen, labels, predictions, protocol):
    """Load only pinned historical arithmetic source, never unverified code."""
    for name, expected in REPLAY_SHA.items():
        need(digest(frozen / "qa_lab" / (name + ".py")) == expected, "Unrecognized frozen replay source: " + name)
    package_name = "_frozen_confirmation_replay"
    previous = {name: module for name, module in sys.modules.items()
                if name == package_name or name.startswith(package_name + ".")}
    package = types.ModuleType(package_name)
    package.__path__ = [str(frozen / "qa_lab")]
    sys.modules[package_name] = package
    try:
        for name in REPLAY_SHA:
            module_name = package_name + "." + name
            spec = importlib.util.spec_from_file_location(module_name, frozen / "qa_lab" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Compile verified text directly: SourceFileLoader.exec_module may
            # create __pycache__ inside the otherwise read-only evidence tree.
            exec(compile((frozen / "qa_lab" / (name + ".py")).read_bytes(), str(spec.origin), "exec"), module.__dict__)
        return module.score_study(labels, predictions, protocol)
    finally:
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                del sys.modules[name]
        sys.modules.update(previous)


def compare_scores(stored, reproduced):
    """Only the legacy set-iteration family macro tolerates summation ordering.

    Frozen metrics.evaluate sums a set of family IDs. Its floating-point order
    varies with PYTHONHASHSEED. This descriptive field never enters acceptance.
    All other score fields (including bootstrap and every gate) remain exact.
    """
    normalized = deepcopy(reproduced)
    for arm in ARMS:
        for seed in SEEDS:
            path = stored["arms"][arm]["per_seed"][str(seed)]
            expected = normalized["arms"][arm]["per_seed"][str(seed)]
            actual_macro = _number(path["family_macro_em"], "family macro EM")
            expected_macro = _number(expected["family_macro_em"], "family macro EM")
            need(abs(actual_macro - expected_macro) <= 1e-15, "Descriptive family macro EM differs")
            expected["family_macro_em"] = actual_macro
    need(stored == normalized, "Frozen scores do not exactly reproduce")


def check_sources(root, study, execution):
    frozen = root / "frozen-source"
    need(study["frozen_files"] == execution["frozen_files"] and execution["frozen_files"], "Frozen source maps differ or are empty")
    need(digest(frozen / PROTOCOL_PATH) == PROTOCOL_SHA, "Frozen statistical/training protocol changed")
    protocol = read(frozen / PROTOCOL_PATH)
    need(study["protocol_sha256"] == execution["protocol_sha256"] == PROTOCOL_SHA, "Protocol receipt differs")
    need(study["protocol_commit"] == execution["protocol_commit"] and
         re.fullmatch(r"[0-9a-f]{40}", study["protocol_commit"]), "Execution commit receipts differ")
    combined = dict(execution["frozen_files"])
    for name, expected in protocol["frozen_inputs"].items():
        need(name not in combined or combined[name] == expected, "Conflicting frozen input hash")
        combined[name] = expected
    for name, expected in combined.items():
        need(digest(frozen / _relative(name)) == _sha(expected), "Frozen source/input changed: " + name)
    actual = {p.relative_to(frozen).as_posix() for p in frozen.rglob("*") if p.is_file()}
    need(actual == set(combined), "Frozen source file set differs")
    return frozen, protocol


def check_cohort(root, frozen, protocol, study, execution, lock):
    cohort = root / "cohort"
    manifest = read(cohort / "manifest.json")
    exclusions = read(frozen / protocol["data"]["exclusions"])
    need(manifest["schema"] == "confirmation-data-v1" and manifest["status"] == "complete", "Cohort incomplete")
    need(manifest["source_sha256"] == exclusions["source"]["sha256"] and
         manifest["exclusions_sha256"] == digest(frozen / protocol["data"]["exclusions"]) and
         manifest["selector_sha256"] == digest(frozen / protocol["data"]["sampler"]) and
         manifest["selection"] == exclusions["selection"], "Cohort selector/source binding changed")
    need(study["cohort_manifest_sha256"] == digest(cohort / "manifest.json"), "Cohort manifest receipt differs")
    input_sha, label_sha = digest(cohort / "inputs.jsonl"), digest(cohort / "sealed-labels.jsonl")
    need(input_sha == manifest["inputs_sha256"] == study["inputs_sha256"] == execution["inputs_sha256"], "Cohort input hash differs")
    need(label_sha == manifest["labels_sha256"] == study["labels_sha256"] == execution["labels_sha256"] == lock["labels_sha256"], "Cohort label hash differs")
    inputs, labels = rows(cohort / "inputs.jsonl"), rows(cohort / "sealed-labels.jsonl")
    need(len(inputs) == len(labels) == manifest["n_questions"] == protocol["data"]["questions"], "Cohort question count differs")
    by_id = {row["id"]: row for row in inputs}
    need(len(by_id) == len(inputs), "Duplicate cohort input ID")
    need({row["id"] for row in labels} == set(by_id) and len({r["id"] for r in labels}) == len(labels), "Cohort label ID coverage differs")
    for row in inputs:
        need(set(row) <= {"id", "context", "question", "article_id", "context_id", "family_id"}, "Prediction inputs contain labels")
    for label in labels:
        need(all(label[key] == by_id[label["id"]][key] for key in ("context", "article_id", "context_id", "family_id")), "Input/label metadata disagrees")
    contexts = {r["context_id"]: r["article_id"] for r in inputs}
    need(len(contexts) == manifest["n_contexts"] == protocol["data"]["contexts"], "Cohort context count differs")
    need(dict(Counter(contexts.values())) == manifest["article_context_counts"], "Cohort article counts differ")
    need(sum(not r["is_impossible"] for r in labels) == manifest["answerable"] == 128 and
         sum(r["is_impossible"] for r in labels) == manifest["unanswerable"] == 128, "Cohort class counts differ")
    return inputs, labels


def check_cache(root, frozen, protocol):
    receipt = read(root / "teacher-cache.receipt.json")
    original_sha = _sha(receipt["original_manifest_sha256"])
    cache = receipt["redacted_manifest"]
    need("artifact_path" not in cache, "Private artifact_path was not redacted")
    need(cache["format"] == "answer-token-full-logprobs-v1" and cache["status"] == "complete" and cache["smoke_limit"] is None,
         "Full cache status differs")
    need(cache["temperature"] == 2.0 and cache["artifact_method"] == "gold_sft" and cache["hard_target_source"] == "gold_reference", "Cache target/objective differs")
    records = cache["records"]
    train_root = frozen / protocol["training"]["artifact"]
    train_rows = rows(train_root / "train.jsonl")
    need(len(records) == len(train_rows) == cache["cached_row_count"] == cache["artifact_row_count"] == 242, "Full cache coverage differs")
    ids = [r["id"] for r in records]
    need(len(set(ids)) == 242 and ids == [r["id"] for r in train_rows], "Cache TRAIN ID order differs")
    need(cache["artifact_manifest_sha256"] == digest(train_root / "manifest.json") and
         cache["train_source_sha256"] == digest(frozen / protocol["training"]["source"] / "train.jsonl"), "Cache TRAIN source identity differs")
    need(cache["student_tokenizer_sha256"] == cache["teacher_tokenizer_sha256"], "Teacher/student tokenizer identities differ")
    _sha(cache["student_tokenizer_sha256"])
    for role in ("student", "teacher"):
        config = read(frozen / "configs/teacher-gated-confirmation-v1" / (role + ".json"))
        need(cache[role + "_model_config"] == config, "Cache runtime configuration differs")
        identity = cache["model_identities"][role]
        identity_manifest = frozen / _relative(identity["manifest"])
        need(identity["model_id"] == config["model_id"] and identity["revision"] == config["revision"] and
             identity["manifest_sha256"] == digest(identity_manifest), "Model identity differs")
        original = read(identity_manifest)
        need(identity["files"] == original.get("files", original), "Model file identity declaration differs")
        for metadata in identity["files"].values():
            _sha(metadata["sha256"]); _integer(metadata["bytes"], "model file bytes", 1)
    cfg = cache["student_model_config"]
    need(cache["student_prompt_config"] == {k: cfg[k] for k in ("model_id", "revision", "system_prompt", "prompt_version")}, "Prompt/token contract differs")
    for record in records:
        need(record["path"] == "records/" + hashlib.sha256(record["id"].encode()).hexdigest() + ".pt", "Cache record path differs")
        _sha(record["sha256"])
        _integer(record["supervised_tokens"], "supervised tokens", 1)
        need(record["vocabulary"] == 151936 and record["supervised_tokens"] < record["sequence_tokens"] <= 2048, "Cache shape metadata differs")
    need(sum(r["supervised_tokens"] for r in records) == 1289, "Cache supervised token count differs")
    payload = sum(r["supervised_tokens"] * r["vocabulary"] * 2 + r["sequence_tokens"] * 8 for r in records)
    total_bytes = _integer(receipt["records_total_bytes"], "records_total_bytes", 1)
    need(payload <= total_bytes <= protocol["budget"]["max_output_bytes"], "Declared cache size contradicts tensor payload or budget")
    return cache, original_sha, total_bytes


def check_training(root, frozen, protocol, execution, cache, cache_sha):
    receipts = {}
    meta = {r["id"]: r for r in cache["records"]}
    for key in protocol["order"]["training"]:
        arm, seed = key.split(":"); seed = int(seed)
        folder = root / "training" / key.replace(":", "-")
        receipt, steps = read(folder / "training.json"), read(folder / "steps.json")
        need(receipt["status"] == "complete" and receipt["phase"] == "complete" and receipt["arm"] == arm and receipt["seed"] == seed,
             "Training run identity/status differs")
        need(receipt["protocol_commit"] == execution["protocol_commit"] and receipt["protocol_sha256"] == PROTOCOL_SHA, "Training execution binding differs")
        expected_python = {k: v for k, v in execution["frozen_files"].items() if k.startswith("qa_lab/") and k.endswith(".py")}
        need(receipt["source_sha256"] == expected_python, "Training source map differs")
        need(receipt["training_config"] == protocol["training"] and receipt["budget"] == protocol["budget"], "Training settings differ")
        need(receipt["student_config"] == cache["student_model_config"] and receipt["model_identities"] == cache["model_identities"], "Training model identity differs")
        need(receipt["teacher_identity_binding"] == "bound_cache_receipt" and receipt["legacy_teacher_unbound"] is False, "Legacy/unbound cache is not confirmation")
        need(receipt["cache_manifest_sha256"] == cache_sha and receipt["verified_artifact_manifest_sha256"] == cache["artifact_manifest_sha256"] and
             receipt["verified_train_source_sha256"] == cache["train_source_sha256"], "Training cache/source binding differs")
        need(execution["adapters"][key]["training_sha256"] == digest(folder / "training.json"), "Training lock hash differs")
        files = receipt["adapter_files"]
        need(set(files) == {"adapter_config.json", "adapter_model.safetensors"}, "Adapter identity declarations differ")
        need({name: value["sha256"] for name, value in files.items()} == execution["adapters"][key]["files"], "Adapter training/execution identities differ")
        for name, value in files.items():
            _sha(value["sha256"]); _integer(value["bytes"], "adapter bytes", 1)
        need(digest(folder / "adapter_config.json") == files["adapter_config.json"]["sha256"] and
             (folder / "adapter_config.json").stat().st_size == files["adapter_config.json"]["bytes"], "Published adapter config differs")
        adapter = read(folder / "adapter_config.json")
        need(adapter["base_model_name_or_path"] == cache["student_model_config"]["model_id"] and
             adapter["revision"] == cache["student_model_config"]["revision"], "Adapter base identity differs")
        order = list(meta)
        random.Random(seed).shuffle(order)
        need(receipt["training_id_order"] == order and
             receipt["training_id_order_sha256"] == hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(), "Training order differs from seed")
        need(len(steps) == receipt["steps_completed"] == 242 and [s["id"] for s in steps] == order, "Training step coverage differs")
        for index, step in enumerate(steps, 1):
            record = meta[step["id"]]
            n = _integer(step["supervised_tokens"], "supervised tokens", 1)
            correct = _integer(step["teacher_top1_correct"], "teacher top1")
            need(step["step"] == index and step["input_tokens"] == record["sequence_tokens"] and n == record["supervised_tokens"] and correct <= n,
                 "Training token/step record differs")
            retained = correct if arm == "gated_kd" else n if arm == "full_kd" else 0
            need(step["kl_retained_tokens"] == retained, "KL gating denominator/count differs")
            for name in ("loss", "hard_ce", "full_kl", "optimized_kl", "grad_norm_before_clip", "clip_scale"):
                _number(step[name], name)
            need(step["hard_ce"] >= 0 and step["grad_norm_before_clip"] >= 0 and 0 < step["clip_scale"] <= 1, "Invalid CE/gradient/clip value")
            optimized = step["optimized_kl"]
            if arm == "gold": need(optimized == 0, "Gold arm optimized KL is nonzero")
            if arm == "full_kd" or (arm == "gated_kd" and retained == n):
                need(optimized == step["full_kl"], "Full-retention KL differs")
            if not retained: need(optimized == 0, "Zero-retention KL differs")
            expected = step["hard_ce"] if arm == "gold" else .5 * step["hard_ce"] + 2 * optimized
            need(math.isclose(step["loss"], expected, rel_tol=2e-6, abs_tol=2e-6), "Float32 loss arithmetic differs")
            scale = min(1., 1. / (step["grad_norm_before_clip"] + 1e-6))
            need(math.isclose(step["clip_scale"], scale, rel_tol=2e-6, abs_tol=2e-7) and
                 step["gradient_clip_applied"] is (step["clip_scale"] < 1), "Gradient clip arithmetic differs")
        for total, field in (("total_input_tokens", "input_tokens"), ("total_supervised_tokens", "supervised_tokens"),
                             ("total_teacher_top1_correct", "teacher_top1_correct"), ("total_kl_retained_tokens", "kl_retained_tokens")):
            need(receipt[total] == sum(s[field] for s in steps), "Training aggregate differs: " + total)
        need(receipt["total_supervised_tokens"] == 1289, "Training supervised count differs")
        need(_number(receipt["run_total_seconds"], "training seconds", 0) <= protocol["budget"]["training_run_seconds"] and
             _number(receipt["max_observed_rss_bytes"], "training RSS", 0) <= protocol["budget"]["max_sampled_rss_bytes"], "Training resource receipt exceeds budget")
        receipts[key] = receipt
    for seed in SEEDS:
        paired = [receipts[f"{arm}:{seed}"] for arm in ARMS]
        for name in ("initial_trainable_state_sha256", "training_id_order_sha256", "cache_manifest_sha256", "total_teacher_top1_correct"):
            need(len({r[name] for r in paired}) == 1, "Matched seed training identities differ: " + name)
    return receipts


def check_predictions(root, protocol, execution, lock, inputs, cache):
    need(lock["execution_lock_sha256"] == digest(root / "execution-lock.json"), "Prediction/execution lock hash differs")
    need(lock["status"] == "all_predictions_locked_before_scoring", "Prediction lock status differs")
    expected_ids = [r["id"] for r in inputs]
    predictions = {arm: {} for arm in ARMS}
    receipts = {}
    for key in protocol["order"]["training"]:
        arm, seed = key.split(":")
        folder = root / "predictions" / key.replace(":", "-")
        receipt, items = read(folder / "run.json"), rows(folder / "predictions.jsonl")
        need(receipt["status"] == "complete" and receipt["key"] == key and receipt["n"] == 256, "Prediction status/key/count differs")
        need(receipt["execution_lock_sha256"] == lock["execution_lock_sha256"] and
             receipt["adapter_files"] == execution["adapters"][key]["files"] and
             receipt["inputs_sha256"] == execution["inputs_sha256"], "Prediction model/input binding differs")
        need(receipt["model_identity"] == cache["model_identities"]["student"], "Prediction base model identity differs")
        need(lock["files"][key]["run_sha256"] == digest(folder / "run.json") and
             lock["files"][key]["predictions_sha256"] == receipt["predictions_sha256"] == digest(folder / "predictions.jsonl"), "Prediction file/lock hashes differ")
        need([r["id"] for r in items] == expected_ids and all(isinstance(r["prediction"], str) for r in items), "Prediction exact coverage/order differs")
        for item in items:
            count = _integer(item["generated_tokens"], "generated tokens", 1)
            need(count == len(item["token_ids"]) <= protocol["generation"]["max_new_tokens"] and
                 0 < item["input_tokens"] <= protocol["generation"]["max_input_tokens"], "Prediction token budget differs")
            need(all(type(token) is int and 0 <= token < 151936 for token in item["token_ids"]), "Invalid predicted token ID")
            need(item["stop_reason"] in {"eos", "max_new_tokens"}, "Unknown prediction stop reason")
            _sha(item["prompt_sha256"])
            for name in ("ttft_seconds", "generation_seconds", "decode_seconds", "end_to_end_seconds"):
                _number(item[name], name, 0)
            if item["decode_tokens_per_second"] is not None:
                _number(item["decode_tokens_per_second"], "decode cost rate", 0)
        need(max(r["input_tokens"] for r in items) == receipt["max_input_tokens_observed"], "Input token preflight receipt differs")
        need(_number(receipt["wall_seconds"], "prediction seconds", 0) <= protocol["budget"]["prediction_run_seconds"], "Prediction time exceeds budget")
        predictions[arm][seed] = items; receipts[key] = receipt
    return predictions, receipts


def verify_archive(root):
    root = Path(root)
    files = check_archive(root)
    study, execution, lock = [read(root / name) for name in ("study.json", "execution-lock.json", "prediction-lock.json")]
    need(study["status"] == "complete" and execution["status"] == "all_adapters_locked_before_predictions", "Study/adapter lock incomplete")
    frozen, protocol = check_sources(root, study, execution)
    expected = {f"{arm}:{seed}" for arm in ARMS for seed in SEEDS}
    order = protocol["order"]["training"]
    need(len(order) == 9 and set(order) == expected and set(execution["adapters"]) == set(lock["files"]) == expected, "Missing/extra/duplicate arm-seed run")
    need({p.name for p in (root / "training").iterdir()} == {k.replace(":", "-") for k in expected} and
         {p.name for p in (root / "predictions").iterdir()} == {k.replace(":", "-") for k in expected}, "Run directory coverage differs")
    inputs, labels = check_cohort(root, frozen, protocol, study, execution, lock)
    cache, cache_sha, cache_bytes = check_cache(root, frozen, protocol)
    training = check_training(root, frozen, protocol, execution, cache, cache_sha)
    predictions, prediction_receipts = check_predictions(root, protocol, execution, lock, inputs, cache)
    start, finish = _time(study["started_utc"]), _time(study["finished_utc"])
    adapter_locked, predictions_locked = _time(execution["created_utc"]), _time(lock["created_utc"])
    previous = _time(cache["created_utc"])
    need(start <= previous, "Cache predates recorded study start")
    for key in order:
        r = training[key]
        need(previous <= _time(r["started_utc"]) <= _time(r["finished_utc"]) <= adapter_locked, "Training/adapter-lock timestamp order differs")
        previous = _time(r["finished_utc"])
    previous = adapter_locked
    for key in order:
        r = prediction_receipts[key]
        need(previous <= _time(r["started_utc"]) <= _time(r["finished_utc"]) <= predictions_locked, "Prediction-before-training or unordered timestamps")
        previous = _time(r["finished_utc"])
    need(predictions_locked <= finish and start < finish, "Study completion predates prediction lock")
    stage_names = ["teacher-cache"] + ["train-" + k.replace(":", "-") for k in order] + ["predict-" + k.replace(":", "-") for k in order]
    need([s["name"] for s in study["stages"]] == stage_names, "Execution stage coverage/order differs")
    for stage in study["stages"]:
        need(stage["returncode"] == 0 and _number(stage["wall_seconds"], "stage wall", 0) <= 900 and
             _number(stage["rss_peak_sampled_bytes"], "stage RSS", 0) <= protocol["budget"]["max_sampled_rss_bytes"], "Stage status/resource bound differs")
        _sha(stage["log_sha256"])
    need(_number(study["wall_seconds"], "study wall", 0) <= protocol["budget"]["total_wall_seconds"], "Study wall budget exceeded")
    need(study["prediction_lock_sha256"] == digest(root / "prediction-lock.json") and study["scores_sha256"] == digest(root / "scores.json"), "Final study output hashes differ")
    stored = read(root / "scores.json")
    reproduced = replay_scores(frozen, labels, predictions, protocol)
    compare_scores(stored, reproduced)
    attempts = root / "attempts"
    if attempts.exists():
        for previous_study in attempts.glob("*/study.json"):
            attempt = read(previous_study)
            need(attempt["status"] == "failed" and _time(attempt["finished_utc"]) <= start,
                 "Prior failed attempt status/time differs")
    return {
        "status": "verified_public_receipts_and_score_replay", "archive_files": len(files),
        "execution_commit": study["protocol_commit"], "training_runs": 9, "prediction_runs": 9,
        "questions_per_run": 256, "cache_records": 242, "cache_declared_record_bytes": cache_bytes,
        "teacher_fp16_distribution_payload_bytes": 1289 * 151936 * 2,
        "primary_accepted": stored["primary"]["accepted"],
        "primary_em_difference": stored["primary"]["em_difference"],
        "primary_interval": stored["primary"]["article_cluster_bootstrap_95_percentile_ci"],
        "limitations": [
            "Replays published scores and recorded identities/order only; does not rerun training or inference and is not independent GPU/model quality confirmation.",
            "Adapter weights and full teacher tensors are omitted. Their SHA256, byte counts, initial-state and gate receipts are declarations, not remeasured content.",
            "Redacting artifact_path changes cache manifest bytes; its original SHA256 is checked between receipts, not reconstructed from the redacted manifest.",
            "Timestamps and source hashes support an auditable process boundary, not cryptographic proof of label concealment or source authenticity.",
            "Float32 training loss/clip arithmetic uses 2e-6 relative tolerance (loss abs 2e-6, clip abs 2e-7). Only per-seed descriptive family_macro_em permits absolute 1e-15 for the frozen set summation's hash-order variability; every other score field and acceptance gate replays exactly.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reports/teacher-gated-confirmation-v1"))
    parser.add_argument("--output", type=Path, help="Optional new JSON file outside the frozen archive")
    args = parser.parse_args()
    result = verify_archive(args.root)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        need(not args.output.resolve().is_relative_to(args.root.resolve()), "Verifier output must be outside the frozen archive")
        with args.output.open("x") as stream:
            stream.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
