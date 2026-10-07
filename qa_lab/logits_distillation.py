"""Hash-bound, answer-token logits distillation for the QA student.

The cache stores the teacher's complete temperature-scaled vocabulary
distribution only at causal positions that predict answer tokens.  It does not
store prompt-token logits and it never reads dev/test labels.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import time

from .common import digest, sha, source_hashes, write_json
from .inference import hardware_chip, load, sync
from .train_artifact import target_tokens, verified_rows


CACHE_FORMAT = "answer-token-full-logprobs-v1"
SUPPORTED_TARGET_METHODS = {"response_distillation", "gold_sft"}


def validate_objective(config):
    required = {"temperature", "hard_ce_weight", "soft_kl_weight"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing objective fields: {sorted(missing)}")
    temperature = float(config["temperature"])
    hard = float(config["hard_ce_weight"])
    soft = float(config["soft_kl_weight"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in (hard, soft)):
        raise ValueError("loss weights must be finite and nonnegative")
    if hard + soft <= 0:
        raise ValueError("at least one loss weight must be positive")
    return temperature, hard, soft


def tokenizer_fingerprint(tokenizer):
    payload = {
        "vocab": sorted(tokenizer.get_vocab().items()),
        "special_tokens_map": tokenizer.special_tokens_map,
    }
    return digest(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def sequence_config(config):
    """Fields that determine token IDs; runtime device/dtype are intentionally excluded."""
    return {key: config[key] for key in ["model_id", "revision", "system_prompt", "prompt_version"]}


def validate_target_method(method):
    """Return a precise target-source label for supported audited artifacts."""
    if method not in SUPPORTED_TARGET_METHODS:
        raise ValueError(
            f"teacher logits cache requires one of {sorted(SUPPORTED_TARGET_METHODS)}; got {method!r}"
        )
    return "teacher_response" if method == "response_distillation" else "gold_reference"


def supervised_prediction_mask(labels):
    """Return mask over logits[:, :-1] that predict labels[:, 1:]."""
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence>=2]")
    return labels[:, 1:] != -100


def distillation_losses(student_logits, labels, teacher_log_probs, objective):
    """Compute hard CE and exact full-vocabulary KL on answer-token positions.

    student_logits: [1, sequence, vocabulary]
    labels: [1, sequence], prompt positions are -100
    teacher_log_probs: [supervised_positions, vocabulary], already scaled by T
    """
    import torch
    import torch.nn.functional as functional

    temperature, hard_weight, soft_weight = validate_objective(objective)
    if student_logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("student_logits and labels must be rank 3 and rank 2")
    if student_logits.shape[:2] != labels.shape:
        raise ValueError("student logits and labels must share batch/sequence dimensions")
    mask = supervised_prediction_mask(labels)
    selected = student_logits[:, :-1, :][mask]
    targets = labels[:, 1:][mask]
    if selected.shape != teacher_log_probs.shape:
        raise ValueError(
            f"teacher/student distribution shape mismatch: {tuple(teacher_log_probs.shape)} != {tuple(selected.shape)}"
        )
    if selected.shape[0] == 0:
        raise ValueError("no supervised answer tokens")
    if not torch.isfinite(teacher_log_probs).all():
        raise ValueError("teacher distribution contains non-finite values")
    hard_ce = functional.cross_entropy(selected.float(), targets)
    student_log_probs = functional.log_softmax(selected.float() / temperature, dim=-1)
    teacher_log_probs = teacher_log_probs.to(device=selected.device, dtype=torch.float32)
    soft_kl = functional.kl_div(
        student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
    )
    total = hard_weight * hard_ce + soft_weight * (temperature ** 2) * soft_kl
    return total, hard_ce, soft_kl, int(selected.shape[0])


def _torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _record_path(folder, row_id):
    return folder / "records" / f"{digest(row_id)}.pt"


def _load_cache_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError("unsupported logits cache format")
    for record in manifest.get("records", []):
        path = folder / record["path"]
        if sha(path) != record["sha256"]:
            raise ValueError(f"logits cache record changed: {record['id']}")
    return manifest


def cache_teacher(args):
    import torch
    from transformers import AutoTokenizer

    objective = json.loads(args.objective.read_text())
    temperature, _, _ = validate_objective(objective)
    student_config = json.loads(args.student_config.read_text())
    teacher_config = json.loads(args.teacher_config.read_text())
    if args.teacher_device:
        teacher_config["device"] = args.teacher_device
    rows, artifact_manifest = verified_rows(args.artifact, args.data_dir)
    target_source = validate_target_method(artifact_manifest["method"])
    full_row_count = len(rows)
    if args.limit is not None:
        if args.limit < 1 or args.limit >= full_row_count:
            raise ValueError("--limit must be positive and smaller than the full artifact")
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "records").mkdir()
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_config["model_id"], revision=student_config["revision"], local_files_only=True)
    teacher_tokenizer, teacher = load(teacher_config)
    student_fp = tokenizer_fingerprint(student_tokenizer)
    teacher_fp = tokenizer_fingerprint(teacher_tokenizer)
    if student_fp != teacher_fp:
        raise ValueError("teacher and student tokenizer mappings differ; logits cannot be aligned")
    teacher.eval()
    records = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        ids, labels = target_tokens(row, student_tokenizer, student_config, objective["max_tokens"])
        x = torch.tensor([ids], device=teacher_config["device"])
        y = torch.tensor([labels], device=teacher_config["device"])
        mask = supervised_prediction_mask(y)
        with torch.inference_mode():
            logits = teacher(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
            selected = logits[:, :-1, :][mask].float()
            log_probs = torch.log_softmax(selected / temperature, dim=-1).to(torch.float16).cpu()
        if not torch.isfinite(log_probs).all():
            raise ValueError(f"non-finite teacher distribution for {row['id']}")
        record = {
            "input_ids": torch.tensor(ids, dtype=torch.int32),
            "labels": torch.tensor(labels, dtype=torch.int32),
            "teacher_log_probs": log_probs,
        }
        path = _record_path(args.output, row["id"])
        torch.save(record, path)
        records.append({
            "id": row["id"],
            "path": str(path.relative_to(args.output)),
            "sha256": sha(path),
            "sequence_tokens": len(ids),
            "supervised_tokens": int(mask.sum().item()),
            "vocabulary": int(log_probs.shape[1]),
        })
        print(f"teacher logits {index + 1}/{len(rows)}", flush=True)
    manifest = {
        "format": CACHE_FORMAT,
        "status": "smoke_complete" if args.limit is not None else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "TRAIN targets only; full-vocabulary teacher distributions at answer-token positions",
        "artifact_method": artifact_manifest["method"],
        "hard_target_source": target_source,
        "temperature": temperature,
        "cache_dtype": "float16 log probabilities; training converts to float32",
        "causal_alignment": "logits[:, :-1] against labels[:, 1:]; prompt labels are -100",
        "student_prompt_config": sequence_config(student_config),
        "student_model_config": student_config,
        "teacher_model_config": teacher_config,
        "student_tokenizer_sha256": student_fp,
        "teacher_tokenizer_sha256": teacher_fp,
        "artifact_manifest_sha256": sha(args.artifact / "manifest.json"),
        "artifact_path": str(args.artifact.resolve()),
        "train_source_sha256": sha(args.data_dir / "train.jsonl"),
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": hardware_chip(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers"]},
        "source_sha256": source_hashes(),
        "records": records,
        "artifact_row_count": full_row_count,
        "cached_row_count": len(records),
        "smoke_limit": args.limit,
    }
    write_json(args.output / "manifest.json", manifest)


def train_student(args):
    import torch
    from peft import LoraConfig, get_peft_model

    objective = json.loads(args.objective.read_text())
    temperature, _, _ = validate_objective(objective)
    train_config = json.loads(args.train_config.read_text())
    student_config = json.loads(args.student_config.read_text())
    if args.student_device:
        student_config["device"] = args.student_device
    cache = _load_cache_manifest(args.cache)
    expected_status = "smoke_complete" if args.smoke else "complete"
    if cache.get("status") != expected_status:
        raise ValueError(f"cache status {cache.get('status')!r} is invalid for this training mode")
    if cache["temperature"] != temperature:
        raise ValueError("cache and training temperatures differ")
    if cache["student_prompt_config"] != sequence_config(student_config):
        raise ValueError("student model revision or prompt differs from logits cache")
    if cache["artifact_manifest_sha256"] != sha(Path(cache["artifact_path"]) / "manifest.json"):
        raise ValueError("source response artifact changed after caching")
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])
    torch.set_num_threads(8)
    tokenizer, model = load(student_config)
    if tokenizer_fingerprint(tokenizer) != cache["student_tokenizer_sha256"]:
        raise ValueError("student tokenizer differs from logits cache")
    model = get_peft_model(model, LoraConfig(
        r=train_config["lora_r"], lora_alpha=train_config["lora_alpha"],
        lora_dropout=train_config["lora_dropout"], target_modules=train_config["target_modules"],
        task_type="CAUSAL_LM"))
    model.train(); model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=train_config["learning_rate"])
    records = list(cache["records"])
    random.shuffle(records)
    log = []
    started = time.perf_counter()
    for step in range(train_config["steps"]):
        meta = records[step % len(records)]
        saved = _torch_load(args.cache / meta["path"])
        ids = saved["input_ids"].to(device=student_config["device"], dtype=torch.long).unsqueeze(0)
        labels = saved["labels"].to(device=student_config["device"], dtype=torch.long).unsqueeze(0)
        teacher_log_probs = saved["teacher_log_probs"]
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        total, hard_ce, soft_kl, supervised = distillation_losses(
            student_logits, labels, teacher_log_probs, objective)
        if not torch.isfinite(total):
            raise ValueError("non-finite distillation loss")
        total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad):
            raise ValueError("non-finite gradient norm")
        optimizer.step(); sync(student_config["device"])
        log.append({
            "step": step + 1, "id": meta["id"], "loss": float(total.detach().cpu()),
            "hard_ce": float(hard_ce.detach().cpu()), "soft_kl": float(soft_kl.detach().cpu()),
            "grad_norm": grad, "input_tokens": int(ids.shape[1]), "supervised_tokens": supervised,
        })
        write_json(args.output / "steps.json", log)
        print(log[-1], flush=True)
    model.save_pretrained(args.output)
    write_json(args.output / "training.json", {
        "status": "smoke_complete" if args.smoke else "complete",
        "method": "full_vocabulary_logits_distillation",
        "seed": train_config["seed"], "steps": len(log), "objective": objective,
        "train_config": train_config, "student_config": student_config,
        "cache_manifest_sha256": sha(args.cache / "manifest.json"),
        "elapsed_seconds": time.perf_counter() - started,
        "total_input_tokens": sum(row["input_tokens"] for row in log),
        "total_supervised_tokens": sum(row["supervised_tokens"] for row in log),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft"]},
        "source_sha256": source_hashes(),
        "limitations": "Teacher log probabilities are cached as float16. Quality requires separate frozen dev/external evaluation; training loss is not adoption evidence.",
    })


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--artifact", type=Path, required=True)
    cache.add_argument("--data-dir", type=Path, default=Path("data/complexity-v1"))
    cache.add_argument("--student-config", type=Path, default=Path("configs/baseline.json"))
    cache.add_argument("--teacher-config", type=Path, required=True)
    cache.add_argument("--teacher-device", choices=["cpu", "mps"],
                       help="Explicit runtime override; never falls back silently")
    cache.add_argument("--objective", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--limit", type=int,
                       help="Smoke only: cache the first N rows and mark the manifest smoke_complete")
    train = sub.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--student-config", type=Path, default=Path("configs/baseline.json"))
    train.add_argument("--student-device", choices=["cpu", "mps"],
                       help="Explicit runtime override; never falls back silently")
    train.add_argument("--train-config", type=Path, required=True)
    train.add_argument("--objective", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--smoke", action="store_true",
                       help="Require a smoke_complete cache; output is never a full experiment")
    args = parser.parse_args()
    cache_teacher(args) if args.command == "cache" else train_student(args)


if __name__ == "__main__":
    main()
