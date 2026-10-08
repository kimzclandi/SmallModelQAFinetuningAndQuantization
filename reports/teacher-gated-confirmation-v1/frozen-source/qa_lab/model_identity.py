"""Offline byte identity for the two pinned logits-distillation checkpoints.

This is an integrity check, not a signature or protection against concurrent
modification. No weights are copied, downloaded, converted or loaded by the CLI.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDENTITIES = {
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"): {
        "role": "student", "manifest": "reports/model-artifacts.json",
        "sha256": "014e6a0442569325a977740ec247dd6438fc2b1f89e53c156d6c6ef023429d27",
    },
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"): {
        "role": "teacher", "manifest": "reports/closure-v1/teacher-model-artifacts.json",
        "sha256": "274df82394a3b16cdbd42b8bb5bb0e000dc49e3a0cf47b9c256cb47979b68645",
    },
}
REQUIRED_FILES = frozenset({"model.safetensors", "config.json", "generation_config.json",
                            "tokenizer.json", "tokenizer_config.json", "merges.txt",
                            "vocab.json", "LICENSE", "README.md"})
CHUNK_BYTES = 1024 * 1024


def _hash_file(path):
    """Bound memory independently of checkpoint size; count the actual bytes."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def expected_identity(config):
    key = (config["model_id"], config["revision"])
    if key not in MODEL_IDENTITIES:
        raise ValueError("No pinned file identity for this model_id/revision")
    identity = MODEL_IDENTITIES[key]
    path = ROOT / identity["manifest"]
    if _hash_file(path)["sha256"] != identity["sha256"]:
        raise ValueError("Pinned model manifest changed")
    manifest = json.loads(path.read_text())
    if "files" in manifest:
        if manifest.get("revision") != key[1]:
            raise ValueError("Model manifest revision differs from configured revision")
        files = manifest["files"]
    else:
        files = manifest
    if set(files) != REQUIRED_FILES:
        raise ValueError("Model manifest has missing or unexpected file identities")
    for name, value in files.items():
        if (set(value) != {"bytes", "sha256"} or type(value["bytes"]) is not int
                or value["bytes"] <= 0 or not isinstance(value["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None):
            raise ValueError(f"Invalid model file metadata: {name}")
    return {"schema": 1, "model_id": key[0], "revision": key[1],
            "manifest": identity["manifest"], "manifest_sha256": identity["sha256"],
            "files": files}


@dataclass(frozen=True)
class VerifiedSnapshot:
    # Local path is internal only. Publish receipt, never this dataclass.
    path: Path
    receipt: dict


def _resolve_local_snapshot(config, cache_dir):
    # The Hub cache lookup is read-only and never contacts the network. Resolve
    # the pinned config entry once, then all loaders use the resulting directory.
    from huggingface_hub import try_to_load_from_cache
    filename = try_to_load_from_cache(config["model_id"], "config.json",
                                      revision=config["revision"], cache_dir=cache_dir)
    if not isinstance(filename, str):
        raise ValueError("Pinned model snapshot is unavailable in the local cache")
    snapshot = Path(filename).parent.resolve()
    model_folder = "models--" + config["model_id"].replace("/", "--")
    if (snapshot.name != config["revision"] or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != model_folder):
        raise ValueError("Local cache did not resolve the configured pinned snapshot")
    return snapshot


def _checked_target(snapshot, name, config):
    path = snapshot / name
    target = path.resolve(strict=True)
    if not target.is_file():
        raise ValueError(f"Model entry is not a regular file: {name}")
    if target.is_relative_to(snapshot):
        return target
    # Normal Hugging Face snapshot entries point to their own model's blobs.
    model_folder = "models--" + config["model_id"].replace("/", "--")
    if (snapshot.name == config["revision"] and snapshot.parent.name == "snapshots"
            and snapshot.parent.parent.name == model_folder
            and target.is_relative_to((snapshot.parent.parent / "blobs").resolve())):
        return target
    raise ValueError(f"Model symlink escapes the snapshot/model blob store: {name}")


def verify_local_model(config, cache_dir=None):
    receipt = expected_identity(config)
    snapshot = _resolve_local_snapshot(config, cache_dir)
    # Both pinned snapshots are flat, nine-file checkpoints. Reject extra files
    # (including alternative weights, tokenizers and chat templates) rather than
    # allow Transformers to consume unverified content.
    if {path.name for path in snapshot.iterdir()} != REQUIRED_FILES:
        raise ValueError("Snapshot has missing or unexpected files")
    for name, metadata in receipt["files"].items():
        target = _checked_target(snapshot, name, config)
        before = target.stat()
        if before.st_size != metadata["bytes"] or _hash_file(target) != metadata:
            raise ValueError(f"Model file identity mismatch: {name}")
        after = target.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino) or _checked_target(snapshot, name, config) != target:
            raise ValueError(f"Model file changed during verification: {name}")
    return VerifiedSnapshot(snapshot, receipt)


def validate_cache_identities(cache, allow_legacy=False):
    if "model_identities" not in cache:
        if not allow_legacy:
            raise ValueError("Legacy cache has no model byte identity receipt; use --allow-legacy-unbound-cache explicitly")
        return "legacy_unbound"
    identities = cache["model_identities"]
    if not isinstance(identities, dict) or set(identities) != {"student", "teacher"}:
        raise ValueError("Invalid cache model identity receipts")
    for role in ("student", "teacher"):
        if identities[role] != expected_identity(cache[f"{role}_model_config"]):
            raise ValueError(f"Cached {role} byte identity differs from pinned manifest")
    return "bound_cache_receipt"


def load_verified_tokenizer(snapshot):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.path), local_files_only=True,
                                              trust_remote_code=False)
    model_id = snapshot.receipt["model_id"]
    tokenizer.name_or_path = model_id
    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["name_or_path"] = model_id
    return tokenizer


def load_verified_model(config, snapshot):
    """Logits-only loader: byte verification must precede calling this function."""
    if (config["model_id"], config["revision"]) != (
            snapshot.receipt["model_id"], snapshot.receipt["revision"]):
        raise ValueError("Verified snapshot and runtime model identity differ")
    import torch
    from transformers import AutoModelForCausalLM
    if config["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no silent device fallback")
    tokenizer = load_verified_tokenizer(snapshot)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot.path), local_files_only=True,
                trust_remote_code=False, use_safetensors=True,
                torch_dtype=getattr(torch, config["dtype"]),
                attn_implementation=config["attention_implementation"]).to(config["device"]).eval()
    # PEFT reads model.__dict__['name_or_path']; its serialized base-model field
    # must remain portable and must not expose the private snapshot directory.
    model.name_or_path = config["model_id"]
    model.config._name_or_path = config["model_id"]
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache-dir", type=Path, help="Existing Hugging Face hub cache directory")
    parser.add_argument("--role", choices=("student", "teacher", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an identity audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identities = {}
    failed_role = error_type = None
    try:
        for (model_id, revision), identity in MODEL_IDENTITIES.items():
            if args.role in ("both", identity["role"]):
                failed_role = identity["role"]
                verified = verify_local_model({"model_id": model_id, "revision": revision}, args.model_cache_dir)
                identities[identity["role"]] = verified.receipt
        failed_role = None
    except Exception as error:
        # Keep completed roles, but never serialize exception text (which may
        # contain local filesystem paths). A failed audit exits nonzero below.
        error_type = type(error).__name__
    result = {"schema": 1, "status": "failed" if error_type else "verified", "scope": "CPU file identity only",
              "model_loading_performed": False, "training_performed": False,
              "model_identities": identities,
              "verifier_sha256": _hash_file(Path(__file__))["sha256"],
              "limitations": "Current file integrity only; no signature, concurrent-modification protection, historical-run revalidation or quality/performance conclusion."}
    if error_type:
        result.update(failed_role=failed_role, error_type=error_type,
                      error="Model file identity verification did not complete")
    # Exclusive creation preserves previous evidence. Receipts contain only
    # logical IDs, repository-relative manifest names and file hashes/sizes.
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "roles": sorted(identities), "model_loading_performed": False}))
    if error_type:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
