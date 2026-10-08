"""Regression checks for cached TRAIN identity, relocation and corrupted masks."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from qa_lab.common import digest
from qa_lab.logits_distillation import (
    CACHE_FORMAT, _load_cache_manifest, validate_cache_lineage, validate_cached_record,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def lineage(tmp_path):
    artifact = tmp_path / "relocated-artifact"
    source = tmp_path / "relocated-source"
    shutil.copytree(ROOT / "data/coverage-v3-gold242", artifact)
    shutil.copytree(ROOT / "data/complexity-v1", source)
    cache = json.loads((ROOT / "reports/logits-distillation-v2/cache-manifest.json").read_text())
    cache["artifact_path"] = "/unavailable/historical/location"
    return cache, artifact, source


def test_frozen_cache_lineage_accepts_relocated_exact_bytes(lineage):
    cache, artifact, source = lineage
    rows = validate_cache_lineage(cache, artifact, source)
    assert len(rows) == 242
    assert list(rows) == [item["id"] for item in cache["records"]]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "eval_id", "order", "count", "tokenizer", "target", "source"])
def test_cache_identity_mutations_fail(lineage, mutation):
    cache, artifact, source = lineage
    if mutation == "duplicate":
        cache["records"][1] = copy.deepcopy(cache["records"][0])
    elif mutation == "missing":
        cache["records"].pop()
        cache["cached_row_count"] -= 1
    elif mutation == "eval_id":
        cache["records"][0]["id"] = json.loads((source / "dev.jsonl").read_text().splitlines()[0])["id"]
    elif mutation == "order":
        cache["records"].reverse()
    elif mutation == "count":
        cache["artifact_row_count"] += 1
    elif mutation == "tokenizer":
        cache["teacher_tokenizer_sha256"] = "different-tokenizer"
    elif mutation == "target":
        cache["hard_target_source"] = "teacher_response"
    else:
        cache["train_source_sha256"] = "different-source"
    with pytest.raises(ValueError):
        validate_cache_lineage(cache, artifact, source)


@pytest.mark.parametrize("tree", ["artifact", "source"])
def test_content_change_rejected_even_when_manifest_unchanged(lineage, tree):
    cache, artifact, source = lineage
    folder = artifact if tree == "artifact" else source
    path = folder / "train.jsonl"
    path.write_text(path.read_text().replace('"context":', '"modified_context":', 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cache_lineage(cache, artifact, source)


def test_smoke_requires_exact_prefix_and_cannot_pass_as_complete(lineage):
    cache, artifact, source = lineage
    cache.update(status="smoke_complete", smoke_limit=2, cached_row_count=2)
    cache["records"] = cache["records"][:2]
    assert len(validate_cache_lineage(cache, artifact, source)) == 2
    cache.update(status="complete", smoke_limit=None)
    with pytest.raises(ValueError, match="coverage"):
        validate_cache_lineage(cache, artifact, source)


@pytest.mark.parametrize("records", [[], [{"id": "a"}, {"id": "a"}],
                                    [{"id": "a", "path": "../outside.pt"}]])
def test_loader_rejects_empty_duplicate_or_noncanonical_before_file_io(tmp_path, records):
    (tmp_path / "manifest.json").write_text(json.dumps({"format": CACHE_FORMAT, "records": records}))
    with pytest.raises(ValueError):
        _load_cache_manifest(tmp_path)


def test_loader_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a cache record")
    folder = tmp_path / "cache"
    (folder / "records").mkdir(parents=True)
    relative = f"records/{digest('a')}.pt"
    (folder / relative).symlink_to(outside)
    (folder / "manifest.json").write_text(json.dumps({"format": CACHE_FORMAT,
        "records": [{"id": "a", "path": relative}]}))
    with pytest.raises(ValueError, match="escapes"):
        _load_cache_manifest(folder)


@pytest.mark.parametrize("mutation", [None, "input_ids", "labels", "supervised_tokens", "nonfinite",
                                      "unnormalized", "vocabulary"])
def test_tensor_preflight_binds_input_and_answer_mask(monkeypatch, mutation):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("qa_lab.logits_distillation.target_tokens",
                        lambda *args: ([0, 1, 2], [-100, -100, 2]))
    saved = {"input_ids": torch.tensor([0, 1, 2], dtype=torch.int32),
             "labels": torch.tensor([-100, -100, 2], dtype=torch.int32),
             "teacher_log_probs": torch.log_softmax(torch.zeros((1, 3)), dim=-1).half()}
    meta = {"id": "train-row", "sequence_tokens": 3, "supervised_tokens": 1, "vocabulary": 3}
    if mutation in ("input_ids", "labels"):
        saved[mutation][1] = 2
    elif mutation == "supervised_tokens":
        meta["supervised_tokens"] = 2
    elif mutation == "nonfinite":
        saved["teacher_log_probs"][0, 0] = float("nan")
    elif mutation == "unnormalized":
        saved["teacher_log_probs"].zero_()
    elif mutation == "vocabulary":
        meta["vocabulary"] = 4
        saved["teacher_log_probs"] = torch.log_softmax(torch.zeros((1, 4)), dim=-1).half()
    if mutation is None:
        validate_cached_record(saved, meta, {}, None, {}, 16, 3)
    else:
        with pytest.raises(ValueError):
            validate_cached_record(saved, meta, {}, None, {}, 16, 3)
