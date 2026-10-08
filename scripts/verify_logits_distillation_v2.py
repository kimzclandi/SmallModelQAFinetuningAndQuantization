"""Offline verification of frozen logits-distillation-v2 evidence."""
import json
from pathlib import Path
import statistics

from qa_lab.common import read_jsonl, sha
from qa_lab.metrics import evaluate
from qa_lab.logits_distillation import validate_cache_lineage


ROOT = Path("reports/logits-distillation-v2")
SEEDS = (20260918, 20260919, 20260920)


def main():
    release = json.loads((ROOT / "release-manifest.json").read_text())
    for name, expected in release.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen evidence changed: {name}")
    summary = json.loads((ROOT / "summary.json").read_text())
    cache = json.loads((ROOT / "cache-manifest.json").read_text())
    if summary["status"] != "complete_development_result":
        raise ValueError("stored status changed")
    if cache["cached_row_count"] != 242 or cache["artifact_method"] != "gold_sft":
        raise ValueError("cache provenance changed")
    validate_cache_lineage(cache, Path("data/coverage-v3-gold242"), Path("data/complexity-v1"))
    protocol = json.loads(Path("configs/logits-distillation-v2/protocol.json").read_text())
    for role in ("student", "teacher"):
        config = cache[f"{role}_model_config"]
        if f"{config['model_id']}@{config['revision']}" != protocol[role]:
            raise ValueError(f"cache {role} identity differs from frozen protocol")
    dev_rows = read_jsonl("data/complexity-v1/dev.jsonl")
    em = []
    for seed in SEEDS:
        folder = ROOT / str(seed)
        training = json.loads((folder / "training.json").read_text())
        steps = json.loads((folder / "steps.json").read_text())
        if training["status"] != "complete" or training["seed"] != seed or len(steps) != 242:
            raise ValueError(f"incomplete fixed training run: {seed}")
        if {step["id"] for step in steps} != {row["id"] for row in read_jsonl("data/coverage-v3-gold242/train.jsonl")}:
            raise ValueError(f"seed {seed} did not cover every TRAIN ID exactly once")
        metrics, scored = evaluate(dev_rows, read_jsonl(folder / "dev.predictions.jsonl"))
        stored = json.loads((folder / "dev.metrics.json").read_text())
        if metrics != {key: value for key, value in stored.items() if key != "performance"}:
            raise ValueError(f"metrics do not reproduce: {seed}")
        if scored != read_jsonl(folder / "dev.scored.jsonl"):
            raise ValueError(f"scored rows do not reproduce: {seed}")
        if stored != summary["runs"][str(seed)]["metrics"]:
            raise ValueError(f"summary metrics differ: {seed}")
        em.append(stored["overall"]["em"])
    aggregate = summary["aggregate"]["overall"]["em"]
    expected = {"mean": statistics.mean(em), "sample_std": statistics.stdev(em),
                "minimum": min(em), "maximum": max(em)}
    if aggregate != expected:
        raise ValueError("aggregate does not reproduce")
    if aggregate["mean"] >= summary["matched_gold242_control"]["overall"]["mean"]:
        raise ValueError("stored conclusion no longer matches the comparison")
    print("Verified: 242-row gold lineage, cache manifest, 3x242-step coverage, 3x74 dev predictions, metrics, hashes and matched-control conclusion.")


if __name__ == "__main__":
    main()
