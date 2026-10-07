"""Offline verification of frozen logits-distillation-v1 evidence."""
import json
from pathlib import Path
import statistics

from qa_lab.common import read_jsonl, sha
from qa_lab.metrics import evaluate


ROOT = Path("reports/logits-distillation-v1")
SEEDS = [20260918, 20260919, 20260920]


def main():
    release = json.loads((ROOT / "release-manifest.json").read_text())
    for name, expected in release.items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen evidence changed: {name}")
    summary = json.loads((ROOT / "summary.json").read_text())
    if summary["status"] != "complete_negative_result" or summary["decision"] != "do_not_advance_to_external_evaluation":
        raise ValueError("stored decision changed")
    rows = read_jsonl("data/complexity-v1/dev.jsonl")
    em = []
    for seed in SEEDS:
        folder = ROOT / str(seed)
        training = json.loads((folder / "training.json").read_text())
        steps = json.loads((folder / "steps.json").read_text())
        if training["status"] != "complete" or training["seed"] != seed or len(steps) != 48:
            raise ValueError(f"incomplete fixed training run: {seed}")
        metrics, scored = evaluate(rows, read_jsonl(folder / "dev.predictions.jsonl"))
        stored_metrics = json.loads((folder / "dev.metrics.json").read_text())
        stored_quality = {key: value for key, value in stored_metrics.items() if key != "performance"}
        if metrics != stored_quality:
            raise ValueError(f"metrics do not reproduce: {seed}")
        if scored != read_jsonl(folder / "dev.scored.jsonl"):
            raise ValueError(f"scored rows do not reproduce: {seed}")
        if stored_metrics != summary["runs"][str(seed)]["metrics"]:
            raise ValueError(f"summary metrics differ: {seed}")
        em.append(stored_metrics["overall"]["em"])
    aggregate = summary["aggregate"]["overall_em"]
    if aggregate != {"n": 3, "mean": statistics.mean(em), "sample_std": statistics.stdev(em),
                     "minimum": min(em), "maximum": max(em)}:
        raise ValueError("aggregate does not reproduce")
    if any(summary["runs"][str(seed)]["metrics"]["answerable"]["em"] >= 19 / 33 for seed in SEEDS):
        raise ValueError("stored stop decision no longer follows the historical continuation boundary")
    print("Verified: 24-row full-vocabulary cache lineage, 3x48-step CPU runs, 3x74 dev predictions, metrics, hashes and negative stop decision.")


if __name__ == "__main__":
    main()
