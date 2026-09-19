"""Export reference-free review inputs separately from reference-based diagnostics."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl, write_json, write_jsonl
from qa_lab.chinese import evaluate_zh
from resumable_teacher import checked_record, record_path, digest


def audit(data, out):
    rows = read_jsonl(data / "train.jsonl")
    cache = data / "teacher-train"
    binding = digest(json.loads((cache / "manifest.json").read_text()))
    predictions = [checked_record(record_path(cache, r["id"]), r, binding) for r in rows]
    metrics, scored = evaluate_zh(rows, predictions)
    queue = []
    for row, pred in zip(rows, predictions):
        answer = pred["prediction"].strip()
        valid = bool(answer) and (answer == "NO_ANSWER" or answer in row["context"])
        queue.append(dict(id=row["id"], context=row["context"], question=row["question"],
                          candidate=pred["prediction"],
                          status="review_required_judge_failed" if valid else "quarantine_format",
                          approved_for_training=False))
    out.mkdir(parents=True, exist_ok=False)
    write_jsonl(out / "review-inputs.jsonl", queue)
    write_jsonl(out / "reference-audit.jsonl", scored)
    write_json(out / "metrics.json", metrics)
    write_json(out / "policy.json", dict(training_allowed=False,
               reason="Registered judge validation failed; no semantic approval assigned.",
               reference_audit="Diagnostic only, not a deployable gold-free selector; all records retained."))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.out), indent=2))
