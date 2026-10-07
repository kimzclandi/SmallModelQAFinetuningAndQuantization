"""Freeze completed logits-distillation-v1 work outputs without new inference."""
import json
from pathlib import Path
import shutil
import statistics

from qa_lab.common import read_jsonl, sha, write_json


SEEDS = [20260918, 20260919, 20260920]
WORK = Path("work")
OUTPUT = Path("reports/logits-distillation-v1")
PROTOCOL = Path("configs/logits-distillation-v1/protocol.json")


def aggregate(values):
    return {
        "n": len(values), "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values), "minimum": min(values), "maximum": max(values),
    }


def paired(before_path, after_path):
    before = {row["id"]: row["em"] for row in read_jsonl(before_path)}
    after = {row["id"]: row["em"] for row in read_jsonl(after_path)}
    if set(before) != set(after):
        raise ValueError("paired prediction IDs differ")
    return {
        "fixes": sorted(key for key in before if before[key] == 0 and after[key] == 1),
        "regressions": sorted(key for key in before if before[key] == 1 and after[key] == 0),
        "delta_em": sum(after[key] - before[key] for key in before) / len(before),
    }


def require_complete(path, kind):
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"{kind} is not complete: {path}")
    return payload


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"preserve frozen output; refusing overwrite: {OUTPUT}")
    cache = require_complete(WORK / "logits-distillation-v1-cache-cpu/manifest.json", "cache")
    if cache["cached_row_count"] != cache["artifact_row_count"] or cache["cached_row_count"] != 24:
        raise ValueError("full 24-row cache required")
    runs = {}
    metrics = {}
    comparisons = {}
    for seed in SEEDS:
        train = WORK / f"logits-distillation-v1-{seed}-cpu"
        dev = WORK / f"logits-distillation-v1-{seed}-cpu-dev"
        training = require_complete(train / "training.json", "training")
        if training["seed"] != seed or training["steps"] != 48:
            raise ValueError("fixed seed/step contract violated")
        metric = json.loads((dev / "dev.metrics.json").read_text())
        if metric["overall"]["n"] != 74:
            raise ValueError("full 74-row dev evaluation required")
        key = str(seed)
        runs[key] = {
            "training": training,
            "metrics": metric,
            "adapter_files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                              for p in train.iterdir() if p.suffix in [".json", ".safetensors"]},
        }
        metrics[key] = metric
        logits_scored = dev / "dev.scored.jsonl"
        comparisons[key] = {
            "gold_to_logits": paired(
                Path(f"reports/teacher-study-v2/gold-{seed}/dev/dev.scored.jsonl"), logits_scored),
            "response_to_logits": paired(
                Path(f"reports/teacher-study-v2/prompted_teacher-{seed}/dev/dev.scored.jsonl"), logits_scored),
        }
    fields = [(split, metric) for split in ["overall", "answerable", "unanswerable"]
              for metric in ["em", "f1"]]
    summary = {
        "experiment": "logits-distillation-v1",
        "status": "complete_negative_result",
        "protocol_sha256": sha(PROTOCOL),
        "scope": "CPU execution, 24 teacher-response training rows, three fixed seeds, 48 steps, 74-row reused dev",
        "cache": {
            "manifest_sha256": sha(WORK / "logits-distillation-v1-cache-cpu/manifest.json"),
            "elapsed_seconds": cache["elapsed_seconds"], "cached_rows": cache["cached_row_count"],
            "temperature": cache["temperature"], "dtype": cache["cache_dtype"],
        },
        "runs": runs,
        "aggregate": {f"{split}_{metric}": aggregate(
            [metrics[str(seed)][split][metric] for seed in SEEDS]) for split, metric in fields},
        "paired": comparisons,
        "historical_context": {
            "gold_sft_overall_em_mean": 0.5540540540540541,
            "prompted_response_distillation_overall_em_mean": 0.5585585585585585,
            "source": "reports/teacher-study-v2/summary.json",
            "warning": "Historical arms ran on MPS; quality comparison shares data/config but wall time is not comparable across devices.",
        },
        "decision": "do_not_advance_to_external_evaluation",
        "decision_reason": "All logits seeds have answerable EM 14/33, below the historical 19/33 continuation boundary. Dev was reused and is not independent confirmation.",
        "limitations": [
            "Teacher log probabilities were cached as float16 and converted to float32 for KL.",
            "Only T=2 and equal CE/KL weights were run; no post-result retuning was performed.",
            "Three seeds describe training variation, not three independent datasets or statistical significance.",
            "CPU execution validates the algorithm path but does not provide CUDA, Ascend, MPS or production performance evidence.",
        ],
    }
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "summary.json", summary)
    for seed in SEEDS:
        destination = OUTPUT / str(seed); destination.mkdir()
        train = WORK / f"logits-distillation-v1-{seed}-cpu"
        dev = WORK / f"logits-distillation-v1-{seed}-cpu-dev"
        for name in ["training.json", "steps.json"]:
            shutil.copyfile(train / name, destination / name)
        for name in ["run.json", "manifest.json", "dev.metrics.json", "dev.scored.jsonl",
                     "dev.failures.jsonl", "dev.predictions.jsonl"]:
            shutil.copyfile(dev / name, destination / name)
    lines = [
        "# Logits distillation v1：完整负结果", "",
        "状态：24条训练数据、3个固定seed、每组48步训练和74题dev推理已完成。没有运行外部集，未进行结果后调参。", "",
        "## 结果", "",
        "| seed | 整体EM | token F1 | 有答案EM | 无答案EM | 训练秒数 | 监督tokens |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        run = summary["runs"][str(seed)]; metric = run["metrics"]; training = run["training"]
        lines.append(f"| {seed} | {metric['overall']['em']:.2%} | {metric['overall']['f1']:.2%} | "
                     f"{metric['answerable']['em']:.2%} | {metric['unanswerable']['em']:.2%} | "
                     f"{training['elapsed_seconds']:.2f} | {training['total_supervised_tokens']} |")
    aggregate_em = summary["aggregate"]["overall_em"]
    lines += ["", f"三seed整体EM均值为{aggregate_em['mean']:.2%}，样本标准差为{aggregate_em['sample_std']:.2%}。"
              "历史同24题的gold-SFT和新提示教师response-distillation均值分别为55.41%和55.86%；"
              "本轮logits方案没有超过两者。", "",
              "三个logits seed的有答案EM均为14/33=42.42%，低于历史继续条件19/33=57.58%。"
              "因此停止在dev，不运行外部集；这不是部署否决，只是当前固定配置没有获得继续证据。", "",
              "## 边界", "",
              "- 教师完整词表log-probability以float16缓存，KL计算时转为float32。",
              "- 只运行预先固定的T=2、CE/KL各0.5；没有在看到结果后调温度或权重。",
              "- dev已经被历史实验使用，不是独立确认集；三seed不代表三个独立数据集。",
              "- 本次CPU耗时只用于记录，不与历史MPS耗时比较，也不推出CUDA/Ascend性能。",
              "- 逐题修复、退化、训练日志、预测和失败样例保存在同目录，summary.json为机器可读汇总。", ""]
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines))
    manifest = {str(path.relative_to(OUTPUT)): sha(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    write_json(OUTPUT / "release-manifest.json", manifest)
    print(json.dumps({"status": summary["status"], "decision": summary["decision"],
                      "overall_em": summary["aggregate"]["overall_em"]}, indent=2))


if __name__ == "__main__":
    main()
