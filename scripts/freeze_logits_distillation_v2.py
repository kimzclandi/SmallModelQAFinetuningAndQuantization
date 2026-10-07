"""Freeze the completed 242-row logits-distillation development study."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from qa_lab.common import read_jsonl, sha, write_json


SEEDS = (20260918, 20260919, 20260920)


def paired(candidate_path, control_path):
    candidate = {row["id"]: row for row in read_jsonl(candidate_path)}
    control = {row["id"]: row for row in read_jsonl(control_path)}
    if candidate.keys() != control.keys() or len(candidate) != 74:
        raise ValueError("candidate/control dev IDs differ")
    fixes = sorted(row_id for row_id in candidate if candidate[row_id]["em"] > control[row_id]["em"])
    regressions = sorted(row_id for row_id in candidate if candidate[row_id]["em"] < control[row_id]["em"])
    unchanged = len(candidate) - len(fixes) - len(regressions)
    return {"fixes": fixes, "regressions": regressions, "unchanged": unchanged}


def percent(value):
    return f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    cache_dir = args.work / "logits-distillation-v2-cache-cpu"
    cache = json.loads((cache_dir / "manifest.json").read_text())
    if cache["status"] != "complete" or cache["cached_row_count"] != 242:
        raise ValueError("expected one complete 242-row teacher cache")
    if cache["artifact_method"] != "gold_sft" or cache["hard_target_source"] != "gold_reference":
        raise ValueError("unexpected v2 target provenance")
    cache_bytes = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
    shutil.copy2(cache_dir / "manifest.json", args.output / "cache-manifest.json")

    rows = {}
    paired_rows = {}
    for seed in SEEDS:
        train_dir = args.work / f"logits-distillation-v2-{seed}-cpu"
        eval_dir = args.work / f"logits-distillation-v2-{seed}-cpu-dev"
        training = json.loads((train_dir / "training.json").read_text())
        metrics = json.loads((eval_dir / "dev.metrics.json").read_text())
        run = json.loads((eval_dir / "run.json").read_text())
        if training["status"] != "complete" or training["steps"] != 242 or training["seed"] != seed:
            raise ValueError(f"incomplete training for seed {seed}")
        if run["status"] != "complete" or metrics["overall"]["n"] != 74:
            raise ValueError(f"incomplete dev evaluation for seed {seed}")
        frozen = args.output / str(seed)
        frozen.mkdir()
        for source, name in [
            (train_dir / "training.json", "training.json"),
            (train_dir / "steps.json", "steps.json"),
            (eval_dir / "run.json", "run.json"),
            (eval_dir / "manifest.json", "manifest.json"),
            (eval_dir / "dev.metrics.json", "dev.metrics.json"),
            (eval_dir / "dev.predictions.jsonl", "dev.predictions.jsonl"),
            (eval_dir / "dev.scored.jsonl", "dev.scored.jsonl"),
            (eval_dir / "dev.failures.jsonl", "dev.failures.jsonl"),
        ]:
            shutil.copy2(source, frozen / name)
        rows[str(seed)] = {"metrics": metrics, "training": {
            "steps": training["steps"],
            "total_supervised_tokens": training["total_supervised_tokens"],
            "elapsed_seconds": training["elapsed_seconds"],
        }}
        paired_rows[str(seed)] = paired(
            eval_dir / "dev.scored.jsonl",
            Path(f"reports/coverage-v3/gold242-{seed}/dev/dev.scored.jsonl"),
        )

    def values(section, metric):
        return [rows[str(seed)]["metrics"][section][metric] for seed in SEEDS]

    aggregate = {}
    for section in ("overall", "answerable", "unanswerable"):
        aggregate[section] = {}
        for metric in ("em", "f1"):
            series = values(section, metric)
            aggregate[section][metric] = {
                "mean": statistics.mean(series),
                "sample_std": statistics.stdev(series),
                "minimum": min(series),
                "maximum": max(series),
            }
    control = json.loads(Path("reports/coverage-v3/summary.json").read_text())["aggregate"]["dev"]["gold242"]
    v1 = json.loads(Path("reports/logits-distillation-v1/summary.json").read_text())["aggregate"]
    summary = {
        "status": "complete_development_result",
        "training_rows": 242,
        "seeds": list(SEEDS),
        "steps_per_seed": 242,
        "cache": {
            "bytes": cache_bytes,
            "elapsed_seconds": cache["elapsed_seconds"],
            "records": cache["cached_row_count"],
            "dtype": cache["cache_dtype"],
        },
        "runs": rows,
        "aggregate": aggregate,
        "matched_gold242_control": control,
        "v1_logits_context": v1,
        "paired_vs_gold242": paired_rows,
        "decision": "More data improved the logits arm relative to v1, but the matched hard-CE-only gold242 control remained better on mean dev EM. No external confirmation was run.",
        "limitations": [
            "The 74-row dev has been reused and is not an independent confirmation set.",
            "Three seeds measure training variation, not three independent datasets.",
            "No post-result temperature, loss-weight, step, or seed selection was performed.",
        ],
    }
    write_json(args.output / "summary.json", summary)

    lines = [
        "# Logits distillation v2：242条训练数据的完整结果",
        "",
        "状态：242条冻结TRAIN数据、3个固定seed、每组242步训练和74题dev推理已完成。没有运行外部集，也没有结果后调参。",
        "",
        "## 结果",
        "",
        "| seed | 整体EM | token F1 | 有答案EM | 无答案EM | 相对同seed gold242修复/回归 | 训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = rows[str(seed)]
        metrics = row["metrics"]
        pair = paired_rows[str(seed)]
        lines.append(
            f"| {seed} | {percent(metrics['overall']['em'])} | {percent(metrics['overall']['f1'])} | "
            f"{percent(metrics['answerable']['em'])} | {percent(metrics['unanswerable']['em'])} | "
            f"{len(pair['fixes'])}/{len(pair['regressions'])} | {row['training']['elapsed_seconds']:.2f} |"
        )
    overall = aggregate["overall"]["em"]
    answerable = aggregate["answerable"]["em"]
    lines += [
        "",
        f"三seed整体EM为{percent(overall['mean'])}±{percent(overall['sample_std'])}，有答案EM均值为{percent(answerable['mean'])}。这里的±是样本标准差，不是置信区间。",
        "",
        f"24条v1 logits方案的整体EM为{percent(v1['overall_em']['mean'])}±{percent(v1['overall_em']['sample_std'])}；扩大到242条后均值提高，但匹配的gold242 hard-CE-only控制组仍为{percent(control['overall']['mean'])}±{percent(control['overall']['sample_std'])}。因此当前证据支持“覆盖扩大有效”，不支持“加入soft targets优于同数据gold SFT”。",
        "",
        "## 成本与边界",
        "",
        f"- 完整词表float16教师缓存包含242条记录，占{cache_bytes / 1024 / 1024:.2f} MiB，CPU生成耗时{cache['elapsed_seconds']:.2f}秒。",
        "- 每个seed恰好训练242步、监督1289 tokens；与历史gold242控制组在训练ID、seed、步数、LoRA配置和学习率上匹配。",
        "- dev已经被历史实验使用，本结果属于开发证据，不是独立确认；没有运行外部集。",
        "- 未依据结果改变T=2、CE/KL各0.5、训练步数或选择seed。",
        "- v2在gold答案位置进行teacher forcing，回答的是教师soft distribution是否能改善gold训练；它不同于v1对教师自由生成响应的蒸馏。",
        "",
        "逐题修复、回归、预测、失败样例和训练日志保存在同目录；`summary.json`为机器可读汇总。",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(args.output / "release-manifest.json", {
        str(path.relative_to(args.output)): sha(path)
        for path in sorted(args.output.rglob("*"))
        if path.is_file() and path.name != "release-manifest.json"
    })


if __name__ == "__main__":
    main()
