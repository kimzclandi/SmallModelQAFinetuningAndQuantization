"""Paired article-cluster arithmetic for a preregistered QA confirmation study.

This module never runs a model. It cannot establish that labels were concealed
before predictions were locked: the execution/identity runner owns that check.
The interval is conditional on the fixed training seeds and observed article
population; it is not a confidence interval over future training procedures.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from .common import read_jsonl, write_json
from .metrics import evaluate


ARMS = ("gold", "full_kd", "gated_kd")
PRIMARY = "gated_kd"
METRICS = ("em", "f1", "format_valid_rate")
GROUPS = ("overall", "answerable", "unanswerable")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def _config(protocol):
    cfg = protocol["scoring"]
    seeds = cfg["seeds"]
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("Exactly three preregistered training seeds are required")
    for seed in seeds:
        _integer(seed, "training seed", 0)
    if len(set(seeds)) != 3:
        raise ValueError("Training seeds must be distinct")
    if not isinstance(cfg["arms"], list) or len(cfg["arms"]) != 3 or set(cfg["arms"]) != set(ARMS):
        raise ValueError("Arms must be exactly gold, full_kd, gated_kd")
    contexts = _integer(cfg["expected_contexts"], "expected_contexts")
    questions = _integer(cfg["expected_questions"], "expected_questions")
    if questions != 2 * contexts:
        raise ValueError("Exactly one question per answerability class per context is required")
    _integer(cfg["bootstrap_repetitions"], "bootstrap_repetitions", 2)
    _integer(cfg["bootstrap_seed"], "bootstrap_seed", 0)
    positives = _integer(cfg["min_positive_seeds"], "min_positive_seeds")
    if positives > len(seeds):
        raise ValueError("min_positive_seeds exceeds the number of training seeds")
    for key in ("min_em_gain", "max_subgroup_drop", "min_format_valid_rate"):
        _rate(cfg[key], key)
    return cfg


def _validate_labels(labels, cfg):
    if not isinstance(labels, list) or len(labels) != cfg["expected_questions"]:
        raise ValueError("Label question count does not match the frozen protocol")
    seen = set()
    contexts = defaultdict(list)
    for row in labels:
        for key in ("id", "article_id", "context_id", "family_id", "context"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Label {key} must be a nonempty string")
        if row["id"] in seen:
            raise ValueError("Duplicate label ID")
        seen.add(row["id"])
        if type(row.get("is_impossible")) is not bool:
            raise ValueError("is_impossible must be a boolean")
        answers = row.get("answers")
        if not isinstance(answers, list) or any(
            not isinstance(a, str) or not a.strip() or a not in row["context"] for a in answers
        ):
            raise ValueError("Gold answers must be nonempty extractive strings")
        if bool(answers) == row["is_impossible"]:
            raise ValueError("Answerability and gold answers disagree")
        contexts[row["context_id"]].append(row)
    if len(contexts) != cfg["expected_contexts"]:
        raise ValueError("Label context count does not match the frozen protocol")
    for rows in contexts.values():
        if len(rows) != 2 or {r["is_impossible"] for r in rows} != {False, True}:
            raise ValueError("Each context needs exactly one answerable and one unanswerable question")
        if len({(r["article_id"], r["context"]) for r in rows}) != 1:
            raise ValueError("Context identity or article membership disagrees")
    if len({r["article_id"] for r in labels}) < 2:
        raise ValueError("Article-cluster bootstrap requires at least two articles")


def _seed_predictions(mapping, seeds):
    if not isinstance(mapping, dict):
        raise ValueError("Predictions for each arm must map seeds to rows")
    normalized = {}
    for key, rows in mapping.items():
        if type(key) is int:
            seed = key
        elif isinstance(key, str) and key in {str(x) for x in seeds}:
            seed = int(key)
        else:
            raise ValueError("Unexpected prediction seed")
        if seed in normalized:
            raise ValueError("Duplicate seed after key normalization")
        if not isinstance(rows, list):
            raise ValueError("Prediction rows must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("prediction"), str):
                raise ValueError("Predictions require string id and prediction fields")
        normalized[seed] = rows
    if set(normalized) != set(seeds):
        raise ValueError("Missing or extra prediction seeds")
    return normalized


def _mean_sd(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _percentile(sorted_values, q):
    """Linear-interpolated empirical quantile (the usual type-7 percentile)."""
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def score_study(labels, predictions_by_arm_seed, protocol):
    """Score complete matched predictions without fitting or selecting methods.

    Labels use qa_lab.metrics fields plus article_id/context_id. Every context
    contains one answerable and one unanswerable question. Prediction input is
    {arm: {seed: [{id, prediction}, ...]}}. All arms/seeds require exact ID coverage.
    The primary candidate is always gated_kd; full_kd remains descriptive.
    """
    cfg = _config(protocol)
    _validate_labels(labels, cfg)
    if not isinstance(predictions_by_arm_seed, dict) or set(predictions_by_arm_seed) != set(ARMS):
        raise ValueError("Predictions must contain all and only the three frozen arms")
    # Stable ID/article ordering makes the result independent of input row order.
    labels = sorted(labels, key=lambda row: row["id"])
    seeds = cfg["seeds"]
    arms, scored = {}, {}
    for arm in ARMS:
        by_seed = _seed_predictions(predictions_by_arm_seed[arm], seeds)
        per_seed, scored[arm] = {}, {}
        for seed in seeds:
            result, items = evaluate(labels, by_seed[seed])
            predictions = {row["id"]: row["prediction"] for row in by_seed[seed]}
            per_seed[str(seed)] = result
            scored[arm][str(seed)] = [dict(
                item, article_id=row["article_id"], context_id=row["context_id"],
                prediction=predictions[row["id"]],
            ) for row, item in zip(labels, items)]
        arms[arm] = {
            "per_seed": per_seed,
            "across_seeds": {
                group: {
                    "n_per_seed": per_seed[str(seeds[0])][group]["n"],
                    **{metric: _mean_sd([per_seed[str(seed)][group][metric] for seed in seeds])
                       for metric in METRICS},
                } for group in GROUPS
            },
        }

    clusters = {}
    for article in sorted({row["article_id"] for row in labels}):
        indices = [i for i, row in enumerate(labels) if row["article_id"] == article]
        clusters[article] = {
            "n_questions": len(indices),
            "n_contexts": len({labels[i]["context_id"] for i in indices}),
            # Sum binary EM differences before division, avoiding roundoff from
            # repeatedly summing 1/3. All seeds/questions remain paired.
            "seed_summed_em_difference": {
                arm: sum(int(scored[arm][str(seed)][i]["em"]) - int(scored["gold"][str(seed)][i]["em"])
                         for seed in seeds for i in indices)
                for arm in ("full_kd", PRIMARY)
            },
        }
    cluster_rows = list(clusters.values())
    rng = random.Random(cfg["bootstrap_seed"])
    bootstrap = {arm: [] for arm in ("full_kd", PRIMARY)}
    for _ in range(cfg["bootstrap_repetitions"]):
        sample = [cluster_rows[rng.randrange(len(cluster_rows))] for _ in cluster_rows]
        denominator = len(seeds) * sum(row["n_questions"] for row in sample)
        for arm, samples in bootstrap.items():
            samples.append(sum(row["seed_summed_em_difference"][arm] for row in sample) / denominator)

    comparisons = {}
    for arm in ("full_kd", PRIMARY):
        per_seed = {
            str(seed): sum(int(candidate["em"]) - int(gold["em"])
                           for candidate, gold in zip(scored[arm][str(seed)], scored["gold"][str(seed)])) / len(labels)
            for seed in seeds
        }
        # Form paired integer differences before dividing. Subtracting two
        # rounded rates (e.g. .98 - 1) could incorrectly fail an inclusive .02
        # noninferiority boundary.
        mean_difference = sum(row["seed_summed_em_difference"][arm] for row in cluster_rows) / (len(seeds) * len(labels))
        subgroup_differences = {}
        for group, impossible in (("answerable", False), ("unanswerable", True)):
            indices = [i for i, row in enumerate(labels) if row["is_impossible"] == impossible]
            subgroup_differences[group] = sum(
                int(scored[arm][str(seed)][i]["em"]) - int(scored["gold"][str(seed)][i]["em"])
                for seed in seeds for i in indices
            ) / (len(seeds) * len(indices))
        values = sorted(bootstrap[arm])
        comparison = {
            "baseline": "gold", "candidate": arm,
            "role": "primary" if arm == PRIMARY else "secondary_descriptive",
            "per_seed_em_difference": per_seed,
            "em_difference": {"mean": mean_difference, "sample_sd": statistics.stdev(per_seed.values())},
            "positive_seed_count": sum(x > 0 for x in per_seed.values()),
            "article_cluster_bootstrap_95_percentile_ci": [_percentile(values, .025), _percentile(values, .975)],
            "subgroup_mean_em_difference": subgroup_differences,
            "format_valid_rate_by_seed": {str(seed): arms[arm]["per_seed"][str(seed)]["overall"]["format_valid_rate"] for seed in seeds},
            "bootstrap_em_differences": bootstrap[arm],
        }
        comparisons[arm] = comparison
    primary = dict(comparisons[PRIMARY])
    criteria = {
        "mean_em_gain": primary["em_difference"]["mean"] >= cfg["min_em_gain"],
        "confidence_lower_bound_positive": primary["article_cluster_bootstrap_95_percentile_ci"][0] > 0,
        "positive_seeds": primary["positive_seed_count"] >= cfg["min_positive_seeds"],
        "answerable_noninferiority": primary["subgroup_mean_em_difference"]["answerable"] >= -cfg["max_subgroup_drop"],
        "unanswerable_noninferiority": primary["subgroup_mean_em_difference"]["unanswerable"] >= -cfg["max_subgroup_drop"],
        "format_valid_every_seed": all(rate >= cfg["min_format_valid_rate"] for rate in primary["format_valid_rate_by_seed"].values()),
    }
    primary.update(criteria=criteria, accepted=all(criteria.values()))
    return {
        "schema_version": 1, "scoring_protocol": dict(cfg), "primary": primary,
        "arms": arms, "comparisons": comparisons, "article_clusters": clusters,
        "per_item_scores": scored,
        "bootstrap_method": {
            "unit": "article", "paired": True, "n_articles": len(clusters),
            "seed": cfg["bootstrap_seed"], "repetitions": cfg["bootstrap_repetitions"],
            "confidence_level": .95, "interval": "percentile", "quantile": "linear_type_7",
            "estimand": "question-weighted EM difference, averaged over the fixed three training seeds",
            "resampling": "sample n_articles articles with replacement; retain every sampled article's questions and all paired seed scores",
        },
        "limitations": [
            "Arithmetic scoring does not establish prediction-lock timing or label concealment; verify the execution receipts separately.",
            "The interval conditions on three fixed training seeds and does not estimate full training-seed uncertainty.",
            "Contexts are newly held out within previously used articles; this is not unseen-topic confirmation.",
            "Public SQuAD may occur in pretraining; unseen local context does not establish no pretraining contamination.",
            "Teacher gating changes both target selection and total KL weight; this study does not separate their causal effects.",
            "full_kd is secondary descriptive reporting and cannot replace the preregistered primary candidate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Arithmetic scorer only: does not prove predictions were locked before label access.")
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, help="JSON mapping arm -> seed -> prediction rows")
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory; never overwrites frozen results")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    predictions = json.loads(args.predictions.read_text())
    labels = read_jsonl(args.labels)
    result = score_study(labels, predictions, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "metrics.json", result)


if __name__ == "__main__":
    main()
