"""Synthetic fixtures only: never load the confirmation labels or predictions."""
from copy import deepcopy
import json
import math
import random
import subprocess
import sys

import pytest

from qa_lab.confirmation_scoring import score_study


SEEDS = [20261009, 20261010, 20261011]
ARMS = ["gold", "full_kd", "gated_kd"]


def fixture(article_sizes=(1, 1), repetitions=200):
    labels = []
    for article, count in enumerate(article_sizes):
        for context in range(count):
            cid = f"a{article}-c{context}"
            for impossible in (False, True):
                labels.append(dict(id=f"{cid}-{int(impossible)}", article_id=f"a{article}",
                                   context_id=cid, family_id=cid, context=f"Alice met Bob at {cid}.",
                                   answers=[] if impossible else ["Alice"], is_impossible=impossible))
    preds = {arm: {str(seed): [dict(id=r["id"], prediction="NO_ANSWER" if r["is_impossible"] else "Alice")
                              for r in labels] for seed in SEEDS} for arm in ARMS}
    protocol = {"scoring": dict(seeds=SEEDS.copy(), arms=ARMS.copy(),
                expected_contexts=sum(article_sizes), expected_questions=len(labels),
                bootstrap_repetitions=repetitions, bootstrap_seed=20261009,
                min_em_gain=.03, min_positive_seeds=2, max_subgroup_drop=.02,
                min_format_valid_rate=.99)}
    return labels, preds, protocol


def test_identical_arms_zero_gain_interval_and_no_acceptance():
    result = score_study(*fixture())
    primary = result["primary"]
    assert primary["em_difference"] == {"mean": 0, "sample_sd": 0}
    assert primary["article_cluster_bootstrap_95_percentile_ci"] == [0, 0]
    assert not primary["accepted"]
    assert not primary["criteria"]["confidence_lower_bound_positive"]
    assert primary["positive_seed_count"] == 0
    for arm in ARMS:
        for group in ("overall", "answerable", "unanswerable"):
            assert result["arms"][arm]["across_seeds"][group]["em"] == {"mean": 1, "sample_sd": 0}


def test_clear_paired_improvement_and_raw_metrics():
    labels, preds, protocol = fixture()
    for rows in preds["gold"].values():
        for row in rows:
            row["prediction"] = "Bob"
    result = score_study(labels, preds, protocol)
    assert result["primary"]["accepted"]
    assert all(result["primary"]["criteria"].values())
    assert result["primary"]["article_cluster_bootstrap_95_percentile_ci"] == [1, 1]
    assert result["arms"]["gold"]["per_seed"][str(SEEDS[0])]["unanswerable"]["em"] == 0
    assert result["comparisons"]["full_kd"]["role"] == "secondary_descriptive"
    assert "accepted" not in result["comparisons"]["full_kd"]
    assert result["per_item_scores"]["gated_kd"][str(SEEDS[0])][0]["article_id"] == "a0"


def test_unequal_article_sizes_use_question_weighted_paired_bootstrap():
    labels, preds, protocol = fixture((1, 3), repetitions=10000)
    # Article 0: +1, article 1: -1 on every question. Overall is -0.5,
    # not zero (the incorrect equal-article mean). Independent 2-cluster draws
    # yield +1, -0.5, -1 with probabilities 1/4, 1/2, 1/4.
    for arm in ("gold", "gated_kd"):
        for rows in preds[arm].values():
            for row in rows:
                correct = (row["id"].startswith("a0") and arm == "gated_kd") or (row["id"].startswith("a1") and arm == "gold")
                if not correct:
                    row["prediction"] = "Bob"
    result = score_study(labels, preds, protocol)
    primary = result["primary"]
    assert primary["em_difference"]["mean"] == -.5
    rng = random.Random(20261009)
    expected = []
    for _ in range(10000):
        chosen = [rng.randrange(2) for _ in range(2)]
        successes = sum(2 if article == 0 else -6 for article in chosen)
        questions = sum(2 if article == 0 else 6 for article in chosen)
        expected.append(successes / questions)
    assert primary["bootstrap_em_differences"] == expected
    assert primary["article_cluster_bootstrap_95_percentile_ci"] == [-1, 1]
    assert set(expected) == {-1, -.5, 1}


def test_sample_sd_is_across_training_seeds_and_partial_f1_is_retained():
    labels, preds, protocol = fixture()
    for row in labels:
        if not row["is_impossible"]:
            row["answers"] = ["Alice met Bob"]
    for index, seed in enumerate(SEEDS):
        for row in preds["gold"][str(seed)][:index * 2]:
            row["prediction"] = "Bob"
    result = score_study(labels, preds, protocol)
    # Answerable "Alice" has EM=0 and F1=.5, unanswerable sentinel earns 1.
    assert result["arms"]["gated_kd"]["across_seeds"]["overall"]["f1"]["mean"] == .75
    assert result["arms"]["gold"]["across_seeds"]["overall"]["em"]["sample_sd"] == .25


def test_one_low_format_seed_fails_even_if_pooled_rate_passes():
    labels, preds, protocol = fixture((25, 25))
    # Two invalid outputs in only one seed: 98% vs pooled 99.333%.
    for row in preds["gated_kd"][str(SEEDS[0])][:2]:
        row["prediction"] = ""
    result = score_study(labels, preds, protocol)
    assert result["arms"]["gated_kd"]["across_seeds"]["overall"]["format_valid_rate"]["mean"] > .99
    assert not result["primary"]["criteria"]["format_valid_every_seed"]


def test_subgroup_harm_and_positive_seed_gates():
    labels, preds, protocol = fixture()
    # Candidate improves answerables but harms all unanswerables.
    for rows in preds["gold"].values():
        for row in rows:
            row["prediction"] = "NO_ANSWER"
    for rows in preds["gated_kd"].values():
        for row in rows:
            row["prediction"] = "Alice"
    result = score_study(labels, preds, protocol)
    assert result["primary"]["subgroup_mean_em_difference"] == {"answerable": 1, "unanswerable": -1}
    assert not result["primary"]["criteria"]["unanswerable_noninferiority"]
    assert not result["primary"]["criteria"]["positive_seeds"]


def test_inclusive_gain_and_subgroup_boundaries_avoid_rate_subtraction_error():
    labels, preds, protocol = fixture((25, 25))
    # 50 answerable + 50 unanswerable. Correctness changes: +4 answerable,
    # -1 unanswerable => overall exactly +.03 and unanswerable exactly -.02.
    answerable_ids = [r["id"] for r in labels if not r["is_impossible"]][:4]
    impossible_id = next(r["id"] for r in labels if r["is_impossible"])
    for rows in preds["gold"].values():
        for row in rows:
            if row["id"] in answerable_ids: row["prediction"] = "Bob"
    for rows in preds["gated_kd"].values():
        for row in rows:
            if row["id"] == impossible_id: row["prediction"] = "Bob"
    primary = score_study(labels, preds, protocol)["primary"]
    assert primary["em_difference"]["mean"] == .03
    assert primary["criteria"]["mean_em_gain"]
    assert primary["subgroup_mean_em_difference"]["unanswerable"] == -.02
    assert primary["criteria"]["unanswerable_noninferiority"]


def test_format_boundary_is_inclusive_and_two_positive_seeds_suffice():
    labels, preds, protocol = fixture((25, 25))
    for seed in SEEDS[:2]:
        for row in preds["gold"][str(seed)][:10]: row["prediction"] = "Bob"
    # One invalid row per 100 predictions exactly meets the .99 gate.
    for rows in preds["gated_kd"].values(): rows[-1]["prediction"] = ""
    primary = score_study(labels, preds, protocol)["primary"]
    assert set(primary["format_valid_rate_by_seed"].values()) == {.99}
    assert primary["criteria"]["format_valid_every_seed"]
    assert primary["positive_seed_count"] == 2
    assert primary["criteria"]["positive_seeds"]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "non_string"])
def test_prediction_coverage_is_exact_for_every_arm_seed(mutation):
    labels, preds, protocol = fixture()
    rows = preds["full_kd"][str(SEEDS[-1])]
    if mutation == "duplicate": rows.append(deepcopy(rows[0]))
    elif mutation == "missing": rows.pop()
    elif mutation == "extra": rows.append(dict(id="unexpected", prediction="Alice"))
    else: rows[0]["prediction"] = None
    with pytest.raises(ValueError): score_study(labels, preds, protocol)


@pytest.mark.parametrize("mutation", ["duplicate_label", "same_class", "article_mismatch", "context_mismatch", "empty_gold", "nonextractive_gold", "not_bool", "empty_article", "wrong_count", "single_article"])
def test_invalid_label_contract_is_rejected(mutation):
    labels, preds, protocol = fixture()
    if mutation == "duplicate_label": labels[1]["id"] = labels[0]["id"]
    elif mutation == "same_class": labels[1].update(is_impossible=False, answers=["Alice"])
    elif mutation == "article_mismatch": labels[1]["article_id"] = "different"
    elif mutation == "context_mismatch": labels[1]["context"] += " different"
    elif mutation == "empty_gold": labels[0]["answers"] = []
    elif mutation == "nonextractive_gold": labels[0]["answers"] = ["absent"]
    elif mutation == "not_bool": labels[0]["is_impossible"] = 0
    elif mutation == "empty_article": labels[0]["article_id"] = ""
    elif mutation == "wrong_count": protocol["scoring"]["expected_contexts"] += 1
    else:
        for row in labels: row["article_id"] = "only"
    with pytest.raises(ValueError): score_study(labels, preds, protocol)


@pytest.mark.parametrize("mutation", ["extra_arm", "missing_arm", "extra_seed", "missing_seed", "seed_alias", "bool_seed", "nan_threshold", "negative_drop"])
def test_protocol_and_arm_seed_contracts(mutation):
    labels, preds, protocol = fixture()
    if mutation == "extra_arm": preds["chosen_after_results"] = preds["gold"]
    elif mutation == "missing_arm": preds.pop("full_kd")
    elif mutation == "extra_seed": preds["gold"]["123"] = preds["gold"][str(SEEDS[0])]
    elif mutation == "missing_seed": preds["gold"].pop(str(SEEDS[0]))
    elif mutation == "seed_alias": preds["gold"][SEEDS[0]] = preds["gold"][str(SEEDS[0])]
    elif mutation == "bool_seed": protocol["scoring"]["seeds"][0] = True
    elif mutation == "nan_threshold": protocol["scoring"]["min_em_gain"] = math.nan
    else: protocol["scoring"]["max_subgroup_drop"] = -.02
    with pytest.raises(ValueError): score_study(labels, preds, protocol)


def test_input_order_does_not_change_result_or_mutate_inputs():
    labels, preds, protocol = fixture((1, 3))
    original = deepcopy((labels, preds, protocol))
    expected = score_study(labels, preds, protocol)
    assert (labels, preds, protocol) == original
    labels.reverse()
    for mapping in preds.values():
        for rows in mapping.values(): rows.reverse()
    assert score_study(labels, preds, protocol) == expected


def test_cli_only_scores_fixture_and_refuses_overwrite(tmp_path):
    labels, preds, protocol = fixture()
    labels_file, preds_file, protocol_file = [tmp_path / f for f in ("labels.jsonl", "preds.json", "protocol.json")]
    labels_file.write_text("".join(json.dumps(row) + "\n" for row in labels))
    preds_file.write_text(json.dumps(preds)); protocol_file.write_text(json.dumps(protocol))
    out = tmp_path / "new-result"
    args = [sys.executable, "-m", "qa_lab.confirmation_scoring", "--labels", str(labels_file),
            "--predictions", str(preds_file), "--protocol", str(protocol_file), "--output", str(out)]
    first = subprocess.run(args, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    before = (out / "metrics.json").read_bytes()
    result = json.loads(before)
    assert not result["primary"]["accepted"]
    assert "Arithmetic scoring does not establish" in result["limitations"][0]
    second = subprocess.run(args, capture_output=True, text=True)
    assert second.returncode != 0
    assert (out / "metrics.json").read_bytes() == before
