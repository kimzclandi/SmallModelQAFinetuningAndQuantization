"""Synthetic-only tests; never select or open a real confirmation cohort."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts import prepare_confirmation_data as prep


def fixture_source(articles=3, contexts=3):
    data = []
    for a in range(articles):
        paragraphs = []
        for c in range(contexts):
            key = f"{a}-{c}"
            unique = hashlib.sha256(key.encode()).hexdigest()
            answer = "value" + unique[:12]
            context = f"{answer} token{unique[12:32]} place{unique[32:]}"
            questions = []
            for flag in [False, True]:
                token = hashlib.sha256(f"question-{key}-{flag}".encode()).hexdigest()
                questions.append({"id": f"q-{key}-{flag}", "question": f"What {token}?",
                                  "is_impossible": flag,
                                  "answers": [] if flag else [{"text": answer, "answer_start": 0}]})
            paragraphs.append({"context": context, "qas": questions})
        data.append({"title": f"article-{a}", "paragraphs": paragraphs})
    return {"data": data}


def fixture_exclusions(n_contexts=3):
    return {
        "schema": "confirmation-exclusions-v1",
        "source": {"sha256": "pending", "url": "https://example.invalid/fixture"},
        "selection": {"seed": 20261009, "n_contexts": n_contexts,
                      "questions_per_context": 2, "answerable_per_context": 1,
                      "unanswerable_per_context": 1, "question_jaccard": .8,
                      "question_sequence_ratio": .9, "selected_context_jaccard": .7},
        "known_normalized_questions": [], "known_question_sha256": [],
        "known_question_ids": [], "known_context_sha256": [], "excluded_articles": [],
    }


def materialize(tmp_path, raw, exclusions):
    source = tmp_path / "source.json"
    config = tmp_path / "exclusions.json"
    source.write_text(json.dumps(raw))
    exclusions["source"]["sha256"] = prep.sha(source.read_bytes())
    config.write_text(json.dumps(exclusions))
    return source, config, {
        "expected_source_sha256": prep.sha(source.read_bytes()),
        "expected_exclusions_sha256": prep.sha(config.read_bytes()),
    }


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_full_fixed_budget_and_label_separation_on_synthetic_source(tmp_path):
    raw, exclusions = fixture_source(4, 40), fixture_exclusions(128)
    source, config, hashes = materialize(tmp_path, raw, exclusions)
    output = tmp_path / "out"
    result = prep.prepare(source, output, config, **hashes)
    inputs, labels = read_rows(output / "inputs.jsonl"), read_rows(output / "sealed-labels.jsonl")
    assert result["n_contexts"] == 128 and result["n_questions"] == 256
    assert result["answerable"] == result["unanswerable"] == 128
    assert set(result["article_context_counts"].values()) == {32}
    assert all(set(row) == prep.INPUT_KEYS for row in inputs)
    assert [row["id"] for row in inputs] == [row["id"] for row in labels]
    assert all(all(a[key] == b[key] for key in ["id", "context_id", "article_id", "family_id", "context"])
               for a, b in zip(inputs, labels))
    assert all(row["family_id"] == row["article_id"] for row in labels)
    assert len({row["context_id"] for row in inputs}) == 128
    assert [row["id"] for row in inputs] == sorted(
        [row["id"] for row in inputs], key=lambda qid: prep.rank(20261009, "input", qid))
    assert result["inputs_sha256"] == prep.sha((output / "inputs.jsonl").read_bytes())
    assert result["labels_sha256"] == prep.sha((output / "sealed-labels.jsonl").read_bytes())


def test_source_iteration_order_does_not_change_selection():
    raw = fixture_source()
    reversed_source = deepcopy(raw)
    reversed_source["data"].reverse()
    for article in reversed_source["data"]:
        article["paragraphs"].reverse()
        for paragraph in article["paragraphs"]:
            paragraph["qas"].reverse()
    assert prep.select_cohort(raw, fixture_exclusions())[0] == prep.select_cohort(
        reversed_source, fixture_exclusions())[0]


@pytest.mark.parametrize("field", ["context", "id", "article"])
def test_history_excludes_the_whole_context_or_article(field):
    raw, exclusions = fixture_source(), fixture_exclusions()
    para = raw["data"][0]["paragraphs"][0]
    if field == "context":
        exclusions["known_context_sha256"] = [prep.text_sha(para["context"])]
    elif field == "id":
        exclusions["known_question_ids"] = [para["qas"][0]["id"]]
    else:
        exclusions["excluded_articles"] = [raw["data"][0]["title"]]
    rows, audit = prep.select_cohort(raw, exclusions)
    assert not any(row["context_id"] == prep.text_sha(para["context"]) for row in rows)
    if field == "article":
        assert not any(row["article_id"] == raw["data"][0]["title"] for row in rows)
        assert audit["exposed_article"] == 3


def test_near_question_thresholds_use_legacy_semantics():
    assert prep.QuestionIndex(["one two three four five"]).related("one two three four five six")
    assert prep.QuestionIndex(["abcdefghijABCDEFGHIJ"]).related("abcdefghijABCDEFGHIX")
    assert not prep.QuestionIndex(["one two three four five"]).related("unrelated sample")


def test_historical_related_question_cannot_fill_missing_class():
    raw, exclusions = fixture_source(), fixture_exclusions(9)
    question = raw["data"][0]["paragraphs"][0]["qas"][0]["question"]
    exclusions["known_normalized_questions"] = [prep.normalize(question)]
    exclusions["known_question_sha256"] = [prep.text_sha(question)]
    with pytest.raises(prep.DataPreparationError, match="insufficient_contexts"):
        prep.select_cohort(raw, exclusions)


def test_related_questions_across_selected_contexts_cannot_fill_budget():
    raw = fixture_source(2, 1)
    first = raw["data"][0]["paragraphs"][0]["qas"]
    second = raw["data"][1]["paragraphs"][0]["qas"]
    for a, b in zip(first, second):
        b["question"] = a["question"]
    with pytest.raises(prep.DataPreparationError, match="insufficient_contexts"):
        prep.select_cohort(raw, fixture_exclusions(2))


def test_intended_opposite_classes_within_one_context_are_not_cross_context_leakage():
    raw = fixture_source(1, 1)
    questions = raw["data"][0]["paragraphs"][0]["qas"]
    questions[1]["question"] = questions[0]["question"] + "x"
    rows, _ = prep.select_cohort(raw, fixture_exclusions(1))
    assert len(rows) == 2 and {row["is_impossible"] for row in rows} == {False, True}


def test_reference_failure_saves_safe_receipt_without_cohort(tmp_path):
    raw = fixture_source()
    raw["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 1
    source, config, hashes = materialize(tmp_path, raw, fixture_exclusions())
    output = tmp_path / "out"
    with pytest.raises(prep.DataPreparationError, match="reference_offset"):
        prep.prepare(source, output, config, **hashes)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_code"] == "reference_offset"
    assert not (output / "inputs.jsonl").exists() and not (output / "sealed-labels.jsonl").exists()
    assert str(tmp_path) not in (output / "failure.json").read_text()


def test_shortfall_fails_without_reducing_fixed_count(tmp_path):
    source, config, hashes = materialize(tmp_path, fixture_source(1, 2), fixture_exclusions(128))
    output = tmp_path / "out"
    with pytest.raises(prep.DataPreparationError, match="insufficient_contexts"):
        prep.prepare(source, output, config, **hashes)
    assert json.loads((output / "failure.json").read_text())["selection_count_reduced"] is False
    assert not (output / "inputs.jsonl").exists()


@pytest.mark.parametrize("which", ["source", "exclusions"])
def test_pinned_identity_rejects_changed_bytes_before_output(tmp_path, which):
    source, config, hashes = materialize(tmp_path, fixture_source(), fixture_exclusions())
    changed = source if which == "source" else config
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(prep.DataPreparationError, match="hash_mismatch"):
        prep.prepare(source, tmp_path / "out", config, **hashes)
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten(tmp_path):
    source, config, hashes = materialize(tmp_path, fixture_source(), fixture_exclusions())
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_text("frozen")
    with pytest.raises(FileExistsError):
        prep.prepare(source, output, config, **hashes)
    assert (output / "keep").read_text() == "frozen"


@pytest.mark.parametrize("kind", ["duplicate_id", "duplicate_context", "invalid_class"])
def test_malformed_source_fails_closed(kind):
    raw = fixture_source()
    a, b = raw["data"][0]["paragraphs"][:2]
    if kind == "duplicate_id":
        b["qas"][0]["id"] = a["qas"][0]["id"]
    elif kind == "duplicate_context":
        b["context"] = a["context"]
    else:
        a["qas"][0]["is_impossible"] = 0
    with pytest.raises(prep.DataPreparationError):
        prep.select_cohort(raw, fixture_exclusions())


def test_public_exclusion_binding_without_opening_official_raw_source():
    encoded = prep.DEFAULT_EXCLUSIONS.read_bytes()
    assert prep.sha(encoded) == prep.EXCLUSIONS_SHA256
    exclusions = json.loads(encoded)
    prep.validate_exclusions(exclusions)
    assert exclusions["source"]["sha256"] == prep.SOURCE_SHA256
    assert exclusions["selection"]["n_contexts"] == 128
    assert exclusions["selection"]["seed"] == 20261009
    assert len(exclusions["excluded_articles"]) == 5
    assert "/Users/" not in encoded.decode()
