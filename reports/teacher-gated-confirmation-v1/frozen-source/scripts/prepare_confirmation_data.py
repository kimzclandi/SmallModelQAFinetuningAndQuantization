"""Prepare a fixed, context-disjoint cohort; no model or network access.

Only run the real source after the study protocol and this program are committed.
Inputs and reference labels are separate; no QA text is printed by the CLI.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8"
EXCLUSIONS_SHA256 = "953a2e5e25075c3dc17e7dac0b3c4c724f93d45ec59b25b08af8c0608973c86b"
DEFAULT_EXCLUSIONS = ROOT / "configs/confirmation-exclusions-v1.json"
INPUT_KEYS = {"id", "context_id", "article_id", "family_id", "context", "question"}


class DataPreparationError(ValueError):
    """A path-free, content-free failure code safe for a public receipt."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def text_sha(text):
    return sha(normalize(text).encode())


def rank(seed, kind, value):
    return sha(f"{seed}:{kind}:{value}".encode()), value


class QuestionIndex:
    """Exact legacy thresholds, with only mathematically safe cheap bounds."""
    def __init__(self, questions=(), jaccard=0.8, sequence=0.9):
        self.rows = []
        self.exact = set()
        self.jaccard = jaccard
        self.sequence = sequence
        for question in questions:
            self.add(question)

    def add(self, question):
        text = normalize(question)
        if text not in self.exact:
            self.exact.add(text)
            self.rows.append((text, set(text.split()), Counter(text)))

    def related(self, question):
        text = normalize(question)
        if text in self.exact:
            return True
        tokens, chars = set(text.split()), Counter(text)
        for old, old_tokens, old_chars in self.rows:
            union = len(tokens | old_tokens)
            if not union or len(tokens & old_tokens) / union >= self.jaccard:
                return True
            total = len(text) + len(old)
            if 2 * min(len(text), len(old)) < self.sequence * total:
                continue
            common = sum(min(n, old_chars.get(c, 0)) for c, n in chars.items())
            if 2 * common < self.sequence * total:
                continue
            if SequenceMatcher(None, text, old, autojunk=False).ratio() >= self.sequence:
                return True
        return False


def validate_exclusions(exclusions):
    if exclusions.get("schema") != "confirmation-exclusions-v1":
        raise DataPreparationError("exclusions_schema")
    questions = exclusions["known_normalized_questions"]
    if len(set(questions)) != len(questions) or any(normalize(q) != q for q in questions):
        raise DataPreparationError("historical_question_identity")
    if sorted(text_sha(q) for q in questions) != exclusions["known_question_sha256"]:
        raise DataPreparationError("historical_question_identity")
    selection = exclusions["selection"]
    if (type(selection["n_contexts"]) is not int or selection["n_contexts"] <= 0
            or selection["questions_per_context"] != 2
            or selection["answerable_per_context"] != 1
            or selection["unanswerable_per_context"] != 1
            or type(selection["seed"]) is not int):
        raise DataPreparationError("selection_budget")


def collect_contexts(raw):
    """Validate schema/reference offsets without displaying reference content."""
    result, ids, context_ids, article_ids = [], set(), set(), set()
    for article in raw["data"]:
        title = article["title"]
        if not isinstance(title, str) or not title or title in article_ids:
            raise DataPreparationError("article_identity")
        article_ids.add(title)
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if not isinstance(context, str) or not normalize(context):
                raise DataPreparationError("empty_context")
            context_id = text_sha(context)
            if context_id in context_ids:
                raise DataPreparationError("duplicate_normalized_context")
            context_ids.add(context_id)
            rows = []
            for question in paragraph["qas"]:
                qid = question["id"]
                if not isinstance(qid, str) or not qid or qid in ids:
                    raise DataPreparationError("question_identity")
                ids.add(qid)
                if not isinstance(question["question"], str) or not normalize(question["question"]):
                    raise DataPreparationError("empty_question")
                flag, answers = question["is_impossible"], question["answers"]
                if type(flag) is not bool or not isinstance(answers, list) or bool(answers) == flag:
                    raise DataPreparationError("answerability_schema")
                for answer in answers:
                    start, text = answer["answer_start"], answer["text"]
                    if (type(start) is not int or start < 0 or not isinstance(text, str)
                            or not text or context[start:start + len(text)] != text):
                        raise DataPreparationError("reference_offset")
                rows.append({"id": qid, "context_id": context_id, "article_id": title,
                             "family_id": title,
                             "context": context, "question": question["question"],
                             "is_impossible": flag,
                             "answers": list(dict.fromkeys(a["text"] for a in answers))})
            result.append({"article_id": title, "context_id": context_id,
                           "context": context, "rows": rows})
    return result


def select_cohort(raw, exclusions):
    validate_exclusions(exclusions)
    policy = exclusions["selection"]
    seed = policy["seed"]
    historical = QuestionIndex(exclusions["known_normalized_questions"],
                               policy["question_jaccard"], policy["question_sequence_ratio"])
    known_ids = set(exclusions["known_question_ids"])
    known_contexts = set(exclusions["known_context_sha256"])
    blocked_articles = set(exclusions["excluded_articles"])
    groups, audit = {}, Counter()
    for context in collect_contexts(raw):
        if (context["context_id"] in known_contexts
                or any(row["id"] in known_ids for row in context["rows"])):
            audit["historical_context"] += 1
            continue
        if context["article_id"] in blocked_articles:
            audit["exposed_article"] += 1
            continue
        eligible = {False: [], True: []}
        for row in context["rows"]:
            if historical.related(row["question"]):
                audit["historical_related_question"] += 1
                continue
            eligible[row["is_impossible"]].append(row)
        if not all(eligible.values()):
            audit["context_missing_class"] += 1
            continue
        context["eligible"] = {
            flag: sorted(rows, key=lambda row: rank(seed, "question", row["id"]))
            for flag, rows in eligible.items()}
        groups.setdefault(context["article_id"], []).append(context)
    for contexts in groups.values():
        contexts.sort(key=lambda row: rank(seed, "context", row["context_id"]))
    articles = sorted(groups, key=lambda article: rank(seed, "article", article))
    selected, selected_contexts, selected_questions = [], [], QuestionIndex(
        (), policy["question_jaccard"], policy["question_sequence_ratio"])
    # Keep the two opposite-answerability questions in one context together;
    # reject related questions across contexts, not the intended paired input.
    for round_id in range(max((len(v) for v in groups.values()), default=0)):
        for article in articles:
            if round_id >= len(groups[article]):
                continue
            context = groups[article][round_id]
            tokens = set(normalize(context["context"]).split())
            if any(len(tokens & old) / len(tokens | old) >= policy["selected_context_jaccard"]
                   for old in selected_contexts):
                audit["selected_related_context"] += 1
                continue
            pair = []
            for flag in [False, True]:
                candidate = next((row for row in context["eligible"][flag]
                                  if not selected_questions.related(row["question"])), None)
                if candidate is None:
                    break
                pair.append(candidate)
            if len(pair) != 2 or normalize(pair[0]["question"]) == normalize(pair[1]["question"]):
                audit["context_without_distinct_pair"] += 1
                continue
            selected.extend(pair)
            selected_contexts.append(tokens)
            for row in pair:
                selected_questions.add(row["question"])
            if len(selected_contexts) == policy["n_contexts"]:
                # Never reveal answerability through fixed answerable-first row order.
                selected.sort(key=lambda row: rank(seed, "input", row["id"]))
                return selected, dict(audit)
    raise DataPreparationError("insufficient_contexts_for_fixed_budget")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                       allow_nan=False) + "\n" for row in rows))


def prepare(source, output, exclusions_path=DEFAULT_EXCLUSIONS, *,
            expected_source_sha256=SOURCE_SHA256, expected_exclusions_sha256=EXCLUSIONS_SHA256):
    source, output, exclusions_path = Path(source), Path(output), Path(exclusions_path)
    source_bytes, exclusions_bytes = source.read_bytes(), exclusions_path.read_bytes()
    if sha(source_bytes) != expected_source_sha256:
        raise DataPreparationError("source_hash_mismatch")
    if sha(exclusions_bytes) != expected_exclusions_sha256:
        raise DataPreparationError("exclusions_hash_mismatch")
    exclusions = json.loads(exclusions_bytes)
    if exclusions["source"]["sha256"] != expected_source_sha256:
        raise DataPreparationError("source_binding_mismatch")
    output.mkdir(parents=True, exist_ok=False)
    stage = "selection"
    try:
        rows, audit = select_cohort(json.loads(source_bytes), exclusions)
        inputs = [{key: row[key] for key in INPUT_KEYS} for row in rows]
        labels = [{key: row[key] for key in ["id", "context_id", "article_id", "context", "answers", "is_impossible"]}
                  | {"family_id": row["article_id"]} for row in rows]
        stage = "serialization"
        write_rows(output / "inputs.jsonl", inputs)
        write_rows(output / "sealed-labels.jsonl", labels)
        context_ids = {row["context_id"] for row in rows}
        manifest = {
            "schema": "confirmation-data-v1", "status": "complete",
            "source_sha256": sha(source_bytes), "exclusions_sha256": sha(exclusions_bytes),
            "selector_sha256": sha(Path(__file__).read_bytes()),
            "inputs_sha256": sha((output / "inputs.jsonl").read_bytes()),
            "labels_sha256": sha((output / "sealed-labels.jsonl").read_bytes()),
            "n_contexts": len(context_ids), "n_questions": len(rows),
            "answerable": sum(not row["is_impossible"] for row in rows),
            "unanswerable": sum(row["is_impossible"] for row in rows),
            "article_context_counts": dict(sorted(Counter(
                {row["context_id"]: row["article_id"] for row in rows}.values()).items())),
            "selection": exclusions["selection"], "exclusion_audit": audit,
            "input_fields": sorted(INPUT_KEYS),
            "label_access": "Use inputs only for generation. Open sealed labels only after all frozen-arm prediction files are complete and hashed.",
            "scope": "New-study context-disjoint evaluation against recorded inputs; all articles have prior project exposure. Not article-independent, absolute blind, official hidden test or pretraining-unseen. Class-balanced sampling is not natural prevalence.",
            "attribution": {"dataset": "SQuAD 2.0 official public dev",
                            "authors": "Pranav Rajpurkar, Robin Jia, Percy Liang; Wikipedia contributors",
                            "source_url": exclusions["source"]["url"],
                            "license": "CC BY-SA 4.0",
                            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/legalcode.en",
                            "changes": "Deterministic context sampling, one answerable and one unanswerable question per context; raw questions/context/references preserved; label-free inputs separated from labels."},
        }
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        write_json(output / "failure.json", {
            "status": "failed", "stage": stage,
            "error_code": str(exc) if isinstance(exc, DataPreparationError) else type(exc).__name__,
            "source_sha256": sha(source_bytes), "exclusions_sha256": sha(exclusions_bytes),
            "selection_count_reduced": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.output, args.exclusions)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": str(exc)
                          if isinstance(exc, DataPreparationError) else type(exc).__name__}))
        return 1
    print(json.dumps({key: result[key] for key in ["status", "n_contexts", "n_questions",
                                                  "inputs_sha256", "labels_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
