# Blind semantic review

Read context, question and raw response only. Judge whether the response answers the question using the passage. NO_ANSWER is correct only if the passage cannot answer. Minor formatting or a noncanonical but equivalent span does not by itself make an answer semantically wrong. Do not consult reference answers, QC decisions, model names or the coordinator file.

For each blind_id fill judgment: correct / incorrect / uncertain. Fill error: none (correct only), wrong_answer, unsupported, false_abstention, missed_abstention, ambiguous_question or uncertain (uncertain judgments). Explain incorrect/uncertain in note. Do not remove rows or edit IDs. Leave uncertain when the question admits unresolved readings.

Use a pseudonymous reviewer_alias, not personal identity. Change origin from pending_human to human_attested only after you personally complete the review. A model-generated review must use synthetic_fixture and cannot count as human review. Keep one independent copy per reviewer; do not inspect another review before finishing. No labels have been filled automatically.
