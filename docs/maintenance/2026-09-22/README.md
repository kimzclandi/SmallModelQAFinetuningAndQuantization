# Selection coverage audit — 2026-09-22

This follow-up starts from the existing candidate QC branch (PR #10), not from default main. It adds descriptive selection analysis without changing QC decisions or any historical data/report/configuration. No new training, teacher inference, human semantic annotation, or model improvement is claimed.

## Problem and change

An equal number of accepted candidates can hide a shift in answerability and question coverage. `qa_lab/selection_profile.py` reports candidate denominators separately from unique IDs, context-question pairs and families; slice acceptance rates; known-answerability denominators; and fixed character-length bands. Empty denominators are null. Length bands are observable proxies, not difficulty labels. Unknown source records remain in the candidate denominator but contribute no source coverage.

`scripts/selection_profile.py` first invokes the existing QC evidence verifier, then writes a new work directory with input-manifest and code hashes. It refuses overwrites through the existing output guard. It neither changes rules nor automatically exports training targets.

## Reproduce

From repository root, Python 3.12 and requirements-ci.lock.txt:

```sh
python -m pytest -q tests/test_selection_profile.py
python scripts/selection_profile.py --run docs/synthetic-qc-20260922/evidence/english-original-train_reference --output work/profile-demo-original
python scripts/selection_profile.py --run docs/synthetic-qc-20260922/evidence/english-prompted-train_reference --output work/profile-demo-prompted
python scripts/acceptance.py --output work/acceptance-demo
```

Use a fresh output name each time. The two profile commands should both report 8/24 accepted; accepted answerable fraction is 0.625 versus 0.25. Unique accepted families are 7 versus 8, from 17 candidate families in each arm. Chinese candidates yield 59/192 accepted, all answerable; none-permission modes accept zero. These are new descriptive computations over old responses, not newly generated samples. The English arms share the same 24 questions; 240 candidates across the three sources are not 240 independent questions.

## Verification and limits

The new regression test first failed to import the missing coverage module; this was a missing executable audit, not an observed model failure. After implementation, five coverage tests and the full 136-test acceptance run passed. All historical checks, including exact QC replay and historical training metrics, passed. Evidence in `evidence/` records same-host execution; no independent-machine reproduction.

Existing training controls remain in `reports/quality-study-20260920/supervised-07/`: equal-count random teacher versus selected teacher changes question selection; random gold uses the same questions as random teacher but changes target lengths. Equal steps do not equal supervised tokens. No new difficulty measurement or semantic false-accept rate exists. Original/new teacher student EM 33.78%→55.86% and answerable EM 51.52%→38.38% belong to the historical 74-question development study, not this QC work. No training method met its adoption gate.

The historical QC files, source snapshots and negative external DRCD result remain unchanged. Human direction supplied scope and constraints; implementation, checks and this report were AI-assisted. This follow-up does not retrospectively assign completion dates or establish personal authorship of prior code.
