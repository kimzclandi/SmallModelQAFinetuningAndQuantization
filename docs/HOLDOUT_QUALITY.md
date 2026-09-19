# Frozen holdout evaluation

Added 2026-09-20 after the two exploratory Chinese data-quality pilots. No new training or prompt tuning occurs in this step.

The evaluation registers all nine already-trained adapters plus the unchanged base student, model/config hashes, seeds, dataset hash and comparison before loading any holdout question for inference. It evaluates the sealed 24 questions, one per article. All are answerable and from the same CMRC source previously indexed by the retrieval project; this is within-source held-out evaluation, not a new external domain or evidence that pretraining did not include the data.

Primary comparison: mean strict EM of semantic filtering minus equal-count random structural-pool selection, across all three fixed seeds. The directional research gate requires the 95% paired article-bootstrap lower bound to be above zero. This is not a deployment approval threshold. Character F1 and comparisons with selected-ID gold and the base student are descriptive; seeds are not selected by test performance.

```bash
HF_HUB_OFFLINE=1 python scripts/holdout_quality.py prepare \
  --parent /path/to/semantic-02 --out work/holdout-evaluation-03
HF_HUB_OFFLINE=1 python scripts/holdout_quality.py run --out work/holdout-evaluation-03
python scripts/holdout_quality.py verify --out work/holdout-evaluation-03
```

Use the cached model environment and MPS hardware of the parent experiment. No downloads or paid calls occur. Existing output directories are rejected, and the first test access is recorded before inference. Once accessed, the dataset is consumed for this protocol, including partial failed runs. The tool does not prevent a caller from explicitly creating another directory; experimental discipline and preserved access records, not a security boundary, enforce one evaluation. Do not use outcomes to tune and claim an independent result on the same questions again.

`access.json` preserves start/end status; every model has complete predictions, scores and metrics. A 900-second budget is checked at example boundaries. The verifier rechecks model/input identities and recomputes scores and paired intervals. It performs no model inference. Intervals condition on the fixed seed and subset choices and do not include model-selection uncertainty.

Training-label review: the prior statement “11 correct answers rejected” means 11 exact reference matches. An AI assistant review found direct support for nine, a mildly ambiguous question for one, and contradictory source wording for one. It is not independent human annotation. Neither labels nor frozen outputs were changed.

## Observed 2026-09-20 outcome

The registered 24-question evaluation completed all 240 predictions in 78.66 seconds. Mean strict EM / character LCS F1: semantic 41.67% / 73.03%; equal-count random 43.06% / 74.62%; selected-ID gold 41.67% / 71.83%; base student 37.50% / 69.98%. Semantic versus random EM difference was -1.39 percentage points, with a paired interval of [-9.72, +6.94]. The primary research gate failed. The development advantage was not confirmed, and the method is not adopted as a demonstrated improvement. These are within-source, answerable-only, small-sample results.

The frozen output package remains local under the dated holdout-03 deliverable, with protocols, original predictions, access ledger and read-only verification. No remote publication has been performed for this follow-up. The test has now been consumed; a changed method requires new independent validation.
