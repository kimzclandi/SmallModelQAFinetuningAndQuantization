# Teacher self-verification follow-up

Added 2026-09-20. This is a new exploratory protocol following inspection of the first Chinese pilot's development results. It does not produce an independent test claim.

The same cached Qwen2.5-1.5B teacher receives only context, question and its original candidate answer. A fixed prompt asks for SUPPORTED or UNSUPPORTED. It must first pass at least 6 of 8 fixed synthetic checks, with both labels emitted. Neither gold nor split metadata appears in the prompt. Exact label plus EOS is required; extra text, truncation or invalid labels do not pass. Self-verification can share the generator's errors and is not a source of ground truth.

The structural pool comes from the unchanged first pilot. Selected teacher outputs are compared against an equal-count random subset of that structural pool and gold answers for the exact same selected IDs. Subsets remain fixed while initialization/training order seeds vary across 20260920, 20260921 and 20260922. All runs perform 64 updates at batch one; input/supervised tokens and repetitions are reported rather than claimed equal. If fewer than 8 samples survive, or fewer than 4 are rejected, the protocol stops before redundant training. Identical semantic/gold targets avoid duplicate training.

From the repository root, with pinned models already cached:

```bash
HF_HUB_OFFLINE=1 python scripts/semantic_quality_pilot.py prepare \
  --parent /path/to/pilot-01 --out work/semantic-02
HF_HUB_OFFLINE=1 python scripts/semantic_quality_pilot.py run --out work/semantic-02
python scripts/verify_semantic_quality.py work/semantic-02
```

`HF_HOME` selects the cache. A working Apple MPS device and the original inference environment are required. No automatic downloads occur. Outputs refuse reuse; failures and partial predictions remain. The runner is a bounded pilot, not a checkpoint-resuming training service. Its 30-minute budget is checked at example/update boundaries, not an OS kill timer. No historical evidence or source data is overwritten.

The protocol and data hashes are frozen before judging. `judge-controls.jsonl` and `judgments.jsonl` preserve raw outputs, token IDs and prompt hashes. `selection.json`, exact training subsets and their hashes link decisions to training. Gold matching is a post-selection audit, not a selection input. Per-seed folders contain adapters, update/token logs, full dev predictions and metrics. The holdout is never inferred.

The verifier rechecks input and adapter hashes, exact subset selection, teacher/gold targets, seeded training order, actual token totals and evaluation pairing. Its exploratory paired bootstrap resamples 24 articles (one dev question each), averaging over the three fixed seeds. Intervals are conditional on these seeds and subsets, omit model-selection uncertainty and are not adjusted for multiple comparisons. Three seeds are not three independent datasets. Do not claim stable superiority solely from higher dev means.
