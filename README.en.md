# SmallModelQAFinetuningAndQuantization

[简体中文](README.md) | **English**

Auditable small-model extractive QA: data isolation → original baseline → gold-SFT and response distillation → coverage controls → same-framework quantization → new-source evaluation of a frozen candidate. Input is a passage and question; output is the shortest verbatim answer or strict `NO_ANSWER`. This is neither retrieval nor closed-book QA.

[![offline-integrity](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/actions/workflows/tests.yml/badge.svg)](https://github.com/kimzclandi/SmallModelQAFinetuningAndQuantization/actions/workflows/tests.yml)

**Across the five historical rounds, no trained candidate passed the adoption gates. Q8 passed English development compression screening and a 96-question Chinese new-source quality-preservation check; business deployment remains unverified.** All quality figures derive from saved per-example predictions, retaining abstention baselines, failures and regressions. Actual distillation used a local Qwen teacher. Manually prepared GPT candidates were audited but never used for training.

## Matched three-arm study on newly held-out contexts

Three fixed seeds per arm were retrained in the same CPU/FP32 loop on 242 TRAIN rows. After locking all nine models, 2,304 predictions were generated for 256 questions from 128 contexts excluded from recorded project inputs; scoring followed the complete prediction lock. Mean normalized EM (sample SD across seeds): **gold-SFT 51.82%±4.53%, full KL 44.79%±1.26%, teacher-gold agreement gated KL 51.82%±2.88%**.

The preregistered gated-versus-gold difference is **0.00 percentage points**, with a paired article-cluster 95% interval **[-2.82,+2.64]**. The candidate failed adoption gates; format validity was 91.41% versus gold's 95.18%, and only one seed improved EM. Always-abstain EM is 50%. All 25 articles have prior project exposure: this is a context holdout, not unseen-article or pretraining-clean validation. No post-result tuning or speedup claim. [Results and failures](docs/TEACHER_GATED_CONFIRMATION_RESULTS.md) · [Frozen protocol](docs/TEACHER_GATED_CONFIRMATION_V1.md) · [Records](reports/teacher-gated-confirmation-v1/) · [Offline replay](scripts/verify_confirmation.py).

## Frozen distillation diagnostics

Additive teacher-cache, loss-arithmetic and paired-error analysis, without retraining or tuning. All three seeds lose unanswerable correct answers versus gold-SFT; weighted KL contributes about 75–77% of scalar loss, not gradient attribution or causal proof. [Evidence and limits](docs/LOGITS_DIAGNOSTICS_V1.md).

[Cache lineage and token/mask preflight](docs/LOGITS_CACHE_INTEGRITY.md) now bind training to frozen TRAIN content, complete ordered IDs and tokenizer identity, while supporting an identical relocated artifact. This is an integrity fix, with no new quality result.

The logits cache/train entry points also perform [checkpoint file identity preflight](docs/MODEL_IDENTITY.md): registered model/revision pairs bind local weights, configuration and tokenizer bytes before loading that exact verified directory. New training with legacy caches requires explicit acknowledgement of unbound teacher bytes; historical runs are not rewritten or retroactively certified.

## Historical v2 logits distillation result and evidence

On 242 frozen TRAIN rows, three fixed seeds × 242 steps achieved **60.36% ± 2.81%** EM on 74 reused dev questions, below the matched gold-only control (**64.86% ± 1.35%**). The objective is `0.5 CE + 0.5 T² KL`, T=2; the full-vocabulary float16 cache is 374.69 MiB. The 24-row v1 failure remains archived. Coverage, steps and target policy changed between v1 and v2, so the difference is not an isolated data-size effect. No external confirmation or post-result tuning was performed. [Evidence map](docs/EVIDENCE_MAP.md) · [v2 records](reports/logits-distillation-v2/RESULTS.md).

## Project history (added 2026-09-20)

According to the maintainer, related early work began locally around June 2026 before consolidation and upload to GitHub. This approximate starting point does not date all current features or experiments. Later implementations, experiments and maintenance retain their actual version and run dates.

## Chinese data-quality comparison (2026-09-20)

Nine LoRA runs totaling 576 steps were completed. The untrained baseline and all nine adapters were evaluated once on a 96-question holdout. Replacing teacher targets with reference answers for the same questions raised mean strict EM from 17.36% to 24.31%, a +6.94-point difference with a paired 95% interval of [+1.04,+13.54]. The F1 difference interval still crosses zero. Training reference labels were used; this is not unlabeled automatic filtering. Neither the 1.5B nor the 3B automatic verifier passed the challenge gate.

**External check: on 96 DRCD questions, random_gold and random_teacher both achieved mean EM of 57.29%; the 95% difference interval was [-5.21,+4.86] points. The positive primary comparison on CMRC did not replicate.** [External evaluation and scope](docs/EXTERNAL_DRCD.md)

[Results, failures and limits](reports/quality-study-20260920/RESULTS.md) · [Offline checks / two-question inference / retraining](docs/QUALITY_STUDY_RELEASE.md) · [Full training-entry rerun](docs/ENTRYPOINT_RETRAIN.md)

## Traceable candidate QC (2026-09-22)

A new offline entry point records accept/reject/review decisions, reference permissions, provenance and rule versions. On 240 saved real teacher candidates, train-reference mode yields 8/11/5 for 24 original English responses, 8/12/4 for 24 revised-prompt responses, and 59/40/93 for 192 Chinese responses (accept/reject/review). Without reference permission, valid spans remain under review. Format compliance is not semantic correctness; near duplicates trigger review rather than automatic deletion.

```bash
python scripts/synthetic_qc.py --preset chinese --label-permission train_reference --output work/qc-01
python scripts/verify_synthetic_qc.py
```

The 59 accepted Chinese inputs and targets exactly match the historical reference-selected training group. Its existing experiments can be reused with their original limitations; no new training or semantic accuracy measurement was performed. Answerable examples make up 62.5% of accepted original English responses but only 25% under the revised prompt. Original candidates and frozen evidence remain unchanged. See the [QC protocol, records and limitations](docs/SYNTHETIC_QC.md) (Chinese).

## Five historical rounds and evidence

| Experiment | Main result | Conclusion and evidence |
|---|---|---|
| 1. Baseline, gold-SFT, response distillation and one data revision | Student test EM 24.51%; gold-SFT 51.96%; distillation 36.27%; added data 48.04% | All three candidates regressed on answerable dev EM and failed gates. Always-abstain test baseline: 52.94%. [Results](reports/closure-v1/RESULTS.md) |
| 2. Teacher prompts × three seeds | Mean dev EM, gold / original teacher / revised teacher: 55.41% / 33.78% / 55.86% | Revised prompt improved aggregate scores but reduced answerable performance. All failed three-seed gates; no new test inference. [Results](reports/teacher-study-v2/RESULTS.md) · [Method](docs/TEACHER_STUDY_V2.md) |
| 3. Repeated 24 questions versus coverage of 242 questions | Across 64 questions from two new articles, three-seed mean EM 36.46% → 44.27% | Same 242 steps, but tokens/epochs/class distribution not controlled. Still below always-abstain 50%; dev gate failed. [Results](reports/coverage-v3/RESULTS.md) · [Method](docs/COVERAGE_V3.md) |
| 4. Same-MLX FP16 / Q4 / Q8 | Q8 weights 988.10 → 525.05MB; one fewer correct dev answer out of 74; fixed-workload decode 266.21 → 313.86 tokens/s | Q8 passed the preset dev compression gate; Q4 failed. Single-machine speed is not a general acceleration promise. [Results/regressions](reports/quantization-v4/RESULTS.md) · [Method](docs/QUANTIZATION_V4.md) |
| 5. Frozen Q8 on a new Chinese source | FP16 and Q8 strict EM both 44/96; character-LCS F1 72.68% / 72.12% | Same correct-ID set, but outputs are not individually lossless. All questions answerable; Chinese abstention untested. [Results](reports/chinese-v5/RESULTS.md) · [Method](docs/CHINESE_V5.md) |

In round one, 4-bit quantization within MLX reduced test EM from 24.51% to 14.71%; smaller files and faster decoding alone are insufficient for adoption. Round four remeasured all three precisions together. Speed ratios are not assembled across rounds or frameworks.

## Data, models and scope

- English data: Computational_complexity_theory in public SQuAD2.0 dev, 418 questions in 48 passage families, split 242/74/102 for train/dev/test after near-duplicate family merging. Project test is not the official hidden test set.
- Round three adds 64 questions from Packet_switching and Prime_number in the same public SQuAD file. Round five uses 96 questions from public CMRC2018 dev, at most one per article. A new source does not imply absence from pretraining.
- Fixed-revision Qwen2.5-0.5B-Instruct student and Qwen2.5-1.5B-Instruct teacher, with local FP32 teacher generation. Baseline means no training by this project. Run records retain model versions, data hashes, source snapshots and configuration.
- Historical hardware: Apple M4 Max, 48GiB unified memory, 40-core GPU, macOS 27.0. PyTorch/MPS training and a separate MLX quantization environment. CUDA/Ascend training, production load and external business generalization are unverified.
- Three seeds represent training variation, not independent test sets. Dev has been reused across rounds. Named custom Chinese metrics are not official CMRC scores. New method selection requires a new protocol and independent data.

## Quick start: offline evidence acceptance

Run from the repository root with Python 3.12. No model, GPU or API key is needed. Initial dependency installation requires network; subsequent acceptance runs offline.

```bash
uv venv .venv-ci --python 3.12
uv pip install --python .venv-ci/bin/python -r requirements-ci.lock.txt
.venv-ci/bin/python scripts/acceptance.py --output work/acceptance-01
```

The entry point runs tests, six offline verification suites, gold controls and teacher auditing. Logs go only to a new `work/` subdirectory; `reports/data/configs` are checked before and after. Missing frozen summaries fail rather than being regenerated. Do not use `python -O`. Linux CI uses the same entry point: **no model downloads, training or fresh inference**. Historical engineering acceptance included 51 tests; Actions records determine the latest executed count.

## Actual model inference and training reproduction

The following loads the student and performs two dev inferences. The first download is about 1GB. For a small student smoke run, allow at least 8GiB available memory and 5GB disk; minimum requirements have not been measured. Historical full training/quantization used a 48GiB Mac.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
.venv/bin/python scripts/download_model.py
HF_HUB_OFFLINE=1 .venv/bin/python -m qa_lab.inference \
  --device cpu --splits dev --limit 2 --output work/smoke-01
```

For MPS, replace `--device cpu` with `--device mps`; unavailable MPS fails rather than silently falling back. Existing output directories are rejected. See the [full reproduction manual](docs/REPRODUCE_CLOSURE.md) for training, teacher generation and MLX quantization. PyTorch and MLX use separate locks/environments; the MLX lock targets macOS arm64. Locks pin versions without wheel hashes and are not cross-platform supply-chain locks.

A fresh-environment CPU smoke run was performed on the same machine. Independent-machine inference/training reproduction is not established. Frozen `reports/`, `data/`, `configs/` and source snapshots must not be overwritten; use isolated output directories.

## Implementation and navigation

| Entry | Responsibility |
|---|---|
| `qa_lab/data.py` | Source hashes, deduplication, near-duplicate families and fixed splits |
| `qa_lab/inference.py`, `metrics.py` | Input allowlist, actual inference, full ID coverage, EM/F1/abstention/format |
| `qa_lab/train.py`, `train_artifact.py`, `closure_data.py` | LoRA, answer-only loss, local teacher data and training boundaries |
| `qa_lab/mlx_experiment.py` | Same-framework conversion, quality comparison and fixed-workload timing |
| `scripts/acceptance.py`, `scripts/verify_*.py` | Tests and read-only recomputation from saved evidence |

[Protocol](docs/PROTOCOL.md) · [Research scope](docs/ROADMAP.md) · [Historical acceptance](docs/RELEASE_AUDIT.md) · [Manual candidate audit](docs/TEACHER_CANDIDATE_AUDIT.md) · [AI assistance/contributions](CONTRIBUTIONS.md) · [Data licensing](DATA_LICENSE.md) · [Third-party attribution](THIRD_PARTY.md)

Linked technical documents retain their original language. `reports/` preserves each round's original report and status. Statements such as “not yet public” or “no online CI” are pre-publication snapshots. Current code, permitted data and records are public; see the CI badge for current status. Model weights, caches and environments are not distributed with the repository.

[2026-09-19 maintenance](docs/maintenance/2026-09-19/README.md) · [2026-09-21 maintenance](docs/maintenance/2026-09-21/README.md)

## TRAIN-only CE/KL gradient diagnostic

The [frozen diagnostic](docs/CE_KL_GRADIENT_DIAGNOSTICS.md) computes five autograd gradients at identical model parameters, with independent logit references and per-parameter reconstruction gates. The first real probe failed the original FP32 numerical contract and stopped: 0 passed, 1 failed, not 192 completed. [Original receipts](reports/ce-kl-gradient-v1/) are preserved. No threshold changes, training updates, or held-out predictions followed. The public raw q_proj block passed; failed v_proj/logit arrays were not archived, so offline verification cannot independently replay the failed coordinates or establish a root cause. This does not establish distillation quality or compression benefits.
