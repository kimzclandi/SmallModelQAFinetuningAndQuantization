# Chinese data-quality distillation pilot

Added 2026-09-20, after the historical experiments. This runner performs actual local teacher generation and student LoRA training. It is separate from the frozen historical evidence and is not included in offline CI.

Scope: 64 answerable training questions, 24 development questions, and 24 sealed holdout questions. Four groups (raw teacher, structural filtering, equal-count random subset, full gold) use 64 optimizer updates each, one seed. This is an exploratory feasibility study; updates are matched but tokens and class/difficulty distributions are not. A same-selected-ID gold control and multi-seed study are not part of this bounded pilot.

## Inputs and execution

Use the existing inference environment in `requirements.lock.txt`. The runner requires cached, pinned Qwen2.5 0.5B and 1.5B Instruct snapshots, a working MPS device, the official CMRC train/dev JSON files and the historical Chinese Evidence Data Engine `data/source-v1` directory. No automatic model or dataset downloads occur. Source license and attribution remain those of CMRC2018, as described in `DATA_LICENSE.md`.

Set `HF_HOME` to the existing cache. From the repository root:

```bash
HF_HUB_OFFLINE=1 python scripts/quality_pilot.py prepare \
  --out work/quality-pilot-01 --repo . \
  --source /path/to/cmrc2018_train.json \
  --olddev /path/to/cmrc2018_dev.json \
  --engine /path/to/chinese-evidence-data-engine/data/source-v1
HF_HUB_OFFLINE=1 python scripts/quality_pilot.py run --out work/quality-pilot-01
python scripts/verify_quality_pilot.py work/quality-pilot-01
```

Preparation hashes and fixes the data/protocol before generation. It excludes historical query article titles, official CMRC dev article titles, and near-duplicates against official dev and selected samples. The actual 2026-09-20 pilot additionally checked all 112 selected questions against all 320 historical retrieval queries: 35,840 comparisons, zero matches under the existing Chinese near-duplicate heuristic. This is not a semantic non-overlap proof. CMRC train was previously indexed in the retrieval project; it is not an unseen source or evidence of absence from model pretraining.

The quality rule requires EOS, nonempty output and an exact context substring; NO_ANSWER is excluded for this answerable-only experiment. It does not read gold labels and does not verify semantic correctness. Raw training deliberately retains nonempty malformed/truncated teacher outputs to measure the unfiltered condition; these are visible in raw prediction records. No result supports Chinese abstention performance.

The run refuses an existing run directory, preserves partial predictions and step logs on failure, and marks failed status. A 1,800-second total compute-stage limit is checked between examples and steps; it is not an OS-level interrupt for a hung kernel. This minimal pilot does not resume optimizer checkpoints. Start any retry in a new prepared output directory; do not delete failed runs.

## Outputs and interpretation

`run/teacher-train.jsonl` contains original teacher outputs. `quality-audit.jsonl` records filtering and post-hoc gold matches separately. Per-group training inputs, hashes, step/token logs, LoRA adapter hashes, development predictions and recomputed metrics remain linked. The holdout is hashed but never passed to inference. The independent verifier checks sample sets, target identity, actual training counts and prediction pairing before recomputing metrics.

Only parameters in the LoRA adapters are updated. Loss reduction is not a quality claim; development-set improvements are exploratory. Future filtering changes require a new protocol and suitable independent validation. No quantization, logits matching, teacher rationales or online reinforcement learning are implemented here.
