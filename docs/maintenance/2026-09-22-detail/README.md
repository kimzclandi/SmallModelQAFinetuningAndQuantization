# QC evidence contracts and origin metadata — 2026-09-22 follow-up

## Failure-first findings

Five new checks failed against the previous implementation: a custom synthetic fixture was described as real saved teacher output; a symlink or extra directory in a run was accepted; an extra file with a matching rewritten manifest was accepted; and an empty code-binding map bypassed source verification. This was an evidence-contract gap, not proof that any historical score was incorrect.

## Changes

- A QC run now requires exactly ten evidence files plus manifest.json, all regular files; symlinked run directories and entries, incidental directories, and extra/missing names are rejected before replay.
- A receipt requires the rule version, source-file map and the exact four implementation-binding keys. Hashes still detect accidental changes, not malicious coordinated forgery or authenticity.
- Custom input defaults to `input_origin={"kind":"custom_unverified"}`. A historical preset label is granted only when input rows, candidates, index, language and source paths match that preset. No claim of model generation is inferred from a file's name or content.
- `baseline/scripts/synthetic_qc.py.txt` preserves exact old entrypoint bytes (SHA-256 `6d7fac7658326c153e3c79b4e875c111586ce31c053107bfcd50d12fb22d0217`). Historical receipts are verified against that fixed archive; new receipts bind current code. Existing QC decisions and historical receipts remain unchanged.

## Verification

Python 3.12.13, existing CI dependencies. From repository root:

```sh
python -m pytest -q tests/test_qc_evidence_contract.py tests/test_quality_control.py
python -m pytest -q
python scripts/acceptance.py --output work/detail-acceptance-new
python scripts/synthetic_qc.py --preset english-original --label-permission train_reference --output work/detail-qc-new
python scripts/selection_profile.py --run work/detail-qc-new --output work/detail-profile-new
```

All 142 tests passed. The offline acceptance checks passed, including replay of all six historical QC runs; the preserved old source hash was verified. A new original-English preset run and selection profile executed: 8/24 accepted, accepted answerable fraction 5/8, unchanged from historical QC. The fixture tests also prove that callers cannot label unrelated data as a named preset.

No new teacher inference, training, semantic review or holdout selection occurred. Old data, configurations, reports, QC evidence and first-round maintenance evidence are unchanged. New evidence is under this directory, with full raw logs retained locally and explicitly sanitized public copies.
