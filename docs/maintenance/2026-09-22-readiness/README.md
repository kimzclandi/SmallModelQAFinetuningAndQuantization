# 2026-09-22 follow-up: QC source-origin readback

This is a post-submission improvement. Earlier records and dates are unchanged.

Three forged or missing provenance declarations passed replay despite rehashed manifests. The verifier now binds a named historical preset to exact projected inputs, source-file coverage, language and known preset. Legacy receipts use their archived entrypoint without inventing origin.

## Reproduce

Python 3.12; install requirements-ci.lock.txt for the Python data repositories. Agent export tests use the standard library. Use a new output directory each time.

```sh
python -m pytest tests/test_qc_evidence_contract.py -q
python scripts/acceptance.py --output work/new-acceptance
```

Expected: tests exit 0; replay and rebuild pass historical contracts. Baseline failure logs and current validation logs are in evidence/. Fresh local clones/environments remain same-host replication, not independent hardware replication.

## Limits

A checksum detects changed bytes, not honest provenance. This remains local consistency verification, not a cryptographic attestation or semantic review.

No new training, human semantic annotation, model-quality uplift or deployment is claimed. For resume mapping and interview questions, see the existing 2026-09-22 maintenance handoff; this addendum changes reliability evidence only.
