"""Descriptive selection shift, separate from QC decisions and model quality.

Lengths are observable proxies only. No evaluation labels or difficulty claims.
"""
from collections import Counter


def fraction(n, d):
    return n / d if d else None


def coverage(records):
    sources = [r['source'] for r in records if r.get('source')]
    return dict(unique_ids=len({r['id'] for r in sources}),
                unique_questions=len({(r['context'], r['question']) for r in sources}),
                unique_families=len({r['family_id'] for r in sources}))


def band(n):
    return '0-127' if n < 128 else '128-511' if n < 512 else '512+'


def profile(records):
    for r in records:
        if r.get('decision') not in {'accept', 'reject', 'review'} or r.get('slice') not in {'answerable', 'unanswerable', 'unknown'}:
            raise ValueError('Invalid QC decision/slice')
    accepted = [r for r in records if r['decision'] == 'accept']
    def stats(rows):
        n = len(rows)
        counts = dict(Counter(r['decision'] for r in rows))
        return dict(n=n, decisions=counts, accept_rate=fraction(counts.get('accept', 0), n))
    slices = {s: stats([r for r in records if r['slice'] == s])
              for s in ('answerable', 'unanswerable', 'unknown')}
    lengths = {}
    for field in ('context', 'question'):
        lengths[field] = {b: stats([r for r in records if r.get('source') and band(len(r['source'][field])) == b])
                          for b in ('0-127', '128-511', '512+')}
    known = [r for r in records if r['slice'] != 'unknown']
    accepted_known = [r for r in accepted if r['slice'] != 'unknown']
    return dict(schema_version='selection-profile-v1', candidate_count=len(records),
                accepted_candidates=len(accepted), accept_rate=fraction(len(accepted),len(records)),
                input_coverage=coverage(records), accepted_coverage=coverage(accepted),
                answerability_known_n=len(known), accepted_answerability_known_n=len(accepted_known),
                answerable_fraction_input=fraction(sum(r['slice']=='answerable' for r in known),len(known)),
                answerable_fraction_accepted=fraction(sum(r['slice']=='answerable' for r in accepted_known),len(accepted_known)),
                slices=slices, character_length_bands=lengths,
                limitations=['Descriptive candidate batch, not unique independent questions.',
                             'Unknown answerability excluded from fractions; denominators reported.',
                             'Length is not measured difficulty; semantic review not performed.',
                             'No training, no causal model-quality claim.'])
