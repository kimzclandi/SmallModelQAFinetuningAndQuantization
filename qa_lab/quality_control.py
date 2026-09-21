"""Conservative, label-permission-aware QC for extractive teacher candidates.

No models, no semantic judge, no normalization of training targets. The public
API consumes train sources, candidates and a label-free split index separately.
"""
from collections import Counter, defaultdict
import hashlib
import json

from .chinese import near as chinese_near
from .data import related

RULE_VERSION = 'extractive-qc-v1'
INDEX_KEYS = {'id', 'split', 'family_id', 'context', 'question'}
EN_THRESHOLDS = dict(context_jaccard=.7, question_jaccard=.8,
                     question_sequence_ratio=.9)


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def validate_index(rows):
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != INDEX_KEYS:
            raise ValueError('Split index must contain exactly label-free INDEX_KEYS')
        if any(not isinstance(row[k], str) or not row[k].strip() for k in INDEX_KEYS):
            raise ValueError('Invalid split-index fields')
        if row['split'] not in {'train', 'dev', 'test', 'holdout'}:
            raise ValueError('Unknown split')
        key = (row['split'], row['id'])
        if key in seen:
            raise ValueError('Duplicate ID within split index')
        seen.add(key)


def leakage_audit(index, language):
    """Reuse historical near rules as review flags, never as deletion rules."""
    if language not in {'en', 'zh'}:
        raise ValueError('language must be en or zh')
    validate_index(index)
    pairs = []
    for i, left in enumerate(index):
        for right in index[:i]:
            if left['split'] == right['split']:
                continue
            exact = [key for key in ('id', 'family_id', 'context')
                     if left[key] == right[key]]
            is_near = (chinese_near(left, right) if language == 'zh'
                       else related(left, right, EN_THRESHOLDS)) if not exact else False
            if exact or is_near:
                pairs.append(dict(left_id=left['id'], left_split=left['split'],
                                  right_id=right['id'], right_split=right['split'],
                                  kind='exact' if exact else 'near', fields=exact))
    return pairs


def audit(sources, candidates, index, *, language, label_permission='none'):
    if label_permission not in {'none', 'train_reference'}:
        raise ValueError('Only none or train_reference permission is supported')
    pairs = leakage_audit(index, language)
    indexed = {(r['split'], r['id']): r for r in index}
    by = {}
    for row in sources:
        if not isinstance(row, dict) or any(
                not isinstance(row.get(k), str) or not row[k].strip() for k in INDEX_KEYS):
            raise ValueError('Source requires nonempty identity/context/question fields')
        if row['split'] != 'train' or row['id'] in by:
            raise ValueError('Sources must be unique training IDs')
        if indexed.get(('train', row['id'])) != {k: row[k] for k in INDEX_KEYS}:
            raise ValueError('Source and split index identity/content mismatch')
        if label_permission == 'none' and {'answers', 'is_impossible'} & row.keys():
            raise ValueError('Remove labels when permission is none')
        by[row['id']] = row
    duplicates = Counter(p.get('id') for p in candidates if isinstance(p, dict)
                         and isinstance(p.get('id'), str))
    records = []
    for position, candidate in enumerate(candidates):
        p = candidate if isinstance(candidate, dict) else {}
        sample_id = p.get('id') if isinstance(p.get('id'), str) else None
        row = by.get(sample_id)
        raw = p.get('prediction')
        reject, review = [], []
        checks = dict(format_valid=None, span_locatable=None, reference_match=None,
                      abstention_correct=None, semantic_correctness='not_assessed')
        if row is None:
            reject.append('unknown_or_invalid_sample_id')
        if sample_id is not None and duplicates[sample_id] > 1:
            reject.append('duplicate_candidate_id')
        if not isinstance(raw, str) or not raw.strip():
            reject.append('empty_or_nonstring_response')
        elif row:
            checks['span_locatable'] = raw in row['context'] if raw != 'NO_ANSWER' else None
            checks['format_valid'] = raw == raw.strip() and (
                raw == 'NO_ANSWER' or checks['span_locatable'])
            if not checks['format_valid']:
                reject.append('not_exact_span_or_sentinel')
            if raw != 'NO_ANSWER' and len(raw) > len(row['context']):
                reject.append('response_longer_than_context')
        if p.get('stop_reason') is None:
            review.append('completion_metadata_unknown')
        elif p['stop_reason'] != 'eos':
            reject.append('generation_not_completed')
        slice_name = 'unknown'
        if row:
            links = [v for v in pairs if (v['left_split'], v['left_id']) == ('train', sample_id)
                     or (v['right_split'], v['right_id']) == ('train', sample_id)]
            if any(v['kind'] == 'exact' for v in links):
                reject.append('cross_split_exact_leakage')
            if any(v['kind'] == 'near' for v in links):
                review.append('cross_split_near_leakage')
            if label_permission == 'none':
                review.append('no_reference_permission')
            else:
                answers, impossible = row.get('answers'), row.get('is_impossible')
                valid = (type(impossible) is bool and isinstance(answers, list)
                         and all(isinstance(a, str) and a.strip() and a in row['context']
                                 for a in answers)
                         and bool(answers) != impossible)
                if not valid:
                    review.append('missing_or_invalid_reference')
                else:
                    slice_name = 'unanswerable' if impossible else 'answerable'
                    checks['reference_match'] = raw == 'NO_ANSWER' if impossible else raw in answers
                    checks['abstention_correct'] = (raw == 'NO_ANSWER') == impossible
                    if isinstance(raw, str) and raw.strip():
                        if (raw == 'NO_ANSWER') != impossible:
                            reject.append('abstention_reference_contradiction')
                        elif not checks['reference_match']:
                            review.append('reference_mismatch_not_semantic_proof')
        record = dict(record_id=stable([position, candidate]), sample_id=sample_id,
                      candidate_position=position, source=row, raw_candidate=candidate,
                      raw_response=raw, output_response=raw, transformations=[],
                      rule_version=RULE_VERSION, label_permission=label_permission,
                      checks=checks, reject_reasons=reject, review_reasons=review,
                      slice=slice_name,
                      provenance={k: p.get('provenance', {}).get(k, 'unknown')
                                  for k in ('model_id', 'revision', 'prompt_version',
                                            'generation_parameters', 'run_id')}
                      if isinstance(p.get('provenance', {}), dict) else {'status': 'unknown'})
        records.append(record)
    # Repeated answers alone (especially NO_ANSWER) are not duplicate examples.
    content = defaultdict(list)
    for r in records:
        if r['source']:
            content[stable([r['source']['context'], r['source']['question']])].append(r)
    for group in content.values():
        if len({r['sample_id'] for r in group}) > 1:
            for r in group:
                r['review_reasons'].append('duplicate_input_different_id')
    for r in records:
        r['decision'] = ('reject' if r['reject_reasons'] else
                         'review' if r['review_reasons'] else 'accept')
    slices = {}
    for name in ('answerable', 'unanswerable', 'unknown'):
        subset = [r for r in records if r['slice'] == name]
        slices[name] = dict(n=len(subset), decisions=dict(Counter(r['decision'] for r in subset)))
    summary = dict(rule_version=RULE_VERSION, n=len(records),
                   decisions=dict(Counter(r['decision'] for r in records)), slices=slices,
                   reasons=dict(Counter(x for r in records for x in r['reject_reasons'] + r['review_reasons'])),
                   missing_candidate_ids=sorted(set(by) - set(duplicates)),
                   cross_split_pairs=len(pairs),
                   format_valid_n=sum(r['checks']['format_valid'] is True for r in records),
                   format_assessed_n=sum(r['checks']['format_valid'] is not None for r in records),
                   semantic_review='Not performed; no measured semantic false acceptance/rejection rate.',
                   accept_meaning='Exact reference agreement plus structural gates; not independently verified semantics.')
    return dict(records=records, leakage=pairs, summary=summary)
