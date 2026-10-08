"""Post-hoc diagnosis of frozen TRAIN teacher distributions and reused dev results.

Never trains, generates predictions, selects hyperparameters or reads a new test set.
Full teacher tensors stay local; published per-token statistics can be reaggregated
offline, while verifying their extraction requires the hash-bound local cache.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa_lab.common import read_jsonl, sha

SEEDS = (20260918, 20260919, 20260920)
FROZEN = ROOT / 'reports/logits-distillation-v2'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def index_rows(rows):
    result = {}
    for row in rows:
        if not isinstance(row['id'], str) or not row['id'] or row['id'] in result:
            raise ValueError('Missing or duplicate row identity')
        if row['em'] not in (0., 1.):
            raise ValueError('Expected binary exact match')
        result[row['id']] = row
    return result


def paired(candidate, control):
    candidate, control = index_rows(candidate), index_rows(control)
    if candidate.keys() != control.keys() or not candidate:
        raise ValueError('Paired ID coverage mismatch')
    transitions = Counter(); fixes = []; regressions = []; groups = {}
    for key, a in candidate.items():
        b = control[key]
        if a['is_impossible'] != b['is_impossible'] or a['family_id'] != b['family_id']:
            raise ValueError('Question metadata differs')
        transitions[b['category'] + ' -> ' + a['category']] += 1
        if a['em'] > b['em']: fixes.append(key)
        if a['em'] < b['em']: regressions.append(key)
        group = 'unanswerable' if a['is_impossible'] else 'answerable'
        counts = groups.setdefault(group, dict(n=0, candidate_correct=0, control_correct=0))
        counts['n'] += 1; counts['candidate_correct'] += int(a['em']); counts['control_correct'] += int(b['em'])
    return dict(n=len(candidate), fixes=sorted(fixes), regressions=sorted(regressions),
                groups=groups, transitions=dict(sorted(transitions.items())))


def loss_diagnostic(steps):
    if len(steps) != 242 or [r['step'] for r in steps] != list(range(1,243)) or len({r['id'] for r in steps}) != 242:
        raise ValueError('Expected full one-pass v2 training trace')
    ce, kl, residuals = [], [], []
    for row in steps:
        values = [row[k] for k in ('hard_ce','soft_kl','loss','grad_norm')]
        if any(not math.isfinite(x) for x in values):
            raise ValueError('Non-finite training evidence')
        ce.append(.5 * row['hard_ce']); kl.append(.5 * 2**2 * row['soft_kl'])
        residuals.append(abs(row['loss'] - ce[-1] - kl[-1]))
    if max(residuals) > 1e-5:
        raise ValueError('CE plus T-squared KL arithmetic differs')
    return dict(mean_weighted_ce=statistics.mean(ce), mean_weighted_kl=statistics.mean(kl),
                max_loss_reconstruction_error=max(residuals),
                kl_share_of_summed_scalar_loss=sum(kl)/(sum(ce)+sum(kl)),
                not_gradient_contribution=True)


def distribution_rows(record, row_id):
    import torch
    labels = record['labels']; ids = record['input_ids']; logp = record['teacher_log_probs']
    if labels.ndim != 1 or ids.ndim != 1 or ids.shape != labels.shape or len(ids) < 2:
        raise ValueError('Invalid causal label/input shape')
    if labels.dtype not in (torch.int32, torch.int64) or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer labels and input IDs')
    positions = torch.nonzero(labels[1:] != -100).flatten()
    targets = labels[1:][positions].long()
    if logp.ndim != 2 or logp.shape[0] != len(targets) or not len(targets) or logp.dtype != torch.float16:
        raise ValueError('Invalid full-vocabulary cache shape or dtype')
    if not torch.isfinite(logp).all() or (targets < 0).any() or (targets >= logp.shape[1]).any():
        raise ValueError('Invalid distribution or target')
    if not torch.equal(labels[1:][positions], ids[1:][positions]):
        raise ValueError('Gold labels do not match causal input tokens')
    raw = logp.double()
    logmass = torch.logsumexp(raw, dim=-1)
    normalized = raw - logmass[:,None]
    probs = normalized.exp()
    entropy = -(probs * normalized).sum(dim=-1)
    top1 = normalized.argmax(dim=-1)
    gold = probs[torch.arange(len(targets)), targets]
    return [dict(id=row_id, prediction_position=int(positions[i]), target_id=int(targets[i]),
                 teacher_top1_id=int(top1[i]), top1_matches_gold=bool(top1[i] == targets[i]),
                 gold_probability=float(gold[i]), entropy_nats=float(entropy[i]),
                 raw_probability_mass=float(logmass[i].exp()), vocabulary=int(logp.shape[1]))
            for i in range(len(targets))]


def aggregate(tokens):
    if not tokens:
        raise ValueError('Empty teacher diagnostics')
    keys = [(r['id'],r['prediction_position']) for r in tokens]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate supervised position')
    for r in tokens:
        if (type(r['top1_matches_gold']) is not bool or
            r['top1_matches_gold'] != (r['target_id'] == r['teacher_top1_id']) or
            not 0 <= r['gold_probability'] <= 1 or
            not math.isfinite(r['raw_probability_mass']) or r['raw_probability_mass'] <= 0 or
            not math.isfinite(r['entropy_nats']) or not 0 <= r['entropy_nats'] <= math.log(r['vocabulary']) + 1e-8):
            raise ValueError('Invalid token diagnostic')
    return dict(training_rows=len({r['id'] for r in tokens}), supervised_tokens=len(tokens),
        top1_gold_agreement=sum(r['top1_matches_gold'] for r in tokens)/len(tokens),
        mean_gold_probability=statistics.mean(r['gold_probability'] for r in tokens),
        mean_entropy_nats=statistics.mean(r['entropy_nats'] for r in tokens),
        maximum_probability_mass_error=max(abs(r['raw_probability_mass']-1) for r in tokens),
        normalization='Diagnostic statistics renormalize float16 cache in float64; training remains unchanged.',
        interpretation='TRAIN gold-forced tokens at T=2, not free-generation quality or proof of why KD lost.')


def frozen_analysis():
    from qa_lab.metrics import evaluate
    dev = read_jsonl(ROOT/'data/complexity-v1/dev.jsonl')
    output = {}
    for seed in SEEDS:
        a = FROZEN/str(seed)
        b = ROOT/f'reports/coverage-v3/gold242-{seed}/dev'
        scored = []
        for folder in (a,b):
            _, computed = evaluate(dev, read_jsonl(folder/'dev.predictions.jsonl'))
            if computed != read_jsonl(folder/'dev.scored.jsonl'):
                raise ValueError('Stored scored rows do not reproduce')
            scored.append(computed)
        output[str(seed)] = dict(paired=paired(*scored), losses=loss_diagnostic(json.loads((a/'steps.json').read_text())))
    return output


def run(cache, output):
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((cache/'manifest.json').read_text())
    if manifest != json.loads((FROZEN/'cache-manifest.json').read_text()):
        raise ValueError('Require exact frozen v2 teacher cache')
    output.mkdir(parents=True, exist_ok=False)
    tokens = []
    for item in manifest['records']:
        path = (cache/item['path']).resolve()
        if not path.is_relative_to(cache.resolve()) or sha(path) != item['sha256']:
            raise ValueError('Cache record identity mismatch')
        record = torch.load(path, map_location='cpu', weights_only=True)
        rows = distribution_rows(record, item['id'])
        if len(rows) != item['supervised_tokens'] or any(r['vocabulary'] != item['vocabulary'] for r in rows):
            raise ValueError('Cache dimensions differ from manifest')
        tokens.extend(rows)
    teacher = aggregate(tokens)
    if teacher['training_rows'] != 242 or teacher['supervised_tokens'] != 1289:
        raise ValueError('Unexpected training coverage')
    save(output/'teacher-token-statistics.json', tokens)
    save(output/'summary.json', dict(teacher=teacher, seeds=frozen_analysis(),
        scope='Post-hoc TRAIN/cache and reused dev diagnosis; no external confirmation or tuning.',
        causal_warning='v1 to v2 changes data, steps and teacher-forced target policy; no data-only causal estimate.'))
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    save(output/'provenance.json', dict(cache_manifest_sha256=sha(cache/'manifest.json'),
        frozen_cache_manifest_sha256=sha(FROZEN/'cache-manifest.json'), source_sha256=sha(__file__),
        torch=torch.__version__, cache_records=manifest['records'],
        extraction_requires_local_cache=True, full_tensors_published=False))
    save(output/'checksums.json', {p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


def verify(output):
    actual = {p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name != 'checksums.json'}
    if actual != json.loads((output/'checksums.json').read_text()):
        raise ValueError('Diagnostic evidence changed')
    provenance = json.loads((output/'provenance.json').read_text())
    frozen = json.loads((FROZEN/'cache-manifest.json').read_text())
    if provenance['cache_records'] != frozen['records'] or provenance['frozen_cache_manifest_sha256'] != sha(FROZEN/'cache-manifest.json'):
        raise ValueError('Cache provenance differs')
    if provenance['source_sha256'] != sha(output/'source.py'):
        raise ValueError('Source archive differs')
    summary = json.loads((output/'summary.json').read_text())
    if summary['teacher'] != aggregate(json.loads((output/'teacher-token-statistics.json').read_text())) or summary['seeds'] != frozen_analysis():
        raise ValueError('Summary does not reproduce')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run','verify'])
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'run':
        if args.cache is None: parser.error('--cache required for extraction')
        run(args.cache, args.output)
    else:
        print(json.dumps(verify(args.output), indent=2))
