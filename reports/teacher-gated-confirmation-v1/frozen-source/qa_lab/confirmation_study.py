"""One frozen, offline quality study; prediction workers never open labels.

This enforces stage/hash boundaries, not adversarial or cryptographic blinding.
CPU timings are run costs, not controlled performance benchmark results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/teacher-gated-confirmation-v1')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_inputs(path, expected):
    if digest(path) != expected:
        raise ValueError('Prediction inputs hash changed')
    rows = jsonl(path)
    allowed = {'id', 'context', 'question', 'article_id', 'context_id', 'family_id'}
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate prediction inputs')
    if any(set(r) - allowed or not {'id', 'context', 'question'} <= set(r) for r in rows):
        raise ValueError('Prediction inputs contain label or unknown fields')
    return rows


def validate_bindings(bindings, root=ROOT):
    for name, expected in bindings.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('Frozen input/source hash changed: ' + name)


def adapter_files(path):
    return {name: digest(path / name) for name in ('adapter_config.json', 'adapter_model.safetensors')}


def verify_paired_training(receipts, protocol):
    for seed in protocol['seeds']:
        paired = [receipts[f'{arm}:{seed}'] for arm in protocol['arms']]
        for field in ('initial_trainable_state_sha256', 'training_id_order_sha256', 'cache_manifest_sha256'):
            if len({r[field] for r in paired}) != 1:
                raise ValueError('Matched training identity differs: ' + field)
        if any(r['steps_completed'] != protocol['training']['steps'] or r['total_supervised_tokens'] != 1289 for r in paired):
            raise ValueError('Training token/step budget differs')


def committed_files(names):
    bindings = {}
    for name in names:
        content = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT, stderr=subprocess.PIPE)
        if content != (ROOT/name).read_bytes():
            raise ValueError('Source/config is not identical to its committed version: ' + name)
        bindings[name] = digest(ROOT/name)
    return bindings


def disk_bytes(output):
    total = 0
    for path in output.rglob('*'):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # Atomic receipt replacement can remove a .tmp between enumeration and stat.
            pass
    return total


def check_global_budget(budget, study_started, output):
    if time.monotonic() - study_started > budget['total_wall_seconds']:
        raise TimeoutError('Frozen study wall-time budget exceeded')
    if disk_bytes(output) > budget['max_output_bytes']:
        raise RuntimeError('Output disk budget exceeded')


def validate_cohort_manifest(manifest, protocol):
    exclusions_path = ROOT / protocol['data']['exclusions']
    exclusions = read(exclusions_path)
    if (manifest.get('schema') != 'confirmation-data-v1' or manifest.get('status') != 'complete'
            or manifest['source_sha256'] != exclusions['source']['sha256']
            or manifest['exclusions_sha256'] != digest(exclusions_path)
            or manifest['selector_sha256'] != digest(ROOT / protocol['data']['sampler'])
            or manifest['selection'] != exclusions['selection']
            or manifest['n_contexts'] != protocol['data']['contexts']
            or manifest['n_questions'] != protocol['data']['questions']):
        raise ValueError('Prepared cohort does not bind the frozen source/selector/exclusions/counts')


def lock_predictions(output, cohort, run_order):
    """Require every planned prediction before the caller may read labels."""
    lock = read(output / 'execution-lock.json')
    if set(lock['adapters']) != set(run_order):
        raise ValueError('Training lock does not cover all planned models')
    files = {}
    for key in run_order:
        folder = output / 'predictions' / key.replace(':', '-')
        receipt = read(folder / 'run.json')
        prediction = folder / 'predictions.jsonl'
        if (receipt['status'] != 'complete' or receipt['execution_lock_sha256'] != digest(output / 'execution-lock.json')
                or receipt['inputs_sha256'] != digest(cohort / 'inputs.jsonl')
                or receipt['predictions_sha256'] != digest(prediction)
                or receipt['key'] != key or receipt['adapter_files'] != lock['adapters'][key]['files']):
            raise ValueError('Incomplete or modified prediction: ' + key)
        rows = jsonl(prediction)
        ids = [r['id'] for r in rows]
        expected = [r['id'] for r in checked_inputs(cohort / 'inputs.jsonl', lock['inputs_sha256'])]
        if (len(ids) != len(set(ids)) or set(ids) != set(expected) or receipt['n'] != len(expected)
                or any(not isinstance(r.get('prediction'), str) for r in rows)):
            raise ValueError('Prediction ID coverage failed')
        files[key] = {'predictions_sha256': digest(prediction), 'run_sha256': digest(folder / 'run.json')}
    return {'status': 'all_predictions_locked_before_scoring', 'created_utc': now(),
            'execution_lock_sha256': digest(output / 'execution-lock.json'),
            'labels_sha256': digest(cohort / 'sealed-labels.jsonl'), 'files': files}


def predict(args):
    """Worker receives inputs and an adapter lock, never a label file."""
    from .inference import generate, QAInput, MemoryMonitor, hardware_chip
    from .model_identity import verify_local_model, load_verified_model
    import torch
    from peft import PeftModel
    lock = read(args.execution_lock)
    validate_bindings(lock['frozen_files'])
    cfg = read(CONFIG / 'student.json')
    rows = checked_inputs(args.inputs, lock['inputs_sha256'])
    if adapter_files(args.adapter) != lock['adapters'][args.key]['files']:
        raise ValueError('Adapter identity changed after training lock')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'started_utc': now(), 'key': args.key,
               'execution_lock_sha256': digest(args.execution_lock), 'inputs_sha256': digest(args.inputs),
               'adapter_files': adapter_files(args.adapter), 'scope': 'Serial offline quality evaluation; costs are not a speed benchmark',
               'hardware': hardware_chip(), 'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')}}
    write(args.output / 'run.json', receipt)
    monitor = MemoryMonitor('cpu')
    monitor.thread.start()
    started = time.perf_counter()
    try:
        torch.set_num_threads(8)
        torch.manual_seed(int(args.key.split(':')[1]))
        snapshot = verify_local_model(cfg, args.cache_dir)
        tokenizer, model = load_verified_model(cfg, snapshot)
        model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True).eval()
        model.config.use_cache = True
        def finite_generation_logits(module, arguments, result):
            if not torch.isfinite(result.logits[:, -1, :]).all():
                raise ValueError('Non-finite next-token logits; generation stopped')
        model.register_forward_hook(finite_generation_logits)
        receipt['model_identity'] = snapshot.receipt
        receipt['parameter_count'] = sum(p.numel() for p in model.parameters())
        receipt['parameter_bytes'] = sum(p.numel() * p.element_size() for p in model.parameters())
        # Structural preflight checks every prompt, without generating or reading labels.
        from .inference import messages
        sizes = [len(tokenizer.apply_chat_template(messages(QAInput(r['context'], r['question']), cfg),
                                                  tokenize=True, add_generation_prompt=True)) for r in rows]
        if max(sizes) > cfg['max_input_tokens']:
            raise ValueError('Heldout input exceeds frozen token budget; no truncation or replacement allowed')
        receipt['max_input_tokens_observed'] = max(sizes)
        for _ in range(2):
            generate(QAInput('A queue follows first in, first out.', 'What order does a queue follow?'), cfg, tokenizer, model)
        with (args.output / 'predictions.jsonl').open('x') as stream:
            for i, row in enumerate(rows):
                result = generate(QAInput(row['context'], row['question']), cfg, tokenizer, model)
                result['id'] = row['id']
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                if (i + 1) % 32 == 0:
                    print(f'{args.key}: {i + 1}/{len(rows)} predictions saved (unscored)', flush=True)
        receipt.update(status='complete', n=len(rows), predictions_sha256=digest(args.output / 'predictions.jsonl'))
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        receipt.update(finished_utc=now(), wall_seconds=time.perf_counter()-started, memory=monitor.values())
        write(args.output / 'run.json', receipt)


def bounded_process(argv, log, budget, stage_seconds, study_started, output):
    """Only terminate this study's own subprocess on the frozen resource caps."""
    import psutil
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    started = time.monotonic()
    peak = 0
    check_global_budget(budget, study_started, output)
    with log.open('x') as f:
        child = subprocess.Popen([sys.executable, *argv], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                try:
                    rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True) if p.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > budget['max_sampled_rss_bytes']:
                    raise RuntimeError('Sampled process RSS budget exceeded')
                if time.monotonic() - started > stage_seconds or time.monotonic() - study_started > budget['total_wall_seconds']:
                    raise TimeoutError('Frozen wall-time budget exceeded')
                check_global_budget(budget, study_started, output)
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError('Study stage failed; see preserved local stage log')
            check_global_budget(budget, study_started, output)
            if time.monotonic() - started > stage_seconds:
                raise TimeoutError('Frozen stage wall-time budget exceeded')
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    return {'wall_seconds': time.monotonic()-started, 'rss_peak_sampled_bytes': peak,
            'log_sha256': digest(log), 'returncode': child.returncode}


def study(args):
    protocol_path = CONFIG / 'protocol.json'
    protocol = read(protocol_path)
    validate_bindings(protocol['frozen_inputs'])
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise ValueError('Tracked code/config changes must be committed before execution')
    tracked = subprocess.check_output(['git', 'ls-files', 'qa_lab', 'scripts/prepare_confirmation_data.py', str(CONFIG),
                                      'configs/confirmation-exclusions-v1.json'], cwd=ROOT, text=True).splitlines()
    required = {'qa_lab/confirmation_study.py', 'qa_lab/confirmation_training.py', 'qa_lab/confirmation_scoring.py',
                'scripts/prepare_confirmation_data.py', 'configs/confirmation-exclusions-v1.json', str(protocol_path),
                str(CONFIG/'student.json'), str(CONFIG/'teacher.json')}
    if not required <= set(tracked):
        raise ValueError('All execution code and protocol must be tracked and committed before model execution')
    bindings = committed_files(tracked)
    cohort_manifest = read(args.cohort / 'manifest.json')
    validate_cohort_manifest(cohort_manifest, protocol)
    # Hash labels without parsing them. No scoring occurs until all nine predictions are locked.
    inputs_sha, labels_sha = digest(args.cohort / 'inputs.jsonl'), digest(args.cohort / 'sealed-labels.jsonl')
    if cohort_manifest['inputs_sha256'] != inputs_sha or cohort_manifest['labels_sha256'] != labels_sha:
        raise ValueError('Prepared cohort integrity failed')
    rows = checked_inputs(args.cohort / 'inputs.jsonl', inputs_sha)
    if len(rows) != protocol['data']['questions']:
        raise ValueError('Prepared cohort count differs from frozen protocol')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'logs').mkdir()
    state = {'status': 'running', 'started_utc': now(), 'protocol_sha256': digest(protocol_path),
             'protocol_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'cohort_manifest_sha256': digest(args.cohort / 'manifest.json'), 'inputs_sha256': inputs_sha,
             'labels_sha256': labels_sha, 'frozen_files': bindings, 'stages': []}
    write(args.output / 'study.json', state)
    started = time.monotonic()
    def run(name, argv, seconds):
        state['active_stage'] = name
        write(args.output / 'study.json', state)
        print('Starting ' + name, flush=True)
        cost = bounded_process(argv, args.output / 'logs' / (name + '.log'), protocol['budget'], seconds, started, args.output)
        state['stages'].append({'name': name, **cost})
        write(args.output / 'study.json', state)
        print('Completed ' + name, flush=True)
    try:
        cache = args.output / 'teacher-cache'
        run('teacher-cache', ['-m', 'qa_lab.logits_distillation', 'cache', '--artifact', protocol['training']['artifact'],
            '--data-dir', protocol['training']['source'], '--objective', 'configs/logits-distillation-v2/objective.json',
            '--student-config', str(CONFIG/'student.json'), '--teacher-config', str(CONFIG/'teacher.json'),
            '--model-cache-dir', str(args.cache_dir), '--output', str(cache)], 900)
        adapters, training_receipts = {}, {}
        for key in protocol['order']['training']:
            arm, seed = key.split(':')
            folder = args.output / 'training' / key.replace(':', '-')
            run('train-' + key.replace(':', '-'), ['-m', 'qa_lab.confirmation_training', '--arm', arm, '--seed', seed,
                '--student-config', str(CONFIG/'student.json'), '--protocol', str(protocol_path), '--cache', str(cache),
                '--artifact', protocol['training']['artifact'], '--cache-dir', str(args.cache_dir), '--output', str(folder)],
                protocol['budget']['training_run_seconds'])
            training = read(folder / 'training.json')
            if training['status'] != 'complete':
                raise ValueError('Training did not complete')
            adapters[key] = {'files': adapter_files(folder), 'training_sha256': digest(folder/'training.json')}
            training_receipts[key] = training
        verify_paired_training(training_receipts, protocol)
        validate_bindings(bindings)
        execution_lock = {'created_utc': now(), 'status': 'all_adapters_locked_before_predictions', 'adapters': adapters,
                          'inputs_sha256': inputs_sha, 'labels_sha256': labels_sha, 'frozen_files': bindings,
                          'protocol_commit': state['protocol_commit'], 'protocol_sha256': digest(protocol_path)}
        write(args.output / 'execution-lock.json', execution_lock)
        for key in protocol['order']['training']:
            suffix = key.replace(':', '-')
            run('predict-' + suffix, ['-m', 'qa_lab.confirmation_study', 'predict', '--inputs', str(args.cohort/'inputs.jsonl'),
                '--execution-lock', str(args.output/'execution-lock.json'), '--key', key,
                '--adapter', str(args.output/'training'/suffix), '--cache-dir', str(args.cache_dir),
                '--output', str(args.output/'predictions'/suffix)], protocol['budget']['prediction_run_seconds'])
        prediction_lock = lock_predictions(args.output, args.cohort, protocol['order']['training'])
        write(args.output/'prediction-lock.json', prediction_lock)
        validate_bindings(bindings)
        if digest(args.cohort/'sealed-labels.jsonl') != labels_sha:
            raise ValueError('Labels changed before unsealing')
        # First label parsing in the study execution: all predictions already exist and are hashed.
        from .confirmation_scoring import score_study
        predictions = {arm: {} for arm in protocol['arms']}
        for key in protocol['order']['training']:
            arm, seed = key.split(':')
            predictions[arm][seed] = jsonl(args.output/'predictions'/key.replace(':', '-')/'predictions.jsonl')
        scores = score_study(jsonl(args.cohort/'sealed-labels.jsonl'), predictions, protocol)
        write(args.output/'scores.json', scores)
        check_global_budget(protocol['budget'], started, args.output)
        state.update(status='complete', scores_sha256=digest(args.output/'scores.json'),
                     prediction_lock_sha256=digest(args.output/'prediction-lock.json'))
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        state.update(finished_utc=now(), wall_seconds=time.monotonic()-started)
        write(args.output/'study.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--cohort', type=Path, required=True)
    run.add_argument('--cache-dir', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('predict')
    for name in ('inputs', 'execution-lock', 'adapter', 'cache-dir', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--key', required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'run':
        study(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
