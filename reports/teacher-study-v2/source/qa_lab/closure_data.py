"""Train-only local teacher responses and one preregistered data repair."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import random
import time
from .common import read_jsonl,write_json,write_jsonl,sha,digest,source_hashes
from .teacher_io import training_rows
from .metrics import evaluate


def pilot_rows(folder,selection):
    rows={r['id']:r for r in training_rows(folder)}
    pick=json.loads(selection.read_text())
    if pick['source_sha256']!=sha(folder/'train.jsonl'):
        raise ValueError('Selection source mismatch')
    if len(set(pick['ids']))!=len(pick['ids']) or not set(pick['ids'])<=set(rows):
        raise ValueError('Invalid training IDs')
    return [rows[i] for i in pick['ids']]


def freeze_artifact(out,rows,manifest):
    out.mkdir(parents=True,exist_ok=False)
    write_jsonl(out/'train.jsonl',rows)
    manifest.update(n=len(rows),train_sha256=sha(out/'train.jsonl'),created_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/'manifest.json',manifest)


def build_repair(folder,selection,out):
    pilot=pilot_rows(folder,selection)
    ids={r['id'] for r in pilot}
    pool=[r for r in training_rows(folder) if r['id'] not in ids and not r['is_impossible']]
    added=sorted(pool,key=lambda r:digest('repair-v1:'+r['id']))[:24]
    if len(added)!=24:raise ValueError('Not enough answerable training rows')
    rows=[dict(id=r['id'],context=r['context'],question=r['question'],split='train',
               target='NO_ANSWER' if r['is_impossible'] else r['answers'][0]) for r in pilot+added]
    freeze_artifact(out,rows,dict(method='gold_data_repair',source_sha256=sha(folder/'train.jsonl'),
        pilot_selection_sha256=sha(selection),added_ids=[r['id'] for r in added],
        answerable_count=36,unanswerable_count=12,
        policy='Added 24 previously unused answerable TRAIN rows by fixed hash; no dev/test text or labels used.'))


def collect_teacher(folder,selection,config,out):
    from .inference import load,generate,QAInput,MemoryMonitor,hardware_chip
    import torch,psutil,platform,importlib.metadata
    rows=pilot_rows(folder,selection);cfg=json.loads(config.read_text())
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);torch.set_num_threads(8)
    meta=dict(status='running',method='local_public_teacher',config=cfg,source_sha256=source_hashes(),
              train_source_sha256=sha(folder/'train.jsonl'),selection_sha256=sha(selection),
              started_utc=datetime.now(timezone.utc).isoformat(),hardware=hardware_chip(),
              total_memory_bytes=psutil.virtual_memory().total,python=platform.python_version(),
              packages={n:importlib.metadata.version(n) for n in ['torch','transformers']},
              filtering='No gold-based filtering; nonempty/noncapped required at artifact build.',
              teacher_license='Apache-2.0, official Qwen2.5-1.5B-Instruct model card')
    write_json(out/'run.json',meta)
    monitor=MemoryMonitor(cfg['device']);monitor.thread.start();t0=time.perf_counter()
    try:
        tok,model=load(cfg)
        def require_finite_logits(module,inputs,output):
            if not torch.isfinite(output.logits[:,-1,:]).all().item():
                raise FloatingPointError('Non-finite teacher logits; do not emit training labels')
        model.register_forward_hook(require_finite_logits)
        for _ in range(2):generate(QAInput('A queue follows first in, first out.','What order does a queue follow?'),cfg,tok,model,monitor)
        preds=[]
        with (out/'predictions.jsonl').open('x') as f:
            for n,r in enumerate(rows):
                result=generate(QAInput(r['context'],r['question']),cfg,tok,model,monitor)
                result.update(id=r['id'],split='train')
                preds.append(result);f.write(json.dumps(result)+'\n');f.flush()
                print(f'teacher {n+1}/{len(rows)}',flush=True)
        metrics,scored=evaluate(rows,preds)
        write_json(out/'train-agreement.json',metrics);write_jsonl(out/'scored.jsonl',scored)
        meta['status']='complete'
    except BaseException as e:
        meta.update(status='failed',error=repr(e));raise
    finally:
        monitor.stop.set();monitor.thread.join()
        meta.update(elapsed_seconds=time.perf_counter()-t0,memory=monitor.values(),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'run.json',meta)
        write_json(out/'manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})


def build_distilled(folder,selection,teacher,out,teacher_config=Path("configs/teacher-local-fp32-v1.json")):
    rows=pilot_rows(folder,selection)
    run=json.loads((teacher/'run.json').read_text())
    for name,h in json.loads((teacher/'manifest.json').read_text()).items():
        if sha(teacher/name)!=h:raise ValueError('Teacher artifacts changed')
    cfg=json.loads(teacher_config.read_text())
    if run['status']!='complete' or run['method']!='local_public_teacher' or run['config']!=cfg:
        raise ValueError('Only frozen local teacher evidence accepted; manual GPT responses are not authorized training data')
    if run['selection_sha256']!=sha(selection) or run['train_source_sha256']!=sha(folder/'train.jsonl'):
        raise ValueError('Teacher provenance mismatch')
    pred=read_jsonl(teacher/'predictions.jsonl');pm={p['id']:p for p in pred}
    if len(pm)!=len(pred) or set(pm)!={r['id'] for r in rows}:raise ValueError('Teacher coverage mismatch')
    out_rows=[]
    for r in rows:
        answer=pm[r['id']]['prediction'].strip()
        if not answer or pm[r['id']]['stop_reason']!='eos':
            raise ValueError('Empty/capped response: stop rather than silently drop/relabel')
        out_rows.append(dict(id=r['id'],context=r['context'],question=r['question'],split='train',target=answer))
    freeze_artifact(out,out_rows,dict(method='response_distillation',source_sha256=sha(folder/'train.jsonl'),
        pilot_selection_sha256=sha(selection),teacher_report=str(teacher),teacher_manifest_sha256=sha(teacher/'manifest.json'),
        teacher_config=cfg,filtering='All 24 responses retained, including gold/format mismatches. Strip boundary whitespace only. No logits or features.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['teacher','distill','repair'])
    p.add_argument('--data-dir',type=Path,default=Path('data/complexity-v1'))
    p.add_argument('--selection',type=Path,default=Path('data/teacher-pilot-v1/manifest.json'))
    p.add_argument('--teacher-config',type=Path,default=Path('configs/teacher-local-fp32-v1.json'))
    p.add_argument('--teacher-report',type=Path,default=Path('reports/closure-v1/teacher'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='teacher':collect_teacher(a.data_dir,a.selection,a.teacher_config,a.output)
    elif a.mode=='repair':build_repair(a.data_dir,a.selection,a.output)
    else:build_distilled(a.data_dir,a.selection,a.teacher_report,a.output,a.teacher_config)

if __name__=='__main__':main()
