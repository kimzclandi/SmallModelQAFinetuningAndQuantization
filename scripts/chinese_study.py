"""Freeze and run a new-source Chinese check; only FP16 and previously selected Q8."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,shutil,sys,subprocess,time,importlib.metadata
from qa_lab.common import sha,read_jsonl,write_json,write_jsonl,source_hashes
from qa_lab.chinese import select,evaluate_zh
from qa_lab.inference import QAInput,messages,MemoryMonitor,hardware_chip
from qa_lab.mlx_experiment import generate,model_files
R=Path('reports/chinese-v5');C=Path('configs/chinese-v5');D=Path('data/chinese-v5')


def check():
    reg=json.loads((R/'preregistration.json').read_text())
    for name,h in reg['files'].items():assert sha(name)==h
    assert sha('reports/quantization-v4/summary.json')==reg['selected_candidate_evidence_sha256']
    return json.loads((C/'protocol.json').read_text())


def prepare(rawdir):
    p=check();src=rawdir/'cmrc2018_dev.json'
    for name,meta in p['source']['files'].items():assert sha(rawdir/Path(name).name)==meta['sha256']
    cfg=json.loads((C/'inference.json').read_text())
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('.cache/mlx/student-fp16',local_files_only=True)
    def count(r):return len(tok.encode(tok.apply_chat_template(messages(QAInput(r['context'],r['question']),cfg),tokenize=False,add_generation_prompt=True),add_special_tokens=False))
    old=sum([read_jsonl(f'data/complexity-v1/{s}.jsonl') for s in ['train','dev','test']],[])+read_jsonl('data/cross-article-v3/test.jsonl')
    rows,audit=select(json.loads(src.read_text()),old,count,p['n'])
    D.mkdir(exist_ok=False);write_jsonl(D/'external.jsonl',rows);shutil.copyfile(rawdir/'LICENCE',D/'SOURCE_LICENSE.txt')
    write_json(D/'manifest.json',dict(source=p['source'],audit=audit,license='CC BY-SA 4.0',changes='Deterministic subset, one per article, input budget and duplicate exclusions; reformatted, raw content/reference strings retained.',data_sha256=sha(D/'external.jsonl'),old_data_sha256={f:sha(f) for f in ['data/complexity-v1/train.jsonl','data/complexity-v1/dev.jsonl','data/complexity-v1/test.jsonl','data/cross-article-v3/test.jsonl']}))
    write_json(R/'model-freeze.json',{v:model_files(Path('.cache/mlx')/f'student-{v}') for v in p['variants']})
    print('Frozen',len(rows),'rows from',audit['selected_articles'],'articles; raw rows',audit['raw_n'])


def infer(variant,output=None,model_path=None,limit=None):
    import mlx.core as mx
    from mlx_lm import load
    p=check();cfg=json.loads((C/'inference.json').read_text());rows=read_jsonl(D/'external.jsonl')
    freeze=json.loads((R/'freeze.json').read_text());assert freeze['data_sha256']==sha(D/'external.jsonl') and freeze['model_freeze_sha256']==sha(R/'model-freeze.json')
    assert variant in p['variants']
    if limit is not None:
        if output is None or limit<1:raise ValueError('A positive limit requires isolated reproduction output')
        rows=rows[:limit]
    folder=output if output is not None else R/variant;folder.mkdir(parents=True,exist_ok=False)
    modelpath=model_path or Path('.cache/mlx')/f'student-{variant}';files=model_files(modelpath)
    frozen=json.loads((R/'model-freeze.json').read_text())[variant]
    if output is None:assert files==frozen
    else:
        # Fresh conversion timestamps/cards may differ; numerical weights and tokenizer must not.
        for name,meta in frozen.items():
            if name not in ('conversion-provenance.json','README.md','LICENSE'):
                if files.get(name)!=meta:raise ValueError('Reproduction numerical/tokenizer file differs: '+name)
    monitor=MemoryMonitor('mlx');monitor.thread.start();t0=time.perf_counter()
    meta=dict(status='running',variant=variant,config=cfg,source_sha256=source_hashes(),hardware=hardware_chip(),data_sha256=sha(D/'external.jsonl'),model_files=files,freeze_sha256=sha(R/'freeze.json'),started_utc=datetime.now(timezone.utc).isoformat(),packages={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','psutil']},kv_quantized=False)
    write_json(folder/'run.json',meta)
    meta.update(reproduction=output is not None,limit=limit,n=len(rows),model_path=str(modelpath),original_file_set_match=files==frozen)
    try:
        mx.random.seed(cfg['seed']);model,tok=load(str(modelpath));mx.eval(model.parameters());mx.synchronize()
        for _ in range(2):generate(QAInput('队列采用先进先出顺序。','队列采用什么顺序？'),cfg,tok,model)
        pred=[]
        with (folder/'predictions.jsonl').open('x') as f:
            for n,r in enumerate(rows):
                q=generate(QAInput(r['context'],r['question']),cfg,tok,model)
                assert q['input_tokens']==r['input_tokens']
                q.update(id=r['id'],split='external');pred.append(q);f.write(json.dumps(q,ensure_ascii=False)+'\n');f.flush();print(variant,n+1,len(rows),flush=True)
        metrics,scored=evaluate_zh(rows,pred);write_json(folder/'metrics.json',metrics);write_jsonl(folder/'scored.jsonl',scored)
        rm={r['id']:r for r in rows};pm={r['id']:r for r in pred}
        write_jsonl(folder/'failures.jsonl',[dict(**s,question=rm[s['id']]['question'],answers=rm[s['id']]['answers'],prediction=pm[s['id']]['prediction']) for s in scored if not s['strict_em']])
        meta['status']='complete'
    except BaseException as e:meta.update(status='failed',error=repr(e));raise
    finally:
        monitor.stop.set();monitor.thread.join();meta.update(elapsed_seconds=time.perf_counter()-t0,memory=dict(rss_peak_sampled_bytes=monitor.rss,mlx_peak_active_bytes=mx.get_peak_memory(),mlx_cache_bytes=mx.get_cache_memory()),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/'run.json',meta);write_json(folder/'manifest.json',{f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name!='manifest.json'})


def run():
    p=check();freeze=json.loads((R/'freeze.json').read_text())
    assert subprocess.check_output(['git','show',freeze['commit']+':'+str(D/'external.jsonl')])==(D/'external.jsonl').read_bytes()
    shutil.copytree('qa_lab',R/'source/qa_lab',ignore=shutil.ignore_patterns('__pycache__'));(R/'source/scripts').mkdir();shutil.copyfile(__file__,R/'source/scripts/chinese_study.py')
    for v in p['variants']:
        args=['scripts/chinese_study.py','infer','--variant',v]
        with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':['.venv-mlx/bin/python',*args]})+'\n')
        with (R/f'{v}.log').open('x') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
    summarize()


def summarize(*, read_only=False):
    p=check();rows=read_jsonl(D/'external.jsonl');out={};scores={}
    for v in p['variants']:out[v],scores[v]=evaluate_zh(rows,read_jsonl(R/v/'predictions.jsonl'))
    a,b=out['fp16'],out['q8'];checks={k:b[k]+tol+1e-12>=a[k] for k,tol in [('strict_em',p['gate']['max_strict_em_drop']),('char_lcs_f1',p['gate']['max_char_lcs_f1_drop']),('format_valid',p['gate']['max_format_drop'])]}
    aa={x['id']:x['strict_em'] for x in scores['fp16']};bb={x['id']:x['strict_em'] for x in scores['q8']}
    s=dict(metrics=out,checks=checks,pass_all=all(checks.values()),fixes=[i for i in aa if aa[i]==0 and bb[i]==1],regressions=[i for i in aa if aa[i]==1 and bb[i]==0],scope=p['scope'])
    path=R/'summary.json'
    if path.exists():assert json.loads(path.read_text())==s
    elif read_only:raise FileNotFoundError('Missing frozen summary: '+str(path))
    else:write_json(path,s)
    print(json.dumps(s,ensure_ascii=False,indent=2));return s


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run','infer','summarize','reproduce']);ap.add_argument('--variant',choices=['fp16','q8']);ap.add_argument('--raw-dir',type=Path,default=Path('work/cmrc2018-source'));ap.add_argument('--model',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--limit',type=int);a=ap.parse_args()
    if a.mode=='prepare':prepare(a.raw_dir)
    elif a.mode=='run':run()
    elif a.mode=='infer':infer(a.variant)
    elif a.mode=='reproduce':
        if a.output is None or a.variant is None:ap.error('reproduce requires --output and --variant')
        infer(a.variant,a.output,a.model,a.limit)
    else:summarize()
