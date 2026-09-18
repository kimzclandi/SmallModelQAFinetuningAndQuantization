"""DEV-only Q4/Q8 screening and same-round balanced-order benchmark."""
from pathlib import Path
import argparse,json,os,shutil,statistics,subprocess,sys
from datetime import datetime,timezone
from qa_lab.common import sha,read_jsonl,write_json
from qa_lab.metrics import evaluate
from scripts.teacher_study import paired
R=Path('reports/quantization-v4');C=Path('configs/quantization-v4.json')
MODELS={v:Path('.cache/mlx')/f'student-{v}' for v in ['fp16','q4','q8']}


def check():
 for f,h in json.loads((R/'preregistration.json').read_text())['files'].items():assert sha(f)==h
 return json.loads(C.read_text())


def gate(base,candidate,limits):
 b,q=base['quality'],candidate['quality'];bp,qp=base['performance'],candidate['performance']
 checks=dict(overall_em=q['overall']['em']+limits['max_overall_em_drop']+1e-12>=b['overall']['em'],
 answerable_em=q['answerable']['em']+limits['max_answerable_em_drop']+1e-12>=b['answerable']['em'],
 f1=q['overall']['f1']+limits['max_f1_drop']+1e-12>=b['overall']['f1'],
 weight_bytes=candidate['weight_bytes']<=base['weight_bytes']*limits['max_weight_ratio'],
 rss=qp['median_rss_bytes']<=bp['median_rss_bytes']*limits['max_median_rss_ratio'],
 decode=qp['median_decode_tps']>=bp['median_decode_tps']*limits['min_median_decode_ratio'],
 ttft=qp['median_ttft_ms']<=bp['median_ttft_ms']*limits['max_median_ttft_ratio'])
 return dict(checks=checks,pass_all=all(checks.values()))


def execute(args,log):
 with (R/'commands.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':['.venv-mlx/bin/python',*args],'log':str(log)})+'\n')
 print('RUN',' '.join(args),flush=True)
 with log.open('x') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1'))


def run():
 p=check();shutil.copytree('qa_lab',R/'source/qa_lab',ignore=shutil.ignore_patterns('__pycache__'))
 (R/'source/scripts').mkdir();shutil.copyfile(__file__,R/'source/scripts/quantization_study.py')
 execute(['-m','qa_lab.mlx_experiment','convert','--model',str(MODELS['fp16']),'--quantize','--bits','8','--output',str(MODELS['q8'])],R/'conversion.log')
 for v in p['variants']:shutil.copyfile(MODELS[v]/'conversion-provenance.json',R/f'{v}-conversion.json')
 common=['--protocol',str(C)]
 execute(['-m','qa_lab.mlx_experiment','parity','--model',str(MODELS['q8']),'--output',str(R/'q8-parity'),*common],R/'parity.log')
 for v in p['variants']:execute(['-m','qa_lab.mlx_experiment','quality','--model',str(MODELS[v]),'--splits','dev','--output',str(R/f'{v}-quality'),*common],R/f'{v}.quality.log')
 for n,v in enumerate(p['benchmark_order'],1):execute(['-m','qa_lab.mlx_experiment','benchmark','--model',str(MODELS[v]),'--output',str(R/f'bench-{n}-{v}'),*common],R/f'bench-{n}-{v}.log')
 summarize()


def summarize(*, read_only=False):
 p=check();rows=read_jsonl('data/complexity-v1/dev.jsonl');result={'split':'dev','n':len(rows),'test_evaluated':False,'variants':{},'gate':{},'paired':{}};scored={}
 for v in p['variants']:
  q,scored[v]=evaluate(rows,read_jsonl(R/f'{v}-quality/dev.predictions.jsonl'))
  runs=[]
  for n,w in enumerate(p['benchmark_order'],1):
   if w!=v:continue
   folder=R/f'bench-{n}-{v}';meta=json.loads((folder/'run.json').read_text());perf=json.loads((folder/'performance.json').read_text())
   runs.append({'run':n,'decode_tps':perf['decode_tokens_per_second'],'ttft_ms':1000*perf['mean_ttft_seconds'],'rss_bytes':meta['memory']['rss_peak_sampled_bytes'],'mlx_peak_active_bytes':meta['memory']['mlx_peak_active_bytes'],'mlx_cache_bytes':meta['memory']['mlx_cache_bytes']})
  m=json.loads((R/f'{v}-quality/run.json').read_text())
  result['variants'][v]=dict(quality=q,weight_bytes=m['weight_bytes'],performance={'runs':runs,**{'median_'+key:statistics.median(x[key] for x in runs) for key in ['decode_tps','ttft_ms','rss_bytes','mlx_peak_active_bytes','mlx_cache_bytes']}})
 for v in ['q4','q8']:
  result['gate'][v]=gate(result['variants']['fp16'],result['variants'][v],p['gate'])
  result['paired'][v]=paired(scored['fp16'],scored[v])
 result['qualifying_local_candidates']=[v for v,g in result['gate'].items() if g['pass_all']]
 out=R/'summary.json'
 if out.exists():assert json.loads(out.read_text())==result
 elif read_only:raise FileNotFoundError('Missing frozen summary: '+str(out))
 else:write_json(out,result)
 print(json.dumps({'gate':result['gate'],'qualifying_local_candidates':result['qualifying_local_candidates']},indent=2))
 return result


if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('mode',choices=['run','summarize']);args=a.parse_args()
 run() if args.mode=='run' else summarize()
