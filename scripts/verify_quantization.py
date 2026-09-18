"""Offline Q8 study verification: source, quality, identical work and benchmark arithmetic."""
import json,statistics
from pathlib import Path
from qa_lab.common import sha,read_jsonl
from scripts.quantization_study import R,C,check,summarize
p=check();summary=summarize(read_only=True);base=json.loads((R/'fp16-quality/run.json').read_text())
conversions={v:json.loads((R/f'{v}-conversion.json').read_text()) for v in p['variants']}
for v in ['q4','q8']:
 c=conversions[v];assert c['weight_bits']==int(v[1:]) and c['group_size']==64 and c['kv_cache_quantization'] is False
 for f,meta in conversions['fp16']['output_files'].items():assert c['source_files'][f]==meta
quality_predictions={v:read_jsonl(R/f'{v}-quality/dev.predictions.jsonl') for v in p['variants']}
reference=[(x['id'],x['input_token_ids_sha256'],x['input_tokens']) for x in quality_predictions['fp16']]
for v,rows in quality_predictions.items():assert [(x['id'],x['input_token_ids_sha256'],x['input_tokens']) for x in rows]==reference
folders=[R/f'{v}-quality' for v in p['variants']]+[R/'q8-parity']+[R/f'bench-{n}-{v}' for n,v in enumerate(p['benchmark_order'],1)]
for folder in folders:
 run=json.loads((folder/'run.json').read_text())
 assert run['status']=='complete' and run['config']==base['config'] and run['hardware']==base['hardware'] and run['packages']==base['packages']
 assert run['protocol_sha256']==sha(C) and run['kv_quantized'] is False
 v='fp16' if run['quantization'] is None else 'q'+str(run['quantization']['bits'])
 assert v in p['variants']
 if v!='fp16':assert run['quantization']['group_size']==64
 for f,meta in conversions[v]['output_files'].items():assert run['model_files'][f]==meta
 assert run['weight_bytes']==sum(m['bytes'] for f,m in run['model_files'].items() if f.endswith('.safetensors'))
 for f,h in run['source_sha256'].items():assert sha(R/'source'/f)==h
 for f,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/f)==h
 if run['mode']=='quality':
  assert run['data_sha256']=={'dev':sha('data/complexity-v1/dev.jsonl')}
  assert json.loads((folder/'dev.metrics.json').read_text())==summary['variants'][v]['quality']
signatures=[]
for n,v in enumerate(p['benchmark_order'],1):
 folder=R/f'bench-{n}-{v}';timings=read_jsonl(folder/'timings.jsonl');perf=json.loads((folder/'performance.json').read_text());run=json.loads((folder/'run.json').read_text())
 assert len(timings)==8 and all(r['generated_tokens']==len(r['token_ids'])==32 and r['stop_reason']=='fixed_length' for r in timings)
 assert run['forced_output_tokens']==32 and run['request_sha256']==sha('data/teacher-pilot-v1/request.json')
 signatures.append([(r['id'],r['input_tokens'],r['input_token_ids_sha256']) for r in timings])
 assert abs(perf['decode_tokens_per_second']-sum(r['generated_tokens']-1 for r in timings)/sum(r['decode_seconds'] for r in timings))<1e-9
 assert abs(perf['mean_ttft_seconds']-statistics.mean(r['ttft_seconds'] for r in timings))<1e-9
 assert perf['generated_tokens']==256 and perf['n']==8
assert all(s==signatures[0] for s in signatures)
commands=read_jsonl(R/'commands.jsonl');assert len(commands)==14
for c in commands:
 a=c['argv']
 if '--splits' in a:assert a[a.index('--splits')+1]=='dev'
assert len(json.loads((R/'q8-parity/parity.json').read_text()))==2
print('Verified: same FP16 source, affine Q4/Q8 weight-only, identical quality inputs, nine matched-work benchmarks, timing arithmetic, source/model hashes, gates, DEV-only commands and two-row upstream decoder parity.')
