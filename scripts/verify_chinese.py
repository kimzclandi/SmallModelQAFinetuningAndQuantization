"""Offline checks for the independently frozen Chinese subset, without model access."""
import json
from pathlib import Path
from qa_lab.common import sha,read_jsonl
from qa_lab.chinese import evaluate_zh,near,normalize_zh
from scripts.chinese_study import R,C,D,check,summarize
p=check();summary=summarize();rows=read_jsonl(D/'external.jsonl');manifest=json.loads((D/'manifest.json').read_text())
freeze=json.loads((R/'freeze.json').read_text());models=json.loads((R/'model-freeze.json').read_text())
assert len(rows)==p['n']==96 and len({normalize_zh(r['source_title']) for r in rows})==96
assert sha(D/'external.jsonl')==manifest['data_sha256']==freeze['data_sha256']
assert sha(R/'model-freeze.json')==freeze['model_freeze_sha256']
assert sha(D/'SOURCE_LICENSE.txt')==p['source']['files']['LICENCE']['sha256']
assert manifest['source']==p['source']
old=[]
for path,h in manifest['old_data_sha256'].items():assert sha(path)==h;old+=read_jsonl(path)
assert not {r['id'] for r in rows}&{r['id'] for r in old}
for n,r in enumerate(rows):
 assert not r['is_impossible'] and r['answers'] and 0<r['input_tokens']<=2048
 assert all(a.strip() and a in r['context'] for a in r['answers'])
 assert not any(near(r,o) for o in old+rows[:n])
base=None;signatures=[]
for variant in p['variants']:
 folder=R/variant;run=json.loads((folder/'run.json').read_text())
 assert run['status']=='complete' and run['config']==json.loads((C/'inference.json').read_text())
 assert run['data_sha256']==sha(D/'external.jsonl') and run['freeze_sha256']==sha(R/'freeze.json')
 assert run['started_utc']>freeze['frozen_utc'] and not run['kv_quantized']
 assert run['model_files']==models[variant]
 previous=json.loads(Path(f'reports/quantization-v4/{variant}-quality/run.json').read_text())
 assert run['model_files']==previous['model_files'],'This is not the previously selected candidate/reference'
 if base is None:base=run
 else:assert run['hardware']==base['hardware'] and run['packages']==base['packages']
 for f,h in run['source_sha256'].items():assert sha(R/'source'/f)==h
 for f,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/f)==h
 pred=read_jsonl(folder/'predictions.jsonl');actual,scored=evaluate_zh(rows,pred)
 assert actual==json.loads((folder/'metrics.json').read_text()) and scored==read_jsonl(folder/'scored.jsonl')
 assert [r['id'] for r in pred]==[r['id'] for r in rows]
 assert [r['input_tokens'] for r in pred]==[r['input_tokens'] for r in rows]
 assert all(len(x['token_ids'])==x['generated_tokens']<=48 for x in pred)
 signatures.append([(r['id'],r['input_token_ids_sha256'],r['prompt_sha256']) for r in pred])
assert signatures[0]==signatures[1]
commands=read_jsonl(R/'commands.jsonl');assert len(commands)==2
for c,v in zip(commands,p['variants']):assert c['argv']==['.venv-mlx/bin/python','scripts/chinese_study.py','infer','--variant',v] and c['utc']>freeze['frozen_utc']
print('Verified: source license/data freeze, 96 article-disjoint rows, old/new near-duplicate isolation, unchanged v4 candidate weights, matched Chinese inputs, source/prediction/metric hashes, two post-freeze runs and non-official metric labels.')
smoke=R/'reproduction-smoke';run=json.loads((smoke/'run.json').read_text())
assert run['status']=='complete' and run['reproduction'] is True and run['limit']==2
assert run['model_files']==models['q8'] and run['n']==2
actual,_=evaluate_zh(rows[:2],read_jsonl(smoke/'predictions.jsonl'));assert actual==json.loads((smoke/'metrics.json').read_text())
for f,h in json.loads((smoke/'manifest.json').read_text()).items():assert sha(smoke/f)==h
validation=json.loads((R/'reproduction-validation.json').read_text())
assert sha(R/'source/scripts/chinese_study-reproduction.py')==validation['script_sha256']
print('Verified isolated two-row reproduction smoke; excluded from primary96-row comparison.')
