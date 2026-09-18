"""Offline closure evidence validation: no model loading, network or new test inference."""
import json
import importlib.util
from pathlib import Path
from qa_lab.common import sha,read_jsonl
from qa_lab.metrics import evaluate
from qa_lab.train_artifact import verified_rows
root=Path('reports/closure-v1')
protocol=json.loads(Path('configs/closure-v1.json').read_text())
reg=json.loads((root/'preregistration.json').read_text())
assert sha('configs/closure-v1.json')==reg['config_sha256']
for path in [root/'teacher',root/'teacher-fp32']:
 run=json.loads((path/'run.json').read_text())
 for name,h in json.loads((path/'manifest.json').read_text()).items():assert sha(path/name)==h
 for file,h in run['source_sha256'].items():assert sha(path/'source'/file)==h,(path,file)
ids=json.loads(Path('data/teacher-pilot-v1/manifest.json').read_text())['ids']
train={r['id']:r for r in read_jsonl('data/complexity-v1/train.jsonl')}
for teacher in ['teacher','teacher-fp32']:
 folder=root/teacher;metrics,scored=evaluate([train[i] for i in ids],read_jsonl(folder/'predictions.jsonl'))
 assert metrics==json.loads((folder/'train-agreement.json').read_text())
for artifact in ['data/distilled-local-v1','data/repair-v1']:verified_rows(Path(artifact),Path('data/complexity-v1'))
for name in ['distilled','repaired']:
 run=json.loads((root/f'{name}-training/training.json').read_text())
 assert run['status']=='complete' and len(run['steps'])==48
 assert run['artifact_sha256']==sha(Path(run['artifact_path'])/'manifest.json')
 for file,h in run['source_sha256'].items():assert sha(root/'source'/file)==h
assert json.loads((root/'selection.json').read_text())['selected']=='baseline'
summary=json.loads((root/'summary.json').read_text())
paths={'baseline':Path('reports/baseline-v1'),'gold24':root/'gold24-test','distilled24':root/'distilled24-test','repaired48':root/'repaired48-test'}
for name,folder in paths.items():
 actual,_=evaluate(read_jsonl('data/complexity-v1/test.jsonl'),read_jsonl(folder/'test.predictions.jsonl'))
 assert actual==summary['test'][name]
for folder in [root/'distilled-dev',root/'repaired-dev',*list(paths.values())[1:],root/'mlx-parity',root/'mlx-fp16-quality',root/'mlx-q4-quality',*[root/f'mlx-bench-{i}-{v}' for i,v in enumerate(['fp16','q4','q4','fp16'],1)]]:
 run=json.loads((folder/'run.json').read_text());assert run['status']=='complete'
 for name,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/name)==h
 for file,h in run['source_sha256'].items():assert sha(root/'source'/file)==h,(folder,file)
 if 'mlx' in folder.name:
  assert not run['kv_quantized']
 if 'test' in run.get('data_sha256',{}):assert sha('data/complexity-v1/test.jsonl')==run['data_sha256']['test']
for variant in ['fp16','q4']:
 for split in ['dev','test']:
  actual,_=evaluate(read_jsonl(f'data/complexity-v1/{split}.jsonl'),read_jsonl(root/f'mlx-{variant}-quality/{split}.predictions.jsonl'))
  assert actual==summary['quantization_quality'][variant][split]
signatures=[]
for i,v in enumerate(['fp16','q4','q4','fp16'],1):
 folder=root/f'mlx-bench-{i}-{v}';pred=read_jsonl(folder/'timings.jsonl')
 assert len(pred)==8 and all(r['generated_tokens']==32 for r in pred)
 signatures.append([(r['id'],r['input_tokens'],r['input_token_ids_sha256']) for r in pred])
 perf=json.loads((folder/'performance.json').read_text())
 assert abs(perf['decode_tokens_per_second']-sum(r['generated_tokens']-1 for r in pred)/sum(r['decode_seconds'] for r in pred))<1e-9
assert all(s==signatures[0] for s in signatures)
print('Closure verified: teacher lineage, training-only boundaries, 48-step runs, frozen selection, quality metrics, identical-work quantization and source snapshots.')

fp=json.loads((root/'mlx-fp16-quality/run.json').read_text())
q=json.loads((root/'mlx-q4-quality/run.json').read_text())
assert fp['config']==q['config'] and fp['hardware']==q['hardware'] and fp['packages']==q['packages']
assert fp['quantization'] is None and q['quantization']['bits']==4 and q['quantization']['group_size']==64
fpconv=json.loads((root/'mlx-fp16-conversion.json').read_text())
qconv=json.loads((root/'mlx-q4-conversion.json').read_text())
for name,meta in fpconv['output_files'].items():
    assert qconv['source_files'][name]==meta, name
for name,meta in qconv['output_files'].items():
    assert q['model_files'][name]==meta,name
print('Quantization lineage verified: same FP16 source weights, same framework/config/hardware, weight-only Q4.')
