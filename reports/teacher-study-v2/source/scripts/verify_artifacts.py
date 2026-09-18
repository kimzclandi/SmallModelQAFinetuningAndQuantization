"""Offline integrity and metric-recomputation gate; no model required."""
import json
from pathlib import Path
from qa_lab.common import sha,read_jsonl
from qa_lab.metrics import evaluate
for folder in [Path('data/complexity-v1'),Path('reports/baseline-v1')]:
 manifest=json.loads((folder/'manifest.json').read_text())
 for file,expected in manifest.get('files',manifest).items():
  assert sha(folder/file)==expected,file
run=json.loads(Path('reports/baseline-v1/run.json').read_text())
assert run['status']=='complete' and run['limit'] is None
for file,expected in run['source_sha256'].items():
 assert sha(Path('reports/baseline-v1/source')/file)==expected,file
for split in ['dev','test']:
 data=Path(f'data/complexity-v1/{split}.jsonl')
 assert sha(data)==run['data_sha256'][split]
 pred=read_jsonl(f'reports/baseline-v1/{split}.predictions.jsonl')
 actual,scored=evaluate(read_jsonl(data),pred)
 saved=json.loads(Path(f'reports/baseline-v1/{split}.metrics.json').read_text())
 assert all(saved[k]==v for k,v in actual.items()),split
 assert scored==read_jsonl(f'reports/baseline-v1/{split}.scored.jsonl')
 print(f'{split}: exact coverage, hashes, metrics and per-item scores verified ({len(pred)} predictions)')
