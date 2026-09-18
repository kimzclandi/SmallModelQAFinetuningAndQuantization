"""Offline audit of fixed data coverage study; no model or network."""
from collections import Counter
import json
from pathlib import Path
from qa_lab.common import read_jsonl,sha
from qa_lab.data import related,jaccard
from qa_lab.train_artifact import verified_rows
from scripts.coverage_study import R,C,D,H,protocol,question_related,summarize
p=protocol();summary=summarize(read_only=True)
m=json.loads((H/'manifest.json').read_text());rows=read_jsonl(H/'test.jsonl')
assert sha(H/'test.jsonl')==m['files']['test.jsonl'] and len(rows)==64
old=[]
for split,h in m['old_split_hashes'].items():
    assert sha(D/f'{split}.jsonl')==h
    old+=read_jsonl(D/f'{split}.jsonl')
assert not {r['id'] for r in rows}&{r['id'] for r in old}
assert {r['source_title'] for r in rows}==set(p['holdout_titles'])
assert Counter((r['source_title'],r['is_impossible']) for r in rows)==Counter({(t,l):16 for t in p['holdout_titles'] for l in [False,True]})
assert all(n==1 for n in Counter((r['context_id'],r['is_impossible']) for r in rows).values())
cfg=json.loads(Path('configs/data.json').read_text())
for i,r in enumerate(rows):
    assert all(not related(r,o,cfg) for o in old)
    for b in rows[:i]:
        assert not question_related(r,b,cfg)
        assert r['context_id']==b['context_id'] or jaccard(r['context'],b['context'])<cfg['context_jaccard']
base=json.loads(Path('configs/baseline.json').read_text())
freeze=json.loads((R/'freeze.json').read_text())
assert sha(R/'selection.json')==freeze['selection_sha256'] and sha(H/'test.jsonl')==freeze['holdout_sha256']
selection=json.loads((R/'selection.json').read_text())
passing=[]
for arm in p['arms']:
    artifact=Path('data/teacher-study-v2-gold' if arm=='gold24' else 'data/coverage-v3-gold242')
    train,_=verified_rows(artifact,D);expected={r['id'] for r in train}
    assert len(train)==(24 if arm=='gold24' else 242)
    passed=[]
    for seed in p['seeds']:
        name=f'{arm}-{seed}';folder=R/name;t=json.loads((folder/'training.json').read_text())
        assert t['status']=='complete' and t['config']==json.loads((C/f'train-{seed}.json').read_text()) and t['base']==base
        assert t['artifact_sha256']==sha(artifact/'manifest.json')
        assert len(t['steps'])==242 and t['total_supervised_tokens']==sum(x['supervised_tokens'] for x in t['steps'])
        count=Counter(x['id'] for x in t['steps']);assert set(count)==expected
        assert set(count.values())<=({10,11} if arm=='gold24' else {1})
        assert sha(folder/'adapter-manifest.json')==selection['adapter_manifests'][name]
        for f,h in t['source_sha256'].items():assert sha(R/'source'/f)==h
        metric=selection['metrics'][name]
        passed.append(metric['overall']['em']>p['dev_gate']['overall_gt'] and metric['answerable']['em']>=p['dev_gate']['answerable_ge'])
        for split,sub,data in [('dev','dev',D/'dev.jsonl'),('test','external',H/'test.jsonl')]:
            folder2=folder/sub;run=json.loads((folder2/'run.json').read_text())
            assert run['status']=='complete' and run['config']==base and run['limit'] is None and run['data_sha256']=={split:sha(data)}
            assert run['adapter']==json.loads((folder/'adapter-manifest.json').read_text())
            for f,h in run['source_sha256'].items():assert sha(R/'source'/f)==h
            for f,h in json.loads((folder2/'manifest.json').read_text()).items():assert sha(folder2/f)==h
            saved=json.loads((folder2/f'{split}.metrics.json').read_text())
            from qa_lab.metrics import evaluate
            actual,_=evaluate(read_jsonl(data),read_jsonl(folder2/f'{split}.predictions.jsonl'))
            for k,v in actual.items():assert saved[k]==v
            if split=='dev':assert actual==selection['metrics'][name]
    if all(passed):passing.append(arm)
assert passing==selection['passing_arms']
folder=R/'baseline/external';run=json.loads((folder/'run.json').read_text())
assert run['status']=='complete' and run['adapter'] is None and run['config']==base and run['data_sha256']=={'test':sha(H/'test.jsonl')}
for f,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/f)==h
commands=read_jsonl(R/'commands.jsonl');assert len(commands)==19
for c in commands:
    a=c['argv']
    if '--splits' in a and a[a.index('--splits')+1]=='test':assert a[a.index('--data-dir')+1]==str(H)
print('Verified: article isolation/near duplicates, gold TRAIN-only coverage, 242-step matched configs, three seeds, pre-holdout freeze, all candidate metrics and adapter hashes, no old-test inference commands.')
for f,h in run['source_sha256'].items():assert sha(R/'source'/f)==h
external_commands=[c for c in commands if '--data-dir' in c['argv']]
assert len(external_commands)==7
assert all(c['utc']>freeze['frozen_utc'] for c in external_commands)
assert selection['holdout_sha256']==sha(H/'test.jsonl')
assert m['source_sha256']==cfg['sha256']
print('Verified: all seven new-article inference commands occurred after the recorded selection freeze.')
