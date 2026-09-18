"""Offline provenance, paired-training and dev metric checks; no test reads/model access."""
from collections import Counter
import json
from pathlib import Path
from qa_lab.common import read_jsonl,sha
from qa_lab.train_artifact import verified_rows
from scripts.teacher_study import ROOT,CONFIG,DATA,SELECTION,check_registration,analyze

protocol=check_registration()
summary=analyze(read_only=True)
expected=json.loads(SELECTION.read_text())['ids']
artifacts={'gold':Path('data/teacher-study-v2-gold'),'original_teacher':Path('data/distilled-local-v1'),
           'prompted_teacher':Path('data/teacher-study-v2-prompted')}
for path in artifacts.values():
    rows,_=verified_rows(path,DATA)
    assert [r['id'] for r in rows]==expected
base=json.loads(Path('configs/baseline.json').read_text())
old_teacher=json.loads(Path('configs/teacher-local-fp32-v1.json').read_text())
new_teacher=json.loads((CONFIG/'teacher.json').read_text())
assert set(old_teacher)==set(new_teacher)
assert {k for k in old_teacher if old_teacher[k]!=new_teacher[k]}=={'system_prompt','prompt_version'}


def source(run):
    for name,h in run['source_sha256'].items():assert sha(ROOT/'source'/name)==h,(name,'source mismatch')


def report_manifest(folder):
    for name,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/name)==h


teacher=json.loads((ROOT/'teacher/run.json').read_text())
assert teacher['status']=='complete' and teacher['config']==new_teacher
source(teacher);report_manifest(ROOT/'teacher')
for seed in protocol['seeds']:
    orders=[]
    for arm in protocol['arms']:
        folder=ROOT/f'{arm}-{seed}'
        t=json.loads((folder/'training.json').read_text());v=json.loads((folder/'dev/run.json').read_text())
        assert t['status']==v['status']=='complete'
        assert t['base']==v['config']==base
        assert t['config']==json.loads((CONFIG/f'train-{seed}.json').read_text())
        assert t['config']['seed']==seed and t['config']['steps']==48
        assert t['artifact_sha256']==sha(artifacts[arm]/'manifest.json')
        assert t['artifact_path']==str(artifacts[arm])
        assert t['selected_ids']==expected
        assert Counter(x['id'] for x in t['steps'])==Counter({i:2 for i in expected})
        orders.append([x['id'] for x in t['steps']])
        assert t['total_supervised_tokens']==sum(x['supervised_tokens'] for x in t['steps'])
        assert t['steps']==json.loads((folder/'steps.json').read_text())
        assert v['limit'] is None and v['data_sha256']=={'dev':sha(DATA/'dev.jsonl')}
        assert v['adapter']['training.json']==sha(folder/'training.json')
        assert v['adapter']['steps.json']==sha(folder/'steps.json')
        for name,meta in json.loads((folder/'adapter-manifest.json').read_text()).items():assert v['adapter'][name]==meta['sha256']
        source(t);source(v);report_manifest(folder/'dev')
        saved=json.loads((folder/'dev/dev.metrics.json').read_text())
        for key,value in summary['runs'][f'{arm}-{seed}']['metrics'].items():assert saved[key]==value
    assert orders[0]==orders[1]==orders[2],'Paired sample order differs'
commands=read_jsonl(ROOT/'commands.jsonl')
assert len(commands)==19
for c in commands:
    assert not any('test'==arg or 'test.jsonl' in arg for arg in c['argv'])
    if 'qa_lab.inference' in c['argv']:
        i=c['argv'].index('--splits');assert c['argv'][i+1]=='dev'
print('Verified: fixed teacher-only prompt change, identical 24 TRAIN IDs, paired shuffle/48 steps, 3 seeds, 9 full DEV runs, target/source/adapter lineage, no new test commands.')
