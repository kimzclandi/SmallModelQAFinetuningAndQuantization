"""Freeze dev selection before test, or summarize all completed preregistered experiments."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from qa_lab.common import read_jsonl,sha,write_json,write_jsonl
from qa_lab.metrics import evaluate
ROOT=Path('reports/closure-v1')
DEV={'baseline':'reports/baseline-v1','gold24':'reports/gold-sft-pilot-v1/eval-dev','distilled24':str(ROOT/'distilled-dev'),'repaired48':str(ROOT/'repaired-dev')}

def checked(path,split):
    folder=Path(path);rows=read_jsonl(f'data/complexity-v1/{split}.jsonl');pred=read_jsonl(folder/f'{split}.predictions.jsonl')
    actual,scored=evaluate(rows,pred);saved=json.loads((folder/f'{split}.metrics.json').read_text())
    assert all(saved[k]==v for k,v in actual.items())
    for name,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/name)==h
    return actual,scored


def paired(before,after,split):
    bm,bs=checked(before,split);am,ass=checked(after,split);b={r['id']:r for r in bs};a={r['id']:r for r in ass}
    return {'fixes':[i for i in b if b[i]['em']==0 and a[i]['em']==1],
            'regressions':[i for i in b if b[i]['em']==1 and a[i]['em']==0],
            'before':bm,'after':am}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['select','report']);a=p.parse_args()
    if a.mode=='select':
        path=ROOT/'selection.json'
        if path.exists():raise FileExistsError(path)
        results={name:checked(folder,'dev')[0] for name,folder in DEV.items()}
        passing=[n for n,m in results.items() if n!='baseline' and m['overall']['em']>19/74 and m['answerable']['em']>=19/33]
        winner=sorted(passing,key=lambda n:(-results[n]['overall']['em'],-results[n]['overall']['f1'],n))[0] if passing else 'baseline'
        write_json(path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'selected':winner,'passing':passing,'dev_metrics':results,
                         'candidate_run_sha256':{n:sha(Path(f)/'run.json') for n,f in DEV.items()},
                         'policy':'Dev only; selection is never changed based on final test. Test includes all preregistered candidates, even rejected ones.'})
        print(json.dumps({'selected':winner,'passing':passing},indent=2));return
    selection=json.loads((ROOT/'selection.json').read_text())
    for n,f in DEV.items():assert sha(Path(f)/'run.json')==selection['candidate_run_sha256'][n]
    test_paths={'baseline':'reports/baseline-v1','gold24':str(ROOT/'gold24-test'),'distilled24':str(ROOT/'distilled24-test'),'repaired48':str(ROOT/'repaired48-test')}
    results={name:checked(folder,'test')[0] for name,folder in test_paths.items()}
    changes={split:{'distill_vs_original':paired(DEV['baseline'] if split=='dev' else test_paths['baseline'],DEV['distilled24'] if split=='dev' else test_paths['distilled24'],split),
                    'distill_vs_gold':paired(DEV['gold24'] if split=='dev' else test_paths['gold24'],DEV['distilled24'] if split=='dev' else test_paths['distilled24'],split),
                    'repair_vs_gold':paired(DEV['gold24'] if split=='dev' else test_paths['gold24'],DEV['repaired48'] if split=='dev' else test_paths['repaired48'],split)} for split in ['dev','test']}
    quant={v:{s:checked(ROOT/f'mlx-{v}-quality',s)[0] for s in ['dev','test']} for v in ['fp16','q4']}
    timing={};token_signatures=[]
    for i,v in enumerate(['fp16','q4','q4','fp16'],1):
        folder=ROOT/f'mlx-bench-{i}-{v}';rows=read_jsonl(folder/'timings.jsonl')
        assert all(r['generated_tokens']==32 and r['stop_reason']=='fixed_length' for r in rows)
        token_signatures.append([(r['id'],r['input_tokens'],r['input_token_ids_sha256']) for r in rows])
        run=json.loads((folder/'run.json').read_text());perf=json.loads((folder/'performance.json').read_text())
        for name,h in json.loads((folder/'manifest.json').read_text()).items():assert sha(folder/name)==h
        timing[f'{i}-{v}']={'performance':perf,'memory':run['memory'],'weight_bytes':run['weight_bytes']}
    assert all(x==token_signatures[0] for x in token_signatures)
    summary={'selected_before_test':selection['selected'],'dev':selection['dev_metrics'],'test':results,'paired':changes,'quantization_quality':quant,'quantization_fixed_work':timing}
    if (ROOT/'summary.json').exists():
        assert json.loads((ROOT/'summary.json').read_text())==summary, 'Existing report differs; create a new version'
    else:
        write_json(ROOT/'summary.json',summary)
    print(json.dumps({'selected_before_test':selection['selected'],'test':{k:v['overall'] for k,v in results.items()},'quantization':{k:v['test']['overall'] for k,v in quant.items()}},indent=2))

if __name__=='__main__':main()
