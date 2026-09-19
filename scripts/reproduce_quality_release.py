"""Small actual CPU smoke or full fixed MPS training from published inputs."""
import argparse,json,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha

def run(stage,output):
    repo=Path(__file__).resolve().parents[1];release=repo/'reports/quality-study-20260920'
    output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((release/'supervised-07/protocol.json').read_text())
    if stage=='smoke':
        from qa_lab.inference import load,generate,QAInput
        import torch
        torch.set_num_threads(4);cfg=dict(cfg['student'],device='cpu');tok,model=load(cfg);rows=read_jsonl(release/'fresh-05/dev.jsonl')[:2]
        predictions=[dict(id=r['id'],**generate(QAInput(r['context'],r['question']),cfg,tok,model)) for r in rows]
        write_jsonl(output/'predictions.jsonl',predictions);write_json(output/'receipt.json',dict(status='complete',n=2,config=cfg,scope='Actual CPU inference only; not quality validation or training'))
    else:
        from supervised_quality_control import run as train
        dest=output/'supervised-07';dest.mkdir()
        fresh=output/'fresh-05';fresh.mkdir();(fresh/'train.jsonl').write_bytes((release/'fresh-05/train.jsonl').read_bytes());shutil.copytree(release/'fresh-05/teacher-train',fresh/'teacher-train')
        for n in [g+'.jsonl' for g in cfg['groups']]+['dev.jsonl']:(dest/n).write_bytes((release/'supervised-07'/n).read_bytes())
        cfg.update(created_utc=datetime.now(timezone.utc).isoformat(),reproduces_protocol_sha256=sha(release/'supervised-07/protocol.json'),code_hashes={n:sha(repo/'scripts'/n) for n in cfg['code_hashes']})
        write_json(dest/'protocol.json',cfg);write_json(dest/'manifest.json',{p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()});train(output)
        from verify_supervised_quality import verify
        write_json(output/'verification.json',verify(output))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.stage,a.output)
