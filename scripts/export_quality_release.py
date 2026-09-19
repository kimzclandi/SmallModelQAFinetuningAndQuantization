"""Copy public evidence byte-for-byte; omit weights, private cache paths and local delivery wrappers."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import sha,write_json

ROUNDS=['calibration-04','score-calibration-04','fresh-05','teacher-3b-06','supervised-07','holdout-08']
OMIT={'delivery-manifest.json','round-four-manifest.json','download-receipt.json','prior-evidence-checks.json'}

def export(source,out):
    out.mkdir(parents=True,exist_ok=False);copied={}
    for sub in ROUNDS:
        for p in sorted((source/sub).rglob('*')):
            if not p.is_file() or p.suffix not in ['.json','.jsonl'] or p.name in OMIT or 'source' in p.relative_to(source).parts:continue
            raw=p.read_text()
            if '/Users/' in raw or '/home/' in raw:raise ValueError('Private path in '+str(p))
            rel=p.relative_to(source);dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);copied[str(rel)]=sha(p)
    write_json(out/'release-manifest.json',dict(scope='Exact copies of public data and records; original local files retained. Excludes adapters/model weights, executable environment paths, local manifests, source duplicates and console logs. Offline checks do not verify excluded weights.',source_sha256=copied,files=copied))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args();export(a.source,a.out)
