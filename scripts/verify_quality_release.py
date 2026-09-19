"""Portable saved-evidence checks; weights excluded, no model loading or inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import sha
from verify_quality_screening import verify as screening
from verify_3b_screen import verify as larger
from verify_supervised_quality import verify as supervised
from final_quality_holdout import verify as holdout

def verify(root):
    manifest=json.loads((root/'release-manifest.json').read_text())
    expected=set(manifest['files'])
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='release-manifest.json'}
    if expected!=actual:raise ValueError('Release file coverage')
    for n,h in manifest['files'].items():
        if sha(root/n)!=h:raise ValueError('Release hash mismatch: '+n)
    a=screening(root);b=larger(root);c=supervised(root,require_adapters=False);d=holdout(root)
    return dict(status='pass',scope='Saved records only; no adapter weights, training or inference verification',screening=a,teacher_3b=b,supervised=c,holdout=d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'reports/quality-study-20260920');a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
