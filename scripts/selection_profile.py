"""Profile a verified QC run without changing decisions or frozen evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qa_lab.common import read_jsonl, write_json, sha
from qa_lab.selection_profile import profile
from scripts.verify_synthetic_qc import verify_run
from scripts.report_output import create_output_directory


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    verify_run(a.run)
    result=profile(read_jsonl(a.run/'records.jsonl'))
    out=create_output_directory(a.output)
    write_json(out/'profile.json',result)
    write_json(out/'receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_manifest_sha256=sha(a.run/'manifest.json'),
        code_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__), ROOT/'qa_lab/selection_profile.py']},
        execution='new descriptive analysis of saved candidates; no new model inference'))
    write_json(out/'manifest.json',{f.name:sha(f) for f in sorted(out.iterdir())})
    print(result)

if __name__=='__main__':main()
