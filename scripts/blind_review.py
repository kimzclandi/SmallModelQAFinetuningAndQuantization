"""Prepare a blind review bundle or analyze declared reviewer annotations."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qa_lab.blind_review import prepare,analyze
from qa_lab.common import sha,read_jsonl,write_json,write_jsonl
from scripts.verify_synthetic_qc import verify_run
from scripts.report_output import create_output_directory

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('prepare');a.add_argument('--run',type=Path,action='append',required=True);a.add_argument('--per-stratum',type=int,default=3);a.add_argument('--seed',type=int,default=20260922);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('analyze');a.add_argument('--bundle',type=Path,required=True);a.add_argument('--annotations',type=Path,action='append',required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args();inputs={}
    if args.mode=='prepare':
        population=[]
        if len({r.name for r in args.run})!=len(args.run):raise ValueError('Distinct batch names required')
        for run in args.run:
            verify_run(run);inputs[run.name]=sha(run/'manifest.json')
            population.extend((run.name,row) for row in read_jsonl(run/'records.jsonl'))
        result=prepare(population,args.per_stratum,args.seed)
        output=create_output_directory(args.output)
        write_json(output/'coordinator-only.json',result)
        blind=output/'reviewer';blind.mkdir()
        write_jsonl(blind/'items.jsonl',result['items'])
        write_json(blind/'annotations.json',result['template'])
        (blind/'INSTRUCTIONS.md').write_text('''# Blind semantic review\n\nRead context, question and raw response only. Judge whether the response answers the question using the passage. NO_ANSWER is correct only if the passage cannot answer. Minor formatting or a noncanonical but equivalent span does not by itself make an answer semantically wrong. Do not consult reference answers, QC decisions, model names or the coordinator file.\n\nFor each blind_id fill judgment: correct / incorrect / uncertain. Fill error: none (correct only), wrong_answer, unsupported, false_abstention, missed_abstention, ambiguous_question or uncertain (uncertain judgments). Explain incorrect/uncertain in note. Do not remove rows or edit IDs. Leave uncertain when the question admits unresolved readings.\n\nUse a pseudonymous reviewer_alias, not personal identity. Change origin from pending_human to human_attested only after you personally complete the review. A model-generated review must use synthetic_fixture and cannot count as human review. Keep one independent copy per reviewer; do not inspect another review before finishing. No labels have been filled automatically.\n''')
    else:
        manifest=json.loads((args.bundle.parent/'manifest.json').read_text())
        if manifest.get(args.bundle.name)!=sha(args.bundle):raise ValueError('Coordinator file differs from prepared manifest')
        result=analyze(json.loads(args.bundle.read_text()),[json.loads(f.read_text()) for f in args.annotations]);inputs={'bundle':sha(args.bundle),**{f'annotation-{i}':sha(f) for i,f in enumerate(args.annotations)}}
        output=create_output_directory(args.output);write_json(output/'analysis.json',result)
    write_json(output/'receipt.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,code_sha256={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'qa_lab/blind_review.py']},execution='Review preparation/analysis only; no human annotation performed by this command.'))
    write_json(output/'manifest.json',{str(f.relative_to(output)):sha(f) for f in sorted(output.rglob('*')) if f.is_file()})
    print(json.dumps({'mode':args.mode,'status':result.get('status','pending_human'),'sample_n':len(result.get('items',[]))}))
if __name__=='__main__':main()
