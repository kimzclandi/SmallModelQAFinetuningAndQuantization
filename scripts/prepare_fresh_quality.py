"""Build disjoint answerable-only data without using predictions or test scores."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qa_lab.common import read_jsonl,write_json,write_jsonl,sha,digest
from qa_lab.chinese import normalize_zh,near
from qa_lab.inference import QAInput,messages

def main(a):
 from transformers import AutoTokenizer
 a.out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.parent/'protocol.json').read_text())['student'];cfg.update(prompt_version='zh-extract-or-abstain-v1',warmup='none')
 tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
 raw=json.loads(a.source.read_text());oldraw=json.loads(a.olddev.read_text());old=[];titles=set()
 for art in oldraw['data']:
  titles.add(normalize_zh(art['title']))
  for p in art['paragraphs']:
   for q in p['qas']:old.append(dict(context=p['context'],question=q['question']))
 prior=[]
 for split in ['train','dev','holdout']:prior+=read_jsonl(a.parent/(split+'.jsonl'))
 old+=prior;titles.update(normalize_zh(r['source_title']) for r in prior)
 historical={r['id'] for n in ['dev.jsonl','holdout.jsonl'] for r in read_jsonl(a.engine/n)}
 for art in raw['data']:
  if any(q['id'] in historical for p in art['paragraphs'] for q in p['qas']):
   titles.add(normalize_zh(art['title']))
   for p in art['paragraphs']:
    for q in p['qas']:
     if q['id'] in historical:old.append(dict(context=p['context'],question=q['question']))
 pool=[]
 for art in raw['data']:
  title=normalize_zh(art['title'])
  if title in titles:continue
  for p in art['paragraphs']:
   for q in p['qas']:
    if not q['answers'] or not all(x['text'].strip() and normalize_zh(x['text']) and p['context'][x['answer_start']:x['answer_start']+len(x['text'])]==x['text'] for x in q['answers']):continue
    r=dict(id=q['id'],context=p['context'],question=q['question'],answers=list(dict.fromkeys(x['text'] for x in q['answers'])),is_impossible=False,source_title=art['title'],family_id=digest(title))
    n=len(tok.apply_chat_template(messages(QAInput(r['context'],r['question']),cfg),tokenize=True,add_generation_prompt=True))
    if n+max(len(tok.encode(x)) for x in r['answers'])+8<=768:r['prompt_tokens']=n;pool.append(r)
 selected=[];families=set();rejections=[]
 for r in sorted(pool,key=lambda r:digest('fresh-quality-20260920:'+r['id'])):
  if r['family_id'] in families:continue
  if any(near(r,o) for o in old+selected):rejections.append(r['id']);continue
  selected.append(r);families.add(r['family_id'])
  if len(selected)==352:break
 if len(selected)!=352:raise ValueError(f'Only {len(selected)} isolated articles available')
 for start,end,split in [(0,192,'train'),(192,256,'dev'),(256,352,'holdout')]:write_jsonl(a.out/(split+'.jsonl'),[dict(r,split=split) for r in selected[start:end]])
 write_json(a.out/'data_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),student=cfg,selection_seed='fresh-quality-20260920',counts={'train':192,'dev':64,'holdout':96},source_sha256=sha(a.source),official_dev_sha256=sha(a.olddev),prior_split_hashes={n:sha(a.parent/(n+'.jsonl')) for n in ['train','dev','holdout']},old_rows_compared=len(old),near_excluded=rejections,eligible_pool=len(pool),isolation='Whole article titles excluded; every selected candidate compared against all prior pilot rows, all official dev rows, historical retrieval queries and earlier selected rows using existing character ngram/question heuristic. Not a semantic proof.',scope='Answerable-only; same previously indexed CMRC source; not a new domain or pretraining-unseen'))
 write_json(a.out/'data_manifest.json',{n:sha(a.out/n) for n in ['train.jsonl','dev.jsonl','holdout.jsonl','data_protocol.json']})
 print('Prepared 192 train, 64 dev, 96 sealed holdout',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['out','parent','source','olddev','engine']:p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
