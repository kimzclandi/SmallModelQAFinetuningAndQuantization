"""Independent smoke: custom timed decoder must match Transformers greedy generate."""
import json
from pathlib import Path
import torch
from qa_lab.inference import QAInput,load,messages,generate
from qa_lab.train import supervised_tokens
from qa_lab.common import read_jsonl
cfg=json.loads(Path('configs/baseline.json').read_text())
torch.set_num_threads(8)
tok,model=load(cfg)
for row in read_jsonl('data/complexity-v1/train.jsonl')[:2]:
 item=QAInput(row['context'],row['question'])
 ours=generate(item,cfg,tok,model)
 prompt=tok.apply_chat_template(messages(item,cfg),tokenize=False,add_generation_prompt=True)
 x=tok(prompt,return_tensors='pt').to(cfg['device'])
 with torch.inference_mode():
  official=model.generate(**x,do_sample=False,max_new_tokens=cfg['max_new_tokens'],temperature=None,top_p=None,top_k=None)
 expected=official[0,x['input_ids'].shape[1]:].tolist()
 assert ours['token_ids']==expected,(ours['token_ids'],expected)
 full,labels=supervised_tokens(row,tok,cfg,2048)
 assert len(full)==len(labels) and labels[0]==-100
 print(json.dumps({'id':row['id'],'matching_tokens':len(expected),'mask_verified':True}))
