"""Download a pinned public model; no API inference or implicit credential use."""
import os
os.environ.setdefault('HF_HOME',str(__import__('pathlib').Path('.cache/huggingface').resolve()))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
cfg=json.loads(Path('configs/baseline.json').read_text())
print(snapshot_download(cfg['model_id'],revision=cfg['revision'],
    allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE','README.md']))
