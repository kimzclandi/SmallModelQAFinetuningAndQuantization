"""Free local open-weight teacher download; no hosted inference or API key."""
import os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path('.cache/huggingface').resolve()))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import json
from huggingface_hub import snapshot_download
c=json.loads(Path('configs/teacher-local-fp32-v1.json').read_text())
print(snapshot_download(c['model_id'],revision=c['revision'],allow_patterns=['*.json','*.safetensors','*.txt','LICENSE','README.md']))
