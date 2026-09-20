"""Offline integrity and metric recomputation for the frozen DRCD transfer run."""
import json
from pathlib import Path
from external_drcd import verify
from verify_quality_release import verify_manifest

root=Path(__file__).resolve().parents[1]/'reports/external-drcd-20260920'
if __name__=='__main__':
    verify_manifest(root)
    print(json.dumps(verify(root),indent=2))
