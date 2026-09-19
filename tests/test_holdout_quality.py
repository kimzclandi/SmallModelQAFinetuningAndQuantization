import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from holdout_quality import check_protocol,paired_analysis
from qa_lab.common import sha

def test_protocol_change_rejected(tmp_path):
 (tmp_path/'protocol.json').write_text('{"frozen":true}')
 (tmp_path/'manifest.json').write_text(json.dumps({'protocol.json':sha(tmp_path/'protocol.json')}))
 assert check_protocol(tmp_path)['frozen']
 (tmp_path/'protocol.json').write_text('{"frozen":false}')
 with pytest.raises(ValueError,match='changed'):check_protocol(tmp_path)

def test_identical_predictions_have_zero_paired_difference():
 rows=[dict(strict_em=1.,char_lcs_f1=1.),dict(strict_em=0.,char_lcs_f1=.2)]
 scores={'baseline':rows}
 for group in ['semantic','random_structural','selected_gold']:
  for seed in [1,2,3]:scores[f'{group}-{seed}']=rows
 for r in paired_analysis(scores,[1,2,3]).values():
  assert r['delta']==0 and r['percentile_95']==[0,0]

def test_paired_direction_is_semantic_minus_control():
 scores={'baseline':[dict(strict_em=0.,char_lcs_f1=0.)]*2}
 for group in ['semantic','random_structural','selected_gold']:
  for seed in [1,2,3]:scores[f'{group}-{seed}']=[dict(strict_em=float(group=='semantic'),char_lcs_f1=float(group=='semantic'))]*2
 assert paired_analysis(scores,[1,2,3])['random_structural:strict_em']['delta']==1.
