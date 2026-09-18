import copy,json
from pathlib import Path
import pytest
from qa_lab.mlx_experiment import convert_models
from scripts.quantization_study import gate


@pytest.mark.parametrize('bits',[2,16])
def test_unsupported_width_fails_before_loading_model(tmp_path,bits):
 with pytest.raises(ValueError,match='widths'):convert_models(tmp_path/'missing',tmp_path/'out',True,bits)


def test_fast_small_model_cannot_bypass_quality_gate():
 limits=json.loads(Path('configs/quantization-v4.json').read_text())['gate']
 # Synthetic unit-test values, not model results.
 base={'quality':{'overall':{'em':.5,'f1':.6},'answerable':{'em':.7}},'weight_bytes':100,
       'performance':{'median_rss_bytes':100,'median_decode_tps':100,'median_ttft_ms':10}}
 candidate=copy.deepcopy(base);candidate['weight_bytes']=50;candidate['performance']['median_rss_bytes']=50
 assert gate(base,candidate,limits)['pass_all']
 candidate['quality']['overall']['em']=.1
 assert not gate(base,candidate,limits)['pass_all']
 assert not gate(base,candidate,limits)['checks']['overall_em']
