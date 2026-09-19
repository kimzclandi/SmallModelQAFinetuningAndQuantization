import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_teacher_cache import audit
from resumable_teacher import produce


def test_review_queue_has_no_gold_and_never_approves_failed_judge(tmp_path):
    rows = [dict(id='a', context='北京和上海', question='在哪里', answers=['北京'], is_impossible=False),
            dict(id='b', context='北京和上海', question='在哪里', answers=['上海'], is_impossible=False)]
    (tmp_path / 'train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    produce(rows, tmp_path / 'teacher-train', {}, {},
            lambda r: dict(id=r['id'], prediction='北京' if r['id']=='a' else '不存在'))
    out = tmp_path / 'audit'
    assert audit(tmp_path, out)['strict_em'] == .5
    queue = [json.loads(x) for x in (out / 'review-inputs.jsonl').read_text().splitlines()]
    assert all('answers' not in r and 'strict_em' not in r and not r['approved_for_training'] for r in queue)
    assert [r['status'] for r in queue] == ['review_required_judge_failed', 'quarantine_format']
    with pytest.raises(FileExistsError):
        audit(tmp_path, out)
