import copy

import pytest

from scripts.v3_wheat_baseline import check_transfer, lr_factor, training_health


def test_schedule_preserves_full_budget_during_smoke():
    assert [lr_factor(s) for s in (0, 100, 1119, 1120, 1399)] == [.01, 1., 1., .1, .1]
    assert lr_factor(50) == pytest.approx(.505)


def test_amp_health_excludes_warmup_but_counts_actual_updates():
    rows = [dict(step=i, skipped=i < 5 or i == 100) for i in range(1400)]
    result = training_health(rows)
    assert result['passed']
    assert result['updates'] == 1394
    assert result['post_warmup_skip_rate'] == pytest.approx(1 / 1300)
    assert not training_health(rows[:10])['passed']


def test_amp_health_strict_one_percent_boundary():
    rows = [dict(step=i, skipped=100 <= i < 113) for i in range(1400)]
    assert not training_health(rows)['passed']
    with pytest.raises(ValueError):
        training_health(rows[1:])


def test_transfer_rejects_extra_missing_or_wrong_shape_keys():
    keys = [f'head.head_series.{i}.class_logits.{kind}' for i in range(6) for kind in ('weight', 'bias')]
    info = dict(loaded=537, skipped_shape=[(k, (), ()) for k in keys],
                skipped_missing_in_model=0, unexpected_in_model=[])
    check_transfer(info, range(549))
    for field, value in [('loaded', 536), ('skipped_shape', info['skipped_shape'][:-1]),
                         ('skipped_missing_in_model', 1), ('unexpected_in_model', ['extra'])]:
        broken = copy.deepcopy(info)
        broken[field] = value
        with pytest.raises(ValueError):
            check_transfer(broken, range(549))
