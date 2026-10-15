"""Reject invalid B1 evidence and enforce every predeclared veto."""
import copy

import pytest

from scripts.v3_backbone_precision import ARMS, arm_order
from scripts.v3_summarize_backbone_precision import audit_matrix, audit_samples, summarize


def evidence():
    manifest = dict(args=dict(train_seeds=[10, 11, 12], eval_seeds=[0, 1, 2], limit=0),
                    image_ids=list(range(750)), full_images=750, warmup=20, measure=200,
                    category_names=['common', 'rare'])
    rows, timing, reference, precision, states = [], [], [], [], []
    for s in (10, 11, 12):
        states.append(dict(train_seed=s, state_unchanged=True, all_float_state_fp32=True,
                           initial_sha256='a' * 64, final_sha256='a' * 64))
        for e in (0, 1, 2):
            for arm in ARMS:
                row = dict(train_seed=s, eval_seed=e, precision=arm, images=750, full_dataset=True,
                           finite_prediction_images=750, measured_images=200, peak_allocated_gb=0.6,
                           AP=67 + s / 10 + e / 100, AP50=88., AP75=73., APs=50., APm=55., APl=70.,
                           latency_p50_ms=30., latency_p95_ms=35., **{'AP-common': 80., 'AP-rare': 45.})
                rows.append(row)
                timing.append(dict(train_seed=s, eval_seed=e, precision=arm, measured_images=200,
                                   latency_p50_ms=30. if arm == 'fp32' else 24., latency_p95_ms=35.))
                if arm == 'fp32':
                    reference.append(dict(row, arm='uniform'))
        for image_id in (0, 1):
            for arm in ARMS:
                names = {'fpn_p' + str(n) for n in range(2, 7)} | {'stem_conv'}
                dtypes = {n: 'torch.float16' if arm == 'fp16' else 'torch.float32' for n in names}
                dtypes.update({n: 'torch.float32' for n in {'head_p' + str(n) for n in range(2, 6)} | {
                    'backbone_input', 'head_boxes', 'head_logits', 'head_coords', 'final_boxes', 'final_scores', 'noise'}})
                precision.append(dict(train_seed=s, image_id=image_id, precision=arm, dtypes=dtypes, finite=True,
                                      fp32_feature_boundary=True, shapes=dict(head_coords=[4, 1, 300, 4], head_logits=[4, 1, 300, 2]),
                                      feature_errors={n: dict(max_abs_error=0.01, relative_l2=0.001) for n in names}))
    return dict(rows=rows, timing=timing, manifest=manifest, completed=dict(evaluations=18, timing_cells=18, checkpoints=3),
                reference=reference, precision=precision, states=states)


def test_complete_protocol_uses_interleaved_timing_for_speed_gate():
    data = evidence()
    audit, metrics = audit_matrix(**data)
    paired, summary, _, gate = summarize(data['rows'], data['timing'], data['manifest'], metrics)
    assert audit['baseline_max_abs_AP_difference'] == 0
    assert summary['fp32_AP'] == pytest.approx(68.11)
    assert summary['p50_reduction_percent'] == pytest.approx(20)
    assert len(paired) == 3 and gate['passed_development_screen']


def test_original_fp64_schedule_buffers_do_not_invalidate_unchanged_state():
    data = evidence()
    for state in data['states']:
        state['all_float_state_fp32'] = False
    audit_matrix(**data)  # File audit additionally compares source checkpoint dtypes.


@pytest.mark.parametrize('corruption', ['subset', 'duplicate', 'missing_timing', 'reference', 'nonfinite',
                                       'state', 'head_precision', 'no_fp16', 'depth', 'feature_error', 'missing_error', 'missing_audit'])
def test_corrupt_evidence_cannot_be_reported(corruption):
    data = evidence()
    if corruption == 'subset':
        data['rows'][0]['full_dataset'] = False
    elif corruption == 'duplicate':
        data['rows'][-1] = copy.deepcopy(data['rows'][0])
    elif corruption == 'missing_timing':
        data['timing'].pop()
    elif corruption == 'reference':
        data['reference'][0]['AP'] += 0.01
    elif corruption == 'nonfinite':
        data['rows'][0]['AP'] = float('nan')
    elif corruption == 'state':
        data['states'][0]['final_sha256'] = 'b' * 64
    elif corruption == 'head_precision':
        data['precision'][1]['dtypes']['head_p2'] = 'torch.float16'
    elif corruption == 'no_fp16':
        data['precision'][1]['dtypes']['stem_conv'] = 'torch.float32'
    elif corruption == 'depth':
        data['precision'][0]['shapes']['head_coords'][0] = 6
    elif corruption == 'feature_error':
        data['precision'][1]['feature_errors']['stem_conv']['relative_l2'] = float('inf')
    elif corruption == 'missing_error':
        data['precision'][1]['feature_errors']['stem_conv'] = {}
    else:
        data['precision'].pop()
    with pytest.raises(ValueError):
        audit_matrix(**data)


@pytest.mark.parametrize('veto', ['mean_AP', 'mean_AP75', 'each_checkpoint_AP', 'worst_category',
                                 'mean_p50', 'each_checkpoint_p50', 'mean_p95'])
def test_each_predeclared_veto_is_enforced(veto):
    data = evidence()
    _, metrics = audit_matrix(**data)
    for row in data['rows']:
        if row['precision'] != 'fp16':
            continue
        if veto == 'mean_AP':
            row['AP'] -= 0.31
        elif veto == 'mean_AP75':
            row['AP75'] -= 0.31
        elif veto == 'each_checkpoint_AP' and row['train_seed'] == 10:
            row['AP'] -= 0.51
        elif veto == 'worst_category':
            row['AP-rare'] -= 1.01
    for row in data['timing']:
        if row['precision'] == 'fp16':
            if veto == 'mean_p50':
                row['latency_p50_ms'] = 26.
            elif veto == 'each_checkpoint_p50':
                row['latency_p50_ms'] = 28. if row['train_seed'] == 10 else 20.
            elif veto == 'mean_p95':
                row['latency_p95_ms'] = 37.
    *_, gate = summarize(data['rows'], data['timing'], data['manifest'], metrics)
    assert gate['checks'][veto] is False
    assert sum(gate['checks'].values()) == 6 and not gate['passed_development_screen']


def test_interleaved_audit_rejects_order_window_and_quantile_corruption():
    ids = list(range(750))
    samples = [dict(index=i, image_id=i, precision=arm, position=p, wall_ms=30., finite=True)
               for i in range(20, 220) for p, arm in enumerate(arm_order(10, 0, i))]
    rows = [dict(precision=arm, latency_p50_ms=30., latency_p95_ms=30.) for arm in ARMS]
    audit_samples(samples, ids, 10, 0, rows, interleaved=True)
    for corruption in ('order', 'window', 'quantile', 'finite'):
        changed = copy.deepcopy(samples)
        if corruption == 'order':
            changed[0]['position'] = 1
        elif corruption == 'window':
            changed[0]['image_id'] = 19
        elif corruption == 'quantile':
            for row in changed:
                row['wall_ms'] = 29.
        else:
            changed[0]['finite'] = False
        with pytest.raises(ValueError):
            audit_samples(changed, ids, 10, 0, rows, interleaved=True)
