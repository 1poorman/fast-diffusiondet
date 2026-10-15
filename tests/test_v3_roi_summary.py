import copy

import pytest

from scripts.v3_roi_checks import roi_events
from scripts.v3_roi_known_counts import ARMS, arm_order
from scripts.v3_summarize_roi import audit_matrix, audit_samples, summarize


def evidence():
    manifest = dict(args=dict(train_seeds=[10, 11, 12], eval_seeds=[0, 1, 2], limit=0),
                    image_ids=list(range(750)), full_images=750, warmup=20, measure=200,
                    profile_images=10, arms=list(ARMS), category_names=['common', 'rare'])
    rows, timing, reference, parity, states, profiles = [], [], [], [], [], []
    for s in (10, 11, 12):
        states.append(dict(train_seed=s, initial_sha256='a' * 64, final_sha256='a' * 64, state_unchanged=True))
        for arm in ARMS:
            syncs = 280 if arm == 'original' else 160
            profiles.append(dict(train_seed=s, arm=arm, images=10, roi_calls=40, prediction_bit_exact=True,
                                 synchronizations={'cudaStreamSynchronize': syncs}, synchronization_count=syncs,
                                 repeat_interleave_calls=80 if arm == 'original' else 0))
        for e in (0, 1, 2):
            parity.append(dict(train_seed=s, eval_seed=e, full_prediction_images=750,
                               interleaved_prediction_calls=440, bit_exact=True))
            for arm in ARMS:
                row = dict(train_seed=s, eval_seed=e, arm=arm, images=750, full_dataset=True, finite_prediction_images=750,
                           measured_images=200, AP=67 + s / 10 + e / 100, AP50=88., AP75=73., APs=50., APm=55., APl=70.,
                           latency_p50_ms=30., latency_p95_ms=35., peak_allocated_gb=0.6,
                           **{'AP-common': 80., 'AP-rare': 45.})
                rows.append(row)
                timing.append(dict(train_seed=s, eval_seed=e, arm=arm, measured_images=200,
                                   latency_p50_ms=30. if arm == 'original' else 27., latency_p95_ms=35.))
                if arm == 'original':
                    reference.append(dict(row, arm='uniform'))
    return dict(rows=rows, timing=timing, manifest=manifest,
                completed=dict(evaluations=18, timing_cells=18, checkpoints=3, profiles=6), reference=reference,
                parity=parity, states=states, profiles=profiles,
                gradient=dict(device='cuda', passed=True, format_bit_exact=True, output_bit_exact=True,
                              gradient_rtol=2e-5, gradient_atol=2e-5, gradient_max_abs=1e-8))


def test_valid_execution_screen_uses_interleaved_timing_and_mechanism():
    data = evidence()
    audit, metrics = audit_matrix(**data)
    paired, summary, classes, gate = summarize(data['rows'], data['timing'], data['profiles'], data['manifest'], metrics)
    assert audit['full_bit_exact_image_pairs'] == 6750
    assert summary['p50_reduction_percent'] == pytest.approx(10)
    assert summary['delta_AP'] == 0 and summary['known_counts_roi_sync_per_image'] == 16
    assert len(paired) == 3 and len(classes) == 2 and gate['passed_development_screen']


@pytest.mark.parametrize('corruption', ['duplicate', 'subset', 'timing', 'reference', 'AP', 'nonfinite', 'parity',
                                       'state', 'profiles', 'profile_counts', 'gradient', 'gradient_tolerance'])
def test_invalid_experiment_cannot_be_reported(corruption):
    data = evidence()
    if corruption == 'duplicate':
        data['rows'][-1] = copy.deepcopy(data['rows'][0])
    elif corruption == 'subset':
        data['rows'][0]['full_dataset'] = False
    elif corruption == 'timing':
        data['timing'][0]['measured_images'] = 199
    elif corruption == 'reference':
        data['reference'][0]['AP'] += 0.01
    elif corruption == 'AP':
        data['rows'][1]['AP-rare'] += 0.01
    elif corruption == 'nonfinite':
        data['rows'][1]['AP'] = float('nan')
    elif corruption == 'parity':
        data['parity'][0]['bit_exact'] = False
    elif corruption == 'state':
        data['states'][0]['final_sha256'] = 'b' * 64
    elif corruption == 'profiles':
        data['profiles'].pop()
    elif corruption == 'profile_counts':
        data['profiles'][0]['synchronization_count'] += 1
    elif corruption == 'gradient':
        data['gradient']['passed'] = False
    else:
        data['gradient']['gradient_atol'] = 0.1
    with pytest.raises(ValueError):
        audit_matrix(**data)


@pytest.mark.parametrize('veto', ['mean_p50', 'each_checkpoint_p50', 'mean_p95', 'synchronization_mechanism'])
def test_predeclared_speed_and_mechanism_gates_are_independent(veto):
    data = evidence()
    _, metrics = audit_matrix(**data)
    for row in data['timing']:
        if row['arm'] != 'known_counts':
            continue
        if veto == 'mean_p50':
            row['latency_p50_ms'] = 28.6
        elif veto == 'each_checkpoint_p50':
            row['latency_p50_ms'] = 29.2 if row['train_seed'] == 10 else 27.
        elif veto == 'mean_p95':
            row['latency_p95_ms'] = 37.
    if veto == 'synchronization_mechanism':
        data['profiles'][1]['repeat_interleave_calls'] = 1
    *_, gate = summarize(data['rows'], data['timing'], data['profiles'], data['manifest'], metrics)
    assert gate['checks'][veto] is False and sum(gate['checks'].values()) == 3
    assert not gate['passed_development_screen']


def test_raw_interleaved_audit_rejects_reordered_images_and_wrong_quantiles():
    ids = list(range(750))
    samples = [dict(index=i, image_id=i, arm=arm, position=pos, wall_ms=30., bit_exact=True)
               for i in range(20, 220) for pos, arm in enumerate(arm_order(10, 0, i))]
    rows = [dict(arm=arm, latency_p50_ms=30., latency_p95_ms=30.) for arm in ARMS]
    audit_samples(samples, ids, 10, 0, rows, True)
    for corruption in ('order', 'quantile', 'parity', 'nonfinite'):
        changed = copy.deepcopy(samples)
        if corruption == 'order':
            changed[0]['position'] = 1
        elif corruption == 'quantile':
            for row in changed:
                row['wall_ms'] = 29.
        elif corruption == 'parity':
            changed[0]['bit_exact'] = False
        else:
            changed[0]['wall_ms'] = float('nan')
        with pytest.raises(ValueError):
            audit_samples(changed, ids, 10, 0, rows, True)


def test_roi_operator_attribution_excludes_other_threads_and_outer_syncs():
    common = dict(ph='X', pid=1, tid=2)
    events = [dict(common, cat='user_annotation', name='R1/roi_pool', ts=10, dur=10),
              dict(common, cat='cpu_op', name='aten::repeat_interleave', ts=11, dur=8),
              dict(common, cat='cuda_runtime', name='cudaStreamSynchronize', ts=12, dur=1),
              dict(common, cat='cuda_runtime', name='cudaStreamSynchronize', ts=12, dur=1, tid=3),
              dict(common, cat='cuda_runtime', name='cudaDeviceSynchronize', ts=21, dur=1)]
    assert roi_events(events) == dict(roi_calls=1, synchronizations={'cudaStreamSynchronize': 1},
                                      synchronization_count=1, repeat_interleave_calls=1)
