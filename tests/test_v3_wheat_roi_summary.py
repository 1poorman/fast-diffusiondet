import copy

import pytest

from scripts.v3_wheat_roi import ARMS, arm_order
from scripts.v3_summarize_wheat_roi import audit_matrix, audit_samples, summarize


@pytest.mark.parametrize('field,value', [('lr', .01), ('skipped', True), ('step_seed', -1),
    ('loss_total', float('nan')), ('grad_norm', float('inf')), ('scale_before', 0),
    ('scale_after', 512), ('image_ids', [9999] * 8)])
def test_training_log_rejects_changed_schedule_scaler_and_inputs(field, value):
    from scripts.v3_summarize_wheat_roi import audit_training_rows
    from scripts.v3_wheat_baseline import lr_factor
    from scripts.v3_head_sweep import image_seed
    rows = [dict(step=i, step_seed=image_seed(10, i), lr=1e-4*lr_factor(i),
                 loss_total=1., losses={'classification': 1.}, grad_norm=1.,
                 gradient_overflow=False, scale_before=256., scale_after=256.,
                 step_s=.7, skipped=False, input_digest='a'*64, image_ids=list(range(8)))
            for i in range(1400)]
    assert audit_training_rows(rows, 10, range(8))['updates'] == 1400
    rows[200][field] = value
    with pytest.raises(ValueError):
        audit_training_rows(rows, 10, range(8))


def evidence():
    manifest = dict(args=dict(train_seeds=[10, 11, 12], eval_seeds=[0, 1, 2], limit=0),
                    image_ids=list(range(123)), full_images=123, warmup=20, measure=103,
                    profile_images=10, arms=list(ARMS), category_names=['common', 'rare'])
    rows, timing, parity, states, profiles = [], [], [], [], []
    for s in (10, 11, 12):
        states.append(dict(train_seed=s, initial_sha256='a' * 64, final_sha256='a' * 64, state_unchanged=True))
        for arm in ARMS:
            syncs = 240 if arm == 'known_counts' else 0
            profiles.append(dict(train_seed=s, arm=arm, images=10, roi_calls=60, prediction_bit_exact=True,
                                 synchronizations={'cudaStreamSynchronize': syncs}, synchronization_count=syncs,
                                 repeat_interleave_calls=0, nonzero_calls=syncs,
                                 kernel_count=1880 if arm == 'known_counts' else 640,
                                 kernels={'example': 1880 if arm == 'known_counts' else 640}))
        for e in (0, 1, 2):
            parity.append(dict(train_seed=s, eval_seed=e, full_prediction_images=123,
                               interleaved_prediction_calls=246, interleaved_bit_exact=True, cross_arm_bit_exact_images=0))
            for arm in ARMS:
                row = dict(train_seed=s, eval_seed=e, arm=arm, images=123, full_dataset=True, finite_prediction_images=123,
                           measured_images=103, AP=67 + s / 10 + e / 100, AP50=88., AP75=73., APs=50., APm=55., APl=70.,
                           latency_p50_ms=30., latency_p95_ms=35., peak_allocated_gb=0.6,
                           **{'AP-common': 80., 'AP-rare': 45.})
                rows.append(row)
                timing.append(dict(train_seed=s, eval_seed=e, arm=arm, measured_images=103,
                                   latency_p50_ms=30. if arm == 'known_counts' else 27., latency_p95_ms=35.))
    return dict(rows=rows, timing=timing, manifest=manifest,
                completed=dict(evaluations=18, timing_cells=18, checkpoints=3, profiles=6),
                parity=parity, states=states, profiles=profiles,
                gradient=dict(passed=True, gradient_fallback_max_abs=1e-8,
                              features=[dict(case=n, rtol=1e-5, atol=2e-5, finite=True, max_abs_error=1e-7)
                                        for n in ('boundaries_ragged', 'empty', 'large_spatial', 'real_channels_300')],
                              native_fallbacks=['noncontiguous', 'autocast']))


def test_valid_execution_screen_uses_interleaved_timing_and_mechanism():
    data = evidence()
    audit, metrics = audit_matrix(**data)
    paired, summary, classes, gate = summarize(data['rows'], data['timing'], data['profiles'], data['manifest'], metrics)
    assert audit['cross_arm_bit_exact_images'] == 0
    assert summary['p50_reduction_percent'] == pytest.approx(10)
    assert summary['delta_AP'] == 0 and summary['gpu_dispatch_roi_sync_per_image'] == 0
    assert len(paired) == 3 and len(classes) == 2 and gate['passed_development_screen']


@pytest.mark.parametrize('corruption', ['duplicate', 'subset', 'timing', 'h4', 'classes', 'nonfinite', 'parity',
                                       'state', 'profiles', 'profile_counts', 'gradient', 'gradient_tolerance'])
def test_invalid_experiment_cannot_be_reported(corruption):
    data = evidence()
    if corruption == 'duplicate':
        data['rows'][-1] = copy.deepcopy(data['rows'][0])
    elif corruption == 'subset':
        data['rows'][0]['full_dataset'] = False
    elif corruption == 'timing':
        data['timing'][0]['measured_images'] = 199
    elif corruption == 'h4':
        data['profiles'][0]['roi_calls'] = 40
    elif corruption == 'classes':
        data['manifest']['category_names'].append('missing')
    elif corruption == 'nonfinite':
        data['rows'][1]['AP'] = float('nan')
    elif corruption == 'parity':
        data['parity'][0]['interleaved_bit_exact'] = False
    elif corruption == 'state':
        data['states'][0]['final_sha256'] = 'b' * 64
    elif corruption == 'profiles':
        data['profiles'].pop()
    elif corruption == 'profile_counts':
        data['profiles'][0]['synchronization_count'] += 1
    elif corruption == 'gradient':
        data['gradient']['passed'] = False
    else:
        data['gradient']['features'][0]['atol'] = 0.1
    with pytest.raises(ValueError):
        audit_matrix(**data)


def test_raw_interleaved_audit_rejects_reordered_images_and_wrong_quantiles():
    ids = list(range(123))
    samples = [dict(index=i, image_id=i, arm=arm, position=pos, wall_ms=30., bit_exact=True)
               for i in range(20, 123) for pos, arm in enumerate(arm_order(10, 0, i))]
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


@pytest.mark.parametrize('corruption', ['model', 'tolerance', 'late_jit', 'cache', 'digest_count', 'missing_checkpoint', 'missing_model'])
def test_prerequisite_and_jit_evidence_cannot_be_silently_replaced(tmp_path, corruption):
    import hashlib
    import json
    from scripts.v3_summarize_wheat_roi import audit_prerequisites
    sources = {'diffusiondet/' + name: 'a' * 64 for name in
               ('roi_gpu_dispatch.py', 'roi_gpu_dispatch_kernel.py', 'roi_pooler.py')}
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'kernel').write_bytes(b'compiled kernel')
    digest = hashlib.sha256(b'compiled kernel').hexdigest()
    manifest = dict(source_sha256=sources, checkpoint_sha256={str(s): 'b' * 64 for s in (10, 11, 12)},
                    triton_cache=str(cache), image_ids=[0])
    prior = [dict(train_seed=s, passed=True, state_unchanged=True,
                  images=32, roi_calls=192, checkpoint_sha256='b' * 64,
                  calls=[dict(max_abs_error=1e-7, rtol=1e-5, atol=2e-5, finite=True)] * 192)
             for s in (10, 11, 12)]
    jit = [dict(train_seed=s, cache_unchanged_during_timing=True, prepared_shapes=[[3, 800, 800]],
                preparation_ms=10., cache_after_preparation={'kernel': digest}) for s in (10, 11, 12)]
    parity = [dict(train_seed=10, eval_seed=0, cross_arm_bit_exact_images=0)]
    digests = {(10, 0, 'known_counts'): {'0': 'a' * 64}, (10, 0, 'gpu_dispatch'): {'0': 'b' * 64}}

    def save():
        for name, value in (('prerequisite.json', prior),
                            ('jit_audit.json', jit), ('parity.json', parity)):
            (tmp_path / name).write_text(json.dumps(value))
    save()
    audit_prerequisites(tmp_path, manifest, digests)
    if corruption == 'model':
        prior[0]['passed'] = False
    elif corruption == 'tolerance':
        prior[0]['calls'][0]['atol'] = .1
    elif corruption == 'late_jit':
        jit[0]['cache_unchanged_during_timing'] = False
    elif corruption == 'cache':
        (cache / 'kernel').write_bytes(b'different kernel')
    elif corruption == 'digest_count':
        parity[0]['cross_arm_bit_exact_images'] = 1
    elif corruption == 'missing_checkpoint':
        jit.pop()
    elif corruption == 'missing_model':
        prior.pop()
    save()
    with pytest.raises(ValueError):
        audit_prerequisites(tmp_path, manifest, digests)
