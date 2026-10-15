"""Audit frozen R2 outputs, timing, provenance and ROI synchronization evidence."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import state_digest
from scripts.v3_head_sweep import sha256
from scripts.v3_roi_dispatch_profile import roi_events
from scripts.v3_roi_gpu_dispatch import ARMS, arm_order
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_summarize_proposals import audit_annotations


def key(row):
    return int(row['train_seed']), int(row['eval_seed']), row['arm']


def average(rows, field):
    return stats.mean(float(row[field]) for row in rows)


def audit_matrix(rows, timing, manifest, completed, reference, parity, states, profiles, gradient):
    args = manifest['args']
    if sorted(args['train_seeds']) != [10, 11, 12] or sorted(args['eval_seeds']) != [0, 1, 2]:
        raise ValueError('R2 requires the three predeclared checkpoint and inference seeds')
    expected = {(s, e, a) for s in args['train_seeds'] for e in args['eval_seeds'] for a in ARMS}
    if any(len(rs) != 18 or {key(r) for r in rs} != expected for rs in (rows, timing)):
        raise ValueError('Incomplete or duplicate R2 matrix')
    if completed != dict(evaluations=18, timing_cells=18, checkpoints=3, profiles=6):
        raise ValueError('Incomplete R2 run')
    if (args['limit'] != 0 or len(manifest['image_ids']) != 750 or len(set(manifest['image_ids'])) != 750
            or manifest['full_images'] != 750 or manifest['warmup'] != 20 or manifest['measure'] != 200
            or manifest['profile_images'] != 10 or manifest['arms'] != list(ARMS)):
        raise ValueError('Need full validation and standard timing/profiler windows')
    metrics = [k for k in rows[0] if k.startswith('AP')]
    if (not {'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'} <= set(metrics)
            or {k[3:] for k in metrics if k.startswith('AP-')} != set(manifest['category_names'])):
        raise ValueError('Incomplete overall or per-class AP')
    for row in rows:
        if (str(row['full_dataset']) != 'True' or int(row['images']) != 750 or int(row['finite_prediction_images']) != 750
                or any(not math.isfinite(float(row[k])) for k in metrics + ['peak_allocated_gb'])):
            raise ValueError('Invalid or partial accuracy/finite prediction evidence')
    for row in rows + timing:
        p50, p95 = float(row['latency_p50_ms']), float(row['latency_p95_ms'])
        if int(row['measured_images']) != 200 or not 0 < p50 <= p95 or not math.isfinite(p95):
            raise ValueError('Invalid timing evidence')
    refs = [r for r in reference if r['arm'] == 'uniform']
    ref = {(int(r['train_seed']), int(r['eval_seed'])): r for r in refs}
    if len(refs) != 9 or set(ref) != {(s, e) for s, e, _ in expected}:
        raise ValueError('Incomplete T1 reference')
    for row in rows:
        if row['arm'] != 'known_counts':
            continue
        for metric in metrics:
            if float(row[metric]) != float(ref[key(row)[:2]][metric]):
                raise ValueError('R2 AP differs from T1 reference')
    if (len(parity) != 9 or {(r['train_seed'], r['eval_seed']) for r in parity} != set(ref)
            or any(r['interleaved_bit_exact'] is not True or r['full_prediction_images'] != 750
                   or r['interleaved_prediction_calls'] != 440
                   or not 0 <= r['cross_arm_bit_exact_images'] <= 750 for r in parity)):
        raise ValueError('Missing full prediction/interleaved parity')
    if (len(states) != 3 or {r['train_seed'] for r in states} != {10, 11, 12}
            or any(r['state_unchanged'] is not True or len(r['initial_sha256']) != 64
                   or r['initial_sha256'] != r['final_sha256'] for r in states)):
        raise ValueError('Frozen checkpoint state audit failed')
    if (len(profiles) != 6 or {(r['train_seed'], r['arm']) for r in profiles} != {(s, a) for s in (10, 11, 12) for a in ARMS}
            or any(r['images'] != 10 or r['roi_calls'] != 40 or r['prediction_bit_exact'] is not True
                   or sum(r['synchronizations'].values()) != r['synchronization_count']
                   or sum(r['kernels'].values()) != r['kernel_count'] or r['kernel_count'] <= 0
                   or any(not isinstance(n, int) or n < 0 for n in [r['repeat_interleave_calls'], r['nonzero_calls'],
                                                                  *r['synchronizations'].values(), *r['kernels'].values()])
                   for r in profiles)):
        raise ValueError('Invalid profiler matrix or execution counts')
    if (gradient['passed'] is not True or len(gradient['features']) != 4
            or {r['case'] for r in gradient['features']} != {'boundaries_ragged', 'empty', 'large_spatial', 'real_channels_300'}
            or set(gradient['native_fallbacks']) != {'noncontiguous', 'autocast'}
            or not math.isfinite(gradient['gradient_fallback_max_abs'])
            or gradient['gradient_fallback_max_abs'] < 0
            or any(r['rtol'] != 1e-5 or r['atol'] != 2e-5 or r['finite'] is not True
                   or not math.isfinite(r['max_abs_error']) or r['max_abs_error'] < 0 for r in gradient['features'])):
        raise ValueError('CUDA numerical/gradient prerequisite failed')
    return dict(evaluations=18, cross_arm_bit_exact_images=sum(r['cross_arm_bit_exact_images'] for r in parity), interleaved_bit_exact_calls=3960,
                profiler_bit_exact_calls=60, frozen_checkpoints=3, baseline_max_abs_AP_difference=0.), metrics


def audit_samples(samples, image_ids, train_seed, eval_seed, rows, interleaved):
    import numpy as np
    if interleaved:
        expected = [(i, image, arm, pos) for i, image in enumerate(image_ids[20:220], 20)
                    for pos, arm in enumerate(arm_order(train_seed, eval_seed, i))]
        actual = [(r['index'], r['image_id'], r['arm'], r['position']) for r in samples]
        if actual != expected or any(r['bit_exact'] is not True for r in samples):
            raise ValueError('Interleaved order/window/parity differs')
    elif [(r['index'], r['image_id']) for r in samples] != list(enumerate(image_ids[20:220], 20)):
        raise ValueError('Sequential timing window differs')
    if any(not math.isfinite(r['wall_ms']) or r['wall_ms'] <= 0 for r in samples):
        raise ValueError('Invalid raw timing')
    for row in rows:
        values = [r['wall_ms'] for r in samples if not interleaved or r['arm'] == row['arm']]
        for field, value in (('latency_p50_ms', np.median(values)), ('latency_p95_ms', np.percentile(values, 95))):
            if not math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Timing quantile differs from raw samples')


def summarize(rows, timing, profiles, manifest, metrics):
    paired = []
    for seed in manifest['args']['train_seeds']:
        row = dict(train_seed=seed)
        for arm in ARMS:
            selected = [r for r in rows if key(r)[0] == seed and r['arm'] == arm]
            timed = [r for r in timing if key(r)[0] == seed and r['arm'] == arm]
            profile = next(r for r in profiles if r['train_seed'] == seed and r['arm'] == arm)
            for metric in metrics + ['peak_allocated_gb', 'latency_p50_ms', 'latency_p95_ms']:
                row[f'{arm}_{metric}'] = average(selected, metric)
            for metric in ('latency_p50_ms', 'latency_p95_ms'):
                row[f'{arm}_interleaved_{metric}'] = average(timed, metric)
            row[f'{arm}_roi_sync_per_image'] = profile['synchronization_count'] / profile['images']
            row[f'{arm}_roi_repeat_calls_per_image'] = profile['repeat_interleave_calls'] / profile['images']
            row[f'{arm}_roi_nonzero_per_image'] = profile['nonzero_calls'] / profile['images']
            row[f'{arm}_roi_kernels_per_image'] = profile['kernel_count'] / profile['images']
        for metric in metrics:
            row['delta_' + metric] = row['gpu_dispatch_' + metric] - row['known_counts_' + metric]
        row['p50_reduction_percent'] = 100 * (1 - row['gpu_dispatch_interleaved_latency_p50_ms'] / row['known_counts_interleaved_latency_p50_ms'])
        row['p95_increase_percent'] = 100 * (row['gpu_dispatch_interleaved_latency_p95_ms'] / row['known_counts_interleaved_latency_p95_ms'] - 1)
        paired.append(row)
    summary = {k: average(paired, k) for k in paired[0] if k != 'train_seed'}
    checks = dict(mean_AP=summary['delta_AP'] >= -0.05,
                  mean_AP75=summary['delta_AP75'] >= -0.05,
                  each_checkpoint_AP=min(r['delta_AP'] for r in paired) >= -0.1,
                  worst_class_AP=min(summary['delta_AP-' + c] for c in manifest['category_names']) >= -0.3,
                  mean_p50=summary['p50_reduction_percent'] >= 5,
                  each_checkpoint_p50=min(r['p50_reduction_percent'] for r in paired) >= 3,
                  mean_p95=summary['p95_increase_percent'] <= 5,
                  synchronization_mechanism=all(r['known_counts_roi_nonzero_per_image'] > 0
                      and r['gpu_dispatch_roi_nonzero_per_image'] == 0
                      and r['gpu_dispatch_roi_repeat_calls_per_image'] == 0
                      and r['gpu_dispatch_roi_sync_per_image'] < r['known_counts_roi_sync_per_image'] for r in paired))
    per_class = [dict(category=c, known_counts_AP=summary['known_counts_AP-' + c], gpu_dispatch_AP=summary['gpu_dispatch_AP-' + c],
                      delta_AP=summary['delta_AP-' + c]) for c in manifest['category_names']]
    return paired, summary, per_class, dict(checks=checks, passed_development_screen=all(checks.values()))


def audit_files(root, manifest, rows, timing, states, profiles):
    from types import SimpleNamespace
    import torch
    import yaml
    for name, digest in manifest['source_sha256'].items():
        if sha256(root / 'source_snapshot' / name) != digest:
            raise ValueError('Source snapshot hash mismatch')
    for name, digest in manifest['reference_sha256'].items():
        if sha256(root / 'reference' / name) != digest:
            raise ValueError('Reference snapshot hash mismatch')
    external = manifest['external_poolers']
    if sha256(root / 'detectron2_poolers_snapshot.py') != external['sha256'] or sha256(external['path']) != external['sha256']:
        raise ValueError('Installed poolers or snapshot changed')
    t1 = json.loads((root / 'reference/manifest.json').read_text())
    if t1['args']['steps'] != 600 or t1['eval_image_ids'] != manifest['image_ids']:
        raise ValueError('Reference training budget or images differ')
    original = Path(manifest['args']['dataset_root']) / 'Strawberry/annotations/instances_val2017.json'
    annotation = Path(manifest['annotation_path'])
    if sha256(original) != t1['annotation_sha256']['val'] or sha256(annotation) != manifest['annotation_sha256']:
        raise ValueError('Annotation hash mismatch')
    data = json.loads(annotation.read_text())
    audit_annotations(json.loads(original.read_text()), data)
    if {r['id'] for r in data['images']} != set(manifest['image_ids']):
        raise ValueError('Evaluation image set differs from annotation')
    cfg = yaml.safe_load((root / 'config.yaml').read_text())
    required = dict(NUM_HEADS=4, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=300,
                    SAMPLE_STEP=1, FORMULATION='vp', SOLVER='ddim', BOX_RENEWAL=False, USE_ENSEMBLE=False,
                    ATTENTION_NEED_WEIGHTS=False, USE_NMS=True, DEEP_SUPERVISION=True, FINAL_CLASS_LOSS='focal',
                    EVAL_BACKBONE_PRECISION='fp32', ROI_POOLER_KNOWN_COUNTS=False, EVAL_ROI_GPU_DISPATCH=False)
    if (any(cfg['MODEL']['DiffusionDet'][k] != v for k, v in required.items()) or cfg['MODEL_EMA']['ENABLED']
            or cfg['SOLVER']['AMP']['ENABLED'] or manifest['tf32'] or manifest['cudnn_benchmark']
            or manifest['cudnn_deterministic'] or manifest['threads'] != 4
            or cfg['MODEL']['ROI_BOX_HEAD']['POOLER_TYPE'] != 'ROIAlignV2'):
        raise ValueError('Config/runtime flags differ from protocol')
    if set(manifest['checkpoint_sha256']) != {'10', '11', '12'}:
        raise ValueError('Missing checkpoint provenance')
    for seed, digest in manifest['checkpoint_sha256'].items():
        folder = Path(manifest['args']['checkpoint_root']) / f'seed{seed}_uniform'
        training = json.loads((folder / 'training_summary.json').read_text())
        if sha256(folder / 'model_final.pth') != digest or training['checkpoint_sha256'] != digest:
            raise ValueError('Checkpoint hash mismatch')
        old = yaml.safe_load((folder / 'config.yaml').read_text())
        if any(cfg[k] != old[k] for k in ('INPUT', 'DATASETS')):
            raise ValueError('Input or dataset settings differ')
        state = torch.load(folder / 'model_final.pth', map_location='cpu')['model']
        if state_digest(SimpleNamespace(state_dict=lambda: state)) != next(s for s in states if str(s['train_seed']) == seed)['initial_sha256']:
            raise ValueError('Loaded model differs from source checkpoint')
        if any(t.is_floating_point() and t.dtype != torch.float32 for name, t in state.items() if name.startswith(('backbone.', 'head.'))):
            raise ValueError('Source backbone/head state is not FP32')
        del state
    artifacts = json.loads((root / 'artifacts.json').read_text())
    expected, digests = {'prerequisite.json', 'prerequisite_manifest.json', 'numerical_audit.json', 'jit_audit.json'}, {}
    for row in rows:
        s, e, arm = key(row)
        folder = f'train{s}_eval{e}_{arm}'
        expected.update(f'{folder}/{name}' for name in ('metrics.json', 'timing.json', 'prediction_digests.json',
                                                        'coco_instances_results.json', 'instances_predictions.pth'))
        saved = json.loads((root / folder / 'metrics.json').read_text())
        if any(str(saved[k]) != str(v) for k, v in row.items()):
            raise ValueError('CSV differs from per-cell metrics')
        audit_samples(json.loads((root / folder / 'timing.json').read_text()), manifest['image_ids'], s, e, [row], False)
        digest = json.loads((root / folder / 'prediction_digests.json').read_text())
        if set(digest) != {str(i) for i in manifest['image_ids']} or any(len(v) != 64 for v in digest.values()):
            raise ValueError('Missing full prediction digests')
        digests[key(row)] = digest
        values = json.loads((root / folder / 'coco_instances_results.json').read_text())
        ids = set(manifest['image_ids'])
        if any(r['image_id'] not in ids or not all(math.isfinite(v) for v in r['bbox'] + [r['score']]) for r in values):
            raise ValueError('Invalid saved predictions')
    for s in manifest['args']['train_seeds']:
        for e in manifest['args']['eval_seeds']:
            name = f'interleaved_train{s}_eval{e}.json'
            expected.add(name)
            audit_samples(json.loads((root / name).read_text()), manifest['image_ids'], s, e,
                          [r for r in timing if key(r)[:2] == (s, e)], True)
    for row in profiles:
        name = f"profile_train{row['train_seed']}_{row['arm']}.json.gz"
        expected.add(name)
        with gzip.open(root / name, 'rt') as f:
            recomputed = roi_events(json.load(f)['traceEvents'])
        if any(row[k] != v for k, v in recomputed.items()):
            raise ValueError('Profiler summary differs from raw trace')
    if set(artifacts) != expected or any(sha256(root / name) != digest for name, digest in artifacts.items()):
        raise ValueError('Missing or corrupted raw artifacts')
    audit_prerequisites(root, manifest, digests)
    return dict(hashed_artifacts=len(artifacts), hashed_sources=len(manifest['source_sha256']),
                installed_poolers_unchanged=True, sequential_timed_calls=3600, interleaved_timed_calls=3600)


def audit_prerequisites(root, manifest, digests):
    prior = json.loads((root / 'prerequisite.json').read_text())
    source = json.loads((root / 'prerequisite_manifest.json').read_text())
    for name in ('roi_gpu_dispatch.py', 'roi_gpu_dispatch_kernel.py', 'roi_pooler.py'):
        path = 'diffusiondet/' + name
        if source['source_sha256'][path] != manifest['source_sha256'][path]:
            raise ValueError('Prerequisite tested different pooler sources')
    model = prior['model']
    if (prior['passed'] is not True or model['passed'] is not True or model['finite'] is not True
            or model['state_unchanged'] is not True or model['images'] != 32 or model['roi_calls'] != 128
            or len(model['calls']) != 128
            or model['checkpoint_sha256'] != manifest['checkpoint_sha256']['10']
            or any(not math.isfinite(r['max_abs_error']) or r['max_abs_error'] < 0 for r in model['calls'])):
        raise ValueError('Real-model prerequisite failed')
    jit = json.loads((root / 'jit_audit.json').read_text())
    if (len(jit) != 3 or {r['train_seed'] for r in jit} != {10, 11, 12}
            or any(r['cache_unchanged_during_timing'] is not True or not r['prepared_shapes']
                   or not r['cache_after_preparation'] or not math.isfinite(r['preparation_ms']) for r in jit)):
        raise ValueError('Missing compilation preparation or changed cache during timing')
    cache = Path(manifest['triton_cache'])
    for row in jit:
        for name, digest in row['cache_after_preparation'].items():
            if sha256(cache / name) != digest:
                raise ValueError('JIT cache hash mismatch')
    parity = json.loads((root / 'parity.json').read_text())
    for row in parity:
        s, e = row['train_seed'], row['eval_seed']
        exact = sum(digests[s, e, 'known_counts'][str(i)] == digests[s, e, 'gpu_dispatch'][str(i)]
                    for i in manifest['image_ids'])
        if exact != row['cross_arm_bit_exact_images']:
            raise ValueError('Cross-arm prediction digest count differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--prefix', default='v3_roi_gpu_dispatch')
    args = parser.parse_args()
    root = args.run_dir
    load = lambda name: json.loads((root / name).read_text())
    manifest, profiles, states = load('manifest.json'), load('profiles.json'), load('state_audit.json')
    rows, timing = read_rows(root / 'metrics.csv'), read_rows(root / 'interleaved.csv')
    audit, metrics = audit_matrix(rows, timing, manifest, load('COMPLETE.json'), read_rows(root / 'reference/metrics.csv'),
                                  load('parity.json'), states, profiles, load('numerical_audit.json'))
    audit.update(audit_files(root, manifest, rows, timing, states, profiles))
    paired, summary, classes, gate = summarize(rows, timing, profiles, manifest, metrics)
    names = ('metrics.csv', 'interleaved.csv', 'manifest.json', 'config.yaml', 'numerical_audit.json', 'parity.json',
             'state_audit.json', 'profiles.json', 'artifacts.json', 'COMPLETE.json', 'prerequisite.json',
             'prerequisite_manifest.json', 'jit_audit.json')
    report = dict(audit=audit, **gate, report_script_sha256=sha256(__file__),
                  evidence_sha256={name: sha256(root / name) for name in names},
                  aggregation='Mean inference seeds within checkpoint, equal checkpoint weights; speed gate averages checkpoint ratios',
                  scope='SDD development execution screen; frozen FP32 H4; profiler runs after all timing, never used for speed gate')
    raw = Path('results/raw')
    raw.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(root / name, raw / (args.prefix + ('.csv' if name == 'metrics.csv' else '_' + name)))
    for name, values in (('paired', paired), ('summary', [summary]), ('per_class', classes)):
        write_rows(raw / f'{args.prefix}_{name}.csv', values)
    (raw / f'{args.prefix}_gate.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['| Pooler | AP | AP75 | Interleaved p50 ms | Interleaved p95 ms | ROI sync/image |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {summary[arm + '_AP']:.3f} | {summary[arm + '_AP75']:.3f} | "
                     f"{summary[arm + '_interleaved_latency_p50_ms']:.3f} | {summary[arm + '_interleaved_latency_p95_ms']:.3f} | "
                     f"{summary[arm + '_roi_sync_per_image']:.1f} |")
    (root / 'summary.md').write_text('\n'.join(lines) + '\n\n' + json.dumps(report, indent=2) + '\n')
    print('\n'.join(lines))
    print(json.dumps(report, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(root.resolve() / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for row in paired:
        axes[0].plot(ARMS, [row[a + '_interleaved_latency_p50_ms'] for a in ARMS], marker='o', label=f"Checkpoint {row['train_seed']}")
    axes[0].set(ylabel='Interleaved model-call p50 (ms)')
    axes[0].legend(fontsize=8)
    axes[1].bar([str(r['train_seed']) for r in paired], [r['p50_reduction_percent'] for r in paired])
    axes[1].axhline(5, color='black', linestyle='--', label='Mean gate 5%')
    axes[1].axhline(3, color='gray', linestyle=':', label='Each checkpoint 3%')
    axes[1].set(xlabel='Checkpoint training seed', ylabel='Interleaved p50 reduction (%)')
    axes[1].legend(fontsize=8)
    axes[2].bar(ARMS, [summary[a + '_roi_sync_per_image'] for a in ARMS])
    axes[2].set(ylabel='ROI synchronizations per image')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('GPU ROI dispatch; frozen FP32 H4/P300\n3 checkpoints × 3 inference seeds; R1 control vs GPU dispatch')
    fig.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'results/figures/{args.prefix}.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
