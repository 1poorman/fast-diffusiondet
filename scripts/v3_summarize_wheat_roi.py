"""Audit W1 training provenance, WHEAT outputs, timing and ROI synchronization."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import state_digest
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_roi_dispatch_profile import roi_events
from scripts.v3_wheat_roi import ARMS, arm_order
from scripts.v3_wheat_baseline import check_transfer, lr_factor, training_health, verify_data
from scripts.v3_summarize_roi_dispatch import audit_samples, summarize
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_summarize_proposals import audit_annotations


def key(row):
    return int(row['train_seed']), int(row['eval_seed']), row['arm']


def audit_matrix(rows, timing, manifest, completed, parity, states, profiles, gradient):
    args = manifest['args']
    if sorted(args['train_seeds']) != [10, 11, 12] or sorted(args['eval_seeds']) != [0, 1, 2]:
        raise ValueError('W1 requires the three predeclared checkpoint and inference seeds')
    expected = {(s, e, a) for s in args['train_seeds'] for e in args['eval_seeds'] for a in ARMS}
    if any(len(rs) != 18 or {key(r) for r in rs} != expected for rs in (rows, timing)):
        raise ValueError('Incomplete or duplicate W1 matrix')
    if completed != dict(evaluations=18, timing_cells=18, checkpoints=3, profiles=6):
        raise ValueError('Incomplete W1 run')
    if (args['limit'] != 0 or len(manifest['image_ids']) != 123 or len(set(manifest['image_ids'])) != 123
            or manifest['full_images'] != 123 or manifest['warmup'] != 20 or manifest['measure'] != 103
            or manifest['profile_images'] != 10 or manifest['arms'] != list(ARMS)):
        raise ValueError('Need full validation and standard timing/profiler windows')
    metrics = [k for k in rows[0] if k.startswith('AP')]
    if (not {'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'} <= set(metrics)
            or {k[3:] for k in metrics if k.startswith('AP-')} != set(manifest['category_names'])):
        raise ValueError('Incomplete overall or per-class AP')
    for row in rows:
        if (str(row['full_dataset']) != 'True' or int(row['images']) != 123 or int(row['finite_prediction_images']) != 123
                or any(not math.isfinite(float(row[k])) for k in metrics + ['peak_allocated_gb'])):
            raise ValueError('Invalid or partial accuracy/finite prediction evidence')
    for row in rows + timing:
        p50, p95 = float(row['latency_p50_ms']), float(row['latency_p95_ms'])
        if int(row['measured_images']) != 103 or not 0 < p50 <= p95 or not math.isfinite(p95):
            raise ValueError('Invalid timing evidence')
    if (len(parity) != 9 or {(r['train_seed'], r['eval_seed']) for r in parity} != {(s, e) for s, e, _ in expected}
            or any(r['interleaved_bit_exact'] is not True or r['full_prediction_images'] != 123
                   or r['interleaved_prediction_calls'] != 246
                   or not 0 <= r['cross_arm_bit_exact_images'] <= 123 for r in parity)):
        raise ValueError('Missing full prediction/interleaved parity')
    if (len(states) != 3 or {r['train_seed'] for r in states} != {10, 11, 12}
            or any(r['state_unchanged'] is not True or len(r['initial_sha256']) != 64
                   or r['initial_sha256'] != r['final_sha256'] for r in states)):
        raise ValueError('Frozen checkpoint state audit failed')
    if (len(profiles) != 6 or {(r['train_seed'], r['arm']) for r in profiles} != {(s, a) for s in (10, 11, 12) for a in ARMS}
            or any(r['images'] != 10 or r['roi_calls'] != 60 or r['prediction_bit_exact'] is not True
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
    return dict(evaluations=18, cross_arm_bit_exact_images=sum(r['cross_arm_bit_exact_images'] for r in parity), interleaved_bit_exact_calls=2214,
                profiler_bit_exact_calls=60, frozen_checkpoints=3), metrics


def audit_training_rows(rows, seed, image_ids):
    health = training_health(rows)
    if len(rows) != 1400 or not health['passed']:
        raise ValueError('Training attempts or health mismatch')
    for index, row in enumerate(rows):
        if (not math.isfinite(row['loss_total'])
                or any(not math.isfinite(v) for v in row['losses'].values())
                or len(row['image_ids']) != 8 or len(row['input_digest']) != 64
                or row['step_seed'] != image_seed(seed, row['step'])
                or not set(row['image_ids']) <= set(image_ids)
                or not math.isclose(row['lr'], 1e-4 * lr_factor(row['step']), rel_tol=1e-12)
                or any(not math.isfinite(row[k]) or row[k] <= 0 for k in ('scale_before', 'scale_after', 'step_s'))
                or (index and row['scale_before'] != rows[index-1]['scale_after'])
                or row['skipped'] != (row['scale_after'] < row['scale_before'])
                or (row['gradient_overflow'] and (not row['skipped'] or row['grad_norm'] is not None))
                or (not row['gradient_overflow'] and (row['grad_norm'] is None
                    or not math.isfinite(row['grad_norm']) or row['grad_norm'] < 0))):
            raise ValueError('Invalid training log evidence')
    return health


def audit_training(root, manifest):
    training = json.loads((root / 'reference/manifest.json').read_text())
    completed = json.loads((root / 'reference/COMPLETE.json').read_text())
    source_root = Path(manifest['args']['checkpoint_root'])
    if (training['args']['steps'] != 1400 or training['args']['train_seeds'] != [10, 11, 12]
            or completed['smoke'] or completed['runs'] != 3 or not completed['training_health_passed']):
        raise ValueError('Training budget or health failed')
    if sha256(training['args']['weights']) != training['pretrain_sha256']:
        raise ValueError('COCO initialization changed')
    for name, digest in training['source_sha256'].items():
        if sha256(source_root / 'source_snapshot' / name) != digest:
            raise ValueError('Training source snapshot changed')
    current_data = verify_data(Path(training['args']['data_audit']))
    if current_data != training['data'] or current_data != manifest['data_audit']:
        raise ValueError('Training/evaluation data audit differs')
    checked = []
    for seed in (10, 11, 12):
        folder = source_root / f'seed{seed}_uniform'
        info = json.loads((folder / 'training_summary.json').read_text())
        if info != next(r for r in completed['summaries'] if r['train_seed'] == seed):
            raise ValueError('Training summary differs from completion record')
        if sha256(folder / 'training.jsonl') != info['training_sha256']:
            raise ValueError('Training log changed')
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]
        if audit_training_rows(rows, seed, current_data['wheat_train']['image_ids']) != info['health']:
            raise ValueError('Training attempts or health mismatch')
        check_transfer(info['transfer'], range(549))
        if info['checkpoint_sha256'] != manifest['checkpoint_sha256'][str(seed)]:
            raise ValueError('Training/evaluation checkpoint differs')
        checked.append(dict(train_seed=seed, **info['health']))
    return checked


def audit_files(root, manifest, rows, timing, states, profiles):
    from contextlib import redirect_stdout
    import io
    from types import SimpleNamespace
    import torch
    import yaml
    from pycocotools.coco import COCO
    from scripts.v3_data_sensitivity import metrics_for_ids, verify_replay
    for name, digest in manifest['source_sha256'].items():
        if sha256(root / 'source_snapshot' / name) != digest:
            raise ValueError('Source snapshot hash mismatch')
    for name, digest in manifest['reference_sha256'].items():
        if sha256(root / 'reference' / name) != digest:
            raise ValueError('Reference snapshot hash mismatch')
    external = manifest['external_poolers']
    if sha256(root / 'detectron2_poolers_snapshot.py') != external['sha256'] or sha256(external['path']) != external['sha256']:
        raise ValueError('Installed poolers or snapshot changed')
    training_audit = audit_training(root, manifest)
    original = Path(manifest['args']['dataset_root']) / 'WHEAT/wheat_seg_strat/annotations/instances_val2017.json'
    annotation = Path(manifest['annotation_path'])
    if sha256(original) != manifest['data_audit']['wheat_val']['annotation_sha256'] or sha256(annotation) != manifest['annotation_sha256']:
        raise ValueError('Annotation hash mismatch')
    data = json.loads(annotation.read_text())
    audit_annotations(json.loads(original.read_text()), data)
    with redirect_stdout(io.StringIO()):
        coco = COCO(str(annotation))
    if [coco.cats[c]['name'] for c in sorted(coco.cats)] != manifest['category_names']:
        raise ValueError('Category mapping differs from evaluator annotation')
    if {r['id'] for r in data['images']} != set(manifest['image_ids']):
        raise ValueError('Evaluation image set differs from annotation')
    cfg = yaml.safe_load((root / 'config.yaml').read_text())
    required = dict(NUM_HEADS=6, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=300,
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
        old = yaml.safe_load((root / 'reference/config.yaml').read_text())
        if any(cfg[k] != old[k] for k in ('INPUT', 'DATASETS')):
            raise ValueError('Input or dataset settings differ')
        state = torch.load(folder / 'model_final.pth', map_location='cpu')['model']
        digest = state_digest(SimpleNamespace(state_dict=lambda: state))
        if (digest != next(s for s in states if str(s['train_seed']) == seed)['initial_sha256']
                or digest != training['final_state_sha256']):
            raise ValueError('Loaded model differs from source checkpoint')
        if any(t.is_floating_point() and t.dtype != torch.float32 for name, t in state.items() if name.startswith(('backbone.', 'head.'))):
            raise ValueError('Source backbone/head state is not FP32')
        del state
    artifacts = json.loads((root / 'artifacts.json').read_text())
    expected, digests = {'prerequisite.json', 'numerical_audit.json', 'jit_audit.json'}, {}
    replay_errors = []
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
        with redirect_stdout(io.StringIO()):
            predictions = coco.loadRes(values)
        replay_errors.append(verify_replay(metrics_for_ids(coco, predictions, manifest['image_ids'],
                                                          manifest['category_names']), row))
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
                full_AP_replays=len(replay_errors), full_replay_max_abs_difference=max(replay_errors),
                installed_poolers_unchanged=True, sequential_timed_calls=1854, interleaved_timed_calls=1854, training=training_audit)


def audit_prerequisites(root, manifest, digests):
    prior = json.loads((root / 'prerequisite.json').read_text())
    if len(prior) != 3 or {r['train_seed'] for r in prior} != {10, 11, 12}:
        raise ValueError('Missing W1 real-model prerequisite')
    for model in prior:
        if (model['passed'] is not True or model['state_unchanged'] is not True
                or model['images'] != 32 or model['roi_calls'] != 192 or len(model['calls']) != 192
                or model['checkpoint_sha256'] != manifest['checkpoint_sha256'][str(model['train_seed'])]
                or any(not math.isfinite(r['max_abs_error']) or r['max_abs_error'] < 0
                       or r['rtol'] != 1e-5 or r['atol'] != 2e-5 or r['finite'] is not True
                       for r in model['calls'])):
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
    parser.add_argument('--prefix', default='v3_wheat_roi')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    root = args.run_dir
    load = lambda name: json.loads((root / name).read_text())
    manifest, profiles, states = load('manifest.json'), load('profiles.json'), load('state_audit.json')
    rows, timing = read_rows(root / 'metrics.csv'), read_rows(root / 'interleaved.csv')
    audit, metrics = audit_matrix(rows, timing, manifest, load('COMPLETE.json'), load('parity.json'), states, profiles, load('numerical_audit.json'))
    audit.update(audit_files(root, manifest, rows, timing, states, profiles))
    paired, summary, classes, gate = summarize(rows, timing, profiles, manifest, metrics)
    names = ('metrics.csv', 'interleaved.csv', 'manifest.json', 'config.yaml', 'numerical_audit.json', 'parity.json',
             'state_audit.json', 'profiles.json', 'artifacts.json', 'COMPLETE.json', 'prerequisite.json',
             'jit_audit.json')
    report = dict(audit=audit, **gate, report_script_sha256=sha256(__file__),
                  evidence_sha256={name: sha256(root / name) for name in names},
                  aggregation='Mean inference seeds within checkpoint, equal checkpoint weights; speed gate averages checkpoint ratios',
                  scope='WHEAT development cross-dataset execution; rebuilt frozen FP32 H6; profiler runs after all timing, never used for speed gate')
    raw = Path('results/raw')
    raw.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(root / name, raw / (args.prefix + ('.csv' if name == 'metrics.csv' else '_' + name)))
    for name in ('manifest.json', 'COMPLETE.json', 'config.yaml'):
        shutil.copyfile(root / 'reference' / name, raw / f'{args.prefix}_training_{name}')
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
    fig.suptitle('WHEAT GPU ROI dispatch; frozen FP32 H6/P300\n3 checkpoints × 3 inference seeds; R1 control vs GPU dispatch')
    fig.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'results/figures/{args.prefix}.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
