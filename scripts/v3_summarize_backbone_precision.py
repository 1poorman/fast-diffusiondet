"""Audit B1 evidence and apply its predeclared accuracy and interleaved-speed gates."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_backbone_precision import ARMS, arm_order
from scripts.v3_head_sweep import sha256
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_summarize_proposals import audit_annotations


def key(row):
    return int(row['train_seed']), int(row['eval_seed']), row['precision']


def average(rows, field):
    return stats.mean(float(row[field]) for row in rows)


def audit_matrix(rows, timing, manifest, completed, reference, precision, states):
    args = manifest['args']
    seeds, eval_seeds = args['train_seeds'], args['eval_seeds']
    if sorted(seeds) != [10, 11, 12] or sorted(eval_seeds) != [0, 1, 2]:
        raise ValueError('B1 requires the predeclared three checkpoint and inference seeds')
    expected = {(s, e, a) for s in seeds for e in eval_seeds for a in ARMS}
    if any(len(rs) != len(expected) or {key(r) for r in rs} != expected for rs in (rows, timing)):
        raise ValueError('Incomplete or duplicate B1 matrix')
    if completed != dict(evaluations=18, timing_cells=18, checkpoints=3):
        raise ValueError('Incomplete B1 run')
    ids = manifest['image_ids']
    if (args['limit'] != 0 or len(ids) != 750 or len(set(ids)) != 750 or manifest['full_images'] != 750
            or manifest['warmup'] != 20 or manifest['measure'] != 200):
        raise ValueError('Need the full 750-image validation and standard timing window')
    if any(str(r['full_dataset']) != 'True' or int(r['images']) != 750
           or int(r['finite_prediction_images']) != 750 for r in rows):
        raise ValueError('Missing full-validation finite prediction audit')
    metrics = [k for k in rows[0] if k.startswith('AP')]
    if not {'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'} <= set(metrics):
        raise ValueError('Incomplete overall AP metrics')
    if {k[3:] for k in metrics if k.startswith('AP-')} != set(manifest['category_names']):
        raise ValueError('Incomplete per-class AP')
    for row in rows:
        if any(not math.isfinite(float(row[k])) for k in metrics + ['peak_allocated_gb']):
            raise ValueError('Non-finite AP or memory metric')
    for row in rows + timing:
        p50, p95 = (float(row[k]) for k in ('latency_p50_ms', 'latency_p95_ms'))
        if int(row['measured_images']) != 200 or not 0 < p50 <= p95 or not math.isfinite(p95):
            raise ValueError('Invalid timing window or quantiles')
    reference = [r for r in reference if r['arm'] == 'uniform']
    ref = {(int(r['train_seed']), int(r['eval_seed'])): r for r in reference}
    if len(reference) != 9 or set(ref) != {(s, e) for s in seeds for e in eval_seeds}:
        raise ValueError('Incomplete or duplicate T1 reference')
    differences = []
    for row in rows:
        if row['precision'] == 'fp32':
            base = ref[key(row)[:2]]
            for metric in metrics:
                difference = abs(float(row[metric]) - float(base[metric]))
                if not math.isfinite(difference) or difference > 1e-5:
                    raise ValueError('FP32 AP differs from T1 reference')
                differences.append(difference)
    if (len(states) != 3 or {r['train_seed'] for r in states} != set(seeds)
            or any(r['state_unchanged'] is not True or not isinstance(r['all_float_state_fp32'], bool)
                   or r['initial_sha256'] != r['final_sha256'] or len(r['initial_sha256']) != 64 for r in states)):
        raise ValueError('Frozen state audit failed')
    if (len(precision) != 12 or {(r['train_seed'], r['image_id'], r['precision']) for r in precision}
            != {(s, i, a) for s in seeds for i in ids[:2] for a in ARMS}):
        raise ValueError('Incomplete precision audit matrix')
    feature_names = {'fpn_p' + str(n) for n in range(2, 7)}
    names = feature_names | {'head_p' + str(n) for n in range(2, 6)} | {
        'stem_conv', 'backbone_input', 'head_boxes', 'head_logits', 'head_coords', 'final_boxes', 'final_scores', 'noise'}
    for row in precision:
        if row['finite'] is not True or row['fp32_feature_boundary'] is not True or set(row['dtypes']) != names:
            raise ValueError('Missing finite precision boundaries')
        for name, dtype in row['dtypes'].items():
            expected_dtype = 'torch.float16' if row['precision'] == 'fp16' and name in feature_names | {'stem_conv'} else 'torch.float32'
            if dtype != expected_dtype:
                raise ValueError('Precision escaped backbone boundary or FP16 was not executed')
        if (row['shapes']['head_coords'] != [4, 1, 300, 4]
                or row['shapes']['head_logits'] != [4, 1, 300, len(manifest['category_names'])]):
            raise ValueError('Executed depth/proposals/classes differ from protocol')
        if row['precision'] == 'fp16':
            errors = row['feature_errors']
            if (set(errors) != feature_names | {'stem_conv'}
                    or any(set(error) != {'max_abs_error', 'relative_l2'} for error in errors.values()) or any(
                    not math.isfinite(v) or v < 0 for error in errors.values() for v in error.values())):
                raise ValueError('Invalid feature error audit')
    return dict(evaluations=18, interleaved_pairs=9, dtype_audits=12, frozen_checkpoints=3,
                finite_validation_image_calls=13500, baseline_max_abs_AP_difference=max(differences)), metrics


def audit_samples(samples, image_ids, train_seed, eval_seed, rows, *, interleaved):
    import numpy as np
    expected = [(i, image_id, arm, position) for i, image_id in enumerate(image_ids[20:220], 20)
                for position, arm in enumerate(arm_order(train_seed, eval_seed, i))]
    if interleaved:
        actual = [(r['index'], r['image_id'], r['precision'], r['position']) for r in samples]
        if actual != expected or any(r['finite'] is not True for r in samples):
            raise ValueError('Interleaved sample pairing/order/finite checks failed')
    elif [(r['index'], r['image_id']) for r in samples] != list(enumerate(image_ids[20:220], 20)):
        raise ValueError('Sequential timing image window differs')
    if any(not math.isfinite(r['wall_ms']) or r['wall_ms'] <= 0 for r in samples):
        raise ValueError('Non-finite or non-positive raw timing')
    for row in rows:
        times = [r['wall_ms'] for r in samples if not interleaved or r['precision'] == row['precision']]
        for name, value in (('latency_p50_ms', np.median(times)), ('latency_p95_ms', np.percentile(times, 95))):
            if not math.isclose(float(row[name]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Timing quantile differs from raw samples')


def summarize(rows, timing, manifest, metrics):
    paired = []
    for seed in manifest['args']['train_seeds']:
        row = dict(train_seed=seed)
        for arm in ARMS:
            selected = [r for r in rows if int(r['train_seed']) == seed and r['precision'] == arm]
            timed = [r for r in timing if int(r['train_seed']) == seed and r['precision'] == arm]
            for metric in metrics + ['peak_allocated_gb', 'latency_p50_ms', 'latency_p95_ms']:
                row[f'{arm}_{metric}'] = average(selected, metric)
            for metric in ('latency_p50_ms', 'latency_p95_ms'):
                row[f'{arm}_interleaved_{metric}'] = average(timed, metric)
        for metric in metrics:
            row[f'delta_{metric}'] = row[f'fp16_{metric}'] - row[f'fp32_{metric}']
        row['p50_reduction_percent'] = 100 * (1 - row['fp16_interleaved_latency_p50_ms'] / row['fp32_interleaved_latency_p50_ms'])
        row['p95_increase_percent'] = 100 * (row['fp16_interleaved_latency_p95_ms'] / row['fp32_interleaved_latency_p95_ms'] - 1)
        paired.append(row)
    summary = {k: average(paired, k) for k in paired[0] if k != 'train_seed'}
    summary['delta_AP_checkpoint_sample_sd'] = stats.stdev(r['delta_AP'] for r in paired)
    per_class = [dict(category=name, fp32_AP=summary[f'fp32_AP-{name}'], fp16_AP=summary[f'fp16_AP-{name}'],
                      delta_AP=summary[f'delta_AP-{name}']) for name in manifest['category_names']]
    worst = min(per_class, key=lambda r: r['delta_AP'])
    summary.update(worst_category=worst['category'], worst_category_delta_AP=worst['delta_AP'])
    checks = dict(mean_AP=summary['delta_AP'] >= -0.3, mean_AP75=summary['delta_AP75'] >= -0.3,
                  each_checkpoint_AP=min(r['delta_AP'] for r in paired) >= -0.5,
                  worst_category=worst['delta_AP'] >= -1,
                  mean_p50=summary['p50_reduction_percent'] >= 15,
                  each_checkpoint_p50=min(r['p50_reduction_percent'] for r in paired) >= 10,
                  mean_p95=summary['p95_increase_percent'] <= 5)
    return paired, summary, per_class, dict(checks=checks, passed_development_screen=all(checks.values()))


def audit_files(root, manifest, rows, timing):
    from collections import Counter
    from types import SimpleNamespace
    import torch
    import yaml
    from scripts.v3_distill import state_digest
    for source, digest in manifest['source_sha256'].items():
        if sha256(root / 'source_snapshot' / source) != digest:
            raise ValueError('Source snapshot hash mismatch: ' + source)
    for name, digest in manifest['reference_sha256'].items():
        if sha256(root / 'reference' / name) != digest:
            raise ValueError('T1 reference snapshot hash mismatch')
    t1 = json.loads((root / 'reference/manifest.json').read_text())
    if t1['args']['steps'] != 600 or t1['eval_image_ids'] != manifest['image_ids']:
        raise ValueError('T1 reference budget or images differ')
    annotation = Path(manifest['annotation_path'])
    original = Path(manifest['args']['dataset_root']) / 'Strawberry/annotations/instances_val2017.json'
    if sha256(annotation) != manifest['annotation_sha256'] or sha256(original) != t1['annotation_sha256']['val']:
        raise ValueError('Annotation hash mismatch')
    data = json.loads(annotation.read_text())
    audit_annotations(json.loads(original.read_text()), data)
    if set(manifest['image_ids']) != {r['id'] for r in data['images']}:
        raise ValueError('Full validation IDs differ from annotations')
    cfg = yaml.safe_load((root / 'config.yaml').read_text())
    required = dict(NUM_HEADS=4, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=300,
                    SAMPLE_STEP=1, FORMULATION='vp', SOLVER='ddim', BOX_RENEWAL=False, USE_ENSEMBLE=False,
                    ATTENTION_NEED_WEIGHTS=False, USE_NMS=True, DEEP_SUPERVISION=True,
                    FINAL_CLASS_LOSS='focal', EVAL_BACKBONE_PRECISION='fp32')
    if (any(cfg['MODEL']['DiffusionDet'][k] != v for k, v in required.items()) or cfg['MODEL_EMA']['ENABLED']
            or cfg['SOLVER']['AMP']['ENABLED'] or manifest['tf32'] or manifest['cudnn_benchmark']
            or manifest['cudnn_deterministic'] or manifest['threads'] != 4 or manifest['precision_modes'] != list(ARMS)):
        raise ValueError('Config or runtime flags differ from B1 protocol')
    if set(manifest['checkpoint_sha256']) != {'10', '11', '12'}:
        raise ValueError('Missing checkpoint provenance')
    states = {str(r['train_seed']): r for r in json.loads((root / 'state_audit.json').read_text())}
    state_dtypes = {}
    for seed, digest in manifest['checkpoint_sha256'].items():
        arm = Path(manifest['args']['checkpoint_root']) / f'seed{seed}_uniform'
        training = json.loads((arm / 'training_summary.json').read_text())
        if sha256(arm / 'model_final.pth') != digest or training['checkpoint_sha256'] != digest:
            raise ValueError('Checkpoint hash differs from T1')
        old = yaml.safe_load((arm / 'config.yaml').read_text())
        if any(cfg[k] != old[k] for k in ('INPUT', 'DATASETS')):
            raise ValueError('Input or dataset settings differ from T1')
        state = torch.load(arm / 'model_final.pth', map_location='cpu')['model']
        if state_digest(SimpleNamespace(state_dict=lambda: state)) != states[seed]['initial_sha256']:
            raise ValueError('Loaded state differs from source checkpoint')
        if any(t.is_floating_point() and t.dtype != torch.float32 for name, t in state.items()
               if name.startswith(('backbone.', 'head.'))):
            raise ValueError('Source backbone/head state is not FP32')
        all_fp32 = all(not t.is_floating_point() or t.dtype == torch.float32 for t in state.values())
        if states[seed]['all_float_state_fp32'] != all_fp32:
            raise ValueError('Stored-state dtypes changed from checkpoint')
        state_dtypes[seed] = dict(Counter(str(t.dtype) for t in state.values()))
        del state
    artifacts = json.loads((root / 'artifacts.json').read_text())
    expected = set()
    for row in rows:
        s, e, arm = key(row)
        folder = f'train{s}_eval{e}_{arm}'
        expected.update(f'{folder}/{name}' for name in ('metrics.json', 'timing.json', 'coco_instances_results.json', 'instances_predictions.pth'))
        actual = json.loads((root / folder / 'metrics.json').read_text())
        if any(str(actual[k]) != str(v) for k, v in row.items()):
            raise ValueError('CSV differs from per-cell metrics')
        audit_samples(json.loads((root / folder / 'timing.json').read_text()), manifest['image_ids'], s, e, [row], interleaved=False)
        predictions = json.loads((root / folder / 'coco_instances_results.json').read_text())
        ids = set(manifest['image_ids'])
        if any(r['image_id'] not in ids or not all(math.isfinite(v) for v in r['bbox'] + [r['score']]) for r in predictions):
            raise ValueError('Invalid saved predictions')
    for s in manifest['args']['train_seeds']:
        for e in manifest['args']['eval_seeds']:
            name = f'interleaved_train{s}_eval{e}.json'
            expected.add(name)
            selected = [r for r in timing if key(r)[:2] == (s, e)]
            audit_samples(json.loads((root / name).read_text()), manifest['image_ids'], s, e, selected, interleaved=True)
    if set(artifacts) != expected or any(sha256(root / name) != digest for name, digest in artifacts.items()):
        raise ValueError('Missing or corrupted evaluation/timing artifacts')
    return dict(hashed_artifacts=len(artifacts), hashed_sources=len(manifest['source_sha256']),
                interleaved_timed_calls=3600, sequential_timed_calls=3600,
                source_state_dtypes=state_dtypes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--prefix', default='v3_backbone_precision')
    args = parser.parse_args()
    root = args.run_dir
    load = lambda name: json.loads((root / name).read_text())
    manifest = load('manifest.json')
    rows, timing = read_rows(root / 'metrics.csv'), read_rows(root / 'interleaved.csv')
    audit, metrics = audit_matrix(rows, timing, manifest, load('COMPLETE.json'), read_rows(root / 'reference/metrics.csv'),
                                  load('precision_audit.json'), load('state_audit.json'))
    audit.update(audit_files(root, manifest, rows, timing))
    paired, summary, per_class, gate = summarize(rows, timing, manifest, metrics)
    names = ('metrics.csv', 'interleaved.csv', 'manifest.json', 'config.yaml', 'precision_audit.json',
             'state_audit.json', 'artifacts.json', 'COMPLETE.json')
    report = dict(audit=audit, **gate, report_script_sha256=sha256(__file__),
                  evidence_sha256={name: sha256(root / name) for name in names},
                  aggregation='Mean inference seeds within checkpoint, then equal checkpoint weights; speed gate averages checkpoint ratios',
                  scope='SDD development screen of frozen H4; precision-only change, not new training or held-out confirmation')
    raw = Path('results/raw')
    raw.mkdir(parents=True, exist_ok=True)
    for name in names:
        suffix = '.csv' if name == 'metrics.csv' else '_' + name
        shutil.copyfile(root / name, raw / (args.prefix + suffix))
    for name, values in (('paired', paired), ('summary', [summary]), ('per_class', per_class)):
        write_rows(raw / f'{args.prefix}_{name}.csv', values)
    (raw / f'{args.prefix}_gate.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['| Mode | AP | AP75 | Interleaved p50 ms | Interleaved p95 ms | Peak allocated GiB |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {summary[arm + '_AP']:.3f} | {summary[arm + '_AP75']:.3f} | "
                     f"{summary[arm + '_interleaved_latency_p50_ms']:.3f} | {summary[arm + '_interleaved_latency_p95_ms']:.3f} | "
                     f"{summary[arm + '_peak_allocated_gb']:.3f} |")
    (root / 'summary.md').write_text('\n'.join(lines) + '\n\n' + json.dumps(report, indent=2) + '\n')
    print('\n'.join(lines))
    print(json.dumps(report, indent=2))
    os.environ.setdefault('MPLCONFIGDIR', str(root.resolve() / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for row in paired:
        axes[0].plot([row[a + '_interleaved_latency_p50_ms'] for a in ARMS], [row[a + '_AP'] for a in ARMS],
                     marker='o', label=f"Checkpoint {row['train_seed']}")
    axes[0].set(xlabel='Interleaved model-call p50 (ms)', ylabel='SDD AP')
    axes[0].legend(fontsize=8)
    seeds = [r['train_seed'] for r in paired]
    for metric in ('AP', 'AP75'):
        axes[1].plot(seeds, [r['delta_' + metric] for r in paired], marker='o', label=metric)
    axes[1].axhline(-0.3, color='gray', linestyle='--', label='Mean floor -0.3')
    axes[1].set(xlabel='Checkpoint training seed', ylabel='FP16 - FP32 AP', xticks=seeds)
    axes[1].legend(fontsize=8)
    axes[2].bar([str(s) for s in seeds], [r['p50_reduction_percent'] for r in paired])
    axes[2].axhline(15, color='black', linestyle='--', label='Mean gate 15%')
    axes[2].axhline(10, color='gray', linestyle=':', label='Each checkpoint 10%')
    axes[2].set(xlabel='Checkpoint training seed', ylabel='Interleaved p50 reduction (%)')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle('Backbone-only FP16; frozen H4/P300, head FP32\n3 checkpoints × 3 inference seeds × 750 development images')
    fig.tight_layout()
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'results/figures/{args.prefix}.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
