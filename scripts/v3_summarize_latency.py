"""Validate L1's complete matrix and distinguish plain latency from perturbed diagnostics."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics as stats
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256
from scripts.v3_latency_profile import ARMS
from scripts.v3_profile_utils import exclusive_rows
from scripts.v3_summarize_distill import read_rows, write_rows


def mean(rows, key):
    return stats.mean(float(row[key]) for row in rows)


def sync_operators(trace):
    """Find the innermost CPU operator/scope containing each CUDA runtime wait."""
    by_thread = defaultdict(list)
    for event in trace:
        if event.get('ph') == 'X' and event.get('cat') in ('cpu_op', 'user_annotation', 'cuda_runtime'):
            by_thread[(event['pid'], event['tid'])].append(event)
    rows = []
    for events in by_thread.values():
        active = []
        for event in sorted(events, key=lambda e: (e['ts'], -e['dur'])):
            if event['cat'] in ('cpu_op', 'user_annotation'):
                active.append(event)
            elif 'Synchronize' in event['name']:
                active = [p for p in active if p['ts'] + p['dur'] + 0.01 >= event['ts'] + event['dur']]
                operators = [p for p in active if p['cat'] == 'cpu_op']
                scopes = [p for p in active if p['cat'] == 'user_annotation' and p['name'].startswith('L1/')]
                rows.append(dict(runtime=event['name'], host_us=event['dur'],
                                 operator=min(operators, key=lambda e: e['dur'])['name'] if operators else 'outside_operator',
                                 scope=min(scopes, key=lambda e: e['dur'])['name'] if scopes else 'outside_call'))
    return rows


def audit_samples(plain, events, image_ids, depth, warmup=20):
    expected_names = Counter(dict(call=1, preprocess=1, backbone=1, head=1, selection_nms=1, postprocess=1,
                                  roi_pool=depth, attention=depth, dynamic_conv=depth, box_decode=depth))
    expected_names.update({f'stage{i}': 1 for i in range(1, depth + 1)})
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Duplicate expected images')
    for records in (plain, events):
        if [row['image_id'] for row in records] != image_ids or [row['index'] for row in records] != list(range(warmup, warmup + len(image_ids))):
            raise ValueError('Incomplete or reordered timing window')
    for original, measured in zip(plain, events):
        if original['prediction_sha256'] != measured['prediction_sha256'] or original['input_hw'] != measured['input_hw']:
            raise ValueError('Instrumentation changed predictions or inputs')
        if any(not math.isfinite(row['wall_ms']) or row['wall_ms'] <= 0 for row in (original, measured)):
            raise ValueError('Invalid model latency')
        spans = measured['spans']
        if Counter(row['name'] for row in spans) != expected_names or spans[0]['name'] != 'call':
            raise ValueError('Incorrect executed stage/component counts')
        if [i for i, row in enumerate(spans) if row['parent'] is None] != [0]:
            raise ValueError('Timing tree needs one root')
        computed = exclusive_rows(spans)
        for row, checked in zip(spans, computed):
            for key in ('self_host_ms', 'self_stream_ms'):
                if not math.isclose(row[key], checked[key], abs_tol=1e-7):
                    raise ValueError('Exclusive duration was not computed from direct children')
        if not math.isclose(sum(s['self_stream_ms'] for s in spans), spans[0]['stream_ms'], abs_tol=1e-5):
            raise ValueError('Nested intervals do not partition the root')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    manifest = json.loads((root / 'manifest.json').read_text())
    protocol = manifest['args']
    if (protocol['train_seeds'] != [10, 11, 12] or protocol['eval_seeds'] != [0, 1, 2]
            or (protocol['warmup'], protocol['measure'], protocol['profile_images']) != (20, 200, 10)
            or len(manifest['image_ids']) != 220 or manifest['full_val_images'] != 750):
        raise ValueError('Final L1 report requires the predeclared complete protocol')
    if json.loads((root / 'COMPLETE.json').read_text()) != dict(cells=27, profiles=9):
        raise ValueError('Incomplete L1 run')
    for name, digest in manifest['source_sha256'].items():
        if sha256(root / 'source_snapshot' / name) != digest:
            raise ValueError(f'Source snapshot differs: {name}')
    if (sha256(root / 'detectron2_poolers_snapshot.py') != manifest['external_poolers']['sha256']
            or sha256(manifest['annotation_path']) != manifest['annotation_sha256']):
        raise ValueError('Dependency or annotation hash mismatch')
    for info in manifest['checkpoints'].values():
        if sha256(info['path']) != info['sha256']:
            raise ValueError('Checkpoint hash mismatch')
    metrics = read_rows(root / 'metrics.csv')
    expected = {(s, e, a, m) for s in (10, 11, 12) for e in (0, 1, 2) for a in ARMS for m in ('plain', 'events')}
    keys = {(int(r['train_seed']), int(r['eval_seed']), r['arm'], r['mode']) for r in metrics}
    if keys != expected or len(metrics) != 54:
        raise ValueError('Incomplete or duplicate latency matrix')
    evidence = {name: sha256(root / name) for name in ('manifest.json', 'config.yaml', 'metrics.csv', 'profiles.json', 'COMPLETE.json')}
    scopes = []
    import yaml
    for seed in (10, 11, 12):
        for e in (0, 1, 2):
            for arm in ARMS:
                cell = root / f'train{seed}_eval{e}_{arm}'
                depth, proposals = (6 if arm.startswith('h6') else 4), int(arm.split('_p')[1])
                if json.loads((cell / 'execution.json').read_text()) != dict(shape=[depth, 1, proposals, 4]):
                    raise ValueError('Actual head output shape differs')
                cfg = yaml.safe_load((cell / 'config.yaml').read_text())['MODEL']['DiffusionDet']
                if (cfg['NUM_HEADS'], cfg['NUM_PROPOSALS'], cfg['INFERENCE_TOPK'], cfg['EVAL_NUM_HEADS']) != (depth, proposals, 300, 0):
                    raise ValueError('Arm configuration differs')
                records = {mode: [json.loads(s) for s in (cell / f'{mode}.jsonl').read_text().splitlines()] for mode in ('plain', 'events')}
                audit_samples(records['plain'], records['events'], manifest['image_ids'][20:], depth)
                for mode in records:
                    values = [r['wall_ms'] for r in records[mode]]
                    row = next(r for r in metrics if (int(r['train_seed']), int(r['eval_seed']), r['arm'], r['mode']) == (seed, e, arm, mode))
                    if int(row['images']) != 200 or any(not math.isclose(float(row[k]), value, abs_tol=1e-7)
                            for k, value in dict(p50_ms=np.median(values), p95_ms=np.percentile(values, 95), mean_ms=np.mean(values)).items()):
                        raise ValueError('Timing quantile summary differs from samples')
                by_scope = defaultdict(list)
                for row in records['events']:
                    for span in row['spans']:
                        by_scope[span['name']].append(span)
                for name, spans in by_scope.items():
                    scopes.append(dict(train_seed=seed, eval_seed=e, arm=arm, scope=name, calls_per_image=len(spans) / 200,
                                       **{key: sum(s[key] for s in spans) / 200 for key in ('host_ms', 'stream_ms', 'self_host_ms', 'self_stream_ms')}))
                for name in ('plain.jsonl', 'events.jsonl', 'config.yaml', 'execution.json'):
                    evidence[str((cell / name).relative_to(root))] = sha256(cell / name)
    profiles = json.loads((root / 'profiles.json').read_text())
    if len(profiles) != 9 or {(p['train_seed'], p['eval_seed'], p['arm']) for p in profiles} != {(s, 0, a) for s in (10, 11, 12) for a in ARMS}:
        raise ValueError('Incomplete operator profile matrix')
    operator_rows, sync_rows, sync_operator_rows = [], [], []
    for profile in profiles:
        if profile['images'] != 10 or profile['bit_exact'] is not True or profile['kernel_count'] <= 0:
            raise ValueError('Invalid operator trace evidence')
        cell = root / f"train{profile['train_seed']}_eval0_{profile['arm']}"
        if sha256(cell / 'trace.json.gz') != profile['trace_sha256']:
            raise ValueError('Operator trace hash mismatch')
        key = dict(train_seed=profile['train_seed'], eval_seed=0, arm=profile['arm'])
        for row in read_rows(cell / 'operators.csv'):
            operator_rows.append(dict(**key, **row))
        aggregated = defaultdict(lambda: [0, 0.0])
        for row in read_rows(cell / 'synchronizations.csv'):
            group = aggregated[(row['scope'], row['runtime'])]
            group[0] += 1
            group[1] += float(row['host_us'])
        for (scope, runtime), (count, duration) in aggregated.items():
            sync_rows.append(dict(**key, scope=scope, runtime=runtime, calls_per_image=count / 10, host_ms_per_image=duration / 10000))
        with gzip.open(cell / 'trace.json.gz', 'rt') as stream:
            trace = json.load(stream)
        mapped = sync_operators(trace['traceEvents'])
        if len(mapped) != profile['sync_calls']:
            raise ValueError('Synchronization attribution is incomplete')
        aggregated_ops = defaultdict(lambda: [0, 0.0])
        for row in mapped:
            group = aggregated_ops[(row['scope'], row['operator'], row['runtime'])]
            group[0] += 1
            group[1] += row['host_us']
        for (scope, operator, runtime), (count, duration) in aggregated_ops.items():
            sync_operator_rows.append(dict(**key, scope=scope, operator=operator, runtime=runtime,
                                           calls_per_image=count / 10, host_ms_per_image=duration / 10000))
        del trace
        for name in ('operators.csv', 'synchronizations.csv'):
            evidence[str((cell / name).relative_to(root))] = sha256(cell / name)
    paired = []
    for seed in (10, 11, 12):
        row = dict(train_seed=seed)
        for arm in ARMS:
            for mode in ('plain', 'events'):
                group = [r for r in metrics if int(r['train_seed']) == seed and r['arm'] == arm and r['mode'] == mode]
                for metric in ('p50_ms', 'p95_ms', 'mean_ms'):
                    row[f'{arm}_{mode}_{metric}'] = mean(group, metric)
            row[f'{arm}_event_overhead_percent'] = 100 * (row[f'{arm}_events_p50_ms'] / row[f'{arm}_plain_p50_ms'] - 1)
        row['P100_p50_reduction_percent'] = 100 * (1 - row['h4_p100_plain_p50_ms'] / row['h4_p300_plain_p50_ms'])
        row['H4_p50_reduction_percent'] = 100 * (1 - row['h4_p300_plain_p50_ms'] / row['h6_p300_plain_p50_ms'])
        paired.append(row)
    summary = []
    for arm in ARMS:
        profile = [p for p in profiles if p['arm'] == arm]
        overheads = []
        for seed in (10, 11, 12):
            for e in (0, 1, 2):
                cell = {r['mode']: float(r['p50_ms']) for r in metrics
                        if int(r['train_seed']) == seed and int(r['eval_seed']) == e and r['arm'] == arm}
                overheads.append(100 * (cell['events'] / cell['plain'] - 1))
        summary.append(dict(arm=arm, **{key: mean(paired, f'{arm}_{key}') for key in ('plain_p50_ms', 'plain_p95_ms', 'plain_mean_ms', 'event_overhead_percent')},
                            event_delta_min_percent=min(overheads), event_delta_max_percent=max(overheads),
                            event_root_stream_ms=mean([r for r in scopes if r['arm'] == arm and r['scope'] == 'call'], 'stream_ms'),
                            kernel_ms_per_image=mean(profile, 'kernel_union_ms') / 10,
                            kernels_per_image=mean(profile, 'kernel_count') / 10))
    report = dict(cells=27, output_parity_images=5400, operator_parity_images=90,
                  P100_p50_reduction_percent=mean(paired, 'P100_p50_reduction_percent'),
                  H4_p50_reduction_percent=mean(paired, 'H4_p50_reduction_percent'),
                  report_script_sha256=sha256(__file__), evidence_sha256=evidence,
                  scope='SDD timing window only; plain latency is the performance result; nested stream spans and profiler kernels are perturbed diagnostics, not additive speedup estimates')
    raw = Path('results/raw')
    for name, rows in (('', metrics), ('_summary', summary), ('_paired', paired), ('_scopes', scopes), ('_operators', operator_rows), ('_sync', sync_rows), ('_sync_operators', sync_operator_rows)):
        write_rows(raw / f'v3_latency{name}.csv', rows)
    for name, suffix in (('manifest.json', '_manifest.json'), ('config.yaml', '_config.yaml'), ('profiles.json', '_profiles.json')):
        shutil.copyfile(root / name, raw / f'v3_latency{suffix}')
    (raw / 'v3_latency_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, P100_p50_reduction_percent=report['P100_p50_reduction_percent'],
                          H4_p50_reduction_percent=report['H4_p50_reduction_percent']), indent=2))

    os.environ.setdefault('MPLCONFIGDIR', str(root.resolve() / 'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(3)
    axes[0].bar(x, [r['plain_p50_ms'] for r in summary], color=['tab:blue', 'tab:orange', 'tab:green'])
    for row in paired:
        axes[0].plot(x, [row[f'{a}_plain_p50_ms'] for a in ARMS], 'k.', alpha=0.6)
    axes[0].set(title='Uninstrumented model latency', ylabel='p50 wall time (ms)')
    groups = [('backbone', 'Backbone'), ('preprocess', 'Preprocess'), ('roi_pool', 'ROI pool'),
              ('attention', 'Attention'), ('dynamic_conv', 'DynamicConv'), ('box_decode', 'Box decode'),
              ('stage_other', 'Other stage work'), ('remaining', 'Sampling / selection / output')]
    bottom = np.zeros(3)
    for index, (group, label) in enumerate(groups):
        values = []
        for arm in ARMS:
            selected = [r for r in scopes if r['arm'] == arm and
                        ((r['scope'].startswith('stage') if group == 'stage_other' else
                          r['scope'] in ('call', 'head', 'selection_nms', 'postprocess') if group == 'remaining' else r['scope'] == group))]
            values.append(sum(r['self_stream_ms'] for r in selected) / 9)
        axes[1].bar(x, values, bottom=bottom, label=label, color=plt.get_cmap('tab20')(index))
        bottom += values
    axes[1].set(title='Instrumented exclusive stream intervals', ylabel='Mean interval (ms); includes enqueue gaps')
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    for ax in axes:
        ax.set_xticks(x, ['H4 P300', 'H4 P100', 'H6 P300'])
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('L1 latency audit: FP32 / TF32 off / batch 1 / RTX 3090\n3 checkpoint pairs × 3 inference seeds; 20 warmup + same 200 timed images')
    fig.tight_layout()
    fig.savefig('results/figures/v3_latency.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
