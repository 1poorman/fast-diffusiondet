"""Re-evaluate frozen R2 predictions after removing exact train/val duplicates."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256
from scripts.v3_summarize_distill import read_rows, write_rows
from scripts.v3_data_readiness import compare_splits


def metrics_for_ids(coco, predictions, ids, names):
    import numpy as np
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
    with redirect_stdout(io.StringIO()):
        evaluator = COCOeval_opt(coco, predictions, 'bbox')
        evaluator.params.imgIds = ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = {name: float(value * 100) for name, value in zip(('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'), evaluator.stats[:6])}
    for index, name in enumerate(names):
        precision = evaluator.eval['precision'][:, :, index, 0, -1]
        valid = precision[precision > -1]
        metrics['AP-' + name] = float(np.mean(valid) * 100) if valid.size else float('nan')
    return metrics


def exact_duplicate_ids(overlaps):
    return sorted({image_id for r in overlaps if (r['left'], r['right']) == ('sdd_train', 'sdd_val')
                   for group in r['groups'] for image_id in group['right_image_ids']})


def verify_replay(metrics, row):
    import math
    if any(not math.isfinite(value) or abs(value - float(row[name])) > 1e-9 for name, value in metrics.items()):
        raise ValueError('Full-set replay differs from frozen R2 AP')
    return max(abs(value - float(row[name])) for name, value in metrics.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=Path('runs/v3/data_readiness_20261015'))
    parser.add_argument('--r2-dir', type=Path, default=Path('runs/v3/roi_gpu_dispatch_20261015'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='v3_data_readiness_sensitivity')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    import torch
    from pycocotools.coco import COCO
    torch.set_num_threads(4)
    root = args.r2_dir
    prior = json.loads((args.audit_dir / 'COMPLETE.json').read_text())
    for name, digest in prior['artifacts_sha256'].items():
        if sha256(args.audit_dir / name) != digest:
            raise ValueError('Data audit artifact hash mismatch')
    overlap = json.loads((args.audit_dir / 'overlap.json').read_text())
    train = json.loads((args.audit_dir / 'sdd_train.json').read_text())['records']
    validation = json.loads((args.audit_dir / 'sdd_val.json').read_text())['records']
    recomputed = compare_splits('sdd_train', 'sdd_val', train, validation)
    if recomputed != [r for r in overlap if (r['left'], r['right']) == ('sdd_train', 'sdd_val')]:
        raise ValueError('Overlap summary differs from hashed image records')
    excluded = exact_duplicate_ids(overlap)
    if len(excluded) != 15:
        raise ValueError('Expected the 15 predeclared duplicated SDD validation images')
    manifest = json.loads((root / 'manifest.json').read_text())
    if sha256(manifest['annotation_path']) != manifest['annotation_sha256']:
        raise ValueError('R2 annotation hash mismatch')
    original_ids = manifest['image_ids']
    retained = sorted(set(original_ids) - set(excluded))
    if len(original_ids) != 750 or len(retained) != 735:
        raise ValueError('Unexpected full or retained validation size')
    rows = read_rows(root / 'metrics.csv')
    expected = {(s, e, a) for s in ('10', '11', '12') for e in ('0', '1', '2') for a in ('known_counts', 'gpu_dispatch')}
    if len(rows) != 18 or {(r['train_seed'], r['eval_seed'], r['arm']) for r in rows} != expected:
        raise ValueError('Incomplete R2 matrix')
    artifacts = json.loads((root / 'artifacts.json').read_text())
    with redirect_stdout(io.StringIO()):
        coco = COCO(manifest['annotation_path'])
    assert [coco.cats[c]['name'] for c in sorted(coco.cats)] == manifest['category_names']
    results, inputs, replay_errors = [], {}, []
    for row in rows:
        s, e, arm = row['train_seed'], row['eval_seed'], row['arm']
        relative = f'train{s}_eval{e}_{arm}/coco_instances_results.json'
        path = root / relative
        if sha256(path) != artifacts[relative]:
            raise ValueError('Frozen prediction hash mismatch')
        inputs[relative] = artifacts[relative]
        with redirect_stdout(io.StringIO()):
            predictions = coco.loadRes(str(path))
        full = metrics_for_ids(coco, predictions, original_ids, manifest['category_names'])
        replay_errors.append(verify_replay(full, row))
        reduced = metrics_for_ids(coco, predictions, retained, manifest['category_names'])
        results.append(dict(train_seed=int(s), eval_seed=int(e), arm=arm, images=735, **reduced))
        print('[d1-sensitivity] ' + json.dumps(dict(train_seed=s, eval_seed=e, arm=arm, AP=reduced['AP'], AP75=reduced['AP75'])), flush=True)
    sources = ['scripts/v3_data_sensitivity.py', 'scripts/v3_data_readiness.py', 'docs/v3/DATA_READINESS.md',
               'scripts/v3_summarize_distill.py', 'scripts/v3_head_sweep.py']
    for name in sources:
        target = args.out_dir / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    from statistics import mean
    paired = []
    metric_names = list(reduced)
    for seed in (10, 11, 12):
        record = dict(train_seed=seed)
        for arm in ('known_counts', 'gpu_dispatch'):
            for field in metric_names:
                record[arm + '_' + field] = mean(r[field] for r in results if r['train_seed'] == seed and r['arm'] == arm)
        for field in metric_names:
            record['delta_' + field] = record['gpu_dispatch_' + field] - record['known_counts_' + field]
        paired.append(record)
    summary = {k: mean(r[k] for r in paired) for k in paired[0] if k != 'train_seed'}
    report = dict(completed=True, evaluations=18, full_replays=18, full_replay_max_abs_difference=max(replay_errors),
                  run_dir=str(args.out_dir), r2_run_dir=str(root), data_audit_dir=str(args.audit_dir),
                  full_images=750, retained_images=735, excluded_image_ids=excluded,
                  prediction_sha256=inputs, annotation_sha256=manifest['annotation_sha256'],
                  overlap_sha256=sha256(args.audit_dir / 'overlap.json'),
                  audit_complete_sha256=sha256(args.audit_dir / 'COMPLETE.json'),
                  r2_metrics_sha256=sha256(root / 'metrics.csv'), source_sha256={p: sha256(p) for p in sources},
                  scope='Post-hoc exact-duplicate sensitivity only; no new inference, no speed retest, no new holdout or revised R2 gate.')
    for name, values in [('metrics', results), ('paired', paired), ('summary', [summary])]:
        write_rows(args.out_dir / (name + '.csv'), values)
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
    for name in ('metrics.csv', 'paired.csv', 'summary.csv', 'COMPLETE.json'):
        shutil.copyfile(args.out_dir / name, Path('results/raw') / (args.prefix + '_' + name))
    print('[d1-summary] ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
