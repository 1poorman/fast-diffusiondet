"""W1: fixed-budget WHEAT H6 rebuild, with no validation-based selection."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import batch_digest, state_digest
from scripts.v3_head_sweep import image_seed, sha256


def lr_factor(step):
    alpha = min(step / 100, 1.)
    return (0.01 * (1 - alpha) + alpha) * (0.1 if step >= 1120 else 1.)


def training_health(rows):
    if not rows or [r['step'] for r in rows] != list(range(len(rows))):
        raise ValueError('Missing or unordered training attempts')
    skipped = sum(r['skipped'] for r in rows)
    tail = rows[100:]
    rate = sum(r['skipped'] for r in tail) / len(tail) if tail else None
    return dict(attempts=len(rows), skipped=skipped, updates=len(rows) - skipped,
                post_warmup_skip_rate=rate, passed=rate is not None and rate < 0.01)


def check_transfer(info, state_keys):
    expected = {f'head.head_series.{i}.class_logits.{kind}'
                for i in range(6) for kind in ('weight', 'bias')}
    if (not info or {r[0] for r in info['skipped_shape']} != expected
            or info['loaded'] != len(state_keys) - len(expected)
            or info['skipped_missing_in_model'] or info['unexpected_in_model']):
        raise ValueError('COCO transfer must cover all tensors except 12 class-logit tensors')


def verify_data(root):
    complete = json.loads((root / 'COMPLETE.json').read_text())
    result = {}
    for split, count in [('wheat_train', 695), ('wheat_val', 123)]:
        path = root / f'{split}.json'
        if sha256(path) != complete['artifacts_sha256'][path.name]:
            raise ValueError('D1 split audit hash mismatch')
        audit = json.loads(path.read_text())
        if sha256(audit['annotation_path']) != audit['annotation_sha256']:
            raise ValueError('Source annotation changed since D1')
        if len(audit['records']) != count:
            raise ValueError('Unexpected WHEAT split size')
        for row in audit['records']:
            if sha256(row['resolved_path']) != row['file_sha256']:
                raise ValueError('Image changed since D1: ' + row['resolved_path'])
        result[split] = dict(audit_sha256=sha256(path), annotation_sha256=audit['annotation_sha256'],
                             image_ids=[r['image_id'] for r in audit['records']])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1', choices=['1'])
    parser.add_argument('--steps', type=int, default=1400, choices=[10, 1400])
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--weights', type=Path, default=Path('pretained/diffdet_coco_res50.pth'))
    parser.add_argument('--data-audit', type=Path, default=Path('runs/v3/data_readiness_20261015'))
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.train_seeds)) != len(args.train_seeds) or not args.weights.is_file():
        parser.error('Need unique seeds and an existing COCO checkpoint')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import detectron2
    import yaml
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, build_detection_train_loader
    from detectron2.data.samplers import TrainingSampler
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.dataset_mapper import DiffusionDetDatasetMapper
    from diffusiondet.data_register import register_all
    from diffusiondet.util.model_ema import add_model_ema_configs
    from diffusiondet.weights import load_transfer_weights

    args.out_dir.mkdir(parents=True, exist_ok=False)
    setup_logger(output=str(args.out_dir))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file('configs/lab/v3/wheat.res50.rebuild.yaml')
    cfg.DATASET_ROOT = str(Path('datasets').resolve())
    cfg.MODEL.DEVICE = 'cuda'
    cfg.OUTPUT_DIR = str(args.out_dir)
    cfg.freeze()
    dd = cfg.MODEL.DiffusionDet
    assert (dd.NUM_HEADS == 6 and dd.NUM_PROPOSALS == 300 and dd.TRAIN_T_DIST == 'uniform'
            and dd.FINAL_CLASS_LOSS == 'focal' and dd.DEEP_SUPERVISION
            and not dd.ROI_POOLER_KNOWN_COUNTS and not dd.EVAL_ROI_GPU_DISPATCH
            and not cfg.MODEL_EMA.ENABLED and cfg.SOLVER.AMP.ENABLED
            and cfg.SOLVER.BASE_LR == 1e-4 and cfg.SOLVER.WEIGHT_DECAY == 1e-4
            and cfg.SOLVER.BACKBONE_MULTIPLIER == 1 and cfg.SOLVER.IMS_PER_BATCH == 8
            and cfg.SOLVER.WARMUP_FACTOR == .01 and cfg.SOLVER.WARMUP_ITERS == 100
            and tuple(cfg.SOLVER.STEPS) == (1120,))
    data = verify_data(args.data_audit)
    register_all(cfg=cfg, dataset_names=['wheat_train', 'wheat_val'])
    records = DatasetCatalog.get('wheat_train')
    assert [r['image_id'] for r in records] == data['wheat_train']['image_ids']
    sources = [Path(__file__).relative_to(Path.cwd()), Path('docs/v3/WHEAT_CONFIRMATION.md'),
               Path('scripts/v3_distill.py'), Path('scripts/v3_head_sweep.py'),
               *sorted(Path('diffusiondet').rglob('*.py'))]
    config = Path('configs/lab/v3/wheat.res50.rebuild.yaml')
    while True:
        sources.append(config)
        parent = yaml.safe_load(config.read_text()).get('_BASE_')
        if parent is None:
            break
        config = (config.parent / parent).resolve().relative_to(Path.cwd())
    for path in sources:
        target = args.out_dir / 'source_snapshot' / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    source_sha256={str(p): sha256(p) for p in sources},
                    pretrain_sha256=sha256(args.weights), data=data,
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), threads=4, tf32=False,
                    git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    budget='attempts; schedule advances on skipped AMP updates; no validation')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out_dir / 'config.yaml').write_text(cfg.dump())

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    summaries = []
    for seed in args.train_seeds:
        out = args.out_dir / f'seed{seed}_uniform'
        out.mkdir()
        seed_all(seed)
        model = build_model(cfg).train()
        transfer = load_transfer_weights(model, path=str(args.weights))
        check_transfer(transfer, model.state_dict())
        initial = state_digest(model)
        seed_all(seed)
        loader = iter(build_detection_train_loader(dataset=records,
            mapper=DiffusionDetDatasetMapper(cfg, is_train=True),
            sampler=TrainingSampler(len(records), seed=seed), total_batch_size=8,
            aspect_ratio_grouping=True, num_workers=0))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.cuda.amp.GradScaler()
        rows = []
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (out / 'training.jsonl').open('w') as log, EventStorage() as storage:
            for step in range(args.steps):
                seed_all(image_seed(seed, step))
                tick = time.perf_counter()
                batch = next(loader)
                digest = batch_digest(batch)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    losses = model(batch)
                    total = sum(losses.values())
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss at seed{seed} attempt{step}')
                old_scale = scaler.get_scale()
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                row = dict(step=step, step_seed=image_seed(seed, step), input_digest=digest,
                           image_ids=[r['image_id'] for r in batch],
                           lr=optimizer.param_groups[0]['lr'], loss_total=float(total.detach()),
                           losses={k: float(v.detach()) for k, v in losses.items()},
                           grad_norm=norm if math.isfinite(norm) else None,
                           gradient_overflow=not math.isfinite(norm), scale_before=old_scale,
                           scale_after=scaler.get_scale(), skipped=scaler.get_scale() < old_scale,
                           step_s=time.perf_counter() - tick)
                if row['gradient_overflow'] and not row['skipped']:
                    raise FloatingPointError('Nonfinite gradients were not skipped')
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                rows.append(row)
                if step % 20 == 0 or step == args.steps - 1:
                    print(f"[w1-train] seed={seed} step={step+1}/{args.steps} "
                          f"loss={row['loss_total']:.4f} skipped={row['skipped']} "
                          f"step_s={row['step_s']:.3f}", flush=True)
                scheduler.step()
                storage.step()
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('Nonfinite final model state')
        checkpoint = out / 'model_final.pth'
        torch.save(dict(model=model.state_dict(), iteration=args.steps-1, train_seed=seed,
                        arm='uniform', config=cfg.dump()), checkpoint)
        summary = dict(train_seed=seed, health=training_health(rows),
                       train_wall_s=time.perf_counter()-start,
                       peak_train_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
                       initial_state_sha256=initial, final_state_sha256=state_digest(model),
                       transfer=transfer, checkpoint_sha256=sha256(checkpoint),
                       training_sha256=sha256(out / 'training.jsonl'))
        (out / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        summaries.append(summary)
        del model, optimizer, scheduler, scaler, loader, params, batch, losses, total
        torch.cuda.empty_cache()
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(dict(runs=len(summaries),
        smoke=args.steps == 10, summaries=summaries,
        training_health_passed=all(s['health']['passed'] for s in summaries)), indent=2) + '\n')


if __name__ == '__main__':
    main()
