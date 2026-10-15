"""B1: frozen H4 FP32 versus backbone-only FP16, with interleaved latency."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_distill import state_digest
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_summarize_distill import write_rows

ARMS = ('fp32', 'fp16')


def arm_order(train_seed, eval_seed, index=0):
    return ARMS if (train_seed + eval_seed + index) % 2 == 0 else ARMS[::-1]


def check_prediction(predictions):
    import torch
    for output in predictions:
        inst = output['instances']
        for tensor in (inst.pred_boxes.tensor, inst.scores):
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError('Non-finite or non-FP32 final prediction')


def audit_precision(model, batch, train_seed):
    """Hooks exist only in this audit; no hooks or feature caching during timing."""
    import torch
    from diffusiondet.proposal_budget import proposal_noise
    captured = {}

    def save(name, tensor):
        captured[name] = tensor.detach().cpu().clone()

    def backbone_hook(module, inputs, outputs):
        save('backbone_input', inputs[0])
        for name, tensor in outputs.items():
            save('fpn_' + name, tensor)

    def head_hook(module, inputs, outputs):
        for name, tensor in zip(model.in_features, inputs[0]):
            save('head_' + name, tensor)
        save('head_boxes', inputs[1])
        save('head_logits', outputs[0])
        save('head_coords', outputs[1])

    handles = [model.backbone.bottom_up.stem.conv1.register_forward_hook(
        lambda module, inputs, output: save('stem_conv', output)),
        model.backbone.register_forward_hook(backbone_hook), model.head.register_forward_hook(head_hook)]
    captures, rows = {}, []
    try:
        with torch.no_grad():
            for arm in ARMS:
                captured = {}
                model.eval_backbone_precision = arm
                torch.manual_seed(image_seed(0, batch[0]['image_id']))
                noise = proposal_noise(1, 300, 300, model.device)
                prediction = model(batch, inference_noise=noise)
                check_prediction(prediction)
                save('final_boxes', prediction[0]['instances'].pred_boxes.tensor)
                save('final_scores', prediction[0]['instances'].scores)
                save('noise', noise)
                if captured['head_coords'].shape[:3] != (4, 1, 300):
                    raise RuntimeError('Expected physical H4 and P300')
                for name, tensor in captured.items():
                    expected = torch.float16 if arm == 'fp16' and (name == 'stem_conv' or name.startswith('fpn_')) else torch.float32
                    if tensor.dtype != expected or not torch.isfinite(tensor).all():
                        raise RuntimeError(f'Precision boundary failed: {arm}/{name}/{tensor.dtype}')
                for name in model.in_features:
                    if not torch.equal(captured['head_' + name], captured['fpn_' + name].float()):
                        raise RuntimeError('Head features differ from FP32-restored FPN outputs')
                captures[arm] = captured
                rows.append(dict(train_seed=train_seed, image_id=batch[0]['image_id'], precision=arm,
                                 dtypes={k: str(v.dtype) for k, v in captured.items()},
                                 shapes={k: list(v.shape) for k, v in captured.items()}, finite=True,
                                 fp32_feature_boundary=True))
    finally:
        for handle in handles:
            handle.remove()
        model.eval_backbone_precision = 'fp32'
    differences = {}
    for name in captures['fp32']:
        if name.startswith('fpn_') or name == 'stem_conv':
            base, actual = captures['fp32'][name].float(), captures['fp16'][name].float()
            differences[name] = dict(max_abs_error=(actual - base).abs().max().item(),
                                     relative_l2=((actual - base).norm() / base.norm().clamp_min(1e-12)).item())
    rows[1]['feature_errors'] = differences
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--checkpoint-root', type=Path, default=Path('runs/v3/time_alignment_20261011'))
    parser.add_argument('--dataset-root', default='datasets')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0 or 0 < args.limit <= 20:
        parser.error('limit must be 0 (full) or greater than 20 (warmup)')
    if any(len(values) != len(set(values)) for values in (args.train_seeds, args.eval_seeds)):
        parser.error('Seed values must be unique')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import detectron2
    import yaml
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
    from detectron2.modeling import build_model
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.proposal_budget import proposal_noise
    from diffusiondet.util.model_ema import add_model_ema_configs

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
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
    cfg.merge_from_file('configs/lab/v3/sdd.res50.student4.yaml')
    cfg.DATASET_ROOT = str(Path(args.dataset_root).resolve())
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.DiffusionDet.INFERENCE_TOPK = 300
    cfg.OUTPUT_DIR = str(args.out_dir)
    cfg.freeze()
    dd = cfg.MODEL.DiffusionDet
    required = dict(NUM_HEADS=4, EVAL_NUM_HEADS=0, NUM_PROPOSALS=300, INFERENCE_TOPK=300,
                    SAMPLE_STEP=1, FORMULATION='vp', SOLVER='ddim', BOX_RENEWAL=False,
                    USE_ENSEMBLE=False, ATTENTION_NEED_WEIGHTS=False, USE_NMS=True,
                    FINAL_CLASS_LOSS='focal', EVAL_BACKBONE_PRECISION='fp32', DEEP_SUPERVISION=True)
    if any(getattr(dd, k) != v for k, v in required.items()) or cfg.MODEL_EMA.ENABLED or cfg.SOLVER.AMP.ENABLED:
        raise ValueError('Config differs from the frozen B1 protocol')
    register_all(cfg=cfg)
    full_records = sorted(DatasetCatalog.get('sdd_val'), key=lambda row: row['image_id'])
    records = full_records[:args.limit] if args.limit else full_records
    loader = build_detection_test_loader(dataset=records, mapper=DatasetMapper(cfg, is_train=False), num_workers=0, batch_size=1)
    metadata = MetadataCatalog.get('sdd_val')
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path('docs/v3/BACKBONE_PRECISION.md'),
               Path('scripts/v3_distill.py'), Path('scripts/v3_head_sweep.py'), Path('scripts/v3_summarize_distill.py'),
               *[Path('diffusiondet') / name for name in ('backbone_precision.py', 'config.py', 'detector.py',
                   'head.py', 'proposal_budget.py', 'solvers/ddim.py', 'solvers/__init__.py', 'solvers/base.py')]]
    config_path = Path('configs/lab/v3/sdd.res50.student4.backbone_fp16.yaml')
    while True:
        sources.append(config_path)
        parent = yaml.safe_load(config_path.read_text()).get('_BASE_')
        if parent is None:
            break
        config_path = (config_path.parent / parent).resolve().relative_to(Path.cwd())
    for source in sources:
        target = args.out_dir / 'source_snapshot' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    checkpoints = {str(s): args.checkpoint_root / f'seed{s}_uniform/model_final.pth' for s in args.train_seeds}
    for path in checkpoints.values():
        training = json.loads((path.parent / 'training_summary.json').read_text())
        if sha256(path) != training['checkpoint_sha256']:
            raise ValueError('T1 checkpoint hash mismatch')
    reference_files = [args.checkpoint_root / name for name in ('manifest.json', 'COMPLETE.json', 'metrics.csv')]
    for source in reference_files:
        (args.out_dir / 'reference').mkdir(exist_ok=True)
        shutil.copyfile(source, args.out_dir / 'reference' / source.name)
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoint_sha256={s: sha256(p) for s, p in checkpoints.items()},
                    source_sha256={str(p): sha256(p) for p in sources},
                    reference_sha256={p.name: sha256(p) for p in reference_files},
                    annotation_path=metadata.json_file, annotation_sha256=sha256(metadata.json_file),
                    image_ids=[r['image_id'] for r in records], full_images=len(full_records),
                    category_names=metadata.thing_classes,
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    tf32=False, cudnn_benchmark=False, cudnn_deterministic=False, threads=4,
                    precision_modes=list(ARMS), warmup=20, measure=min(200, len(records) - 20),
                    protocol='Frozen T1 uniform H4/P300/NFE1/focal; only backbone autocast differs; head FP32; no hooks during timing',
                    timing='Synchronized wall time including shared full noise pool and entire model call; excludes loader/CPU resize/seeding/audits',
                    primary_speed_evidence='Per-image AB/BA interleaved timing, 20 warmup + 200 measured per arm')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out_dir / 'config.yaml').write_text(cfg.dump())
    rows, timing, audits, states, artifacts = [], [], [], [], {}

    def invoke(model, batch, arm, seed):
        model.eval_backbone_precision = arm
        torch.manual_seed(image_seed(seed, batch[0]['image_id']))
        torch.cuda.synchronize()
        start = time.perf_counter()
        noise = proposal_noise(1, 300, 300, model.device)
        prediction = model(batch, inference_noise=noise)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        check_prediction(prediction)
        return prediction, elapsed

    def remember(path):
        artifacts[str(path.relative_to(args.out_dir))] = sha256(path)

    with torch.no_grad():
        for train_seed in args.train_seeds:
            model = build_model(cfg).eval()
            state = torch.load(checkpoints[str(train_seed)], map_location='cpu')
            if state.get('arm') != 'uniform' or state.get('train_seed') != train_seed:
                raise ValueError('Expected matching T1 uniform checkpoint')
            model.load_state_dict(state['model'], strict=True)
            del state
            initial_digest = state_digest(model)
            for index, batch in enumerate(loader):
                if index >= 2:
                    break
                audits.extend(audit_precision(model, batch, train_seed))
            (args.out_dir / 'precision_audit.json').write_text(json.dumps(audits, indent=2) + '\n')
            for seed in args.eval_seeds:
                for arm in arm_order(train_seed, seed):
                    arm_dir = args.out_dir / f'train{train_seed}_eval{seed}_{arm}'
                    arm_dir.mkdir()
                    evaluator = COCOEvaluator('sdd_val', tasks=('bbox',), distributed=False, output_dir=str(arm_dir))
                    evaluator.reset()
                    samples = []
                    torch.cuda.reset_peak_memory_stats()
                    for index, batch in enumerate(loader):
                        prediction, elapsed = invoke(model, batch, arm, seed)
                        if 20 <= index < 220:
                            samples.append(dict(index=index, image_id=batch[0]['image_id'], wall_ms=elapsed))
                        evaluator.process(batch, prediction)
                    metrics = evaluator.evaluate(img_ids=manifest['image_ids'])['bbox']
                    times = [r['wall_ms'] for r in samples]
                    row = dict(train_seed=train_seed, eval_seed=seed, precision=arm, images=len(records),
                               full_dataset=len(records) == len(full_records), **metrics,
                               latency_p50_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times, 95)),
                               measured_images=len(times), peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                               finite_prediction_images=len(records))
                    rows.append(row)
                    (arm_dir / 'metrics.json').write_text(json.dumps(row, indent=2) + '\n')
                    (arm_dir / 'timing.json').write_text(json.dumps(samples) + '\n')
                    for name in ('coco_instances_results.json', 'instances_predictions.pth', 'metrics.json', 'timing.json'):
                        remember(arm_dir / name)
                    write_rows(args.out_dir / 'metrics.csv', rows)
                    print('[b1-eval] ' + json.dumps(row), flush=True)
                samples = []
                for index, batch in enumerate(loader):
                    if index >= 220:
                        break
                    order = arm_order(train_seed, seed, index)
                    for position, arm in enumerate(order):
                        prediction, elapsed = invoke(model, batch, arm, seed)
                        if index >= 20:
                            samples.append(dict(index=index, image_id=batch[0]['image_id'], precision=arm,
                                                position=position, wall_ms=elapsed, finite=True))
                path = args.out_dir / f'interleaved_train{train_seed}_eval{seed}.json'
                path.write_text(json.dumps(samples) + '\n')
                remember(path)
                for arm in ARMS:
                    times = [r['wall_ms'] for r in samples if r['precision'] == arm]
                    timing.append(dict(train_seed=train_seed, eval_seed=seed, precision=arm,
                                       latency_p50_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times, 95)),
                                       measured_images=len(times)))
                write_rows(args.out_dir / 'interleaved.csv', timing)
                print('[b1-timing] ' + json.dumps(timing[-2:]), flush=True)
            final_digest = state_digest(model)
            if final_digest != initial_digest:
                raise RuntimeError('Frozen model state changed during evaluation')
            states.append(dict(train_seed=train_seed, initial_sha256=initial_digest, final_sha256=final_digest,
                               state_unchanged=True, all_float_state_fp32=all(
                                   not t.is_floating_point() or t.dtype == torch.float32 for t in model.state_dict().values())))
            (args.out_dir / 'state_audit.json').write_text(json.dumps(states, indent=2) + '\n')
            del model, prediction
            torch.cuda.empty_cache()
    (args.out_dir / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(dict(evaluations=len(rows), timing_cells=len(timing),
                                                               checkpoints=len(states))) + '\n')


if __name__ == '__main__':
    main()
