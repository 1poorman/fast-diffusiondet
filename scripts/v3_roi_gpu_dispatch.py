"""R2: frozen FP32 H4, GPU level dispatch, and separate profiler audit."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_backbone_precision import check_prediction
from scripts.v3_distill import state_digest
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_profile_utils import prediction_digest
from scripts.v3_roi_dispatch_checks import synthetic_audit
from scripts.v3_roi_dispatch_profile import profile_model
from scripts.v3_summarize_distill import write_rows

ARMS = ('known_counts', 'gpu_dispatch')


def arm_order(train_seed, eval_seed, index=0):
    return ARMS if (train_seed + eval_seed + index) % 2 == 0 else ARMS[::-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--checkpoint-root', type=Path, default=Path('runs/v3/time_alignment_20261011'))
    parser.add_argument('--dataset-root', default='datasets')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--prerequisite', type=Path, default=Path('runs/v3/roi_dispatch_prerequisite_20261015'))
    args = parser.parse_args()
    if args.limit < 0 or 0 < args.limit < 30 or 0 not in args.eval_seeds:
        parser.error('Need limit 0 (full) or >=30, and eval seed 0 for profiling')
    if any(len(values) != len(set(values)) for values in (args.train_seeds, args.eval_seeds)):
        parser.error('Seed values must be unique')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TRITON_CACHE_DIR'] = str(args.out_dir.resolve() / 'triton_cache')
    import numpy as np
    import torch
    import detectron2
    import triton
    import yaml
    import detectron2.modeling.poolers as upstream
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
    from detectron2.modeling import build_model
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.proposal_budget import proposal_noise
    from diffusiondet.roi_pooler import KnownCountROIPooler
    from diffusiondet.roi_gpu_dispatch import GPUDispatchROIPooler
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
                    FINAL_CLASS_LOSS='focal', EVAL_BACKBONE_PRECISION='fp32', DEEP_SUPERVISION=True,
                    ROI_POOLER_KNOWN_COUNTS=False, EVAL_ROI_GPU_DISPATCH=False)
    if (any(getattr(dd, k) != v for k, v in required.items()) or cfg.MODEL_EMA.ENABLED or cfg.SOLVER.AMP.ENABLED
            or cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE != 'ROIAlignV2'):
        raise ValueError('Config differs from the frozen R2 protocol')
    register_all(cfg=cfg)
    full_records = sorted(DatasetCatalog.get('sdd_val'), key=lambda row: row['image_id'])
    records = full_records[:args.limit] if args.limit else full_records
    loader = build_detection_test_loader(dataset=records, mapper=DatasetMapper(cfg, is_train=False), num_workers=0, batch_size=1)
    metadata = MetadataCatalog.get('sdd_val')
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path('docs/v3/ROI_GPU_DISPATCH.md'),
               *[Path('scripts') / name for name in ('v3_roi_dispatch_checks.py', 'v3_roi_dispatch_profile.py', 'v3_backbone_precision.py',
                   'v3_distill.py', 'v3_head_sweep.py', 'v3_profile_utils.py', 'v3_summarize_distill.py')],
               *[Path('diffusiondet') / name for name in ('roi_gpu_dispatch.py', 'roi_gpu_dispatch_kernel.py', 'roi_pooler.py', 'backbone_precision.py', 'config.py',
                   'detector.py', 'head.py', 'proposal_budget.py', 'solvers/ddim.py', 'solvers/__init__.py', 'solvers/base.py')]]
    config_path = Path('configs/lab/v3/sdd.res50.student4.roi_gpu_dispatch.yaml')
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
    prerequisite = json.loads((args.prerequisite / 'COMPLETE.json').read_text())
    prior = json.loads((args.prerequisite / 'manifest.json').read_text())
    if not (prerequisite['passed'] and prerequisite['model']['passed'] and prerequisite['model']['images'] == 32
            and prerequisite['model']['roi_calls'] == 128):
        raise ValueError('Missing real-model numerical prerequisite')
    for name in ('roi_gpu_dispatch.py', 'roi_gpu_dispatch_kernel.py', 'roi_pooler.py'):
        source = 'diffusiondet/' + name
        if sha256(source) != prior['source_sha256'][source]:
            raise ValueError('Prerequisite tested a different pooler source')
    (args.out_dir / 'prerequisite.json').write_text(json.dumps(prerequisite, indent=2) + '\n')
    shutil.copyfile(args.prerequisite / 'manifest.json', args.out_dir / 'prerequisite_manifest.json')
    external = Path(inspect.getfile(upstream))
    shutil.copyfile(external, args.out_dir / 'detectron2_poolers_snapshot.py')
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
                    external_poolers=dict(path=str(external), sha256=sha256(external)),
                    reference_sha256={p.name: sha256(p) for p in reference_files},
                    annotation_path=metadata.json_file, annotation_sha256=sha256(metadata.json_file),
                    image_ids=[r['image_id'] for r in records], full_images=len(full_records), category_names=metadata.thing_classes,
                    torch=torch.__version__, triton=triton.__version__, triton_cache=os.environ['TRITON_CACHE_DIR'], detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    tf32=False, cudnn_benchmark=False, cudnn_deterministic=False, threads=4,
                    arms=list(ARMS), warmup=20, measure=min(200, len(records) - 20), profile_images=10,
                    protocol='Frozen T1 uniform H4/P300/NFE1/focal FP32; R1 known-count control versus GPU level dispatch and ROIAlignV2',
                    timing='Synchronized full-noise+model call; excludes loader/CPU resize/seeding/digests; no hooks or profiler',
                    primary_speed_evidence='Per-image AB/BA interleaved timing, 20 warmup + 200 measured per arm')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out_dir / 'config.yaml').write_text(cfg.dump())
    (args.out_dir / 'numerical_audit.json').write_text(json.dumps(synthetic_audit('cuda'), indent=2) + '\n')
    rows, timing, parity, states, profiles, artifacts = [], [], [], [], [], {}

    def invoke(model, batch, seed):
        torch.manual_seed(image_seed(seed, batch[0]['image_id']))
        noise = proposal_noise(1, 300, 300, model.device)
        prediction = model(batch, inference_noise=noise)
        check_prediction(prediction)
        return prediction

    def timed(model, batch, seed):
        torch.manual_seed(image_seed(seed, batch[0]['image_id']))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        prediction = model(batch, inference_noise=proposal_noise(1, 300, 300, model.device))
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - tick)
        check_prediction(prediction)
        return prediction, elapsed

    def remember(path):
        artifacts[str(path.relative_to(args.out_dir))] = sha256(path)

    def load_model(train_seed):
        model = build_model(cfg).eval()
        state = torch.load(checkpoints[str(train_seed)], map_location='cpu')
        if state.get('arm') != 'uniform' or state.get('train_seed') != train_seed:
            raise ValueError('Expected matching T1 uniform checkpoint')
        model.load_state_dict(state['model'], strict=True)
        if len(model.head.head_series) != 4 or model.num_proposals != 300:
            raise RuntimeError('Executed H/P differs from protocol')
        return model

    def make_poolers(model):
        kwargs = dict(output_size=cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            scales=[1 / model.backbone.output_shape()[k].stride for k in cfg.MODEL.ROI_HEADS.IN_FEATURES],
            sampling_ratio=cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO, pooler_type=cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE)
        return dict(known_counts=KnownCountROIPooler(**kwargs).eval(), gpu_dispatch=GPUDispatchROIPooler(**kwargs).eval())

    def cache_state():
        cache = Path(os.environ['TRITON_CACHE_DIR'])
        return {str(p.relative_to(cache)): sha256(p) for p in sorted(cache.rglob('*')) if p.is_file()}

    jit_rows = []

    with torch.no_grad():
        for train_seed in args.train_seeds:
            model = load_model(train_seed)
            poolers = make_poolers(model)
            initial = state_digest(model)
            # Prepare each input shape before any timing; reject late JIT compilation.
            seen = set()
            tick = time.perf_counter()
            model.head.box_pooler = poolers['gpu_dispatch']
            for batch in loader:
                shape = tuple(batch[0]['image'].shape)
                if shape not in seen:
                    invoke(model, batch, 0)
                    seen.add(shape)
            torch.cuda.synchronize()
            ready_cache = cache_state()
            jit_rows.append(dict(train_seed=train_seed, prepared_shapes=sorted(seen),
                                 preparation_ms=1000 * (time.perf_counter() - tick), cache_after_preparation=ready_cache))
            for seed in args.eval_seeds:
                digests = {}
                for arm in arm_order(train_seed, seed):
                    model.head.box_pooler = poolers[arm]
                    arm_dir = args.out_dir / f'train{train_seed}_eval{seed}_{arm}'
                    arm_dir.mkdir()
                    evaluator = COCOEvaluator('sdd_val', tasks=('bbox',), distributed=False, output_dir=str(arm_dir))
                    evaluator.reset()
                    samples, hashes = [], {}
                    torch.cuda.reset_peak_memory_stats()
                    for index, batch in enumerate(loader):
                        prediction, elapsed = timed(model, batch, seed)
                        if 20 <= index < 220:
                            samples.append(dict(index=index, image_id=batch[0]['image_id'], wall_ms=elapsed))
                        hashes[str(batch[0]['image_id'])] = prediction_digest(prediction)
                        evaluator.process(batch, prediction)
                    metrics = evaluator.evaluate(img_ids=manifest['image_ids'])['bbox']
                    times = [r['wall_ms'] for r in samples]
                    row = dict(train_seed=train_seed, eval_seed=seed, arm=arm, images=len(records),
                               full_dataset=len(records) == len(full_records), **metrics,
                               latency_p50_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times, 95)),
                               measured_images=len(times), peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
                               finite_prediction_images=len(records))
                    rows.append(row)
                    digests[arm] = hashes
                    for name, data in (('metrics.json', row), ('timing.json', samples), ('prediction_digests.json', hashes)):
                        (arm_dir / name).write_text(json.dumps(data) + '\n')
                    for name in ('metrics.json', 'timing.json', 'prediction_digests.json', 'coco_instances_results.json', 'instances_predictions.pth'):
                        remember(arm_dir / name)
                    write_rows(args.out_dir / 'metrics.csv', rows)
                    print('[r2-eval] ' + json.dumps(row), flush=True)
                exact_pairs = sum(digests['known_counts'][str(i)] == digests['gpu_dispatch'][str(i)] for i in manifest['image_ids'])
                samples = []
                for index, batch in enumerate(loader):
                    if index >= 220:
                        break
                    for position, arm in enumerate(arm_order(train_seed, seed, index)):
                        model.head.box_pooler = poolers[arm]
                        prediction, elapsed = timed(model, batch, seed)
                        if prediction_digest(prediction) != digests[arm][str(batch[0]['image_id'])]:
                            raise RuntimeError('Interleaved output differs from full validation')
                        if index >= 20:
                            samples.append(dict(index=index, image_id=batch[0]['image_id'], arm=arm, position=position,
                                                wall_ms=elapsed, bit_exact=True))
                path = args.out_dir / f'interleaved_train{train_seed}_eval{seed}.json'
                path.write_text(json.dumps(samples) + '\n')
                remember(path)
                for arm in ARMS:
                    times = [r['wall_ms'] for r in samples if r['arm'] == arm]
                    timing.append(dict(train_seed=train_seed, eval_seed=seed, arm=arm,
                                       latency_p50_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times, 95)),
                                       measured_images=len(times)))
                parity.append(dict(train_seed=train_seed, eval_seed=seed, full_prediction_images=len(records),
                                   interleaved_prediction_calls=2 * min(220, len(records)), interleaved_bit_exact=True,
                                   cross_arm_bit_exact_images=exact_pairs))
                write_rows(args.out_dir / 'interleaved.csv', timing)
                (args.out_dir / 'parity.json').write_text(json.dumps(parity, indent=2) + '\n')
                print('[r2-timing] ' + json.dumps(timing[-2:]), flush=True)
            if cache_state() != ready_cache:
                raise RuntimeError('JIT cache changed during timed evaluation')
            jit_rows[-1]['cache_unchanged_during_timing'] = True
            (args.out_dir / 'jit_audit.json').write_text(json.dumps(jit_rows, indent=2) + '\n')
            final = state_digest(model)
            if final != initial:
                raise RuntimeError('Frozen state changed')
            states.append(dict(train_seed=train_seed, initial_sha256=initial, final_sha256=final, state_unchanged=True))
            (args.out_dir / 'state_audit.json').write_text(json.dumps(states, indent=2) + '\n')
            del model, poolers, prediction
            torch.cuda.empty_cache()
        # No profiler is started until ALL accuracy and latency cells have finished.
        for train_seed in args.train_seeds:
            model = load_model(train_seed)
            poolers = make_poolers(model)
            initial = state_digest(model)
            for arm in arm_order(train_seed, 0):
                model.head.box_pooler = poolers[arm]
                reference = json.loads((args.out_dir / f'train{train_seed}_eval0_{arm}/prediction_digests.json').read_text())
                target = args.out_dir / f'profile_train{train_seed}_{arm}.json.gz'
                record = profile_model(model, loader, invoke, 0, reference, target)
                profiles.append(dict(train_seed=train_seed, arm=arm, **record))
                remember(target)
                (args.out_dir / 'profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
                print('[r2-profile] ' + json.dumps(profiles[-1]), flush=True)
            if state_digest(model) != initial:
                raise RuntimeError('Profiler changed frozen model state')
            del model, poolers
            torch.cuda.empty_cache()
    for name in ('prerequisite.json', 'prerequisite_manifest.json', 'numerical_audit.json', 'jit_audit.json'):
        remember(args.out_dir / name)
    (args.out_dir / 'artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(dict(evaluations=len(rows), timing_cells=len(timing),
                                                               checkpoints=len(states), profiles=len(profiles))) + '\n')


if __name__ == '__main__':
    main()
