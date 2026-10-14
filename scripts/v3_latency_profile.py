"""L1: plain latency, nested stream spans, and separate CPU/CUDA operator traces."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_profile_utils import EventScopes, ScopedCalls, instrument, interval_union, prediction_digest, sync_scope_rows
from scripts.v3_summarize_distill import write_rows

ARMS = ('h4_p300', 'h4_p100', 'h6_p300')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--measure', type=int, default=200)
    parser.add_argument('--profile-images', type=int, default=10)
    parser.add_argument('--h4-root', type=Path, default=Path('runs/v3/time_alignment_20261011'))
    parser.add_argument('--h6-root', type=Path, default=Path('runs/v3/quality_20261013'))
    parser.add_argument('--dataset-root', default='datasets')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 1 or not 1 <= args.profile_images <= args.measure:
        parser.error('Need positive warmup and 1 <= profile-images <= measure')
    if any(len(s) != len(set(s)) for s in (args.train_seeds, args.eval_seeds)) or 0 not in args.eval_seeds:
        parser.error('Unique seeds required, including eval seed 0 for operator profiling')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import numpy as np
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
    from detectron2.modeling import build_model
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
    if (dd.SAMPLE_STEP != 1 or dd.SOLVER != 'ddim' or dd.FORMULATION != 'vp'
            or dd.BOX_RENEWAL or dd.USE_ENSEMBLE or dd.ATTENTION_NEED_WEIGHTS
            or cfg.MODEL_EMA.ENABLED or cfg.SOLVER.AMP.ENABLED):
        raise ValueError('L1 requires the existing FP32/no_weights/NFE1 protocol')
    register_all(cfg=cfg)
    full_records = sorted(DatasetCatalog.get('sdd_val'), key=lambda row: row['image_id'])
    records = full_records[:args.warmup + args.measure]
    if len(records) != args.warmup + args.measure:
        raise ValueError('Not enough images for the requested timing window')
    loader = build_detection_test_loader(dataset=records, mapper=DatasetMapper(cfg, is_train=False), num_workers=0, batch_size=1)
    checkpoints = {f'{seed}_{depth}': (args.h4_root / f'seed{seed}_uniform/model_final.pth' if depth == 4
                                      else args.h6_root / f'seed{seed}_h6_focal/model_final.pth')
                   for seed in args.train_seeds for depth in (4, 6)}
    for path in checkpoints.values():
        summary = json.loads((path.parent / 'training_summary.json').read_text())
        if sha256(path) != summary['checkpoint_sha256']:
            raise ValueError('Reference checkpoint hash mismatch')
    import inspect
    import detectron2.modeling.poolers as poolers
    sources = [Path('scripts/v3_latency_profile.py'), Path('scripts/v3_profile_utils.py'),
               Path('scripts/v3_head_sweep.py'), Path('scripts/v3_summarize_distill.py'),
               Path('docs/v3/LATENCY_PROFILE.md'),
               *[Path('diffusiondet') / name for name in ('config.py', 'detector.py', 'head.py', 'proposal_budget.py',
                   'solvers/ddim.py', 'solvers/__init__.py', 'solvers/base.py')]]
    # Follow the complete config inheritance chain.
    import yaml
    config_path = Path('configs/lab/v3/sdd.res50.student4.yaml')
    while True:
        sources.append(config_path)
        parent = yaml.safe_load(config_path.read_text()).get('_BASE_')
        if parent is None:
            break
        config_path = (config_path.parent / parent).resolve().relative_to(Path.cwd())
    for source in sources:
        dst = args.out_dir / 'source_snapshot' / source
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)
    external = Path(inspect.getfile(poolers))
    shutil.copyfile(external, args.out_dir / 'detectron2_poolers_snapshot.py')
    manifest = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    checkpoints={k: dict(path=str(v), sha256=sha256(v)) for k, v in checkpoints.items()},
                    source_sha256={str(p): sha256(p) for p in sources},
                    external_poolers=dict(path=str(external), sha256=sha256(external)),
                    annotation_path=MetadataCatalog.get('sdd_val').json_file,
                    annotation_sha256=sha256(MetadataCatalog.get('sdd_val').json_file),
                    image_ids=[r['image_id'] for r in records], full_val_images=len(full_records),
                    torch=torch.__version__, detectron2=detectron2.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    fp32=True, tf32=False, cudnn_benchmark=False, cudnn_deterministic=False,
                    timing='Full 300-noise pool prefix + model call; no loader/resize/seeding/digest; batch1 synchronized wall time',
                    segments='Nested event spans include GPU work and enqueue gaps; host spans include dispatch/waits; direct-child subtraction',
                    scope='Latency diagnosis, not AP evaluation or new training; profiler times are perturbed')
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.out_dir / 'config.yaml').write_text(cfg.dump())
    metrics, profiles = [], []

    def invoke(model, batch, proposals):
        noise = proposal_noise(1, proposals, 300, model.device)
        return model(batch, inference_noise=noise)

    def profile_model(model, proposals, cell_dir, key, plain_hashes):
        batches = []
        for index, batch in enumerate(loader):
            if index < args.warmup:
                torch.manual_seed(image_seed(0, batch[0]['image_id']))
                invoke(model, batch, proposals)
            else:
                batches.append(batch)
            if len(batches) == args.profile_images:
                break
        outputs = []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as prof:
            with ScopedCalls(lambda name: torch.profiler.record_function('L1/' + name)) as patches:
                instrument(model, patches)
                for batch in batches:
                    torch.manual_seed(image_seed(0, batch[0]['image_id']))
                    torch.cuda.synchronize()
                    with torch.profiler.record_function('L1/call'):
                        outputs.append(invoke(model, batch, proposals))
                    torch.cuda.synchronize()
        for batch, prediction in zip(batches, outputs):
            if prediction_digest(prediction) != plain_hashes[batch[0]['image_id']]:
                raise ValueError('Profiler output is not bit exact')
        operators = [dict(name=event.key, count=event.count, self_cpu_us=event.self_cpu_time_total,
                          self_cuda_us=event.self_cuda_time_total, cpu_total_us=event.cpu_time_total)
                     for event in prof.key_averages()]
        write_rows(cell_dir / 'operators.csv', operators)
        trace_path = cell_dir / 'trace.json'
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        device_events = [e for e in trace['traceEvents'] if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
        if not device_events:
            raise ValueError('Profiler did not capture CUDA activity')
        kernel_events = [e for e in device_events if e['cat'] == 'kernel']
        sync_events = [e for e in trace['traceEvents'] if e.get('cat') == 'cuda_runtime' and 'Synchronize' in e.get('name', '')]
        write_rows(cell_dir / 'synchronizations.csv', sync_scope_rows(trace['traceEvents']))
        profile = dict(**key, images=len(batches), kernel_count=len(kernel_events),
                       kernel_sum_ms=sum(e['dur'] for e in kernel_events) / 1000,
                       kernel_union_ms=interval_union([(e['ts'], e['ts'] + e['dur']) for e in kernel_events]) / 1000,
                       device_union_ms=interval_union([(e['ts'], e['ts'] + e['dur']) for e in device_events]) / 1000,
                       sync_calls=len(sync_events), sync_cpu_ms=sum(e['dur'] for e in sync_events) / 1000,
                       bit_exact=True)
        compressed = cell_dir / 'trace.json.gz'
        with trace_path.open('rb') as source, gzip.open(compressed, 'wb') as target:
            shutil.copyfileobj(source, target)
        trace_path.unlink()  # Temporary uncompressed trace created by this invocation.
        profile['trace_sha256'] = sha256(compressed)
        profiles.append(profile)
        (args.out_dir / 'profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')
        del prof, trace, device_events, kernel_events, sync_events, batches, outputs, prediction

    with torch.no_grad():
        for train_seed in args.train_seeds:
            for eval_seed in args.eval_seeds:
                order = ARMS if (train_seed + eval_seed) % 2 == 0 else tuple(reversed(ARMS))
                for arm in order:
                    depth, proposals = (6 if arm.startswith('h6') else 4), int(arm.split('_p')[1])
                    arm_cfg = cfg.clone()
                    arm_cfg.defrost()
                    arm_cfg.MODEL.DiffusionDet.NUM_HEADS = depth
                    arm_cfg.MODEL.DiffusionDet.NUM_PROPOSALS = proposals
                    arm_cfg.freeze()
                    model = build_model(arm_cfg).eval()
                    state = torch.load(checkpoints[f'{train_seed}_{depth}'], map_location='cpu')
                    expected_arm = 'uniform' if depth == 4 else 'h6_focal'
                    if state['train_seed'] != train_seed or state['arm'] != expected_arm or state['iteration'] != 599:
                        raise ValueError('Incorrect checkpoint metadata')
                    model.load_state_dict(state['model'], strict=True)
                    del state
                    cell_dir = args.out_dir / f'train{train_seed}_eval{eval_seed}_{arm}'
                    cell_dir.mkdir()
                    (cell_dir / 'config.yaml').write_text(arm_cfg.dump())
                    key = dict(train_seed=train_seed, eval_seed=eval_seed, arm=arm)
                    plain_hashes = {}
                    shape_holder = {}
                    shape_hook = model.head.register_forward_hook(
                        lambda module, inputs, output: shape_holder.update(shape=list(output[1].shape)))
                    for mode in ('plain', 'events'):
                        durations = []
                        with (cell_dir / f'{mode}.jsonl').open('w') as stream:
                            for index, batch in enumerate(loader):
                                image_id = batch[0]['image_id']
                                torch.manual_seed(image_seed(eval_seed, image_id))
                                meter = EventScopes()
                                torch.cuda.synchronize()
                                if mode == 'events':
                                    with ScopedCalls(meter.scope) as patches:
                                        instrument(model, patches)
                                        start = time.perf_counter()
                                        with meter.scope('call'):
                                            prediction = invoke(model, batch, proposals)
                                        torch.cuda.synchronize()
                                        wall_ms = 1000 * (time.perf_counter() - start)
                                    spans = meter.finish()
                                    counts = Counter(row['name'] for row in spans)
                                    if any(counts[f'stage{h}'] != 1 for h in range(1, depth + 1)) or counts['head'] != 1 or counts['roi_pool'] != depth:
                                        raise ValueError('Incorrect executed head/ROI counts')
                                else:
                                    start = time.perf_counter()
                                    prediction = invoke(model, batch, proposals)
                                    torch.cuda.synchronize()
                                    wall_ms = 1000 * (time.perf_counter() - start)
                                    spans = None
                                if mode == 'plain' and index == 0:
                                    shape_hook.remove()
                                    if shape_holder['shape'] != [depth, 1, proposals, 4]:
                                        raise ValueError('Actual head depth/proposal shape differs')
                                    (cell_dir / 'execution.json').write_text(json.dumps(shape_holder) + '\n')
                                if index >= args.warmup:
                                    digest = prediction_digest(prediction)
                                    if mode == 'plain':
                                        plain_hashes[image_id] = digest
                                    elif digest != plain_hashes[image_id]:
                                        raise ValueError('Instrumented output is not bit exact')
                                    row = dict(index=index, image_id=image_id, wall_ms=wall_ms, prediction_sha256=digest,
                                               input_hw=list(batch[0]['image'].shape[-2:]))
                                    if spans is not None:
                                        row['spans'] = spans
                                    stream.write(json.dumps(row) + '\n')
                                    durations.append(wall_ms)
                                del prediction
                        metrics.append(dict(**key, mode=mode, images=len(durations),
                                            p50_ms=float(np.median(durations)), p95_ms=float(np.percentile(durations, 95)),
                                            mean_ms=float(np.mean(durations))))
                        write_rows(args.out_dir / 'metrics.csv', metrics)
                        print('[l1] ' + json.dumps(metrics[-1]), flush=True)
                    del model
                    torch.cuda.empty_cache()
        # CUPTI tracing runs only after all unprofiled measurements have finished.
        for train_seed in args.train_seeds:
            for arm in ARMS:
                depth, proposals = (6 if arm.startswith('h6') else 4), int(arm.split('_p')[1])
                arm_cfg = cfg.clone()
                arm_cfg.defrost()
                arm_cfg.MODEL.DiffusionDet.NUM_HEADS = depth
                arm_cfg.MODEL.DiffusionDet.NUM_PROPOSALS = proposals
                arm_cfg.freeze()
                model = build_model(arm_cfg).eval()
                state = torch.load(checkpoints[f'{train_seed}_{depth}'], map_location='cpu')
                model.load_state_dict(state['model'], strict=True)
                del state
                cell_dir = args.out_dir / f'train{train_seed}_eval0_{arm}'
                plain_hashes = {r['image_id']: r['prediction_sha256'] for r in
                                map(json.loads, (cell_dir / 'plain.jsonl').read_text().splitlines())}
                profile_model(model, proposals, cell_dir, dict(train_seed=train_seed, eval_seed=0, arm=arm), plain_hashes)
                del model
                torch.cuda.empty_cache()
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(dict(cells=len(metrics) // 2, profiles=len(profiles))) + '\n')


if __name__ == '__main__':
    main()
