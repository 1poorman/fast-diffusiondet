"""R2 prerequisite numerical checks; no AP or speed selection from this script."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import sha256


def synthetic_audit(device):
    import torch
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.structures import Boxes
    from diffusiondet.roi_gpu_dispatch import GPUDispatchROIPooler
    rows = []
    kwargs = dict(output_size=7, scales=[0.25, 0.125, 0.0625, 0.03125], sampling_ratio=2, pooler_type='ROIAlignV2')
    native, candidate = ROIPooler(**kwargs).eval(), GPUDispatchROIPooler(**kwargs).eval()
    torch.manual_seed(83)
    for name, channels, height, width, counts in (
        ('boundaries_ragged', 8, 64, 80, [11, 0, 3]),
        ('empty', 8, 64, 80, [0, 0]),
        ('large_spatial', 32, 200, 264, [11, 3]),
        ('real_channels_300', 256, 200, 256, [300]),
    ):
        features = [torch.rand(len(counts), channels, height // 2**i, width // 2**i, device=device) * 2 - 1 for i in range(4)]
        if max(counts) <= 11:
            sizes = torch.tensor([0., 55.9, 56., 111.9, 112., 223.9, 224., 448., 900., -20., float('nan')], device=device)
            coords = torch.stack((sizes * 0 - 8, sizes * 0 - 5, sizes - 8, sizes - 5), dim=1)
            boxes = [Boxes(coords[:n]) for n in counts]
        else:
            start = torch.rand(max(counts), 2, device=device) * 1000 - 100
            size = torch.rand(max(counts), 2, device=device) * 900
            boxes = [Boxes(torch.cat([start[:n], start[:n] + size[:n]], dim=1)) for n in counts]
        with torch.no_grad():
            expected = native(features, boxes)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            actual = candidate(features, boxes)
            torch.cuda.synchronize()
            cold_ms = 1000 * (time.perf_counter() - tick)
        difference = (actual - expected).abs()
        row = dict(case=name, shape=list(actual.shape), max_abs_error=float(difference.max()) if actual.numel() else 0.,
                   relative_l2=float(difference.norm() / expected.norm().clamp_min(1e-12)),
                   first_call_ms=cold_ms, finite=bool(torch.isfinite(actual).all()), rtol=1e-5, atol=2e-5)
        rows.append(row)
        print('[r2-feature] ' + json.dumps(row), flush=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        if not row['finite']:
            raise RuntimeError('Non-finite R2 pooled features')
    # Autograd must use native kernels even when an eval-only option is configured.
    features = [torch.rand(2, 8, 64 // 2**i, 80 // 2**i, device=device, requires_grad=True) for i in range(4)]
    copies = [f.detach().clone().requires_grad_(True) for f in features]
    sizes = torch.tensor([56., 112., 224., 448.], device=device)
    coords = torch.stack((sizes * 0, sizes * 0, sizes, sizes), dim=1)
    boxes = [Boxes(coords), Boxes(coords[:2])]
    expected, actual = native(features, boxes), candidate(copies, boxes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    errors = []
    for a, b in zip(features, copies):
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5)
        errors.append(float((a.grad - b.grad).abs().max()))
    fallback = []
    # Strided features and autocast must use native ROIAlign, even under no_grad.
    with torch.no_grad():
        strided = [f.detach().transpose(-1, -2) for f in features]
        torch.testing.assert_close(candidate(strided, boxes), native(strided, boxes), rtol=0, atol=0)
        fallback.append('noncontiguous')
        with torch.autocast('cuda', dtype=torch.float16):
            torch.testing.assert_close(candidate(features, boxes), native(features, boxes), rtol=0, atol=0)
        fallback.append('autocast')
    return dict(features=rows, gradient_fallback_max_abs=max(errors), native_fallbacks=fallback, passed=True)


def model_audit(out_dir, checkpoint):
    """Compare every executed ROI call on real inputs; exclude audits from timing."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
    from detectron2.modeling import build_model
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.proposal_budget import proposal_noise
    from diffusiondet.roi_gpu_dispatch import GPUDispatchROIPooler
    from diffusiondet.util.model_ema import add_model_ema_configs
    from scripts.v3_backbone_precision import check_prediction
    from scripts.v3_distill import state_digest
    from scripts.v3_head_sweep import image_seed
    from scripts.v3_profile_utils import prediction_digest
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file('configs/lab/v3/sdd.res50.student4.yaml')
    cfg.DATASET_ROOT = str(Path('datasets').resolve())
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.DiffusionDet.INFERENCE_TOPK = 300
    cfg.freeze()
    register_all(cfg=cfg)
    records = sorted(DatasetCatalog.get('sdd_val'), key=lambda r: r['image_id'])[:32]
    loader = build_detection_test_loader(dataset=records, mapper=DatasetMapper(cfg, is_train=False), num_workers=0, batch_size=1)
    model = build_model(cfg).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
    initial = state_digest(model)
    native = model.head.box_pooler
    candidate = GPUDispatchROIPooler(
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
        [1 / model.backbone.output_shape()[k].stride for k in cfg.MODEL.ROI_HEADS.IN_FEATURES],
        cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO, cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE).eval()
    calls, predictions = [], []
    candidate_forward = candidate.forward
    def audit_pooler(features, boxes):
        expected = native(features, boxes)
        actual = candidate_forward(features, boxes)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)
        calls.append(dict(max_abs_error=float((actual - expected).abs().max()),
                          relative_l2=float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
                          feature_shapes=[list(f.shape) for f in features]))
        return actual
    with torch.no_grad():
        for batch in loader:
            outputs = {}
            for arm in ('native', 'dispatch'):
                torch.manual_seed(image_seed(0, batch[0]['image_id']))
                noise = proposal_noise(1, 300, 300, model.device)
                model.head.box_pooler = native if arm == 'native' else candidate
                if arm == 'dispatch':
                    candidate.forward = audit_pooler
                    try:
                        result = model(batch, inference_noise=noise)
                    finally:
                        candidate.forward = candidate_forward
                else:
                    result = model(batch, inference_noise=noise)
                check_prediction(result)
                outputs[arm] = prediction_digest(result)
            predictions.append(dict(image_id=batch[0]['image_id'], **outputs))
    if state_digest(model) != initial:
        raise RuntimeError('Model smoke changed frozen state')
    return dict(images=len(records), calls=calls, roi_calls=len(calls), predictions=predictions,
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                state_sha256=initial, state_unchanged=True, finite=True, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--model-smoke', action='store_true')
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('runs/v3/time_alignment_20261011/seed10_uniform/model_final.pth'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['TRITON_CACHE_DIR'] = str(args.out_dir.resolve() / 'triton_cache')
    import torch
    import triton
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path('diffusiondet/roi_gpu_dispatch.py'),
               Path('diffusiondet/roi_gpu_dispatch_kernel.py'), Path('diffusiondet/roi_pooler.py'), Path('docs/v3/ROI_GPU_DISPATCH.md')]
    for source in sources:
        target = args.out_dir / 'source_snapshot' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    manifest = dict(torch=torch.__version__, triton=triton.__version__, gpu=args.gpu,
                    source_sha256={str(s): sha256(s) for s in sources},
                    cache=os.environ['TRITON_CACHE_DIR'])
    (args.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        result = synthetic_audit('cuda')
        if args.model_smoke:
            result['model'] = model_audit(args.out_dir, args.checkpoint)
    except Exception as error:
        (args.out_dir / 'FAILED.json').write_text(json.dumps(dict(error=str(error), passed=False), indent=2) + '\n')
        raise
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(result, indent=2) + '\n')
    print('[r2-checks] passed; model images=' + str(result.get('model', {}).get('images', 0)), flush=True)


if __name__ == '__main__':
    main()
