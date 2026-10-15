import pytest
import torch
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec

from diffusiondet import add_diffusiondet_config
from diffusiondet.head import DynamicHead
from diffusiondet.roi_gpu_dispatch import GPUDispatchROIPooler


@pytest.mark.parametrize('mode', ['train', 'eval_grad', 'eval_no_grad', 'unsupported_ratio', 'single_level'])
def test_native_fallback_preserves_outputs_gradients_and_state(mode):
    levels = 1 if mode == 'single_level' else 4
    kwargs = dict(output_size=3, scales=[0.25 / 2**i for i in range(levels)],
                  sampling_ratio=1 if mode == 'unsupported_ratio' else 2, pooler_type='ROIAlignV2')
    original, candidate = ROIPooler(**kwargs), GPUDispatchROIPooler(**kwargs)
    if mode != 'train':
        original.eval()
        candidate.eval()
    a = [torch.rand(2, 4, 32 // 2**i, 40 // 2**i, requires_grad=True) for i in range(levels)]
    b = [f.detach().clone().requires_grad_(True) for f in a]
    boxes = [Boxes(torch.tensor([[0., 0., 56., 56.], [0., 0., 224., 224.]])), Boxes(torch.zeros(0, 4))]
    with torch.set_grad_enabled(mode != 'eval_no_grad'):
        expected, actual = original(a, boxes), candidate(b, boxes)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if expected.requires_grad:
            expected.square().sum().backward()
            actual.square().sum().backward()
            for x, y in zip(a, b):
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    candidate.load_state_dict(original.state_dict(), strict=True)


def test_cpu_no_grad_does_not_import_optional_triton_kernel(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.endswith('roi_gpu_dispatch_kernel'):
            pytest.fail('CPU fallback attempted to import CUDA kernel')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    pooler = GPUDispatchROIPooler(3, [0.25, 0.125, 0.0625, 0.03125], 2, 'ROIAlignV2').eval()
    with torch.no_grad():
        pooler([torch.rand(1, 4, 32 // 2**i, 40 // 2**i) for i in range(4)], [Boxes(torch.zeros(0, 4))])


def test_input_shape_validation_is_preserved():
    pooler = GPUDispatchROIPooler(3, [0.25, 0.125, 0.0625, 0.03125], 2, 'ROIAlignV2').eval()
    with torch.no_grad(), pytest.raises(AssertionError):
        pooler([torch.rand(1, 4, 32, 32)], [Boxes(torch.zeros(0, 4))])


def test_config_switch_keeps_full_head_outputs_gradients_and_checkpoint_keys():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5']
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 3
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = 'ROIAlignV2'
    dd = cfg.MODEL.DiffusionDet
    dd.NUM_HEADS, dd.NUM_CLASSES, dd.HIDDEN_DIM, dd.DIM_FEEDFORWARD, dd.DIM_DYNAMIC = 2, 7, 32, 64, 8
    shape = {f'p{i + 2}': ShapeSpec(channels=32, stride=4 * 2**i) for i in range(4)}
    torch.manual_seed(8)
    reference = DynamicHead(cfg, shape).eval()
    dd.EVAL_ROI_GPU_DISPATCH = True
    actual = DynamicHead(cfg, shape).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert type(reference.box_pooler) is ROIPooler and isinstance(actual.box_pooler, GPUDispatchROIPooler)
    features = [torch.rand(2, 32, 64 // 2**i, 80 // 2**i) for i in range(4)]
    sizes = torch.tensor([0., 55.9, 56., 111.9, 112., 223.9, 224., 448.])
    coords = torch.stack((sizes * 0 - 8, sizes * 0 - 5, sizes - 8, sizes - 5), dim=1)
    boxes = coords.unsqueeze(0).repeat(2, 1, 1)
    args = features, boxes, torch.tensor([999, 999]), None
    expected, result = reference(*args), actual(*args)
    for left, right in zip(expected, result):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    sum(x.square().mean() for x in expected).backward()
    sum(x.square().mean() for x in result).backward()
    for left, right in zip(reference.parameters(), actual.parameters()):
        if left.grad is not None:
            assert torch.isfinite(right.grad).all()
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
