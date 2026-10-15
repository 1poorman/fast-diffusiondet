import copy

import pytest
import torch
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
from detectron2.structures import Boxes

from diffusiondet import add_diffusiondet_config
from diffusiondet.head import DynamicHead
from diffusiondet.roi_pooler import KnownCountROIPooler, known_count_pooler_format


def boxes_for_counts(counts, device='cpu'):
    sizes = torch.tensor([0., 55.9, 56., 111.9, 112., 223.9, 224., 448.], device=device)
    boxes = torch.stack((torch.full_like(sizes, -8), torch.full_like(sizes, -5), sizes - 8, sizes - 5), dim=1)
    return [Boxes(boxes[:n].clone()) for n in counts]


@pytest.mark.parametrize('counts', [(8,), (8, 0, 3), (0, 0), (1, 2, 7)])
def test_format_matches_values_order_dtype_and_box_gradients(counts):
    a = boxes_for_counts(counts)
    b = copy.deepcopy(a)
    for box in a + b:
        box.tensor.requires_grad_(True)
    expected, actual = convert_boxes_to_pooler_format(a), known_count_pooler_format(b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for left, right in zip(a, b):
        torch.testing.assert_close(left.tensor.grad, right.tensor.grad, rtol=0, atol=0)


@pytest.mark.parametrize('levels,counts', [(1, (8,)), (4, (8, 0, 3)), (4, (0, 0)), (4, ())])
def test_pooler_outputs_and_feature_gradients_match(levels, counts):
    torch.manual_seed(8)
    kwargs = dict(output_size=3, scales=[1 / (4 * 2**i) for i in range(levels)], sampling_ratio=2, pooler_type='ROIAlignV2')
    original, optimized = ROIPooler(**kwargs), KnownCountROIPooler(**kwargs)
    a = [torch.rand(len(counts), 4, 64 // 2**i, 80 // 2**i, requires_grad=True) for i in range(levels)]
    b = [x.detach().clone().requires_grad_(True) for x in a]
    boxes = boxes_for_counts(counts)
    expected, actual = original(a, boxes), optimized(b, boxes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if expected.requires_grad:
        expected.square().sum().backward()
        actual.square().sum().backward()
        for left, right in zip(a, b):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize('case', ['not_lists', 'levels', 'batch'])
def test_input_validation_remains_enabled(case):
    pooler = KnownCountROIPooler(3, [0.25], 2, 'ROIAlignV2')
    features, boxes = [torch.rand(1, 4, 16, 16)], boxes_for_counts((1,))
    if case == 'not_lists':
        features = tuple(features)
    elif case == 'levels':
        features = features * 2
    else:
        boxes = boxes * 2
    with pytest.raises(AssertionError):
        pooler(features, boxes)


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
    dd.ROI_POOLER_KNOWN_COUNTS = True
    actual = DynamicHead(cfg, shape).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert type(reference.box_pooler) is ROIPooler and isinstance(actual.box_pooler, KnownCountROIPooler)
    features = [torch.rand(2, 32, 64 // 2**i, 80 // 2**i) for i in range(4)]
    boxes = torch.stack([box.tensor for box in boxes_for_counts((8, 8))])
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


def test_tracing_uses_upstream_format_path(monkeypatch):
    import diffusiondet.roi_pooler as implementation
    monkeypatch.setattr(implementation, 'is_fx_tracing', lambda: True)
    monkeypatch.setattr(implementation, 'known_count_pooler_format', lambda _: pytest.fail('eager path during tracing'))
    pooler = KnownCountROIPooler(3, [0.25], 2, 'ROIAlignV2')
    pooler([torch.rand(1, 4, 16, 16)], boxes_for_counts((1,)))
