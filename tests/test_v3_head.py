"""Guard checkpoint compatibility, actual skipped work, and SDPA numerics."""
import pytest
import torch
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec

from diffusiondet import add_diffusiondet_config
from diffusiondet.head import DynamicHead


def make_head(depth=0):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2"]
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 3
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 2
    dd = cfg.MODEL.DiffusionDet
    dd.NUM_HEADS = 3
    dd.EVAL_NUM_HEADS = depth
    dd.NUM_CLASSES = 7
    dd.HIDDEN_DIM = 32
    dd.DIM_FEEDFORWARD = 64
    dd.DIM_DYNAMIC = 8
    return DynamicHead(cfg, {"p2": ShapeSpec(channels=32, stride=4)})


def inputs():
    torch.manual_seed(4)
    features = [torch.rand(2, 32, 16, 20)]
    xy = torch.rand(2, 8, 2) * 24
    boxes = torch.cat((xy, xy + 24), dim=-1)
    return features, boxes, torch.tensor([999, 999]), None


def test_depth_is_exact_prefix_and_skips_later_work():
    head = make_head().eval()
    args = inputs()
    calls = []
    hooks = [m.register_forward_hook(lambda m, i, o, j=j: calls.append(j))
             for j, m in enumerate(head.head_series)]
    with torch.no_grad():
        full = head(*args)
        calls.clear()
        head.eval_num_heads = 2
        early = head(*args)
    assert calls == [0, 1]
    for all_stages, prefix in zip(full, early):
        torch.testing.assert_close(all_stages[:2], prefix, rtol=0, atol=0)
    for hook in hooks:
        hook.remove()
    # Inference settings must never silently remove a training auxiliary loss.
    head.train()
    classes, boxes = head(*args)
    assert classes.shape[0] == boxes.shape[0] == 3
    (classes.square().mean() + boxes.mean()).backward()
    assert head.head_series[-1].bboxes_delta.weight.grad is not None


def test_no_weights_attention_preserves_parameters_outputs_and_gradients():
    head = make_head().eval()
    args = inputs()
    reference = head(*args)
    loss = sum(x.square().mean() for x in reference)
    loss.backward()
    grad = head.head_series[0].self_attn.in_proj_weight.grad.clone()
    state_keys = set(head.state_dict())
    head.zero_grad(set_to_none=True)
    for stage in head.head_series:
        stage.attention_need_weights = False
    actual = head(*args)
    sum(x.square().mean() for x in actual).backward()
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(grad, head.head_series[0].self_attn.in_proj_weight.grad,
                               rtol=2e-3, atol=2e-4)
    assert set(head.state_dict()) == state_keys


@pytest.mark.parametrize("depth", [-1, 4])
def test_invalid_depth_fails(depth):
    with pytest.raises(ValueError, match="EVAL_NUM_HEADS"):
        make_head(depth)
