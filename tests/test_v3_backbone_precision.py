import copy

import pytest
import torch
from torch import nn

from diffusiondet.backbone_precision import backbone_features


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, images):
        return {'p2': self.conv(images)}


def test_default_preserves_outputs_gradients_and_state():
    torch.manual_seed(4)
    a = TinyBackbone()
    b = copy.deepcopy(a)
    x = torch.rand(2, 3, 8, 8)
    original = a(x)['p2']
    actual = backbone_features(b, x)['p2']
    torch.testing.assert_close(original, actual, rtol=0, atol=0)
    original.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(a.conv.weight.grad, b.conv.weight.grad, rtol=0, atol=0)
    assert set(a.state_dict()) == set(b.state_dict())


def test_inference_setting_does_not_enable_autocast_during_training():
    backbone = TinyBackbone().train()
    x = torch.rand(1, 3, 8, 8)
    expected = backbone(x)['p2']
    actual = backbone_features(backbone, x, 'fp16', training=True)['p2']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.isfinite(backbone.conv.weight.grad).all()


def test_default_respects_existing_training_autocast():
    backbone = TinyBackbone().train()
    x = torch.rand(1, 3, 8, 8)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = backbone(x)['p2']
        actual = backbone_features(backbone, x, 'fp16', training=True)['p2']
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_opt_in_rejects_cpu_instead_of_silently_changing_the_experiment():
    with pytest.raises(ValueError, match='CUDA FP32'):
        backbone_features(TinyBackbone().eval(), torch.rand(1, 3, 8, 8), 'fp16')


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match='EVAL_BACKBONE_PRECISION'):
        backbone_features(TinyBackbone(), torch.rand(1, 3, 8, 8), 'fp8')
