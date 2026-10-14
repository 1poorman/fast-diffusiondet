"""Quality targets, gradient boundaries, and final-only training behavior."""
import pytest
import torch
from torch import nn
from detectron2.config import get_cfg
from fvcore.nn import sigmoid_focal_loss

from diffusiondet import add_diffusiondet_config
from diffusiondet.loss import SetCriterionDynamicK
from diffusiondet.quality import aligned_iou, matched_quality_loss, quality_focal


def test_binary_targets_recover_original_focal_values_and_gradients():
    logits = torch.tensor([[-100., -3., 0., 2., 100.]], requires_grad=True)
    hard = torch.tensor([[0., 1., 0., 1., 1.]])
    actual = quality_focal(logits, hard, hard.bool())
    reference = sigmoid_focal_loss(logits, hard, alpha=0.25, gamma=2, reduction="none")
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    a = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(reference.sum(), logits)[0]
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_quality_target_is_detached_and_has_minimum_at_predicted_quality():
    target = torch.tensor([[0.2, 0.7]], requires_grad=True)
    logits = torch.logit(target.detach()).requires_grad_()
    loss = quality_focal(logits, target, torch.ones_like(target, dtype=torch.bool)).sum()
    loss.backward()
    assert target.grad is None
    assert loss.item() < 1e-12
    assert logits.grad.abs().max() < 1e-7


def test_aligned_iou_handles_disjoint_degenerate_and_scale_changes():
    a = torch.tensor([[0., 0., 2., 2.], [3., 3., 4., 4.], [1., 1., 1., 1.]])
    b = torch.tensor([[0., 0., 1., 2.]] * 3)
    torch.testing.assert_close(aligned_iou(a, b), torch.tensor([0.5, 0., 0.]))
    torch.testing.assert_close(aligned_iou(a * 5, b * 5), aligned_iou(a, b))


def example():
    outputs = dict(pred_logits=torch.tensor([[[1., -1.], [-2., 2.], [0., 0.]]], requires_grad=True),
                   pred_boxes=torch.tensor([[[0., 0., 5., 10.], [20., 0., 30., 10.], [50., 50., 60., 60.]]], requires_grad=True))
    targets = [dict(labels=torch.tensor([0, 1]), boxes_xyxy=torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]))]
    indices = [(torch.tensor([True, True, False]), torch.tensor([0, 1]))]
    return outputs, targets, indices


def test_gt_alignment_and_detached_boxes_with_correct_positive_normalization():
    outputs, targets, indices = example()
    loss, info = matched_quality_loss(outputs, targets, indices)
    assert info["quality_positives"] == 2
    assert info["quality_mean_iou"].item() == 0.75
    q = torch.tensor([[[0.5, 0.], [0., 1.], [0., 0.]]])
    pos = torch.tensor([[[True, False], [False, True], [False, False]]])
    torch.testing.assert_close(loss, quality_focal(outputs["pred_logits"], q, pos).sum() / 2)
    loss.backward()
    assert outputs["pred_boxes"].grad is None
    assert outputs["pred_logits"].grad is not None
    assert (outputs["pred_logits"].grad[0, 2] > 0).all()
    shuffled_targets = [dict(labels=targets[0]["labels"].flip(0), boxes_xyxy=targets[0]["boxes_xyxy"].flip(0))]
    shuffled_indices = [(indices[0][0], torch.tensor([1, 0]))]
    torch.testing.assert_close(matched_quality_loss(outputs, shuffled_targets, shuffled_indices)[0], loss)


def test_empty_gt_matches_original_negative_focal_and_has_finite_gradients():
    outputs, _, _ = example()
    targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes_xyxy=torch.empty(0, 4))]
    # The existing Dynamic-K matcher emits float indices for the empty case.
    indices = [(torch.zeros(3, dtype=torch.bool), torch.empty(0))]
    loss, info = matched_quality_loss(outputs, targets, indices)
    expected = sigmoid_focal_loss(outputs["pred_logits"], torch.zeros_like(outputs["pred_logits"]),
                                  alpha=0.25, gamma=2, reduction="sum")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(outputs["pred_logits"].grad).all()
    assert info["quality_positives"] == 0


class FixedMatcher(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices
        self.calls = 0

    def forward(self, outputs, targets):
        self.calls += 1
        return self.indices, None


def criterion(mode, indices):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    cfg.MODEL.DiffusionDet.FINAL_CLASS_LOSS = mode
    return SetCriterionDynamicK(cfg, 2, FixedMatcher(indices), {}, 0.1, ["labels"], True)


def test_quality_changes_only_final_loss_and_keeps_auxiliary_matching_and_state():
    outputs, targets, indices = example()
    outputs["aux_outputs"] = [{k: v for k, v in outputs.items()}]
    original, quality = criterion("focal", indices), criterion("quality_focal", indices)
    a, b = original(outputs, targets), quality(outputs, targets)
    assert a["loss_ce"].item() != b["loss_ce"].item()
    torch.testing.assert_close(a["loss_ce_0"], b["loss_ce_0"], rtol=0, atol=0)
    assert original.matcher.calls == quality.matcher.calls == 2
    assert set(original.state_dict()) == set(quality.state_dict())
    assert quality.last_quality_stats["quality_positives"] == 2


def test_unknown_classification_mode_fails():
    with pytest.raises(ValueError, match="FINAL_CLASS_LOSS"):
        criterion("unknown", [])


@pytest.mark.parametrize("kwargs", [{"alpha": -0.1}, {"alpha": 1.1}, {"beta": 0}])
def test_invalid_quality_loss_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        quality_focal(torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.bool), **kwargs)
