import copy

import pytest
import torch
from torch import nn

from diffusiondet.distillation import bernoulli_kl, load_truncated_student, quality_distillation_loss


def example():
    student = {
        "pred_boxes": torch.tensor([[[1., 1., 9., 9.], [21., 1., 29., 9.], [50., 50., 60., 60.]]], requires_grad=True),
        "pred_logits": torch.tensor([[[1., -1.], [-1., 1.], [0., 0.]]], requires_grad=True),
    }
    teacher = {
        "pred_boxes": torch.tensor([[[20., 0., 30., 10.], [0., 0., 10., 10.], [2., 2., 8., 8.]]], requires_grad=True),
        "pred_logits": torch.tensor([[[-5., 5.], [5., -5.], [5., -5.]]], requires_grad=True),
    }
    targets = [{"labels": torch.tensor([0, 1]), "boxes_xyxy": torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]]),
                "image_size_xyxy": torch.tensor([100., 100., 100., 100.])}]
    si = [(torch.tensor([True, True, False]), torch.tensor([0, 1]))]
    ti = [(torch.tensor([True, True, True]), torch.tensor([1, 0, 0]))]
    return student, teacher, targets, si, ti


def test_gt_alignment_is_invariant_to_teacher_query_order_and_stops_teacher_grad():
    args = example()
    losses, stats = quality_distillation_loss(*args)
    assert stats["kd_pairs"] == 2
    assert stats["kd_teacher_iou"] > stats["kd_student_iou"]
    perm = torch.tensor([1, 2, 0])
    teacher = {k: v[:, perm] for k, v in args[1].items()}
    ti = [(torch.ones(3, dtype=torch.bool), args[4][0][1][perm])]
    shuffled, _ = quality_distillation_loss(args[0], teacher, args[2], args[3], ti)
    for k in losses:
        torch.testing.assert_close(losses[k], shuffled[k])
    sum(losses.values()).backward()
    for tensor in args[1].values():
        assert tensor.grad is None
    for tensor in args[0].values():
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.equal(tensor.grad[0, 2], torch.zeros_like(tensor.grad[0, 2]))


@pytest.mark.parametrize("reason", ["empty_gt", "equal_teacher", "wrong_class", "no_teacher_matches"])
def test_empty_or_unreliable_teacher_produces_differentiable_zero(reason):
    s, t, targets, si, ti = example()
    if reason == "empty_gt":
        targets[0]["labels"] = torch.empty(0, dtype=torch.long)
    elif reason == "equal_teacher":
        t["pred_boxes"] = s["pred_boxes"]
        t["pred_logits"] = s["pred_logits"]
        ti = si
    elif reason == "wrong_class":
        t["pred_logits"] = -t["pred_logits"]
    else:
        ti = [(torch.zeros(3, dtype=torch.bool), torch.empty(0, dtype=torch.long))]
    losses, stats = quality_distillation_loss(s, t, targets, si, ti)
    assert stats["kd_pairs"] == 0
    total = sum(losses.values())
    assert total.item() == 0
    total.backward()
    for tensor in s.values():
        assert torch.isfinite(tensor.grad).all()


def test_bernoulli_kl_matches_torch_distribution_and_handles_extreme_logits():
    s = torch.tensor([[2., -3., 0.]], requires_grad=True)
    t = torch.tensor([[4., -1., 1.]], requires_grad=True)
    dist = torch.distributions.Bernoulli
    reference = torch.distributions.kl_divergence(dist(logits=t / 2), dist(logits=s / 2)).sum(-1) * 4
    torch.testing.assert_close(bernoulli_kl(s, t), reference)
    extreme = bernoulli_kl(torch.tensor([[1000., -1000.]], requires_grad=True), torch.tensor([[-1000., 1000.]]))
    assert torch.isfinite(extreme).all()
    extreme.sum().backward()


def test_box_distillation_is_image_scale_invariant():
    args = example()
    original, _ = quality_distillation_loss(*args)
    s, t, targets, si, ti = example()
    s["pred_boxes"] = s["pred_boxes"] * 3
    t["pred_boxes"] = t["pred_boxes"] * 3
    targets[0]["boxes_xyxy"] *= 3
    targets[0]["image_size_xyxy"] *= 3
    scaled, _ = quality_distillation_loss(s, t, targets, si, ti)
    torch.testing.assert_close(original["loss_kd_box"], scaled["loss_kd_box"])


def test_truncation_loads_only_known_stage_prefixes_and_rejects_missing_weights():
    model = nn.Module()
    model.head = nn.Module()
    model.head.num_heads = 2
    model.head.head_series = nn.ModuleList([nn.Linear(3, 2), nn.Linear(3, 2)])
    state = copy.deepcopy(model.state_dict())
    state["head.head_series.2.weight"] = torch.randn(2, 3)
    state["head.head_series.2.bias"] = torch.randn(2)
    assert len(load_truncated_student(model, state)) == 2
    del state["head.head_series.0.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_truncated_student(model, state)


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"min_iou": -1}, {"min_iou": 2}])
def test_invalid_hyperparameters_fail(kwargs):
    with pytest.raises(ValueError):
        quality_distillation_loss(*example(), **kwargs)
