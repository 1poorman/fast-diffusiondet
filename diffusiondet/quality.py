"""Opt-in quality focal classification with the original hard-label alpha weights."""
import torch
import torch.nn.functional as F


def aligned_iou(boxes, targets):
    """Aligned xyxy IoU, including disjoint or degenerate boxes, without an NxN matrix."""
    intersection = (torch.minimum(boxes[..., 2:], targets[..., 2:])
                    - torch.maximum(boxes[..., :2], targets[..., :2])).clamp(min=0).prod(-1)
    area_a = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(-1)
    area_b = (targets[..., 2:] - targets[..., :2]).clamp(min=0).prod(-1)
    return (intersection / (area_a + area_b - intersection).clamp(min=1e-7)).clamp(0, 1)


def quality_focal(logits, quality, positives, *, alpha=0.25, beta=2.0):
    """Unreduced QFL; binary targets recover sigmoid focal, including alpha weighting."""
    if not 0 <= alpha <= 1 or beta <= 0:
        raise ValueError("Require alpha in [0,1] and beta > 0")
    quality = quality.detach()
    weight = torch.where(positives, alpha, 1 - alpha)
    return F.binary_cross_entropy_with_logits(logits, quality, reduction="none") * (
        logits.sigmoid() - quality).abs().pow(beta) * weight


def matched_quality_loss(outputs, targets, indices, *, alpha=0.25, beta=2.0):
    """Dynamic-K matched GT quality targets; classification cannot backpropagate into boxes."""
    logits = outputs["pred_logits"]
    quality = torch.zeros_like(logits)
    positives = torch.zeros_like(logits, dtype=torch.bool)
    count = 0
    iou_sum = logits.new_zeros(())
    with torch.no_grad():
        for b, (mask, gt_indices) in enumerate(indices):
            if len(gt_indices) == 0:
                continue
            labels = targets[b]["labels"][gt_indices]
            q = aligned_iou(outputs["pred_boxes"][b, mask].detach(),
                            targets[b]["boxes_xyxy"][gt_indices].detach())
            queries = torch.arange(logits.shape[1], device=logits.device)[mask]
            quality[b, queries, labels] = q.to(logits.dtype)
            positives[b, queries, labels] = True
            count += len(gt_indices)
            iou_sum += q.sum()
    losses = quality_focal(logits, quality, positives, alpha=alpha, beta=beta)
    return losses.sum() / max(1, count), dict(quality_positives=count,
                                             quality_mean_iou=iou_sum / max(1, count))
