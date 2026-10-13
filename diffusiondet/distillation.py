"""GT-aligned, quality-gated distillation for a truncated DiffusionDet student.

The frozen teacher may have a different proposal ordering/count. For each GT,
pick its highest-IoU matched teacher query, then supervise matched student queries
only where that teacher is better. Background queries are never KD targets.
"""
import re

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, generalized_box_iou


def load_truncated_student(model, state_dict):
    """Drop only unused RCNN stages; all remaining parameters must match exactly."""
    kept, dropped = {}, []
    for key, value in state_dict.items():
        match = re.match(r"head\.head_series\.(\d+)\.", key)
        if match and int(match[1]) >= model.head.num_heads:
            dropped.append(key)
        else:
            kept[key] = value
    model.load_state_dict(kept, strict=True)
    return dropped


def bernoulli_kl(student_logits, teacher_logits, temperature=2.0):
    """Stable independent-class KL(teacher || student), per query, with T^2."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_logits.float() / temperature
    t = teacher_logits.detach().float() / temperature
    p = t.sigmoid()
    kl = p * (F.logsigmoid(t) - F.logsigmoid(s))
    kl += (1 - p) * (F.logsigmoid(-t) - F.logsigmoid(-s))
    return kl.sum(dim=-1) * temperature ** 2


def quality_distillation_loss(student, teacher, targets, student_indices, teacher_indices,
                              temperature=2.0, min_iou=0.5):
    """Return unweighted cls/box losses and detached diagnostic scalars.

    Matching format is the repository's Dynamic-K (foreground mask, GT indices).
    Box KD = 5 * normalized-coordinate L1 + 2 * GIoU loss. Normalize by the
    number of accepted positive pairs, not the full proposal/background count.
    Single-device training is used by the V3 pilot runner.
    """
    if not 0 <= min_iou <= 1:
        raise ValueError("min_iou must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    cls_sum = student["pred_logits"].sum() * 0.0
    box_sum = student["pred_boxes"].sum() * 0.0
    accepted = positives = 0
    teacher_iou_sum = student_iou_sum = 0.0
    for b, target in enumerate(targets):
        if len(target["labels"]) == 0:
            continue
        s_mask, s_gt = student_indices[b]
        t_mask, t_gt = teacher_indices[b]
        s_ids = torch.where(s_mask)[0]
        t_ids = torch.where(t_mask)[0]
        positives += len(s_ids)
        if len(s_ids) == 0 or len(t_ids) == 0:
            continue
        with torch.no_grad():
            gt_boxes = target["boxes_xyxy"]
            t_boxes = teacher["pred_boxes"][b].detach()
            t_logits = teacher["pred_logits"][b].detach()
            t_iou = box_iou(t_boxes[t_ids], gt_boxes).gather(1, t_gt[:, None]).squeeze(1)
            s_iou = box_iou(student["pred_boxes"][b, s_ids].detach(), gt_boxes).gather(1, s_gt[:, None]).squeeze(1)
            # A GT-aligned teacher representative; no query-index correspondence.
            best_query = torch.full((len(gt_boxes),), -1, device=t_ids.device, dtype=torch.long)
            best_iou = torch.zeros(len(gt_boxes), device=t_ids.device)
            for gt_id in t_gt.unique():
                candidates = torch.where(t_gt == gt_id)[0]
                best = candidates[t_iou[candidates].argmax()]
                best_query[gt_id] = t_ids[best]
                best_iou[gt_id] = t_iou[best]
            selected_teacher = best_query[s_gt]
            selected_iou = best_iou[s_gt]
            keep = (selected_teacher >= 0) & (selected_iou >= min_iou) & (selected_iou > s_iou)
            # Do not transfer a teacher's wrong class as a positive soft target.
            predicted_class = t_logits[selected_teacher.clamp(min=0)].argmax(-1)
            keep &= predicted_class == target["labels"][s_gt]
            s_keep = s_ids[keep]
            t_keep = selected_teacher[keep]
            n = int(keep.sum())
            teacher_iou_sum += float(selected_iou[keep].sum())
            student_iou_sum += float(s_iou[keep].sum())
        if not n:
            continue
        accepted += n
        cls_sum = cls_sum + bernoulli_kl(student["pred_logits"][b, s_keep],
                                        t_logits[t_keep], temperature).sum()
        s_boxes = student["pred_boxes"][b, s_keep]
        teach_boxes = t_boxes[t_keep]
        size = target["image_size_xyxy"]
        l1 = F.l1_loss(s_boxes / size, teach_boxes / size, reduction="none").sum(-1)
        giou = 1 - generalized_box_iou(s_boxes, teach_boxes).diag()
        box_sum = box_sum + (5 * l1 + 2 * giou).sum()
    denom = max(accepted, 1)
    return {"loss_kd_cls": cls_sum / denom, "loss_kd_box": box_sum / denom}, {
        "kd_pairs": accepted, "student_positives": positives,
        "kd_teacher_iou": teacher_iou_sum / denom, "kd_student_iou": student_iou_sum / denom,
    }
