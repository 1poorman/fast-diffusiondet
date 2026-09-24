# fast-diffusiondet M1: 统一采样器接口
# ========================================
# DenoiseFn 协议 + NFE 计数器 + 各 solver 共享的结果收尾逻辑。
#
# 唯一的去噪函数接口（蓝图 §3.1）：
#     denoise_fn(x, t) -> (pred_noise, x_start)
# 其中 x ∈ (B, P, 4)（归一化 [-scale, scale] 空间），t 为离散时间索引
# 的 long 张量（shape (B,)）。闭包内部完成 detector.py:170-184 的
# clamp -> 反归一化 -> head forward -> 反变换，除 DDIM 外所有 solver
# 都"看不见" images_whwh / backbone_feats / batched_inputs。
import torch

from detectron2.layers import batched_nms
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, Instances


class DenoiseFn:
    """包装 detector.model_predictions 的可调用对象。

    职责：
      1. 统计 NFE（每次调用 = 1 次 head forward，蓝图 §2.2）；
      2. 缓存每步的 outputs_class / outputs_coord，供 ensemble 与最终
         推理复用（R3：denoise_fn 只返回 box 相关量，cls 不参与 ODE）。
    """

    def __init__(self, predict_fn):
        # predict_fn(x, t) -> (preds, outputs_class, outputs_coord)
        # preds 为 detector.model_predictions 返回的 ModelPrediction namedtuple。
        self.predict_fn = predict_fn
        self.nfe = 0
        self.last_pred_noise = None
        self.last_x_start = None
        self.last_outputs_class = None
        self.last_outputs_coord = None

    def call_model_predictions(self, x, t):
        """与 detector.model_predictions 完全一致的返回形式（DDIM 逐 bit 路径专用）。"""
        self.nfe += 1
        preds, outputs_class, outputs_coord = self.predict_fn(x, t)
        self.last_pred_noise = preds.pred_noise
        self.last_x_start = preds.pred_x_start
        self.last_outputs_class = outputs_class
        self.last_outputs_coord = outputs_coord
        return preds, outputs_class, outputs_coord

    def __call__(self, x, t):
        """ODE solver 用的简化接口：返回 (pred_noise, x_start)。"""
        preds, outputs_class, outputs_coord = self.call_model_predictions(x, t)
        return preds.pred_noise, preds.pred_x_start


def collect_results(detector, batched_inputs, images, ensemble_score, ensemble_label,
                    ensemble_coord, outputs_class, outputs_coord, do_postprocess=True):
    """采样结束后的共享收尾（对应 detector.py:247-274 的非 DDIM 复刻）。

    DDIM 路径为保逐 bit 一致不经过这里（solvers/ddim.py 内含逐行拷贝）。
    """
    if detector.use_ensemble and detector.sampling_timesteps > 1:
        box_pred_per_image = torch.cat(ensemble_coord, dim=0)
        scores_per_image = torch.cat(ensemble_score, dim=0)
        labels_per_image = torch.cat(ensemble_label, dim=0)
        if detector.use_nms:
            keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
            box_pred_per_image = box_pred_per_image[keep]
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]

        result = Instances(images.image_sizes[0])
        result.pred_boxes = Boxes(box_pred_per_image)
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        results = [result]
    else:
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = detector.inference(output["pred_logits"], output["pred_boxes"], images.image_sizes)

    if do_postprocess:
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    return results
