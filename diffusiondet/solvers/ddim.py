# fast-diffusiondet M1: DDIM 采样器（从 detector.py 逐行抽取）
# ========================================
# 来源：detector.py 的 DiffusionDet.ddim_sample（原 186-274 行），逐行拷贝。
# 唯一改动：
#   1. `self.model_predictions(...)` -> `denoise_fn.call_model_predictions(...)`
#      （DenoiseFn 闭包内部调用同一个 method，运算顺序不变）；
#   2. `self.*` -> `detector.*`；
#   3. 初始噪声支持注入（FEP §2.3，M2 用；默认仍为 torch.randn）。
# M1 Gate：与原实现同 seed 下 box 输出 max|Δ| < 1e-5。
import torch

from detectron2.layers import batched_nms
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, Instances


@torch.no_grad()
def ddim_sample(detector, denoise_fn, batch, shape, batched_inputs, images,
                do_postprocess=True, noise=None):
    """DDIM 采样（eta=1，含 box renewal 与 ensemble，均沿用原实现语义）。"""
    total_timesteps, sampling_timesteps, eta, objective = (
        detector.num_timesteps, detector.sampling_timesteps,
        detector.ddim_sampling_eta, detector.objective)

    # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), ..., (1, 0), (0, -1)]

    img = noise if noise is not None else torch.randn(shape, device=detector.device)

    ensemble_score, ensemble_label, ensemble_coord = [], [], []
    x_start = None
    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=detector.device, dtype=torch.long)
        self_cond = None  # detector.self_condition 恒为 False
        preds, outputs_class, outputs_coord = denoise_fn.call_model_predictions(img, time_cond)
        pred_noise, x_start = preds.pred_noise, preds.pred_x_start

        if detector.box_renewal:  # filter
            score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
            threshold = 0.5
            score_per_image = torch.sigmoid(score_per_image)
            value, _ = torch.max(score_per_image, -1, keepdim=False)
            keep_idx = value > threshold
            num_remain = torch.sum(keep_idx)

            pred_noise = pred_noise[:, keep_idx, :]
            x_start = x_start[:, keep_idx, :]
            img = img[:, keep_idx, :]
        if time_next < 0:
            img = x_start
            continue

        alpha = detector.alphas_cumprod[time]
        alpha_next = detector.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise_t = torch.randn_like(img)    ## sample noise

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise_t

        if detector.box_renewal:  # filter
            # replenish with randn boxes
            img = torch.cat((img, torch.randn(1, detector.num_proposals - num_remain, 4, device=img.device)), dim=1)
        if detector.use_ensemble and detector.sampling_timesteps > 1:
            box_pred_per_image, scores_per_image, labels_per_image = detector.inference(
                outputs_class[-1], outputs_coord[-1], images.image_sizes)
            ensemble_score.append(scores_per_image)
            ensemble_label.append(labels_per_image)
            ensemble_coord.append(box_pred_per_image)

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
        box_cls = output["pred_logits"]
        box_pred = output["pred_boxes"]
        results = detector.inference(box_cls, box_pred, images.image_sizes)
    if do_postprocess:
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
    return results
