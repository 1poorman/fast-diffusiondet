# -*- coding: utf-8 -*-
"""
DDIM solver —— 基线。

这是 ``diffusiondet/detector.py`` 原 ``ddim_sample``（旧版 186-274 行）的**逐行搬移**，
包括 Box Renewal 与 Ensemble 的全部逻辑，以及**完全一致的 RNG 调用顺序**：
    (1) 初始 ``torch.randn(shape)``
    (2) 每步 ``torch.randn_like(img)``（eta>0 的随机项）
    (3) 每步 renewal 补齐 ``torch.randn(1, P - num_remain, 4)``
任何顺序变动都会破坏 M1 的「逐 bit 等价」Gate。

注意 DiffusionDet 默认 ``ddim_sampling_eta = 1.0``（detector.py:97），
**基线其实是随机采样器（DDPM 形式）而非确定性 DDIM**。
做 ODE 求解器对比时应使用 ``eta=0``，这一点在 FEP 里单独处理。
"""

import torch

from .base import BoxSolver

__all__ = ["DDIMSolver"]


class DDIMSolver(BoxSolver):
    name = "ddim"

    def __init__(self, denoise_fn, shape, device,
                 alphas_cumprod, num_timesteps=None, sampling_timesteps=1,
                 eta=1.0, box_renewal=True, use_ensemble=True,
                 num_proposals=None, ensemble_cb=None, **kwargs):
        super().__init__(denoise_fn, shape, device, **kwargs)
        batch, p, _ = shape
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.num_timesteps = int(num_timesteps or self.alphas_cumprod.shape[0])
        self.sampling_timesteps = int(sampling_timesteps)
        self.eta = float(eta)
        self.box_renewal = bool(box_renewal)
        self.use_ensemble = bool(use_ensemble)
        self.num_proposals = int(num_proposals or p)
        self.ensemble_cb = ensemble_cb  # callable(outputs_class, outputs_coord) -> (boxes, scores, labels)

    @torch.no_grad()
    def sample(self, x_init=None, **kwargs):
        batch = self.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.eta

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=self.device) if x_init is None else x_init

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        last_outputs = (None, None)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            pred_noise, x_start, outputs_class, outputs_coord = self.denoise(img, time_cond)
            last_outputs = (outputs_class, outputs_coord)

            if self.box_renewal:  # filter
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

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)  # sample noise
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            if self.box_renewal:  # filter
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4,
                                                  device=img.device)), dim=1)

            if self.use_ensemble and self.sampling_timesteps > 1 and self.ensemble_cb is not None:
                box_pred_per_image, scores_per_image, labels_per_image = self.ensemble_cb(
                    outputs_class, outputs_coord)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # 注意：这里返回的是**列表**（与 detector.py 原实现一致），
        # concat + NMS 由 detector 负责，不要把后处理逻辑搬进 solver。
        ensemble = None
        if self.use_ensemble and self.sampling_timesteps > 1 and ensemble_score:
            ensemble = (ensemble_score, ensemble_label, ensemble_coord)
        return {"x_final": img, "x_start": x_start, "ensemble": ensemble,
                "last_outputs": last_outputs}
