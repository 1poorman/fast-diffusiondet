# -*- coding: utf-8 -*-
"""
噪声调度 (M1)。

DiffusionDet 用的是标准 VP：``x_t = sqrt(abar_t) * x0 + sqrt(1-abar_t) * eps``，
满足 ``alpha^2 + sigma^2 = 1``。

高阶求解器（Heun / DPM-Solver-v3）工作在 EDM 的规范坐标里：
    x_hat = x / alpha_t = x0 + sigma_hat * eps,   sigma_hat = sigma_t / alpha_t
这样 ``alpha=1, sigma=sigma_hat``，正好是 EDM 的形式，
可直接复用 EDM 的 Euler/Heun 与 DPM-Solver-v3 的 NoiseScheduleEDM。

注意：DiffusionDet 的 head 只接受**离散** timestep(0..999)，
所以从连续的 sigma_hat 反查 t 时必须取整到最近的离散档位。
"""

import torch

__all__ = ["VPSchedule"]


class VPSchedule:
    def __init__(self, alphas_cumprod, device="cuda"):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.alphas_cumprod = alphas_cumprod.to(self.device).float()
        self.num_timesteps = int(self.alphas_cumprod.shape[0])

        self.alpha = self.alphas_cumprod.sqrt()
        self.sigma = (1.0 - self.alphas_cumprod).sqrt()
        # EDM canonical: x_hat = x/alpha = x0 + sigma_hat * eps
        self.sigma_hat = self.sigma / self.alpha
        # half-logSNR: lambda_t = log(alpha/sigma) = -log(sigma_hat)
        self.lambda_t = torch.log(self.alpha) - torch.log(self.sigma)

    # ---------- 离散 <-> 连续 ----------
    def sigma_hat_at(self, t):
        """t: 离散 timestep（int 或 LongTensor）-> sigma_hat"""
        return self.sigma_hat[t]

    def t_from_sigma_hat(self, sigma_hat):
        """连续 sigma_hat -> 最近的离散 timestep（LongTensor）。"""
        sh = torch.as_tensor(sigma_hat, device=self.device).float().reshape(-1)
        idx = torch.searchsorted(self.sigma_hat, sh)
        idx = idx.clamp(1, self.num_timesteps - 1)
        left = (idx - 1).clamp(0, self.num_timesteps - 1)
        # 取两边更近的一档
        d_left = (sh - self.sigma_hat[left]).abs()
        d_right = (self.sigma_hat[idx] - sh).abs()
        return torch.where(d_left <= d_right, left, idx).long()

    @property
    def sigma_hat_min(self):
        return float(self.sigma_hat[0])

    @property
    def sigma_hat_max(self):
        return float(self.sigma_hat[-1])
