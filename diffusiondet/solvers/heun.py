# -*- coding: utf-8 -*-
"""
EDM 风格 Heun / Euler 采样器 (M1)。

直接搬自 ``edm-main/generate.py:25-60`` 的 ``edm_sampler``，去掉了
churn（``S_churn=0``，即完全确定性）与时间步的 float64 强制。

在 EDM 规范坐标下（见 schedule.py）：
    x_hat = x / alpha_t = x0 + sigma_hat * eps
    ODE:  dx_hat / d sigma_hat = (x_hat - D(x_hat; sigma_hat)) / sigma_hat
其中 ``D`` 就是 denoise_fn（DiffusionDet 的 head 直接预测 x0，天然吻合）。

NFE = num_steps（Euler）或 2*num_steps - 1（Heun，每步多一次校正）。

注意：DiffusionDet 的 head 只吃离散 timestep，
因此每一步都要用 ``schedule.t_from_sigma_hat`` 把连续 sigma_hat 取整到最近档位。
"""

import torch

from .base import BoxSolver

__all__ = ["HeunSolver", "karras_sigma_steps"]


def karras_sigma_steps(num_steps, sigma_min, sigma_max, rho=7.0, device="cuda"):
    """EDM 的 rho 幂律时间步（edm-main/generate.py:36）。"""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1.0 / rho)
               + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    return torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()


class HeunSolver(BoxSolver):
    name = "heun"

    def __init__(self, denoise_fn, shape, device, schedule,
                 num_steps=18, sigma_min=None, sigma_max=None, rho=7.0,
                 solver="heun", **kwargs):
        super().__init__(denoise_fn, shape, device, **kwargs)
        self.schedule = schedule
        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min if sigma_min is not None else schedule.sigma_hat_min)
        self.sigma_max = float(sigma_max if sigma_max is not None else schedule.sigma_hat_max)
        self.rho = float(rho)
        assert solver in ("euler", "heun"), solver
        self.solver = solver

    def _denoise(self, x, sigma_hat):
        """连续 sigma_hat -> 离散 timestep -> head 前向。"""
        t = self.schedule.t_from_sigma_hat(
            torch.full((x.shape[0],), float(sigma_hat), device=x.device))
        out = self.denoise(x, t)
        # 兼容返回 ModelPrediction 或直接返回 x0 两种协议
        if isinstance(out, tuple):
            return out[1]
        return out

    @torch.no_grad()
    def sample(self, x_init=None, **kwargs):
        batch = self.shape[0]
        t_steps = karras_sigma_steps(self.num_steps, self.sigma_min, self.sigma_max,
                                     self.rho, self.device)
        if x_init is None:
            x_next = torch.randn(self.shape, device=self.device) * t_steps[0]
        else:
            x_next = x_init * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            denoised = self._denoise(x_cur, float(t_cur))
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            if self.solver == "heun" and i < self.num_steps - 1:
                denoised = self._denoise(x_next, float(t_next))
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return {"x_final": x_next, "x_start": x_next, "ensemble": None}
