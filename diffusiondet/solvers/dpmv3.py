# -*- coding: utf-8 -*-
"""
DPM-Solver-v3 的 box 版适配层 (M1)。

关键点：
1. 工作在 EDM 规范坐标（x_hat, sigma_hat），因此用官方的 ``NoiseScheduleEDM``
   （alpha=1, sigma=t, lambda=-log t），并显式给出 t_start / t_end。
2. 官方求解器内部统一使用 **eps 预测**。而我们 head 给出的是 x0。
   注意 ``detector.predict_noise_from_start`` 在 VP 参数化下算出的正是
       eps = (x - alpha*x0)/sigma = (x_hat - x0)/sigma_hat
   恰好等于 sigma_hat 空间里的 eps，可以直接喂给求解器，无需二次转换。
3. 统计量：M1 用退化的占位文件（见 statistics.py）；M3 再换成自算的 EMS。

⚠ 已知限制（风险 R14）：DiffusionDet 的 head 只接受离散 timestep，
   连续 sigma_hat 必须取整，而高阶多步法对这种取整极其敏感 ——
   实测 order=3 在真实 1000 档下发散，**默认 order=2**。
"""

import torch

from .base import BoxSolver
from .statistics import ensure_statistics_dir
from .dpm_solver_v3 import DPM_Solver_v3, NoiseScheduleEDM

__all__ = ["DPMSolverV3Solver"]


class DPMSolverV3Solver(BoxSolver):
    name = "dpm_solver_v3"

    def __init__(self, denoise_fn, shape, device, schedule,
                 steps=10, order=2, skip_type="logSNR", degenerated=True,
                 statistics_dir="statistics/degenerated", statistics_steps=1200,
                 t_start=None, t_end=None, use_corrector=False, **kwargs):
        super().__init__(denoise_fn, shape, device, **kwargs)
        self.schedule = schedule
        self.steps = int(steps)
        self.order = int(order)
        self.use_corrector = bool(use_corrector)
        self.last_outputs = (None, None)

        ensure_statistics_dir(statistics_dir, statistics_steps=statistics_steps, tail=(1, 1))
        self.solver = DPM_Solver_v3(
            statistics_dir=statistics_dir,
            noise_schedule=NoiseScheduleEDM(),
            steps=self.steps,
            t_start=schedule.sigma_hat_max if t_start is None else t_start,
            t_end=schedule.sigma_hat_min if t_end is None else t_end,
            skip_type=skip_type,
            degenerated=degenerated,
            device=self.device,
        )

    def _model_fn(self, x, t_continuous):
        """DPM-Solver-v3 要求 model_fn(x, t) -> eps。"""
        t = self.schedule.t_from_sigma_hat(t_continuous.to(x.device))
        pred_noise, x_start, outputs_class, outputs_coord = self.denoise(x, t)
        self.last_outputs = (outputs_class, outputs_coord)
        return pred_noise

    @torch.no_grad()
    def sample(self, x_init=None, **kwargs):
        if x_init is None:
            x_init = torch.randn(self.shape, device=self.device)
        x = x_init * self.schedule.sigma_hat_max
        out = self.solver.sample(
            x, self._model_fn,
            order=self.order,
            p_pseudo=False,
            use_corrector=self.use_corrector,
            c_pseudo=True,
            lower_order_final=True,
        )
        return {"x_final": out, "x_start": out, "ensemble": None,
                "last_outputs": self.last_outputs}
