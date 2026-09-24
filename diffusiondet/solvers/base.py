# -*- coding: utf-8 -*-
"""
采样器公共抽象 (M1)。

核心约定：**去噪函数协议**
    denoise_fn(x, t) -> ModelPrediction / Tensor
其中 ``x`` 形状恒为 ``(B, P, 4)``（归一化 box，值域 ``[-scale, scale]``），
``t`` 是时间步张量（离散 timestep，形状 ``(B,)``）。

这样三个 solver 都**看不见** ``backbone_feats`` / ``images_whwh`` / ``batched_inputs``，
可以互相替换与对拍；NFE 由计数器统一统计（backbone 只算一次，不计入）。
"""

import torch
from collections import namedtuple

__all__ = ["ModelPrediction", "NFECounter", "BoxSolver"]

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class NFECounter:
    """包裹 denoise_fn，统计 head 前向次数（= NFE）。"""

    def __init__(self, fn):
        self.fn = fn
        self.nfe = 0

    def __call__(self, x, t):
        self.nfe += 1
        return self.fn(x, t)

    def reset(self):
        self.nfe = 0


class BoxSolver:
    """所有 box 采样器的基类。"""

    name = "base"

    def __init__(self, denoise_fn, shape, device, **kwargs):
        """
        Args:
            denoise_fn: callable(x, t) -> 预测结果
            shape: (B, P, 4)
            device: torch.device / str
        """
        self.denoise = NFECounter(denoise_fn)
        self.shape = tuple(shape)
        self.device = torch.device(device) if isinstance(device, str) else device

    @torch.no_grad()
    def sample(self, x_init=None, **kwargs):
        raise NotImplementedError

    @property
    def nfe(self):
        return self.denoise.nfe
