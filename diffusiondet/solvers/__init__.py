# -*- coding: utf-8 -*-
"""
fast-diffusiondet 采样器包 (M1)。

    from diffusiondet.solvers import build_solver
"""

from .base import BoxSolver, NFECounter, ModelPrediction
from .schedule import VPSchedule
from .ddim import DDIMSolver
from .heun import HeunSolver
from .dpmv3 import DPMSolverV3Solver
from .statistics import make_degenerated_statistics, ensure_statistics_dir

__all__ = [
    "BoxSolver", "NFECounter", "ModelPrediction",
    "VPSchedule", "DDIMSolver", "HeunSolver", "DPMSolverV3Solver",
    "make_degenerated_statistics", "ensure_statistics_dir",
    "SAMPLER_REGISTRY", "register_sampler", "build_solver",
]

SAMPLER_REGISTRY = {}


def register_sampler(name):
    def _wrap(cls):
        SAMPLER_REGISTRY[name] = cls
        return cls
    return _wrap


register_sampler("ddim")(DDIMSolver)
register_sampler("heun")(HeunSolver)
register_sampler("euler")(HeunSolver)          # euler 与 heun 同类，靠 solver= 参数区分
register_sampler("dpm_solver_v3")(DPMSolverV3Solver)


def build_solver(name, denoise_fn, shape, device, **kwargs):
    """
    统一构造入口。

    Args:
        name: 'ddim' | 'heun' | 'dpm_solver_v3'
        denoise_fn: callable(x, t)
        shape: (B, P, 4)
        device: 'cuda' / 'cpu'
    """
    if name not in SAMPLER_REGISTRY:
        raise KeyError("未知采样器 '{}'，可用: {}".format(name, list(SAMPLER_REGISTRY.keys())))
    return SAMPLER_REGISTRY[name](denoise_fn=denoise_fn, shape=shape, device=device, **kwargs)
