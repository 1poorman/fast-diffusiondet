# fast-diffusiondet M1: 采样器统一入口
# SAMPLER_REGISTRY + 分发函数。detector.forward 的推理分支经由此处
# 按 MODEL.DiffusionDet.SOLVER 选择采样器。

from .base import DenoiseFn, collect_results
from .schedule import VPSchedule, build_time_pairs
from .ddim import ddim_sample
from .heun import heun_sample
from .edm import edm_sample
from .dpm_solver_v3 import (
    DPM_Solver_v3,
    NoiseScheduleVP,
    model_wrapper,
    dpm_v3_sample,
)

__all__ = [
    "SAMPLER_REGISTRY",
    "register_sampler",
    "run_sampler",
    "DenoiseFn",
    "collect_results",
    "VPSchedule",
    "build_time_pairs",
    "DPM_Solver_v3",
    "NoiseScheduleVP",
    "model_wrapper",
]

SAMPLER_REGISTRY = {}


def register_sampler(name):
    def _wrap(fn):
        SAMPLER_REGISTRY[name] = fn
        return fn
    return _wrap


register_sampler("ddim")(ddim_sample)
register_sampler("heun")(lambda *a, **kw: heun_sample(*a, order=2, **kw))
register_sampler("euler")(lambda *a, **kw: heun_sample(*a, order=1, **kw))
register_sampler("dpm_v3")(dpm_v3_sample)
register_sampler("edm_heun")(lambda *a, **kw: edm_sample(*a, order=2, **kw))
register_sampler("edm_euler")(lambda *a, **kw: edm_sample(*a, order=1, **kw))


def run_sampler(detector, batched_inputs, backbone_feats, images_whwh, images,
                do_postprocess=True, noise=None):
    """detector.forward 推理分支的统一入口。

    依据 detector.solver_name 分发；denoise_fn 闭包绑定 backbone 特征与
    images_whwh，ODE 类 solver 不再直接接触这些量（蓝图 §3.1）。
    FORMULATION=edm 时 solver 名可带 edm_ 前缀（E1 矩阵）。
    """
    name = detector.solver_name
    if name not in SAMPLER_REGISTRY:
        raise KeyError(f"Unknown solver '{name}', available: {sorted(SAMPLER_REGISTRY)}")
    if name != "ddim" and detector.box_renewal:
        raise ValueError(
            "MODEL.DiffusionDet.BOX_RENEWAL must be False for non-DDIM solvers "
            "(renewal breaks multi-step ODE history, see blueprint R1).")

    batch = images_whwh.shape[0]
    shape = (batch, detector.num_proposals, 4)
    denoise_fn = detector.make_denoise_fn(backbone_feats, images_whwh)
    return SAMPLER_REGISTRY[name](
        detector, denoise_fn, batch, shape, batched_inputs, images,
        do_postprocess=do_postprocess, noise=noise)
