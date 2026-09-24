# -*- coding: utf-8 -*-
"""
DPM-Solver-v3 的 EMS 统计量 (M1 先用退化版，M3 再算真统计量)。

官方实现要求目录下必须有 ``l.npz``(key ``l``) 与 ``sb.npz``(keys ``s``,``b``)，
因为 ``statistics_steps = l.shape[0] - 1`` 决定了 logSNR 网格密度。

``degenerated=True`` 时求解器会强行把 ``l=1, s=0, b=0``（退化为 DPM-Solver++），
**不需要真实统计量**，但文件仍必须存在。M1 就用这种占位文件跑通链路。

形状约定（本项目相对官方的关键适配）：
    官方是逐像素统计 ``(N+1, C, H, W)``；本项目 box 张量是 ``(B, P, 4)``。
    取 ``(N+1, 1, 1)``（各向同性）或 ``(N+1, 1, 4)``（逐坐标 cx/cy/w/h），
    索引后为 ``(B, 1, 1)`` / ``(B, 1, 4)``，都能与 ``(B, P, 4)`` 正确广播。
"""

import os

import numpy as np

__all__ = ["make_degenerated_statistics", "ensure_statistics_dir"]


def make_degenerated_statistics(out_dir, statistics_steps=1200, tail=(1, 1), force=False):
    """
    生成占位统计量（l 全 1，s/b 全 0）。

    Returns:
        out_dir
    """
    os.makedirs(out_dir, exist_ok=True)
    l_path = os.path.join(out_dir, "l.npz")
    sb_path = os.path.join(out_dir, "sb.npz")
    if os.path.isfile(l_path) and os.path.isfile(sb_path) and not force:
        return out_dir

    shape = (statistics_steps + 1,) + tuple(tail)
    np.savez(l_path, l=np.ones(shape, dtype=np.float64))
    np.savez(sb_path, s=np.zeros(shape, dtype=np.float64), b=np.zeros(shape, dtype=np.float64))
    return out_dir


def ensure_statistics_dir(stats_dir, statistics_steps=1200, tail=(1, 1)):
    """若目录里没有统计量文件，就补一份退化版，保证求解器能构造。"""
    if stats_dir is None:
        return None
    l_path = os.path.join(stats_dir, "l.npz")
    sb_path = os.path.join(stats_dir, "sb.npz")
    if os.path.isfile(l_path) and os.path.isfile(sb_path):
        return stats_dir
    return make_degenerated_statistics(stats_dir, statistics_steps=statistics_steps, tail=tail)
