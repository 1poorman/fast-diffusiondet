# -*- coding: utf-8 -*-
"""
解析解收敛阶验证 (M1 Gate)。

思路：用一个**有闭式解**的合成去噪器，绕开真实网络，直接检验求解器数学。

设数据分布为 N(mu, s^2 I)，则在 EDM 规范坐标下 x_hat = x0 + sigma_hat * eps 的
后验均值去噪器是
    D(x_hat; sigma_hat) = (s^2 * x_hat + sigma_hat^2 * mu) / (s^2 + sigma_hat^2)
代入 ODE  dx_hat/d sigma_hat = (x_hat - D) / sigma_hat 得
    d(x_hat - mu)/d sigma_hat = sigma_hat * (x_hat - mu) / (s^2 + sigma_hat^2)
令 y = x_hat - mu，则 ln y = 0.5 ln(s^2 + sigma_hat^2) + C，于是
    y(sigma_hat) = y(sigma_hat_max) * sqrt(s^2 + sigma_hat^2) / sqrt(s^2 + sigma_hat_max^2)
sigma_hat -> 0 时：
    x_hat(0) = mu + (x_hat_max - mu) * s / sqrt(s^2 + sigma_hat_max^2)      <- 闭式解

用它检验：Euler 应为 O(h)，Heun 应为 O(h^2)，DPM-Solver-v3 应 >= O(h^2)。
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from diffusiondet.detector import cosine_beta_schedule
from diffusiondet.solvers import VPSchedule, HeunSolver, ensure_statistics_dir


def build_schedule(device="cuda"):
    betas = cosine_beta_schedule(1000)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return VPSchedule(alphas_cumprod, device=device)


def build_fine_schedule(n_levels, sigma_min, sigma_max, device="cuda"):
    """
    构造一个「时间档位极其密集」的等价 VP schedule，用于验证：
    DPM-Solver-v3 高阶失效是否来自「连续 sigma_hat 被取整到离散 timestep」。
    由 sigma_hat = sigma/alpha 且 alpha^2+sigma^2=1 反解：alphas_cumprod = 1/(1+sigma_hat^2)
    """
    sh = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), n_levels, device=device)
    alphas_cumprod = 1.0 / (1.0 + sh ** 2)
    return VPSchedule(alphas_cumprod, device=device)


def make_gaussian_denoiser(schedule, s=0.5, mu=0.0):
    """返回 denoise_fn(x, t_discrete) -> x0 估计。"""
    def denoise_fn(x, t):
        sh = schedule.sigma_hat[t].to(x.device).to(x.dtype).reshape(-1, 1, 1)
        s2 = s * s
        return (s2 * x + sh * sh * mu) / (s2 + sh * sh)
    return denoise_fn


def exact_solution(x_init, schedule, s=0.5, mu=0.0):
    sh_max = schedule.sigma_hat_max
    return mu + (x_init * sh_max - mu) * s / math.sqrt(s * s + sh_max * sh_max)


def fit_order(ns, errs):
    """对 log(err) ~ -order * log(N) 做最小二乘，返回阶数。"""
    xs = [math.log(n) for n in ns]
    ys = [math.log(max(e, 1e-12)) for e in errs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = sum((a - mx) ** 2 for a in xs)
    return -num / den


def run_heun(schedule, solver_name, ns, shape, device, s=0.5, mu=0.0, seed=0):
    errs = []
    denoise_fn = make_gaussian_denoiser(schedule, s=s, mu=mu)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_init = torch.randn(shape, generator=g).to(device)
    gt = exact_solution(x_init, schedule, s=s, mu=mu)
    for n in ns:
        solver = HeunSolver(denoise_fn=denoise_fn, shape=shape, device=device,
                            schedule=schedule, num_steps=n, solver=solver_name)
        out = solver.sample(x_init=x_init)
        errs.append((out["x_final"] - gt).abs().max().item())
    return errs


def run_dpmv3(schedule, ns, shape, device, order=3, s=0.5, mu=0.0, seed=0,
              stats_dir="statistics/degenerated", statistics_steps=1200,
              skip_type="logSNR", t_start=None, t_end=None):
    from diffusiondet.solvers.dpm_solver_v3 import DPM_Solver_v3, NoiseScheduleEDM
    ensure_statistics_dir(stats_dir, statistics_steps=statistics_steps, tail=(1, 1))

    denoise_fn = make_gaussian_denoiser(schedule, s=s, mu=mu)

    def model_fn(x, t_continuous):
        """DPM-Solver-v3 内部统一用 eps：eps = (x - x0) / sigma_hat"""
        t = schedule.t_from_sigma_hat(t_continuous.to(x.device))
        x0 = denoise_fn(x, t)
        sh = schedule.sigma_hat[t].to(x.device).to(x.dtype).reshape(-1, 1, 1)
        return (x - x0) / sh

    errs, nfes = [], []
    g = torch.Generator(device="cpu").manual_seed(seed)
    x_init = torch.randn(shape, generator=g).to(device)
    gt = exact_solution(x_init, schedule, s=s, mu=mu)
    for n in ns:
        dpm = DPM_Solver_v3(
            statistics_dir=stats_dir,
            noise_schedule=NoiseScheduleEDM(),
            steps=n,
            t_start=schedule.sigma_hat_max if t_start is None else t_start,
            t_end=schedule.sigma_hat_min if t_end is None else t_end,
            skip_type=skip_type,
            degenerated=True,
            device=device,
        )
        x = dpm.sample(x_init * schedule.sigma_hat_max, model_fn,
                       order=order, p_pseudo=False, use_corrector=False,
                       c_pseudo=True, lower_order_final=True)
        errs.append((x - gt).abs().max().item())
        nfes.append(n)
    return errs, nfes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--skip-dpmv3", action="store_true")
    ap.add_argument("--fine-schedule", type=int, default=0,
                    help=">0 时用该档位数构造超密集 schedule，用于验证取整误差的影响")
    ap.add_argument("--dpm-configs", nargs="+",
                    default=["full,3,logSNR,auto,auto", "full,2,logSNR,auto,auto",
                             "full,1,logSNR,auto,auto", "edm,3,logSNR,80.0,0.002"],
                    help="name,order,skip_type,t_start,t_end（auto=用 schedule 的 sigma_hat 范围）")
    args = ap.parse_args()

    device = args.device
    shape = (2, 16, 4)  # (B, P, 4)
    schedule = build_schedule(device)
    if args.fine_schedule:
        schedule = build_fine_schedule(args.fine_schedule, schedule.sigma_hat_min,
                                       schedule.sigma_hat_max, device)
        print("using FINE schedule with {} levels (rounding error ~{:.2e} rel)".format(
            args.fine_schedule,
            (math.log(schedule.sigma_hat_max / schedule.sigma_hat_min) / args.fine_schedule)))

    print("sigma_hat_min = {:.6f}   sigma_hat_max = {:.4f}".format(
        schedule.sigma_hat_min, schedule.sigma_hat_max))
    print("{:<22} {:<10} {:<40}".format("solver", "order", "errors (N=" + str(args.ns) + ")"))

    results = {}

    for name in ("euler", "heun"):
        errs = run_heun(schedule, name, args.ns, shape, device)
        o = fit_order(args.ns, errs)
        results[name] = o
        print("{:<22} {:<10.2f} {}".format(
            name, o, " ".join("{:.3e}".format(e) for e in errs)))

    if not args.skip_dpmv3:
        for cfg in args.dpm_configs:
            name, order, skip_type, t0, t1 = cfg.split(",")
            errs, _ = run_dpmv3(schedule, args.ns, shape, device, order=int(order),
                                skip_type=skip_type,
                                t_start=float(t0) if t0 != "auto" else None,
                                t_end=float(t1) if t1 != "auto" else None)
            o = fit_order(args.ns, errs)
            key = "dpmv3[{},o{},{}]".format(name, order, skip_type)
            results[key] = o
            print("{:<22} {:<10.2f} {}".format(
                key[:22], o, " ".join("{:.3e}".format(e) for e in errs)))

    print()
    print("--- Gate ---")
    ok = True
    euler_ok = 0.7 <= results.get("euler", 0) <= 1.3
    heun_ok = results.get("heun", 0) >= 1.6
    print("euler  ~ O(h)   : {:<5} (measured {:.2f}, expect 0.7~1.3)".format(
        "PASS" if euler_ok else "FAIL", results.get("euler", float('nan'))))
    print("heun   >= O(h^2): {:<5} (measured {:.2f}, expect >=1.6)".format(
        "PASS" if heun_ok else "FAIL", results.get("heun", float('nan'))))
    ok = euler_ok and heun_ok
    if not args.skip_dpmv3:
        dpm_keys = [k for k in results if k.startswith("dpmv3[full")]
        dpm_ok = False
        for k in dpm_keys:
            good = results[k] >= 1.6
            print("dpmv3  >= O(h^2): {:<5} {:<28} measured {:.2f}".format(
                "PASS" if good else "FAIL", k, results[k]))
            dpm_ok = dpm_ok or good
        ok = ok and dpm_ok
    print("[test_convergence] {}".format("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
