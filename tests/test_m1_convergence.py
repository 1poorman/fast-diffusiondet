# M1 Gate（二）：解析解收敛阶 + ODE solver 的 NFE 计数
# conda activate fastdiff && python -m pytest tests/test_m1_convergence.py -v -x
#
# 用已知闭式解的线性高斯去噪器验证收敛阶：
#   x0 ~ N(0, sd^2 I)（sd=1），后验均值 x0_hat = alpha * x（利用 alpha^2+sigma^2=1）
#   PF-ODE 解析解（w=sigma/alpha 参数化）：z(w) = x/alpha = sqrt(sd^2+w^2) * u，方向不变
#   x_init = alpha(w_max) * z(w_max) = u（sd=1, w_max=8 时恰好 = u），x_true = sd * u
# 阶数测量：步数 k 减半序列 err(k)，p = log2(err(k)/err(2k))
#   Euler O(h) -> p≈1；Heun O(h^2) -> p≈2；DPMv3(order=3, deg) -> p>=1.7
import math
import types

import pytest
import torch

from diffusiondet.solvers import dpm_v3_sample, heun_sample


def _make_gaussian_schedule(T, w_max=8.0, w_min=1e-3):
    """abar 表（与真实 VP 一致：index 0 最干净 w=w_min，index T-1 最噪 w=w_max）。"""
    w = torch.logspace(math.log10(w_min), math.log10(w_max), T)
    return 1.0 / (1.0 + w ** 2)


def _fake_detector(schedule, sampling_timesteps, order=3, skip_type="logSNR"):
    d = types.SimpleNamespace()
    d.device = torch.device("cpu")
    d.num_timesteps = schedule.shape[0]
    d.sampling_timesteps = sampling_timesteps
    d.alphas_cumprod = schedule
    beta = 1.0 - schedule[1:] / schedule[:-1]
    d.betas = torch.cat([torch.tensor([1.0 - schedule[0].item()]), beta])
    d.box_renewal = False
    d.use_ensemble = False
    d.num_proposals = 8
    d.scale = 1.0
    d.use_nms = False
    d.dpm_order = order
    d.dpm_skip_type = skip_type
    d.dpm_degenerated = True
    d.dpm_stats_dir = ""
    d.heun_max_dw = float("inf")  # 合成高斯去噪器无 clamp 病理，关闭步长守卫以测纯 Heun 数学
    return d


class _GaussianDenoiseFn:
    """返回 (eps_hat, x0_hat)：x0_hat = alpha*x（后验均值），eps_hat = (x - alpha*x0_hat)/sigma。"""

    def __init__(self, schedule):
        self.schedule = schedule
        self.nfe = 0
        self.last_outputs_class = None
        self.last_outputs_coord = None

    def __call__(self, x, t):
        self.nfe += 1
        idx = int(t[0].item())
        alpha = self.schedule[idx].sqrt()
        sigma = (1.0 - self.schedule[idx]).sqrt()
        x0 = alpha * x
        return (x - alpha * x0) / sigma, x0


def _setup(T=512):
    torch.manual_seed(0)
    schedule = _make_gaussian_schedule(T)
    u = torch.randn(1, 8, 4)
    u = u / u.norm()
    x_init = u.clone()  # sd=1, w_max=8: x_init = sqrt(65)/sqrt(65) * u
    x_true = u.clone()  # w->0: x = sd * u
    return schedule, x_init, x_true


def _run_ode(order, k, schedule, x_init):
    det = _fake_detector(schedule, sampling_timesteps=k)
    fn = _GaussianDenoiseFn(schedule)
    img, _, _ = heun_sample(det, fn, 1, (1, 8, 4), None, None,
                            order=order, do_postprocess=False, noise=x_init.clone())
    return img


def test_convergence_euler():
    schedule, x_init, x_true = _setup()
    errs = []
    for k in [8, 16, 32, 64]:
        img = _run_ode(1, k, schedule, x_init)
        errs.append((img - x_true).abs().max().item())
    p = math.log2(errs[0] / errs[1])
    print(f"euler errs={errs}, measured order={p:.3f}")
    assert abs(p - 1.0) < 0.3, f"euler measured order {p:.3f}, expect ~1.0"


def test_convergence_heun():
    schedule, x_init, x_true = _setup()
    errs = []
    for k in [8, 16, 32, 64]:
        img = _run_ode(2, k, schedule, x_init)
        errs.append((img - x_true).abs().max().item())
    p = math.log2(errs[0] / errs[1])
    print(f"heun errs={errs}, measured order={p:.3f}")
    assert abs(p - 2.0) < 0.3, f"heun measured order {p:.3f}, expect ~2.0"


class _MixtureDenoiseFn:
    """非线性去噪器：x0 ~ 0.5N(m1,s^2)+0.5N(m2,s^2)（逐坐标独立）。

    后验均值 x0_hat = sum_i r_i * mean_i（软指派），对 x 非线性，
    破坏 DPMv3 在线性高斯问题上的指数积分器精确性，可测收敛阶。
    """

    def __init__(self, schedule, m1=-0.6, m2=0.6, s=0.5):
        self.schedule = schedule
        self.m = torch.tensor([m1, m2]).view(2, 1, 1)  # (2,1,1) 广播到 (B,P,4)
        self.s = s
        self.nfe = 0
        self.last_outputs_class = None
        self.last_outputs_coord = None

    def __call__(self, x, t):
        self.nfe += 1
        idx = int(t[0].item())
        abar = self.schedule[idx]
        alpha = abar.sqrt()
        sigma = (1.0 - abar).sqrt()
        # 分量后验：likelihood x ~ N(alpha*x0, sigma^2)，prior x0 ~ N(m_i, s^2)
        v = 1.0 / (alpha ** 2 / sigma ** 2 + 1.0 / self.s ** 2)          # 标量（两分量同方差）
        mean_i = v * (alpha * x / sigma ** 2 + self.m / self.s ** 2)      # (2,B,P,4)
        logpdf = -0.5 * ((x - alpha * self.m) ** 2) / (alpha ** 2 * self.s ** 2 + sigma ** 2)
        r = torch.softmax(logpdf, dim=0)                                  # (2,B,P,4)
        x0 = (r * mean_i).sum(dim=0)
        return (x - alpha * x0) / sigma, x0


def test_convergence_dpm_v3():
    """Gate 2: DPMv3(order=3, degenerated) 收敛阶 >= O(h^2)-0.3。

    参考解：同一 schedule 上 steps=256 的 DPMv3（确定性，无随机源）。
    注意：参考步数不能接近调度表长度 T（此处 512），否则 inverse_lambda
    分段线性插值分辨率不足会产生重复 λ，使 Vandermonde 矩阵奇异
    （实际使用 NFE<=10 远小于 N=1000，不会触发）。
    """
    torch.manual_seed(0)
    T = 512
    schedule = _make_gaussian_schedule(T)
    u = torch.randn(1, 8, 4)
    w_max = 8.0
    sigma_max = (1.0 - schedule[T - 1]).sqrt()
    x_init = u * sigma_max  # 典型噪声起点（同一 x_T 保证与参考解同轨迹）

    def run(k):
        det = _fake_detector(schedule, sampling_timesteps=k, order=3)
        fn = _MixtureDenoiseFn(schedule)
        img, _, _ = dpm_v3_sample(det, fn, 1, (1, 8, 4), None, None,
                                  do_postprocess=False, noise=x_init.clone())
        assert torch.isfinite(img).all()
        return img

    ref = run(256)
    errs = []
    nfes = []
    for k in [4, 8, 16, 32]:
        det = _fake_detector(schedule, sampling_timesteps=k, order=3)
        fn = _MixtureDenoiseFn(schedule)
        img, _, _ = dpm_v3_sample(det, fn, 1, (1, 8, 4), None, None,
                                  do_postprocess=False, noise=x_init.clone())
        errs.append((img - ref).abs().max().item())
        nfes.append(fn.nfe)
    # 用最细两档测阶：粗步数段受离散索引 snap（连续 t -> 最近离散 index）干扰
    p = math.log2(errs[-2] / errs[-1])
    print(f"dpm_v3 errs={errs}, nfe={nfes}, measured order={p:.3f}")
    assert p >= 1.7, f"dpm_v3 measured order {p:.3f}, expect >= O(h^2)-0.3"
    assert all(e2 < e1 for e1, e2 in zip(errs[:-1], errs[1:])), f"errs not monotone: {errs}"


def test_nfe_ode_solvers():
    schedule = _make_gaussian_schedule(256, w_max=8.0, w_min=1e-3)
    x_init = torch.randn(1, 8, 4)
    # heun: 2N-1；euler: N
    for order, expected in [(2, 7), (1, 4)]:
        det = _fake_detector(schedule, sampling_timesteps=4)
        fn = _GaussianDenoiseFn(schedule)
        heun_sample(det, fn, 1, (1, 8, 4), None, None, order=order,
                    do_postprocess=False, noise=x_init.clone())
        assert fn.nfe == expected, f"order={order}: nfe={fn.nfe} != {expected}"
    # dpm_v3: NFE = steps
    det = _fake_detector(schedule, sampling_timesteps=4, order=3)
    fn = _MixtureDenoiseFn(schedule)
    dpm_v3_sample(det, fn, 1, (1, 8, 4), None, None,
                  do_postprocess=False, noise=x_init.clone())
    assert fn.nfe == 4, f"dpm_v3 nfe={fn.nfe}, expect 4"
