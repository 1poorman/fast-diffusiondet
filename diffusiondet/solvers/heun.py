# fast-diffusiondet M1: Euler / Heun 采样器（VP 离散调度，EDM 数学形式）
# ========================================
# 设计（蓝图 §3.3 移植说明 + 本机修订）：
#   * 数学形式来自 edm-main/generate.py 的 edm_sampler / ablation_sampler：
#     1 阶 Euler 与 2 阶 Heun（S_churn=0，完全确定性），NFE = N（Euler）/ 2N-1（Heun）。
#   * 修订：不采用 ablation_sampler 的"线性 VP 拟合"路径（它会用
#     vp_beta_d/beta_min 近似我们的 cosine 调度），而是直接在离散 cosine
#     VP 调度上积分 PF-ODE，杜绝拟合误差。
#   * 参数化：z = x/α，w = σ/α = e^{-λ}。PF-ODE 为 dz/dw = ε̂(z·α, w)，
#     其中 ε̂ = (x - α·x0_hat)/σ = pred_noise（与 predict_noise_from_start 一致）。
#     该参数化下 1 阶 Euler 与 DDIM(eta=0) 严格等价（自检性质）。
#   * 时间网格与 DDIM 完全同一份（build_time_pairs），保证同 NFE 公平对比。
#   * 最后一个区间 (0, -1)：直接取 x0 预测（与 DDIM 的 time_next<0 分支一致）。
import torch

from .base import collect_results
from .schedule import build_time_pairs


@torch.no_grad()
def heun_sample(detector, denoise_fn, batch, shape, batched_inputs, images,
                order=2, do_postprocess=True, noise=None):
    """VP 离散调度上的 Euler(order=1) / Heun(order=2) 采样。

    主矩阵（FEP §2.3）要求 BOX_RENEWAL=False，此处不支持 renewal；
    ensemble 语义与 DDIM 对齐（最后一步的输出不参与 ensemble）。
    """
    assert order in (1, 2), "order must be 1 (euler) or 2 (heun)"
    assert not detector.box_renewal, \
        "Box renewal is not supported for ODE solvers (FEP requires it off); use SOLVER=ddim."
    device = detector.device
    ac = detector.alphas_cumprod
    time_pairs, _ = build_time_pairs(detector.num_timesteps, detector.sampling_timesteps)

    img = noise if noise is not None else torch.randn(shape, device=device)

    ensemble_score, ensemble_label, ensemble_coord = [], [], []
    outputs_class = outputs_coord = None
    for i, (time, time_next) in enumerate(time_pairs):
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        pred_noise, x_start = denoise_fn(img, time_cond)  # pred_noise == ε̂ == (z - x0_hat)/w
        outputs_class, outputs_coord = denoise_fn.last_outputs_class, denoise_fn.last_outputs_coord

        if time_next < 0:
            img = x_start
            continue

        alpha = ac[time].sqrt()
        alpha_next = ac[time_next].sqrt()
        w = (1.0 - ac[time]).sqrt() / alpha          # w_cur = σ/α
        w_next = (1.0 - ac[time_next]).sqrt() / alpha_next

        # z 空间 Euler 步（等价于 DDIM eta=0 的更新）
        z = img / alpha
        z_prime = z + (w_next - w) * pred_noise

        # 步长守卫：cosine 调度末端 w 可达 O(10^4)，DDIM 网格首步 |Δw| 巨大。
        # 二阶校正在超大步长上会放大 clamp 去噪器的不一致性（实测 AP 从 47 崩到 5.6），
        # 故 |Δw| 超过 heun_max_dw 的区间退回一阶（多阶 ramp-up 的标准做法）。
        # 实际 NFE 由 DenoiseFn 计数器如实记录。
        heun_max_dw = getattr(detector, "heun_max_dw", 1.0)
        do_correction = order == 2 and (w - w_next).item() <= heun_max_dw

        if do_correction:
            # Heun 二阶校正：在 t_next 处再评一次 ε̂
            t_next_cond = torch.full((batch,), time_next, device=device, dtype=torch.long)
            pred_noise_next, _ = denoise_fn(z_prime * alpha_next, t_next_cond)
            z_next = z + (w_next - w) * 0.5 * (pred_noise + pred_noise_next)
        else:
            z_next = z_prime
        img = z_next * alpha_next

        if detector.use_ensemble and detector.sampling_timesteps > 1:
            box_pred_per_image, scores_per_image, labels_per_image = detector.inference(
                outputs_class[-1], outputs_coord[-1], images.image_sizes)
            ensemble_score.append(scores_per_image)
            ensemble_label.append(labels_per_image)
            ensemble_coord.append(box_pred_per_image)

    if not do_postprocess:
        return img, outputs_class, outputs_coord
    return collect_results(detector, batched_inputs, images, ensemble_score, ensemble_label,
                           ensemble_coord, outputs_class, outputs_coord, do_postprocess)
