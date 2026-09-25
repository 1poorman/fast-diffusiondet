# fast-diffusiondet M4: EDM σ 空间采样器（Karras 网格，Euler/Heun）
# ========================================
# E1-HEUN 主力求解器。与 heun.py（VP 离散网格）的区别：
#   * 时间变量是 σ 本身（VE：x = x0 + σ·ε），Karras ρ=7 幂律网格 σ_max -> σ_min；
#   * denoise_fn(x, σ) 返回 (ε̂, D)，D = c_skip·x + c_out·F 为预条件 x0 预测，
#     ε̂ = (x − D)/σ；
#   * 最后一步 σ_min -> 0：直接返回 D(x, σ_min)（EDM 官方做法）；
#   * HEUN_MAX_DW 守卫在 σ 空间重新标定：σ_max=4，Karras 网格首步 Δσ≈1.7，
#     守卫阈值用 |Δσ| > heun_max_sigma_step 时退一阶（默认 1.0 覆盖首步，
#     其余步 Δσ < 1 全部二阶；M4 实测后可放开）。
import torch

from .base import collect_results


def karras_sigmas(sigma_min, sigma_max, rho, num_steps, device):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    sig = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
           * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sig, torch.zeros_like(sig[:1])])  # 末位 0


@torch.no_grad()
def edm_sample(detector, denoise_fn, batch, shape, batched_inputs, images,
               order=2, do_postprocess=True, noise=None):
    """EDM σ 空间 Euler(order=1)/Heun(order=2) 采样。NFE = N（Euler）/ 2N-1（Heun）。"""
    assert order in (1, 2)
    device = detector.device
    num_steps = detector.sampling_timesteps
    sigmas = karras_sigmas(detector.edm_sigma_min, detector.edm_sigma_max,
                           detector.edm_karras_rho, num_steps, device)
    heun_max = getattr(detector, "heun_max_sigma_step", 1.0)

    img = noise if noise is not None else torch.randn(shape, device=device) * detector.edm_sigma_max

    ensemble_score, ensemble_label, ensemble_coord = [], [], []
    outputs_class = outputs_coord = None
    for i in range(num_steps):
        sigma_cur = sigmas[i].item()
        t_cond = torch.full((batch,), sigma_cur, device=device, dtype=torch.float32)
        pred_noise, x_start = denoise_fn(img, t_cond)  # ε̂=(x-D)/σ, D=x0 预测
        outputs_class, outputs_coord = denoise_fn.last_outputs_class, denoise_fn.last_outputs_coord
        sigma_next = sigmas[i + 1].item()

        # Euler 步（σ 方向）
        d_cur = (img - x_start) / sigma_cur
        x_next = img + (sigma_next - sigma_cur) * d_cur

        if order == 2 and i < num_steps - 1 and (sigma_next - sigma_cur) <= heun_max:
            t_next_cond = torch.full((batch,), sigma_next, device=device, dtype=torch.float32)
            pred_noise_next, _ = denoise_fn(x_next, t_next_cond)
            d_next = (x_next - pred_noise_next * sigma_next) / sigma_next  # (x_next − D)/σ_next
            x_next = img + (sigma_next - sigma_cur) * 0.5 * (d_cur + d_next)
        img = x_next

        if detector.use_ensemble and detector.sampling_timesteps > 1:
            box_pred_per_image, scores_per_image, labels_per_image = detector.inference(
                outputs_class[-1], outputs_coord[-1], images.image_sizes)
            ensemble_score.append(scores_per_image)
            ensemble_label.append(labels_per_image)
            ensemble_coord.append(box_pred_per_image)

    # 终点投影：D(x, σ_min)（EDM 官方：最后 σ=0 一步直接取 D）
    t_final = torch.full((batch,), detector.edm_sigma_min, device=device, dtype=torch.float32)
    _, img = denoise_fn(img, t_final)

    if not do_postprocess:
        return img, outputs_class, outputs_coord
    return collect_results(detector, batched_inputs, images, ensemble_score, ensemble_label,
                           ensemble_coord, outputs_class, outputs_coord, do_postprocess)
