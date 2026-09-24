# fast-diffusiondet M1: VP 噪声调度与时间网格
# ========================================
# DiffusionDet 的前向满足标准 VP：x_t = α_t·x0 + σ_t·ε，且 α² + σ² = 1，
# 其中 α_t = sqrt(ᾱ_t)、σ_t = sqrt(1-ᾱ_t)，ᾱ 来自 cosine beta schedule。
# 本模块只封装离散索引 i ∈ [0, N-1] 上的查表；连续时间（DPM-Solver-v3）
# 由 NoiseScheduleVP 处理。
import torch


class VPSchedule:
    """离散 VP 调度查表（alphas_cumprod 张量的薄包装）。"""

    def __init__(self, alphas_cumprod):
        assert alphas_cumprod.dim() == 1
        self.alphas_cumprod = alphas_cumprod
        self.num_timesteps = alphas_cumprod.shape[0]

    def alpha(self, i):
        """α_i = sqrt(ᾱ_i)"""
        return self.alphas_cumprod[i].sqrt()

    def sigma(self, i):
        """σ_i = sqrt(1 - ᾱ_i)"""
        return (1.0 - self.alphas_cumprod[i]).sqrt()

    def w(self, i):
        """w_i = σ_i / α_i = e^{-λ_i}（DDIM/DPM-Solver++ 的步长参数化）"""
        a = self.alphas_cumprod[i]
        return (1.0 - a).sqrt() / a.sqrt()

    def lambda_(self, i):
        """λ_i = log(α_i / σ_i)，半 log-SNR"""
        a = self.alphas_cumprod[i]
        return 0.5 * (torch.log(a) - torch.log(1.0 - a))


def build_time_pairs(total_timesteps, sampling_timesteps):
    """构造与 detector.ddim_sample 完全相同的时间网格（保证同 NFE 公平对比）。

    返回 (time_pairs, times)：
      times:      [T-1, ..., 1, 0, -1] 共 sampling_timesteps+1 个（降序）
      time_pairs: [(T-1, T-2), ..., (1, 0), (0, -1)] 共 sampling_timesteps 个
    """
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    return time_pairs, times
