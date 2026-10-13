"""Opt-in common-random-number sampler for paired VP time-distribution trials."""
import torch


class PairedTimeSampler:
    """Keep the uniform arm's original RNG stream; isolate only the mixture gate.

    Every image draws the original device-side randint, including forced-tmax
    images. The gate uses a private CPU generator in both arms, so it cannot
    shift augmentation, Gaussian noise, placeholder boxes or dropout draws.
    Reset per batch with a seed distinct from the batch's ordinary RNG seed.
    The caller chooses the actual distribution through the model's config.
    """

    def __init__(self):
        self.generator = torch.Generator(device="cpu")
        self.reset(0)

    def reset(self, seed):
        self.generator.manual_seed(seed)
        self.base_timesteps = []
        self.gates = []
        self.timesteps = []

    def __call__(self, num_timesteps, device, distribution, high_probability):
        if num_timesteps < 1 or distribution not in ("uniform", "mix_tmax"):
            raise ValueError("Requires positive num_timesteps and uniform/mix_tmax")
        if not 0 <= high_probability <= 1:
            raise ValueError("high_probability must be in [0, 1]")
        base = torch.randint(0, num_timesteps, (1,), device=device).long()
        gate = torch.rand((), generator=self.generator).item()
        t = (torch.full_like(base, num_timesteps - 1)
             if distribution == "mix_tmax" and gate < high_probability else base)
        self.base_timesteps.append(base.item())
        self.gates.append(gate)
        self.timesteps.append(t.item())
        return t
