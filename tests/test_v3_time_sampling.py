"""Protect a one-factor experiment: t changes, noise and padding must not."""
import random
from types import SimpleNamespace

import pytest
import torch

from diffusiondet.detector import DiffusionDet
from diffusiondet.time_sampling import PairedTimeSampler


def stub(distribution="uniform", probability=0.5, paired=True):
    obj = SimpleNamespace(device=torch.device("cpu"), num_timesteps=1000,
                          num_proposals=8, scale=2.0, formulation="vp",
                          train_t_dist=distribution, train_t_high_prob=probability,
                          train_time_sampler=PairedTimeSampler() if paired else None,
                          train_diffusion_observer=None)
    obj.sample_train_t = lambda: DiffusionDet.sample_train_t(obj)
    # Exercise real proposal preparation without building a backbone.
    obj.q_sample = lambda x_start, t, noise: (1 - t.float() / 1000) * x_start + noise
    return obj


@pytest.mark.parametrize("num_gt", [0, 3, 8, 12])
def test_only_t_changes_including_empty_padding_and_subsample_paths(num_gt):
    gt = torch.linspace(0.1, 0.9, max(1, num_gt * 4))[:num_gt * 4].reshape(num_gt, 4)
    traces = []
    for distribution in ("uniform", "mix_tmax"):
        model = stub(distribution, 1.0)
        captured = []
        model.train_diffusion_observer = lambda x, eps, t: captured.append((x.clone(), eps.clone()))
        torch.manual_seed(42)
        random.seed(42)
        boxes, noise, t = DiffusionDet.prepare_diffusion_concat(model, gt)
        traces.append((captured[0], t, torch.get_rng_state(), random.getstate(), boxes))
    for a, b in zip(traces[0][0], traces[1][0]):
        assert torch.equal(a, b)
    assert traces[0][1].item() != traces[1][1].item() == 999
    assert torch.equal(traces[0][2], traces[1][2])
    assert traces[0][3] == traces[1][3]
    assert not torch.equal(traces[0][4], traces[1][4])


def test_paired_uniform_preserves_legacy_output_and_global_rng():
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    results = []
    for paired in (False, True):
        torch.manual_seed(31)
        model = stub(paired=paired)
        outputs = [DiffusionDet.prepare_diffusion_concat(model, gt) for _ in range(8)]
        results.append((outputs, torch.get_rng_state()))
    for a, b in zip(results[0][0], results[1][0]):
        for x, y in zip(a, b):
            assert torch.equal(x, y)
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_endpoints_empirical_distribution_and_shared_random_draws(probability):
    sampler = PairedTimeSampler()
    all_rows = []
    for distribution in ("uniform", "mix_tmax"):
        sampler.reset(89)
        torch.manual_seed(41)
        for _ in range(12000):
            t = sampler(1000, "cpu", distribution, probability)
            assert t.dtype == torch.long and t.shape == (1,)
        all_rows.append((sampler.base_timesteps[:], sampler.gates[:], sampler.timesteps[:]))
    assert all_rows[0][:2] == all_rows[1][:2]
    assert all_rows[0][0] == all_rows[0][2]
    for base, gate, actual in zip(*all_rows[1]):
        assert actual == (999 if gate < probability else base)
    rate = sum(t == 999 for t in all_rows[1][2]) / 12000
    assert abs(rate - (probability + (1-probability)/1000)) < 0.02
    assert abs(sum(all_rows[0][2]) / 12000 - 499.5) < 12


@pytest.mark.parametrize("distribution,probability", [("invalid", 0.5), ("uniform", -0.1),
                                                       ("mix_tmax", 1.1), ("mix_tmax", float("nan"))])
def test_invalid_sampling_settings_fail(distribution, probability):
    with pytest.raises(ValueError):
        PairedTimeSampler()(1000, "cpu", distribution, probability)
