"""Common initial noise for paired constant-query-budget evaluation."""
import torch


def proposal_noise(batch_size, proposals, pool_size, device):
    """Draw the full pool each time: equal seeds alone do not pair varying shapes."""
    if batch_size < 1 or not 1 <= proposals <= pool_size:
        raise ValueError("Require positive batch size and 1 <= proposals <= pool_size")
    pool = torch.randn((batch_size, pool_size, 4), device=device)
    return pool[:, :proposals].contiguous()
