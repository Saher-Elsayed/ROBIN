"""Unit tests for empirical CVaR and CVaR-shaped advantage."""

import torch

from robin.cvar import cvar_advantage, empirical_cvar, value_at_risk


def test_cvar_lower_tail_under_uniform():
    """CVaR at beta=0.2 over a uniform [0,1] sample is approximately 0.1."""
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 1000)
    cvar = empirical_cvar(x, beta=0.2)
    assert 0.05 <= cvar.item() <= 0.15


def test_cvar_below_var():
    """CVaR_beta should never exceed VaR_beta."""
    torch.manual_seed(0)
    x = torch.randn(500)
    for beta in (0.05, 0.1, 0.2, 0.5):
        assert empirical_cvar(x, beta=beta).item() <= value_at_risk(x, beta=beta).item() + 1e-5


def test_cvar_monotone_in_beta():
    """Larger beta -> larger (less risk-averse) CVaR."""
    torch.manual_seed(1)
    x = torch.randn(2000)
    last = empirical_cvar(x, beta=0.05).item()
    for beta in (0.1, 0.2, 0.5):
        cur = empirical_cvar(x, beta=beta).item()
        assert cur >= last - 1e-6
        last = cur


def test_cvar_advantage_shape():
    """cvar_advantage returns (T,) when given (T, K_theta)."""
    rets = torch.randn(16, 20)
    vals = torch.randn(16)
    adv = cvar_advantage(rets, vals, beta=0.2)
    assert adv.shape == (16,)


def test_cvar_batch():
    """Batched CVaR matches per-row CVaR."""
    torch.manual_seed(2)
    x = torch.randn(8, 100)
    batch = empirical_cvar(x, beta=0.2)
    per_row = torch.stack([empirical_cvar(x[i], beta=0.2) for i in range(8)])
    assert torch.allclose(batch, per_row, atol=1e-6)
