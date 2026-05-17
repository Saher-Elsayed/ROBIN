"""Empirical CVaR (Conditional Value-at-Risk) and CVaR-shaped advantage.

Reference: Rockafellar & Uryasev, "Optimization of conditional value-at-risk",
J. Risk 2(3), 2000.
"""

from __future__ import annotations

import torch
from torch import Tensor


def empirical_cvar(returns: Tensor, beta: float = 0.2) -> Tensor:
    """Empirical CVaR at confidence level beta over a batch of returns.

    Given a sample {R_1, ..., R_n}, returns:

        CVaR_beta(R) = E[R | R <= VaR_beta(R)]

    Parameters
    ----------
    returns : Tensor of shape (n,) or (batch, n)
        Sampled returns over K seeds and |Theta| corners.
    beta : float in (0, 1)
        Tail probability. The lower beta, the more risk-averse.

    Returns
    -------
    Tensor of shape () or (batch,) — the lower-tail mean.
    """
    if returns.dim() == 1:
        returns = returns.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    n = returns.size(-1)
    k = max(1, int(beta * n))
    # sort ascending; take the smallest k
    sorted_returns, _ = torch.sort(returns, dim=-1)
    cvar = sorted_returns[..., :k].mean(dim=-1)

    return cvar.squeeze(0) if squeeze else cvar


def cvar_advantage(
    returns_per_step: Tensor, value_estimates: Tensor, beta: float = 0.2
) -> Tensor:
    """CVaR-shaped advantage estimator for DR-PPO.

    Replaces the mean return in the standard advantage with the empirical
    CVaR at level beta, which targets the lower tail of the return
    distribution induced by (seed, corner) ambiguity.

        A_hat_t = CVaR_beta({R_{k,theta}}) - V_phi(z_t)

    Parameters
    ----------
    returns_per_step : Tensor of shape (T, K_theta)
        Per-step returns sampled across seed/corner pairs.
    value_estimates : Tensor of shape (T,)
        Value-head estimates V_phi(z_t).
    beta : float
        CVaR confidence level.

    Returns
    -------
    Tensor of shape (T,) — advantages.
    """
    cvar = empirical_cvar(returns_per_step, beta=beta)  # (T,)
    advantages = cvar - value_estimates
    return advantages


def value_at_risk(returns: Tensor, beta: float = 0.2) -> Tensor:
    """Empirical Value-at-Risk: the beta-quantile of the return distribution."""
    sorted_returns, _ = torch.sort(returns, dim=-1)
    n = sorted_returns.size(-1)
    idx = max(0, int(beta * n) - 1)
    return sorted_returns[..., idx]
