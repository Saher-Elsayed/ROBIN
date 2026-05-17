"""Value head V_phi(z) — scalar return estimator for the critic side of PPO."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Map latent state z_t -> scalar value estimate V_phi(z_t)."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = self.net(z).squeeze(-1)
        return v
