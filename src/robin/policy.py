"""Policy head (categorical over |A|=192 directive bundles) and slack-regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from robin.encoder import TimingGraphEncoder


class ROBINPolicy(nn.Module):
    """Full ROBIN policy: encoder + categorical policy head + slack head.

    The action space is the Cartesian product of:
      - 4 synthesis strategies
      - 3 phys-opt flags
      - 4 pblock variants
      - 2 retiming modes
      - 2 route-effort settings
    pruned of vendor-flagged incompatible pairs, yielding |A| = 192.
    """

    def __init__(
        self,
        n_actions: int = 192,
        node_in_dim: int = 16,
        tabular_in_dim: int = 32,
        latent_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = TimingGraphEncoder(
            node_in_dim=node_in_dim,
            tabular_in_dim=tabular_in_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_actions),
        )
        self.slack_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
        )
        self.n_actions = n_actions
        self.latent_dim = latent_dim

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode -> (logits, predicted_wns, latent).

        Returns
        -------
        logits : (B, n_actions)
        wns_hat : (B,)
        z : (B, latent_dim)
        """
        z = self.encoder(node_features, edge_index, tabular, batch=batch)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        logits = self.policy_head(z)
        wns_hat = self.slack_head(z).squeeze(-1)
        return logits, wns_hat, z

    def sample(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: torch.Tensor,
        batch: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; return (action, log_prob, entropy, wns_hat)."""
        logits, wns_hat, _ = self.forward(node_features, edge_index, tabular, batch)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, wns_hat

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: torch.Tensor,
        actions: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score stored trajectories during PPO update."""
        logits, wns_hat, _ = self.forward(node_features, edge_index, tabular, batch)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), wns_hat
