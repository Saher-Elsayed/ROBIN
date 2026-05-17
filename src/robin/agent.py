"""DR-PPO agent: PPO with CVaR-shaped advantage and Huber slack regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from robin.cvar import cvar_advantage
from robin.policy import ROBINPolicy
from robin.value import ValueHead


@dataclass
class DRPPOConfig:
    n_actions: int = 192
    latent_dim: int = 128
    node_in_dim: int = 16
    tabular_in_dim: int = 32
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    clip_eps: float = 0.2
    beta_cvar: float = 0.2
    c_value: float = 0.5
    c_entropy: float = 0.01
    c_slack: float = 0.5
    n_ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    minibatch_size: int = 64
    grad_clip: float = 0.5


class DRPPOAgent:
    """DR-PPO trainer with CVaR-shaped advantage.

    Implements the clipped surrogate objective of Eq. (7) in the paper:

        L^CLIP(theta) = E_t [ min(rho_t * A_hat_t,
                                  clip(rho_t, 1-eps, 1+eps) * A_hat_t) ]
        rho_t = pi_theta(a_t|z_t) / pi_theta_old(a_t|z_t)

    Combined loss:
        L = -L^CLIP + c_v * L^V - c_e * H[pi_theta] + c_s * L^slack
    """

    def __init__(self, cfg: DRPPOConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or DRPPOConfig()
        self.device = torch.device(device)
        self.policy = ROBINPolicy(
            n_actions=self.cfg.n_actions,
            node_in_dim=self.cfg.node_in_dim,
            tabular_in_dim=self.cfg.tabular_in_dim,
            latent_dim=self.cfg.latent_dim,
        ).to(self.device)
        self.value = ValueHead(latent_dim=self.cfg.latent_dim).to(self.device)
        self.optim_policy = Adam(self.policy.parameters(), lr=self.cfg.lr_policy)
        self.optim_value = Adam(self.value.parameters(), lr=self.cfg.lr_value)

    def update(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns_per_step: torch.Tensor,  # (T, K_theta) for CVaR
        wns_targets: torch.Tensor,        # (T,) for slack regression
        batch: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One PPO update across n_ppo_epochs minibatches."""
        cfg = self.cfg

        # CVaR-shaped advantage
        with torch.no_grad():
            _, _, z = self.policy(node_features, edge_index, tabular, batch=batch)
            v_old = self.value(z)
            advantages = cvar_advantage(returns_per_step, v_old, beta=cfg.beta_cvar)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns_for_value = returns_per_step.mean(dim=-1)  # standard return target

        # PPO epochs
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "slack_loss": 0.0}
        for _ in range(cfg.n_ppo_epochs):
            new_log_probs, entropy, wns_hat = self.policy.evaluate_actions(
                node_features, edge_index, tabular, actions, batch=batch
            )
            _, _, z_new = self.policy(node_features, edge_index, tabular, batch=batch)
            values = self.value(z_new)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns_for_value)
            slack_loss = F.huber_loss(wns_hat, wns_targets, delta=0.05)
            ent = entropy.mean()

            loss = (
                policy_loss
                + cfg.c_value * value_loss
                - cfg.c_entropy * ent
                + cfg.c_slack * slack_loss
            )

            self.optim_policy.zero_grad()
            self.optim_value.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            nn.utils.clip_grad_norm_(self.value.parameters(), cfg.grad_clip)
            self.optim_policy.step()
            self.optim_value.step()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(ent.item())
            stats["slack_loss"] += float(slack_loss.item())

        for k in stats:
            stats[k] /= cfg.n_ppo_epochs
        return stats

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])
