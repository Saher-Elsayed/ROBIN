"""Smoke tests for the DR-PPO agent."""

import torch

from robin.agent import DRPPOAgent, DRPPOConfig


def test_agent_construction():
    cfg = DRPPOConfig(latent_dim=64, node_in_dim=16, tabular_in_dim=32)
    a = DRPPOAgent(cfg=cfg, device="cpu")
    assert a.policy.n_actions == cfg.n_actions
    assert a.value is not None


def test_save_and_load(tmp_path):
    cfg = DRPPOConfig(latent_dim=64)
    a = DRPPOAgent(cfg=cfg, device="cpu")
    ckpt = tmp_path / "p.pt"
    a.save(str(ckpt))
    b = DRPPOAgent(cfg=cfg, device="cpu")
    b.load(str(ckpt))
    for p1, p2 in zip(a.policy.parameters(), b.policy.parameters()):
        assert torch.allclose(p1, p2)
