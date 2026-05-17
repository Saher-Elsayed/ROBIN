"""Unit tests for the encoder, policy, and value heads."""

import torch

from robin.encoder import GATLayer, TimingGraphEncoder
from robin.policy import ROBINPolicy
from robin.value import ValueHead


def _toy_graph(n_nodes=10, n_edges=20, node_in_dim=16):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, node_in_dim)
    e = torch.randint(0, n_nodes, (2, n_edges))
    return x, e


def test_gat_layer_shape():
    x, e = _toy_graph()
    layer = GATLayer(16, 32, n_heads=4)
    out = layer(x, e)
    assert out.shape == (10, 32)


def test_encoder_produces_latent():
    x, e = _toy_graph()
    tabular = torch.randn(32)
    enc = TimingGraphEncoder(node_in_dim=16, tabular_in_dim=32, latent_dim=128)
    z = enc(x, e, tabular)
    assert z.shape == (128,)


def test_policy_sample():
    x, e = _toy_graph()
    tabular = torch.randn(32)
    pol = ROBINPolicy(n_actions=192)
    a, lp, ent, wns = pol.sample(x, e, tabular)
    assert a.shape == (1,)
    assert 0 <= a.item() < 192
    assert lp.shape == (1,)
    assert ent.shape == (1,)
    assert wns.shape == (1,)


def test_policy_evaluate_actions():
    x, e = _toy_graph()
    tabular = torch.randn(32)
    pol = ROBINPolicy(n_actions=192)
    actions = torch.tensor([42])
    lp, ent, wns = pol.evaluate_actions(x, e, tabular, actions)
    assert lp.shape == (1,)
    assert ent.shape == (1,)
    assert wns.shape == (1,)


def test_value_head_scalar():
    z = torch.randn(4, 128)
    v = ValueHead(latent_dim=128)
    out = v(z)
    assert out.shape == (4,)
