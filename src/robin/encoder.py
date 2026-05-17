"""Graph-attention encoder for the post-synthesis timing graph + tabular fusion.

Architecture (matches paper §V):
    Timing graph G_t = (V, E) ---> GAT layer 1 (H=4, d=32)
                              ---> GAT layer 2 (H=4, d=64) ---> h_G_t in R^64
    Tabular x_t in R^32 (utilization, congestion, tool-version one-hot,
                         device-family one-hot, last action embed)
    [h_G_t || x_t] ---> 3-layer MLP (GELU) ---> z_t in R^128
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Graph-attention layer (Velickovic et al. 2018) with edge-attention."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        if self.head_dim * n_heads != out_dim:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by n_heads ({n_heads})")
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim), edge_index: (2, E)  ->  (N, out_dim)."""
        N = x.size(0)
        h = self.W(x).view(N, self.n_heads, self.head_dim)  # (N, H, d_h)
        src, dst = edge_index[0], edge_index[1]
        alpha_src = (h * self.a_src.unsqueeze(0)).sum(-1)  # (N, H)
        alpha_dst = (h * self.a_dst.unsqueeze(0)).sum(-1)
        e = F.leaky_relu(alpha_src[src] + alpha_dst[dst], negative_slope=0.2)

        # softmax over incoming edges for each destination node
        e = e - e.max()
        e = e.exp()
        denom = torch.zeros(N, self.n_heads, device=x.device).index_add_(0, dst, e)
        alpha = e / (denom[dst] + 1e-9)
        alpha = self.dropout(alpha)

        # aggregate
        out = torch.zeros(N, self.n_heads, self.head_dim, device=x.device)
        msg = h[src] * alpha.unsqueeze(-1)
        out.index_add_(0, dst, msg)
        return out.view(N, self.out_dim)


class TimingGraphEncoder(nn.Module):
    """Two-layer GAT encoder + MLP fusion -> latent z_t in R^128."""

    def __init__(
        self,
        node_in_dim: int = 16,
        tabular_in_dim: int = 32,
        latent_dim: int = 128,
        gat1_out: int = 32,
        gat2_out: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gat1 = GATLayer(node_in_dim, gat1_out, n_heads=n_heads, dropout=dropout)
        self.gat2 = GATLayer(gat1_out, gat2_out, n_heads=n_heads, dropout=dropout)
        self.graph_pool_dim = gat2_out  # mean-pool over nodes

        self.fusion = nn.Sequential(
            nn.Linear(self.graph_pool_dim + tabular_in_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        tabular: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one graph (or batch) to a latent vector.

        Parameters
        ----------
        node_features : (N, node_in_dim)
        edge_index    : (2, E)
        tabular       : (tabular_in_dim,) or (B, tabular_in_dim)
        batch         : optional (N,) tensor mapping nodes to graphs

        Returns
        -------
        z : (latent_dim,) or (B, latent_dim)
        """
        h = F.gelu(self.gat1(node_features, edge_index))
        h = self.gat2(h, edge_index)

        if batch is None:
            graph_emb = h.mean(dim=0, keepdim=True)  # (1, gat2_out)
            tabular_b = tabular.unsqueeze(0) if tabular.dim() == 1 else tabular
        else:
            B = int(batch.max().item()) + 1
            graph_emb = torch.zeros(B, h.size(-1), device=h.device)
            counts = torch.zeros(B, 1, device=h.device)
            graph_emb.index_add_(0, batch, h)
            counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float).unsqueeze(-1))
            graph_emb = graph_emb / counts.clamp_min(1)
            tabular_b = tabular

        z = self.fusion(torch.cat([graph_emb, tabular_b], dim=-1))
        return z.squeeze(0) if batch is None and z.size(0) == 1 else z
