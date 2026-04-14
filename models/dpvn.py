"""DP-Distilled Value Network (DPVN).

Architecture:
    LOB(40) -> structured ask/bid embedding -> compact transformer -> value head
    forward(x) returns (B, 3) raw V-values for actions {-1, 0, +1}.

Trained with Huber loss against truncated-horizon Q targets bootstrapped from
the DP-optimal trajectory. Inference uses a deterministic spread-aware argmax
applied outside the model (see evaluate_trading.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


_N_LEVELS = 10
_LOB_FEATURES = 40


class LOBLevelEmbedding(nn.Module):
    """Structured LOB embedding preserving bid-ask symmetry.

    Input: (B, W, 40) raw LOB with per-level layout
        [ask_price, ask_vol, bid_price, bid_vol] x 10 levels.

    Output: (B, W, hidden_dim) structured token embedding.
    """

    def __init__(self, hidden_dim: int, n_levels: int = _N_LEVELS):
        super().__init__()
        self.n_levels = n_levels
        side_dim = max(hidden_dim // 4, 4)
        self.side_dim = side_dim
        self.side_enc = nn.Sequential(
            nn.Linear(2, side_dim),
            nn.GELU(),
            nn.Linear(side_dim, side_dim),
        )
        flat_dim = 2 * n_levels * side_dim + n_levels * 2
        self.proj = nn.Linear(flat_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, _ = x.shape
        x = x.reshape(b, w, self.n_levels, 4)
        ask = x[..., 0:2]
        bid = x[..., 2:4]
        ask_enc = self.side_enc(ask)
        bid_enc = self.side_enc(bid)
        diff = ask - bid
        z = torch.cat(
            [
                ask_enc.flatten(-2),
                bid_enc.flatten(-2),
                diff.flatten(-2),
            ],
            dim=-1,
        )
        return self.proj(z)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block: RMSNorm -> SDPA -> MLP."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_dropout = dropout
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, d = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(b, w, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, w, d)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DPVN(nn.Module):
    """DP-Distilled Value Network.

    forward(x) -> (B, 3) raw values V(s, a) for a in {-1, 0, +1}.
    The spread-aware decision rule is applied outside the model at inference.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 4,
        seq_size: int = 128,
        num_features: int = _LOB_FEATURES,
        num_heads: int = 4,
        dataset_type: str | None = None,
        use_fast_attention: bool = True,
        num_horizons: int = 1,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert num_horizons == 1, "DPVN v1 is single-horizon only"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features_in = num_features
        self.embed = LOBLevelEmbedding(hidden_dim)
        self.register_buffer("pos_emb", self._sinusoidal_pe(seq_size, hidden_dim), persistent=False)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 3)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, hidden_dim: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, hidden_dim)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Take only the first 40 LOB columns; ignore appended diff/message features.
        x_lob = x[..., :_LOB_FEATURES]
        h = self.embed(x_lob)
        h = h + self.pos_emb.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        last = self.norm(h[:, -1])
        return self.value_head(last)
