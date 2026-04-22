"""DAVN - Dual-Axis Value Network.

A successor to DPVN that keeps the DP-distilled Q-target training paradigm but
adds three architectural improvements:

1. Fused input (shared with DPVN-F via ``FusedInputProjection``):
   LOB embedding + engineered aux features + in-model spread features.
2. Dual-axis transformer trunk: alternating temporal and feature-axis
   self-attention (a la TLOB) instead of DPVN's temporal-only stack.
3. Attention-pool readout: a learned query aggregates all W timesteps
   instead of ``h[:, -1, :]``.

Output: (B, 3) raw V-values for actions {-1, 0, +1}. Training loss and
inference decision rule are unchanged from DPVN.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.dpvn import (
    FusedInputProjection,
    SelfAttentionBlock,
    _ENGINEERED_FEATURES,
    _LOB_FEATURES,
    _SPREAD_FEATURES,
)


class FeatureAxisBlock(nn.Module):
    """Self-attention over the feature (hidden) axis.

    Input shape : (B, W, D)
    Strategy    : transpose to (B, D, W), treat each hidden dim as a token with W
                  features, apply standard SDPA, transpose back.
    The head-dim is W / num_heads — with W=128 and num_heads=4 that is 32, a
    normal value. Compute is O(B * D^2 * W) which is cheaper than the temporal
    block's O(B * W^2 * D) at our D=64, W=128 sizes.
    """

    def __init__(self, hidden_dim: int, seq_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert seq_size % num_heads == 0, "seq_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = seq_size // num_heads
        self.attn_dropout = dropout
        self.norm1 = nn.RMSNorm(seq_size)
        self.qkv = nn.Linear(seq_size, seq_size * 3)
        self.proj = nn.Linear(seq_size, seq_size)
        self.norm2 = nn.RMSNorm(seq_size)
        self.mlp = nn.Sequential(
            nn.Linear(seq_size, seq_size * 4),
            nn.GELU(),
            nn.Linear(seq_size * 4, seq_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, W, D) -> (B, D, W) so attention runs over the D hidden dims as tokens.
        y = x.transpose(1, 2)
        b, d, w = y.shape
        h = self.norm1(y)
        qkv = self.qkv(h).reshape(b, d, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, d, w)
        y = y + self.proj(attn_out)
        y = y + self.mlp(self.norm2(y))
        return y.transpose(1, 2)  # back to (B, W, D)


class DualBlock(nn.Module):
    """One temporal attention block followed by one feature-axis block."""

    def __init__(self, hidden_dim: int, seq_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.temporal = SelfAttentionBlock(hidden_dim, num_heads, dropout=dropout)
        self.feature = FeatureAxisBlock(hidden_dim, seq_size, num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.feature(x)
        return x


class AttentionPool(nn.Module):
    """Single-query attention pooling over the time axis.

    Initialised so the first forward pass resembles DPVN's last-token readout,
    giving a smooth starting point when comparing to DPVN-F.
    """

    def __init__(self, hidden_dim: int, seq_size: int, bias_last: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Learnable query. Zero-init so the pre-bias softmax is uniform; the
        # learnable position bias below supplies the initial last-token prior.
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        # Position bias over W tokens; initialised to strongly favour t = W-1 so
        # the starting behaviour matches DPVN's h[:, -1, :] readout.
        bias = torch.zeros(seq_size)
        if bias_last:
            bias[-1] = 4.0
        self.pos_bias = nn.Parameter(bias)
        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, W, D)
        scores = torch.einsum("bwd,d->bw", h, self.query) * self.scale  # (B, W)
        scores = scores + self.pos_bias
        weights = torch.softmax(scores, dim=-1)  # (B, W)
        pooled = torch.einsum("bw,bwd->bd", weights, h)
        return pooled


class DAVN(nn.Module):
    """Dual-Axis Value Network.

    forward(x) -> (B, 3) raw V-values for actions {-1, 0, +1}.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        seq_size: int = 128,
        num_features: int = _LOB_FEATURES,
        num_heads: int = 4,
        dataset_type: str | None = None,
        use_fast_attention: bool = True,
        num_horizons: int = 1,
        dropout: float = 0.0,
        all_features: bool = True,
        davn_dual_axis: bool = True,
        davn_attn_pool: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert num_horizons == 1, "DAVN v1 is single-horizon only"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features_in = num_features
        self.all_features = all_features
        self.davn_dual_axis = davn_dual_axis
        self.davn_attn_pool = davn_attn_pool

        # Input: always fused. all_features=False is an ablation that drops aux
        # engineered channels but keeps in-model spread features.
        self.embed = FusedInputProjection(hidden_dim, num_features)
        self.register_buffer(
            "pos_emb", self._sinusoidal_pe(seq_size, hidden_dim), persistent=False
        )

        if davn_dual_axis:
            self.blocks = nn.ModuleList(
                [
                    DualBlock(hidden_dim, seq_size, num_heads, dropout=dropout)
                    for _ in range(num_layers)
                ]
            )
        else:
            # Ablation: temporal-only trunk, matched attention op count (2*num_layers).
            self.blocks = nn.ModuleList(
                [
                    SelfAttentionBlock(hidden_dim, num_heads, dropout=dropout)
                    for _ in range(num_layers * 2)
                ]
            )

        self.norm = nn.RMSNorm(hidden_dim)
        if davn_attn_pool:
            self.pool = AttentionPool(hidden_dim, seq_size)
        else:
            self.pool = None  # falls back to last-token readout

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
        if self.all_features:
            h = self.embed(x[..., : self.num_features_in])
        else:
            # Ablation path: zero out the aux_extra channels but keep in-model spread.
            h = self.embed(x[..., :_LOB_FEATURES])
        h = h + self.pos_emb.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        if self.pool is not None:
            pooled = self.pool(h)
        else:
            pooled = h[:, -1]
        return self.value_head(self.norm(pooled))

    def input_breakdown(self) -> dict:
        """Per-timestep channel accounting. DAVN always uses fused input; the
        ``all_features=False`` ablation path still feeds the LOB slice through
        ``FusedInputProjection``, so spread and engineered channels are always
        present.
        """
        aux_extra = max(self.num_features_in - _LOB_FEATURES, 0) if self.all_features else 0
        ignored = max(self.num_features_in - _LOB_FEATURES, 0) if not self.all_features else 0
        aux_in = aux_extra + _SPREAD_FEATURES + _ENGINEERED_FEATURES
        return {
            "mode": "davn_fused" if self.all_features else "davn_lob_only",
            "input_tensor_cols": self.num_features_in,
            "lob_cols": _LOB_FEATURES,
            "aux_extra_cols": aux_extra,
            "spread_in_model": _SPREAD_FEATURES,
            "engineered_in_model": _ENGINEERED_FEATURES,
            "aux_proj_in_dim": aux_in,
            "ignored_input_cols": ignored,
            "effective_channels": _LOB_FEATURES + aux_in,
        }
