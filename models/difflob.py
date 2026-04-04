"""
DiffLOB: Differential Attention for Limit Order Book Prediction.

Replaces standard self-attention in the dual-path LOB transformer with
differential attention, which computes (A1 - λ·A2)·V to cancel common-mode
noise from uninformative timesteps and features.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.bin import BiN
from models.mlplob import MLP
from models.tlob import _build_head


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope(dim, max_len=256, base=10000.0):
    """Precompute cos/sin tables for rotary position embeddings."""
    half = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half).float() * 2 / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)  # (max_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to q and k tensors.

    Uses half-split (not interleaved) to handle both even and odd dimensions.
    If dim is odd, the last element is left unrotated.

    Args:
        q, k: (B, H, T, d)
        cos, sin: (T, d//2) precomputed tables
    """
    T = q.shape[2]
    half = cos.shape[-1]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    def _rotate(x):
        x1 = x[..., :half]
        x2 = x[..., half:2 * half]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        if x.shape[-1] > 2 * half:
            # Odd dimension: pass through the last element unrotated
            rotated = torch.cat([rotated, x[..., 2 * half:]], dim=-1)
        return rotated

    return _rotate(q), _rotate(k)


# ---------------------------------------------------------------------------
# Differential Attention
# ---------------------------------------------------------------------------

class DiffAttention(nn.Module):
    """Differential attention: (A1 - λ·A2) V.

    Splits Q,K into two sub-groups, computes two softmax attention maps,
    takes their difference to cancel common-mode noise.
    """

    def __init__(self, hidden_dim, num_heads, layer_idx, use_rope=False, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.attn_dropout_p = dropout

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        sub_head_dim = head_dim // 2
        self.sub_head_dim = sub_head_dim

        # Q, K, V projections (same total params as standard attention)
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)

        # λ parameterization (per head)
        # λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
        lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(sub_head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(sub_head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(sub_head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(sub_head_dim) * 0.1)

        # QK-Norm: learned temperature per head (initialized to √sub_head_dim)
        self.temperature = nn.Parameter(
            torch.ones(num_heads) * math.sqrt(sub_head_dim)
        )

        # GroupNorm on attention output (stabilizes small diff values)
        self.group_norm = nn.GroupNorm(num_heads, hidden_dim)

    def forward(self, x, rope_cos=None, rope_sin=None):
        B, T, _ = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.wq(x).view(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.wk(x).view(B, T, H, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, hd).transpose(1, 2)

        # Split Q, K into two sub-heads
        q1, q2 = q.chunk(2, dim=-1)  # each (B, H, T, hd//2)
        k1, k2 = k.chunk(2, dim=-1)

        # Apply RoPE (temporal attention only)
        if self.use_rope and rope_cos is not None:
            q1, k1 = apply_rope(q1, k1, rope_cos, rope_sin)
            q2, k2 = apply_rope(q2, k2, rope_cos, rope_sin)

        # QK-Norm: L2 normalize + learned temperature
        q1 = F.normalize(q1, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        q2 = F.normalize(q2, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        # Scale by learned temperature (replaces 1/√d)
        temp = self.temperature.view(1, H, 1, 1)

        # Two attention maps via FlashAttention
        drop_p = self.attn_dropout_p if self.training else 0.0
        attn1_out = F.scaled_dot_product_attention(
            q1 * temp, k1, v, dropout_p=drop_p, is_causal=False,
        )
        attn2_out = F.scaled_dot_product_attention(
            q2 * temp, k2, v, dropout_p=drop_p, is_causal=False,
        )

        # Compute λ (scalar)
        lambda_ = (
            torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
            - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
            + self.lambda_init
        )

        # Differential output: (A1 - λ·A2) V
        out = attn1_out - lambda_ * attn2_out  # (B, H, T, hd)

        # GroupNorm (channel-wise normalization)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, D)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)

        return self.wo(out)


# ---------------------------------------------------------------------------
# DiffTransformerLayer
# ---------------------------------------------------------------------------

class DiffTransformerLayer(nn.Module):
    """Single transformer layer with differential attention + MLP."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        final_dim,
        layer_idx=0,
        use_rope=False,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_residual = final_dim == hidden_dim
        self.resid_dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(hidden_dim)
        self.attn = DiffAttention(
            hidden_dim, num_heads, layer_idx,
            use_rope=use_rope, dropout=dropout,
        )
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim, dropout=dropout)

    def forward(self, x, rope_cos=None, rope_sin=None):
        res = x
        x = self.resid_dropout(self.attn(x, rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if self.use_residual:
            x = x + res
        return x


# ---------------------------------------------------------------------------
# DiffLOB Model
# ---------------------------------------------------------------------------

class DiffLOB(nn.Module):
    """Dual-path differential attention transformer for LOB prediction.

    Same interface as TLOB — drop-in replacement.
    """

    def __init__(
        self,
        hidden_dim,
        num_layers,
        seq_size,
        num_features,
        num_heads,
        is_sin_emb,
        dataset_type,
        use_fast_attention=True,
        num_horizons=1,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.num_horizons = num_horizons

        # Input processing
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)

        # RoPE buffers for temporal attention
        # sub_head_dim = (hidden_dim // num_heads) // 2
        sub_head_dim_temporal = (hidden_dim // num_heads) // 2
        rope_cos, rope_sin = precompute_rope(sub_head_dim_temporal, max_len=seq_size)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Build alternating temporal/spatial layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            t_final = hidden_dim // 4 if is_last else hidden_dim
            s_final = seq_size // 4 if is_last else seq_size

            # Temporal attention (with RoPE)
            self.layers.append(
                DiffTransformerLayer(
                    hidden_dim, num_heads, t_final,
                    layer_idx=i, use_rope=True, dropout=dropout,
                )
            )
            # Spatial attention (no RoPE)
            self.layers.append(
                DiffTransformerLayer(
                    seq_size, num_heads, s_final,
                    layer_idx=i, use_rope=False, dropout=dropout,
                )
            )

        # Classification heads
        total_dim = (hidden_dim // 4) * (seq_size // 4)
        if num_horizons == 1:
            self.final_layers = _build_head(total_dim, dropout=dropout)
            self.heads = None
        else:
            self.final_layers = None
            self.heads = nn.ModuleList(
                [nn.ModuleList(_build_head(total_dim, dropout=dropout))
                 for _ in range(num_horizons)]
            )

    def _encode(self, input):
        """Shared encoder: input → flat representation."""
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat(
                [input[:, :, :41], input[:, :, 42:]], dim=2
            )
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input

        # BiN normalization
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")

        # Embedding
        x = self.emb_layer(x)

        # Alternating temporal/spatial layers with permute
        for i, layer in enumerate(self.layers):
            is_temporal = i % 2 == 0
            if is_temporal:
                x = layer(x, rope_cos=self.rope_cos, rope_sin=self.rope_sin)
            else:
                x = layer(x)
            x = x.permute(0, 2, 1)

        x = rearrange(x, "b s f -> b (f s) 1")
        x = x.reshape(x.shape[0], -1)
        return x

    def forward(self, input, store_att=False):
        x = self._encode(input)

        if self.num_horizons == 1:
            for layer in self.final_layers:
                x = layer(x)
            return x
        else:
            outputs = []
            for head in self.heads:
                h = x
                for layer in head:
                    h = layer(h)
                outputs.append(h)
            return outputs
