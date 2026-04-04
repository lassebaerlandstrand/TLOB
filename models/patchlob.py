"""
PatchLOB: Patched Transformer with LOB-Structured Embedding.

Exploits the natural spatial structure of limit order books (depth levels,
bid-ask symmetry) and compresses temporal sequences via patching to reduce
redundancy while capturing multi-scale dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.bin import BiN
from models.difflob import DiffTransformerLayer, precompute_rope
from models.tlob import _build_head

N_LOB_RAW = 40
N_LEVELS = 10
FEATS_PER_LEVEL = 4  # sell_p, sell_v, buy_p, buy_v


# ---------------------------------------------------------------------------
# LOB-Structured Embedding
# ---------------------------------------------------------------------------

class LOBStructuredEmbedding(nn.Module):
    """Embed LOB features respecting depth-level structure and bid-ask symmetry.

    Reshapes flat LOB features into (10 levels, feats_per_level), splits into
    ask/bid sides, processes with a shared encoder, and computes explicit
    cross-side interactions (spread/imbalance signal).
    """

    def __init__(self, num_features, d_side=16):
        super().__init__()
        self.num_features = num_features
        self.has_diffs = num_features >= 2 * N_LOB_RAW
        self.n_extra = (
            num_features - 2 * N_LOB_RAW if self.has_diffs
            else num_features - N_LOB_RAW
        )
        feats_per_side = FEATS_PER_LEVEL // 2  # 2 raw (price, vol)
        if self.has_diffs:
            feats_per_side *= 2  # + 2 diffs

        self.d_side = d_side
        self.d_level = d_side * 3  # ask + bid + (ask - bid)

        # Build index tensors for gathering LOB features per level
        level_idx = []
        for i in range(N_LEVELS):
            base_raw = FEATS_PER_LEVEL * i
            raw = [base_raw, base_raw + 1, base_raw + 2, base_raw + 3]
            if self.has_diffs:
                base_diff = num_features - N_LOB_RAW + FEATS_PER_LEVEL * i
                diff = [base_diff, base_diff + 1, base_diff + 2, base_diff + 3]
                level_idx.append(raw + diff)
            else:
                level_idx.append(raw)
        self.register_buffer("level_idx", torch.tensor(level_idx))  # (10, 8) or (10, 4)

        # Indices within each level's features for ask/bid split
        if self.has_diffs:
            # [sell_p, sell_v, buy_p, buy_v, Δsell_p, Δsell_v, Δbuy_p, Δbuy_v]
            self.ask_idx = [0, 1, 4, 5]
            self.bid_idx = [2, 3, 6, 7]
        else:
            # [sell_p, sell_v, buy_p, buy_v]
            self.ask_idx = [0, 1]
            self.bid_idx = [2, 3]

        # Shared encoder for each side (ask/bid processed identically)
        self.side_encoder = nn.Linear(feats_per_side, d_side)

        # Cross-level convolution (captures patterns across depth levels)
        self.cross_level_conv = nn.Conv1d(
            self.d_level, self.d_level, kernel_size=3, padding=1
        )

        # Extra features (Battery order messages)
        if self.n_extra > 0:
            self.extra_proj = nn.Linear(self.n_extra, d_side)
            self.out_proj = nn.Linear(self.d_level + d_side, self.d_level)

    def forward(self, x):
        B, T, _ = x.shape

        # Gather LOB features per level: (B, T, 10, 8) or (B, T, 10, 4)
        lob = x[:, :, self.level_idx]

        # Split ask/bid sides
        ask = lob[:, :, :, self.ask_idx]
        bid = lob[:, :, :, self.bid_idx]

        # Shared side encoder
        ask_emb = self.side_encoder(ask)  # (B, T, 10, d_side)
        bid_emb = self.side_encoder(bid)

        # Level representation: [ask, bid, ask-bid]
        # The difference explicitly captures spread/imbalance
        level = torch.cat([ask_emb, bid_emb, ask_emb - bid_emb], dim=-1)

        # Cross-level convolution
        level = level.reshape(B * T, N_LEVELS, self.d_level)
        level = level.permute(0, 2, 1)  # (B*T, d_level, 10)
        level = F.gelu(self.cross_level_conv(level))
        level = level.mean(dim=-1)  # (B*T, d_level) — pool across levels
        level = level.view(B, T, self.d_level)

        # Handle extra features (Battery order messages)
        if self.n_extra > 0:
            extra_start = N_LOB_RAW
            extra_end = N_LOB_RAW + self.n_extra
            extra = x[:, :, extra_start:extra_end]
            extra = self.extra_proj(extra)
            level = self.out_proj(torch.cat([level, extra], dim=-1))

        return level


# ---------------------------------------------------------------------------
# Temporal Patcher
# ---------------------------------------------------------------------------

class TemporalPatcher(nn.Module):
    """Compress temporal sequence via strided convolution."""

    def __init__(self, d_in, d_out, patch_size=4):
        super().__init__()
        self.patch_conv = nn.Conv1d(d_in, d_out, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, d_in, T)
        x = self.patch_conv(x)         # (B, d_out, T//P)
        return x.transpose(1, 2)       # (B, T//P, d_out)


# ---------------------------------------------------------------------------
# PatchLOB Model
# ---------------------------------------------------------------------------

class PatchLOB(nn.Module):
    """Dual-path differential attention transformer with LOB-structured
    embedding and temporal patching.

    Same interface as TLOB/DiffLOB — drop-in replacement.
    """

    PATCH_SIZE = 4
    D_SIDE = 16

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
        self.dataset_type = dataset_type
        self.num_horizons = num_horizons

        num_patches = seq_size // self.PATCH_SIZE
        assert seq_size % self.PATCH_SIZE == 0
        assert num_patches % 4 == 0

        # Input processing
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)

        # LOB-Structured Embedding
        self.lob_embedding = LOBStructuredEmbedding(num_features, d_side=self.D_SIDE)
        d_spatial = self.lob_embedding.d_level  # 48

        # Temporal Patching
        self.patcher = TemporalPatcher(d_spatial, hidden_dim, patch_size=self.PATCH_SIZE)

        # RoPE for temporal attention
        sub_head_dim = (hidden_dim // num_heads) // 2
        rope_cos, rope_sin = precompute_rope(sub_head_dim, max_len=num_patches)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Dual-path DiffTransformer blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            t_final = hidden_dim // 4 if is_last else hidden_dim
            s_final = num_patches // 4 if is_last else num_patches

            self.layers.append(
                DiffTransformerLayer(
                    hidden_dim, num_heads, t_final,
                    layer_idx=i, use_rope=True, dropout=dropout,
                )
            )
            self.layers.append(
                DiffTransformerLayer(
                    num_patches, num_heads, s_final,
                    layer_idx=i, use_rope=False, dropout=dropout,
                )
            )

        # Classification heads
        total_dim = (hidden_dim // 4) * (num_patches // 4)
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
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat(
                [input[:, :, :41], input[:, :, 42:]], dim=2
            )
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input

        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")

        x = self.lob_embedding(x)
        x = self.patcher(x)

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
