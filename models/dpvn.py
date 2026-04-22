"""DP-Distilled Value Network (DPVN).

Architecture:
    LOB(40) -> structured ask/bid embedding -> compact transformer -> value head
    forward(x) returns (B, 3) raw V-values for actions {-1, 0, +1}.

Trained with Huber loss against truncated-horizon Q targets bootstrapped from
the DP-optimal trajectory. Inference uses a deterministic spread-aware argmax
applied outside the model (see evaluate_trading.py).

When constructed with ``all_features=True`` the model additionally consumes
engineered aux features appended after the 40 LOB columns and three in-model
spread features computed from LOB columns 0, 2. This "DPVN-F" variant is the
fair baseline for DAVN so architecture and input gains are attributable
independently.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


_N_LEVELS = 10
_LOB_FEATURES = 40
_SPREAD_FEATURES = 3  # [z_half_spread, ewma(span=8), log(ewma(span=32) + 1e-6)]
_ENGINEERED_FEATURES = 8  # [ofi_top{1,3,5,10}, {bid,ask}_vwap{3,10}_rel]


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


def compute_spread_features(x_lob: torch.Tensor) -> torch.Tensor:
    """Derive three in-model spread features from the top-of-book ask/bid.

    Input : (B, W, >=4) with LOB cols [ask_price, ask_vol, bid_price, bid_vol, ...].
    Output: (B, W, 3) with channels [z_half_spread, ewma(span=8), log(ewma(span=32) + 1e-6)].

    EWMAs are causal (only use past values) and computed in a single pass. The
    sequence length W is modest (128), so the O(W) loop is not a bottleneck.
    """
    ask = x_lob[..., 0]
    bid = x_lob[..., 2]
    z_hs = torch.abs(ask - bid) * 0.5  # (B, W)

    def _ewma(series: torch.Tensor, span: int) -> torch.Tensor:
        alpha = 2.0 / (span + 1)
        out = torch.empty_like(series)
        running = series[..., 0]
        out[..., 0] = running
        for t in range(1, series.shape[-1]):
            running = alpha * series[..., t] + (1.0 - alpha) * running
            out[..., t] = running
        return out

    ema_short = _ewma(z_hs, span=8)
    ema_long = _ewma(z_hs, span=32)
    log_ema_long = torch.log(ema_long + 1e-6)
    return torch.stack([z_hs, ema_short, log_ema_long], dim=-1)  # (B, W, 3)


def compute_engineered_features(x_lob: torch.Tensor) -> torch.Tensor:
    """Derive LOB-only microstructure features per timestep.

    Chosen to match the Battery ``_msg`` engineered set so BTC gets the same
    signal family in-model (without any preprocessing-time cache). Values are
    computed on whatever LOB scale the model sees (z-scored here), so the
    numerical values differ from Battery's preprocessing-time versions; the
    linear ``aux_proj`` absorbs that difference.

    Features (8 total):
      - ``ofi_top{1,3,5,10}`` — Σ_{l<N}(Δbid_vol_l − Δask_vol_l); positive = net buy pressure.
      - ``{bid,ask}_vwap_{3,10}_rel`` — ``(VWAP_N − mid)``; side-specific depth-weighted skew.

    Input : (B, W, 40) LOB ``[ask_p, ask_v, bid_p, bid_v] × 10 levels``.
    Output: (B, W, 8)
    """
    b, w, _ = x_lob.shape
    lob = x_lob.reshape(b, w, _N_LEVELS, 4)
    ask_p = lob[..., 0]
    ask_v = lob[..., 1]
    bid_p = lob[..., 2]
    bid_v = lob[..., 3]
    mid = 0.5 * (ask_p[..., 0] + bid_p[..., 0])  # (B, W)

    # OFI: first timestep has no prior, leave Δ as zero for that slice.
    d_bid_v = torch.zeros_like(bid_v)
    d_ask_v = torch.zeros_like(ask_v)
    d_bid_v[:, 1:] = bid_v[:, 1:] - bid_v[:, :-1]
    d_ask_v[:, 1:] = ask_v[:, 1:] - ask_v[:, :-1]
    ofi_per_level = d_bid_v - d_ask_v  # (B, W, 10)
    ofi_top1 = ofi_per_level[..., 0]
    ofi_top3 = ofi_per_level[..., :3].sum(-1)
    ofi_top5 = ofi_per_level[..., :5].sum(-1)
    ofi_top10 = ofi_per_level.sum(-1)

    eps = 1e-8
    bid_v3 = bid_v[..., :3].sum(-1)
    ask_v3 = ask_v[..., :3].sum(-1)
    bid_v10 = bid_v.sum(-1)
    ask_v10 = ask_v.sum(-1)
    bid_pv3 = (bid_p[..., :3] * bid_v[..., :3]).sum(-1)
    ask_pv3 = (ask_p[..., :3] * ask_v[..., :3]).sum(-1)
    bid_pv10 = (bid_p * bid_v).sum(-1)
    ask_pv10 = (ask_p * ask_v).sum(-1)
    bid_vwap3_rel = bid_pv3 / (bid_v3 + eps) - mid
    ask_vwap3_rel = ask_pv3 / (ask_v3 + eps) - mid
    bid_vwap10_rel = bid_pv10 / (bid_v10 + eps) - mid
    ask_vwap10_rel = ask_pv10 / (ask_v10 + eps) - mid

    return torch.stack(
        [
            ofi_top1, ofi_top3, ofi_top5, ofi_top10,
            bid_vwap3_rel, ask_vwap3_rel, bid_vwap10_rel, ask_vwap10_rel,
        ],
        dim=-1,
    )


class FusedInputProjection(nn.Module):
    """Additive fusion of LOB embedding, aux features, in-model spread features,
    and in-model engineered features.

    Used by DPVN (when ``all_features=True``) and DAVN. Keeps the LOB path
    byte-identical to DPVN's structured embedding while adding a second
    projection path for the non-LOB channels.

    ``aux_extra`` (cols beyond the first 40) differs by dataset:
      - BTC: 40 first-order LOB diff columns (from ``maybe_add_diff_features``).
      - Battery: 18 engineered + 40 diff = 58 columns.

    On top of that we ALWAYS compute ``compute_spread_features`` (3 ch) and
    ``compute_engineered_features`` (8 ch) in-model. For Battery this means
    some per-timestep stats (e.g. OFI, VWAP_rel) are duplicated with the
    preprocessed ``aux_extra`` — that is a minor capacity cost the linear
    ``aux_proj`` absorbs, in exchange for BTC getting the same signal family
    without a preprocessing change.
    """

    def __init__(self, hidden_dim: int, num_features: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_features_in = num_features
        self.lob_embed = LOBLevelEmbedding(hidden_dim)
        aux_dim = (
            max(num_features - _LOB_FEATURES, 0)
            + _SPREAD_FEATURES
            + _ENGINEERED_FEATURES
        )
        self.aux_dim = aux_dim
        self.aux_proj = nn.Linear(aux_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LOB path: first 40 columns through the structured embedding.
        x_lob = x[..., :_LOB_FEATURES]
        e_lob = self.lob_embed(x_lob)

        # Aux path: optional cached aux columns + always-on in-model features.
        aux_extra = x[..., _LOB_FEATURES:self.num_features_in]
        spread = compute_spread_features(x_lob)
        engineered = compute_engineered_features(x_lob)
        parts = [spread, engineered]
        if aux_extra.shape[-1] > 0:
            parts.insert(0, aux_extra)
        aux_cat = torch.cat(parts, dim=-1)
        e_aux = self.aux_proj(aux_cat)

        return e_lob + e_aux


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
        all_features: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert num_horizons == 1, "DPVN v1 is single-horizon only"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features_in = num_features
        self.all_features = all_features
        if all_features:
            self.embed = FusedInputProjection(hidden_dim, num_features)
        else:
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
        if self.all_features:
            # Fused-input DPVN-F: model sees LOB + aux features + in-model spread features.
            h = self.embed(x[..., :self.num_features_in])
        else:
            # Default DPVN: structurally drop anything beyond the 40 LOB cols.
            h = self.embed(x[..., :_LOB_FEATURES])
        h = h + self.pos_emb.unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        last = self.norm(h[:, -1])
        return self.value_head(last)

    def input_breakdown(self) -> dict:
        """Describe how the input tensor is split between embedding paths.

        Counts the actual per-timestep channels the model consumes, including
        in-model-computed features that never appear in ``num_features``.
        """
        if not self.all_features:
            return {
                "mode": "dpvn_baseline",
                "input_tensor_cols": self.num_features_in,
                "lob_cols": _LOB_FEATURES,
                "aux_extra_cols": 0,
                "spread_in_model": 0,
                "engineered_in_model": 0,
                "aux_proj_in_dim": 0,
                "ignored_input_cols": max(self.num_features_in - _LOB_FEATURES, 0),
                "effective_channels": _LOB_FEATURES,
            }
        aux_extra = max(self.num_features_in - _LOB_FEATURES, 0)
        aux_in = aux_extra + _SPREAD_FEATURES + _ENGINEERED_FEATURES
        return {
            "mode": "dpvn_fused",
            "input_tensor_cols": self.num_features_in,
            "lob_cols": _LOB_FEATURES,
            "aux_extra_cols": aux_extra,
            "spread_in_model": _SPREAD_FEATURES,
            "engineered_in_model": _ENGINEERED_FEATURES,
            "aux_proj_in_dim": aux_in,
            "ignored_input_cols": 0,
            "effective_channels": _LOB_FEATURES + aux_in,
        }
