"""FuseLOB: Dual-stream (Events + Snapshots) LOB model with gated fusion.

Architecture:
    Stream 1 (Events):    (B, T, E, 7) -> EventEncoder -> (B, T, d)
    Stream 2 (Snapshots): (B, T, F)    -> SnapEncoder  -> (B, T, d)
                                 |               |
                            Gated Fusion -> (B, T, d)
                                 |
                   Temporal Transformer (alternating spatial/temporal) -> (B, T', d')
                                 |
                   Multi-Horizon Classification Heads -> (B, 3) x num_horizons
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.bin import BiN
from models.tlob import TransformerLayer, _build_head, sinusoidal_positional_embedding


# ---------------------------------------------------------------------------
# Event Embedding
# ---------------------------------------------------------------------------

class EventEmbedding(nn.Module):
    """Map 7 raw event features to d_event-dimensional embedding.

    Feature order (from preprocessing/events.py):
        0: action_code  (int 0-7)
        1: side          (int 0-1)
        2: price_relative (float)
        3: quantity_log   (float)
        4: time_delta     (float)
        5: revision_flag  (int 0-1)
        6: is_aggressive  (int 0-1)
    """

    def __init__(self, d_event: int = 64):
        super().__init__()
        # Categorical embeddings: total 32 dims
        self.action_emb = nn.Embedding(8, 16)
        self.side_emb = nn.Embedding(2, 8)
        self.revision_emb = nn.Embedding(2, 4)
        self.aggression_emb = nn.Embedding(2, 4)
        cat_dim = 16 + 8 + 4 + 4  # 32

        # Continuous projection: 3 features -> remaining dims
        self.cont_proj = nn.Linear(3, d_event - cat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (*, E, 7) float32 -> (*, E, d_event)"""
        action = self.action_emb(x[..., 0].long())
        side = self.side_emb(x[..., 1].long())
        continuous = self.cont_proj(x[..., 2:5])
        revision = self.revision_emb(x[..., 5].long())
        aggression = self.aggression_emb(x[..., 6].long())
        return torch.cat([action, side, revision, aggression, continuous], dim=-1)


# ---------------------------------------------------------------------------
# Event Attention Layer (maskless — uses zero-padding for FlashAttention)
# ---------------------------------------------------------------------------

class EventAttentionLayer(nn.Module):
    """Bidirectional self-attention for event sequences.

    No explicit attention mask — padded positions are pre-zeroed so their Q/K/V
    are ~0, contributing negligibly to attention output. This enables FlashAttention
    via F.scaled_dot_product_attention without any mask argument.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attn_dropout_p = dropout

        self.norm1 = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, E, d) -> (B, E, d). No mask needed — padding is pre-zeroed."""
        B, E, d = x.shape
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).reshape(B, E, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, E, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, E, self.num_heads, self.head_dim).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).contiguous().reshape(B, E, d)
        x = self.resid_dropout(self.out_proj(x))
        x = x + residual

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


# ---------------------------------------------------------------------------
# Perceiver Compression (maskless SDPA)
# ---------------------------------------------------------------------------

class PerceiverCompression(nn.Module):
    """Compress event sequences to fixed-size via cross-attention with learned queries.

    Uses F.scaled_dot_product_attention directly (no nn.MultiheadAttention) to
    enable FlashAttention dispatch. Padding handled via zero-padding strategy.
    """

    def __init__(self, d_model: int, n_queries: int = 8, n_heads: int = 4, dropout: float = 0.0,
                 pool_output: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pool_output = pool_output
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.RMSNorm(d_model)
        self.attn_dropout_p = dropout

    def forward(self, x: torch.Tensor, has_events: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:          (B, E, d) event encoder output (padding pre-zeroed)
            has_events: (B,) bool, True if window has any events

        Returns:
            If pool_output=True:  (B, d) compressed representation (mean-pooled over queries)
            If pool_output=False: (B, K, d) all K query outputs preserved
        """
        B, E, d = x.shape
        M = self.queries.shape[1]
        queries = self.queries.expand(B, -1, -1)

        q = self.q_proj(queries).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, E, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, E, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().reshape(B, M, d)
        compressed = self.out_proj(out)
        compressed = self.norm(compressed + queries)

        # Zero all-padded windows (no data-dependent branching)
        if has_events is not None:
            compressed = compressed * has_events[:, None, None].float()

        if self.pool_output:
            return compressed.mean(dim=1)  # (B, d)
        return compressed  # (B, K, d)


# ---------------------------------------------------------------------------
# Event Encoder (full pipeline)
# ---------------------------------------------------------------------------

class EventEncoder(nn.Module):
    """Embed + self-attention + Perceiver compression for event windows.

    Uses zero-padding strategy: padded event positions are zeroed before attention,
    eliminating the need for explicit attention masks and enabling FlashAttention.
    """

    def __init__(
        self,
        d_event: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_events: int = 64,
        n_queries: int = 8,
        dropout: float = 0.0,
        pool_output: bool = True,
    ):
        super().__init__()
        self.pool_output = pool_output
        self.n_queries = n_queries
        self.embedding = EventEmbedding(d_event)

        pos_emb = sinusoidal_positional_embedding(max_events, d_event)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))  # (1, E, d)

        self.event_layers = nn.ModuleList([
            EventAttentionLayer(d_event, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.compression = PerceiverCompression(
            d_event, n_queries, n_heads, dropout=dropout, pool_output=pool_output,
        )

    def forward(self, event_features: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            event_features: (B, T, E, 7)
            event_mask:     (B, T, E) bool, True=real event

        Returns:
            If pool_output=True:  (B, T, d) per-window compressed event representations
            If pool_output=False: (B, T, K, d) per-window multi-token event representations
        """
        B, T, E, _ = event_features.shape

        # Reshape to process all windows at once
        x = event_features.reshape(B * T, E, -1)       # (B*T, E, 7)
        mask = event_mask.reshape(B * T, E)             # (B*T, E)
        mask_float = mask.unsqueeze(-1).float()         # (B*T, E, 1)

        # Embed and zero padded positions
        x = self.embedding(x)                           # (B*T, E, d)
        x = x + self.pos_emb[:, :E, :]
        x = x * mask_float                              # zero padding ONCE

        # Self-attention layers (no mask — FlashAttention)
        for layer in self.event_layers:
            x = layer(x)

        # Re-zero before compression (clean up residual leakage)
        x = x * mask_float

        # Perceiver compression
        has_events = mask.any(dim=-1)                   # (B*T,)
        out = self.compression(x, has_events=has_events)

        if self.pool_output:
            return out.reshape(B, T, -1)                # (B, T, d)
        return out.reshape(B, T, self.n_queries, -1)    # (B, T, K, d)


# ---------------------------------------------------------------------------
# Snapshot Encoder (lightweight)
# ---------------------------------------------------------------------------

class SnapshotEncoder(nn.Module):
    """Lightweight LOB snapshot encoder: BiN + embedding + 2-layer transformer."""

    def __init__(
        self,
        num_features: int,
        seq_size: int,
        d_model: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        dropout: float = 0.0,
        use_fast_attention: bool = True,
    ):
        super().__init__()
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, d_model)

        pos_emb = sinusoidal_positional_embedding(seq_size, d_model)
        self.register_buffer("pos_encoder", pos_emb.unsqueeze(0))  # (1, T, d)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model, n_heads, d_model,
                use_fast_attention=use_fast_attention,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, T, d)"""
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")
        x = self.emb_layer(x)
        x = x + self.pos_encoder
        for layer in self.layers:
            x, _ = layer(x)
        return x


# ---------------------------------------------------------------------------
# Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """Learned per-dimension gate between event and snapshot representations."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(2 * d_model, d_model)

    def forward(
        self, event_repr: torch.Tensor, snap_repr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            event_repr: (B, T, d)
            snap_repr:  (B, T, d)

        Returns:
            fused: (B, T, d)
            gate:  (B, T, d) sigmoid values (for logging/interpretability)
        """
        concat = torch.cat([event_repr, snap_repr], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        fused = gate * event_repr + (1 - gate) * snap_repr
        return fused, gate


# ---------------------------------------------------------------------------
# FuseLOB (top-level model)
# ---------------------------------------------------------------------------

class FuseLOB(nn.Module):
    """Dual-stream LOB model fusing raw order events with LOB snapshots.

    The event stream uses bidirectional self-attention + Perceiver compression
    (all using FlashAttention via maskless SDPA with zero-padding).
    The snapshot stream uses BiN + lightweight transformer.
    Streams are fused via a learned gate, then processed by a temporal
    transformer with alternating spatial/temporal attention (same as TLOB).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        seq_size: int,
        num_features: int,
        num_heads: int,
        is_sin_emb: bool,
        dataset_type: str,
        use_fast_attention: bool = True,
        num_horizons: int = 1,
        dropout: float = 0.0,
        # FuseLOB-specific
        max_events_per_window: int = 64,
        n_event_features: int = 7,
        n_perceiver_queries: int = 8,
        event_encoder_layers: int = 2,
        snap_encoder_layers: int = 2,
        event_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_horizons = num_horizons
        self.dataset_type = dataset_type
        self.use_fast_attention = use_fast_attention

        d_model = hidden_dim

        # Stream 1: Event encoder
        self.event_encoder = EventEncoder(
            d_event=d_model,
            n_heads=event_heads,
            n_layers=event_encoder_layers,
            max_events=max_events_per_window,
            n_queries=n_perceiver_queries,
            dropout=dropout,
        )

        # Stream 2: Snapshot encoder
        self.snap_encoder = SnapshotEncoder(
            num_features=num_features,
            seq_size=seq_size,
            d_model=d_model,
            n_heads=num_heads,
            n_layers=snap_encoder_layers,
            dropout=dropout,
            use_fast_attention=use_fast_attention,
        )

        # Gated fusion
        self.fusion = GatedFusion(d_model)

        # Temporal transformer — alternating spatial (feature) / temporal (sequence)
        # Same pattern as TLOB: even layers attend over hidden_dim, odd over seq_size
        self.temporal_layers = nn.ModuleList()
        for i in range(num_layers):
            is_last = i == num_layers - 1
            feat_final = d_model // 4 if is_last else d_model
            seq_final = seq_size // 4 if is_last else seq_size
            # Feature (spatial) attention
            self.temporal_layers.append(
                TransformerLayer(
                    d_model, num_heads, feat_final,
                    use_fast_attention=use_fast_attention,
                    dropout=dropout,
                )
            )
            # Temporal (sequence) attention
            self.temporal_layers.append(
                TransformerLayer(
                    seq_size, num_heads, seq_final,
                    use_fast_attention=use_fast_attention,
                    dropout=dropout,
                )
            )

        # Classification heads (same as TLOB)
        total_dim = (d_model // 4) * (seq_size // 4)
        if num_horizons == 1:
            self.final_layers = _build_head(total_dim, dropout=dropout)
            self.heads = None
        else:
            self.final_layers = None
            self.heads = nn.ModuleList([
                nn.ModuleList(_build_head(total_dim, dropout=dropout))
                for _ in range(num_horizons)
            ])

        # Gate logging (stored as tensor to avoid torch.compile graph break from .item())
        self._last_gate_mean = torch.tensor(0.0)

    def forward(
        self,
        snapshot_input: torch.Tensor,
        event_features: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        store_att: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Args:
            snapshot_input: (B, T, F) LOB snapshot features
            event_features: (B, T, E, 7) raw event tokens, or None for snapshot-only
            event_mask:     (B, T, E) bool mask, or None
            store_att:      unused, kept for API compatibility

        Returns:
            Single-horizon: (B, 3) logits
            Multi-horizon:  list of num_horizons (B, 3) logit tensors
        """
        # Snapshot stream
        snap_repr = self.snap_encoder(snapshot_input)  # (B, T, d)

        # Event stream (optional for ablation)
        if event_features is not None and event_mask is not None:
            event_repr = self.event_encoder(event_features, event_mask)  # (B, T, d)
            fused, gate = self.fusion(event_repr, snap_repr)
            self._last_gate_mean = gate.detach().mean()
        else:
            fused = snap_repr

        # Temporal transformer with alternating feature/temporal attention
        x = fused
        for layer in self.temporal_layers:
            x, _ = layer(x)
            x = x.permute(0, 2, 1)

        # Flatten
        x = rearrange(x, "b s f -> b (f s) 1")
        x = x.reshape(x.shape[0], -1)

        # Classification
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
