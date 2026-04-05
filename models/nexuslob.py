"""NexusLOB: Deep Cross-Modal Attention for LOB Prediction.

Combines PatchLOB's efficient snapshot encoding (LOB-structured embedding +
temporal patching) with FuseLOB's event encoder, connected by cross-attention
fusion at every temporal layer instead of a one-shot gate.

Architecture:
    Snapshots: BiN -> LOBStructuredEmbedding -> TemporalPatch -> (B, T', d)
    Events:    EventEncoder(K queries) -> TemporalPatch -> (B, T', d)
                              |                    |
               Deep Cross-Attention Fusion (at every layer)
                              |
               DiffAttention Temporal/Spatial Layers -> (B, T'', d'')
                              |
               Multi-Horizon Classification Heads -> (B, 3) x num_horizons
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.bin import BiN
from models.difflob import DiffTransformerLayer, precompute_rope
from models.fuselob import EventEncoder
from models.patchlob import LOBStructuredEmbedding, TemporalPatcher
from models.tlob import _build_head, sinusoidal_positional_embedding


# ---------------------------------------------------------------------------
# Cross-Attention Fusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Cross-attention where snapshot queries attend to event key-values.

    Uses residual gating (initialized at 0) so the model starts as a pure
    snapshot model and gradually learns to incorporate events.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout_p = dropout

        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

        # Residual gate: tanh(0) = 0, so cross-attention has no effect at init
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, snap: torch.Tensor, event_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            snap:     (B, T', d) snapshot representations (queries)
            event_kv: (B, T', d) event representations (keys/values)

        Returns:
            (B, T', d) snapshot + gated cross-attention output
        """
        B, T, d = snap.shape

        q = self.q_proj(self.norm_q(snap))
        kv_normed = self.norm_kv(event_kv)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)

        # Reshape for multi-head attention
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cross_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False,
        )
        cross_out = cross_out.transpose(1, 2).contiguous().reshape(B, T, d)
        cross_out = self.resid_dropout(self.out_proj(cross_out))

        return snap + self.gate.tanh() * cross_out


# ---------------------------------------------------------------------------
# NexusLOB Model
# ---------------------------------------------------------------------------

class NexusLOB(nn.Module):
    """Deep cross-modal attention LOB model.

    Snapshot stream uses LOB-structured embedding + temporal patching (from PatchLOB).
    Event stream uses self-attention + multi-token Perceiver (from FuseLOB).
    Streams are fused via cross-attention at every temporal layer (new).
    Temporal layers use DiffAttention + RoPE (from PatchLOB/DiffLOB).
    """

    PATCH_SIZE = 4
    D_SIDE = 16  # LOBStructuredEmbedding d_side -> d_level = 48

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
        # Event params
        max_events_per_window: int = 64,
        n_event_features: int = 7,
        n_perceiver_queries: int = 4,
        event_encoder_layers: int = 2,
        event_heads: int = 4,
        # NexusLOB-specific
        patch_size: int = 4,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_horizons = num_horizons
        self.dataset_type = dataset_type
        self.n_perceiver_queries = n_perceiver_queries
        self.patch_size = patch_size

        num_patches = seq_size // patch_size
        d_model = hidden_dim

        # ---- Snapshot Stream ----
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.lob_embedding = LOBStructuredEmbedding(num_features, d_side=self.D_SIDE)
        d_spatial = self.lob_embedding.d_level  # 48

        self.snap_patcher = TemporalPatcher(d_spatial, d_model, patch_size=patch_size)

        pos_emb = sinusoidal_positional_embedding(num_patches, d_model)
        self.register_buffer("snap_pos_emb", pos_emb.unsqueeze(0))  # (1, T', d)

        # ---- Event Stream ----
        self.event_encoder = EventEncoder(
            d_event=d_model,
            n_heads=event_heads,
            n_layers=event_encoder_layers,
            max_events=max_events_per_window,
            n_queries=n_perceiver_queries,
            dropout=dropout,
            pool_output=False,  # Keep all K query tokens
        )

        # Temporal patching for events: (B, T, K, d) -> (B, T', K, d)
        # Process each query channel independently through shared patcher
        self.event_patcher = TemporalPatcher(d_model, d_model, patch_size=patch_size)

        # Project K*d -> d after patching
        self.event_proj = nn.Linear(n_perceiver_queries * d_model, d_model)

        # ---- RoPE for temporal DiffAttention ----
        sub_head_dim = (d_model // num_heads) // 2
        rope_cos, rope_sin = precompute_rope(sub_head_dim, max_len=num_patches)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # ---- Dual-Path Layers: DiffAttn (temporal) + CrossAttn + DiffAttn (spatial) ----
        self.temporal_layers = nn.ModuleList()
        self.cross_attn_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()

        for i in range(num_layers):
            is_last = i == num_layers - 1
            t_final = d_model // 4 if is_last else d_model
            s_final = num_patches // 4 if is_last else num_patches

            self.temporal_layers.append(
                DiffTransformerLayer(
                    d_model, num_heads, t_final,
                    layer_idx=i, use_rope=True, dropout=dropout,
                )
            )
            self.cross_attn_layers.append(
                CrossAttentionFusion(d_model, n_heads=cross_attn_heads, dropout=dropout)
            )
            self.spatial_layers.append(
                DiffTransformerLayer(
                    num_patches, num_heads, s_final,
                    layer_idx=i, use_rope=False, dropout=dropout,
                )
            )

        # ---- Classification Heads ----
        total_dim = (d_model // 4) * (num_patches // 4)
        if num_horizons == 1:
            self.final_layers = _build_head(total_dim, dropout=dropout)
            self.heads = None
        else:
            self.final_layers = None
            self.heads = nn.ModuleList([
                nn.ModuleList(_build_head(total_dim, dropout=dropout))
                for _ in range(num_horizons)
            ])

        # Gate logging (stored as tensor to avoid torch.compile graph break)
        self._last_cross_gate_means = [torch.tensor(0.0)] * num_layers

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
        # ---- Snapshot Stream ----
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat(
                [snapshot_input[:, :, :41], snapshot_input[:, :, 42:]], dim=2
            )
            order_type = snapshot_input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x_snap = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x_snap = snapshot_input

        # BiN normalization
        x_snap = rearrange(x_snap, "b s f -> b f s")
        x_snap = self.norm_layer(x_snap)
        x_snap = rearrange(x_snap, "b f s -> b s f")

        # LOB-structured embedding + temporal patching
        x_snap = self.lob_embedding(x_snap)             # (B, T, 48)
        x_snap = self.snap_patcher(x_snap)              # (B, T', d)
        x_snap = x_snap + self.snap_pos_emb             # + positional embedding

        # ---- Event Stream ----
        if event_features is not None and event_mask is not None:
            # EventEncoder with multi-token output: (B, T, K, d)
            event_repr = self.event_encoder(event_features, event_mask)

            B, T, K, d = event_repr.shape
            # Temporal patching per query channel
            # Reshape: (B, T, K, d) -> (B*K, T, d) -> patcher -> (B*K, T', d) -> (B, T', K, d)
            event_repr = event_repr.permute(0, 2, 1, 3).reshape(B * K, T, d)
            event_repr = self.event_patcher(event_repr)  # (B*K, T', d)
            T_prime = event_repr.shape[1]
            event_repr = event_repr.reshape(B, K, T_prime, d).permute(0, 2, 1, 3)  # (B, T', K, d)

            # Project K*d -> d
            event_repr = event_repr.reshape(B, T_prime, K * d)
            event_kv = self.event_proj(event_repr)       # (B, T', d)
            has_events = True
        else:
            event_kv = None
            has_events = False

        # ---- Dual-Path Layers ----
        x = x_snap
        for i in range(self.num_layers):
            # 1. Cross-attention: snap attends to events (before dimension compression)
            if has_events:
                x = self.cross_attn_layers[i](x, event_kv)
                self._last_cross_gate_means[i] = self.cross_attn_layers[i].gate.detach()

            # 2. Temporal DiffAttention + RoPE (may compress d on last layer)
            x = self.temporal_layers[i](x, rope_cos=self.rope_cos, rope_sin=self.rope_sin)

            # 3. Permute for spatial attention
            x = x.permute(0, 2, 1)

            # 4. Spatial DiffAttention (may compress T' on last layer)
            x = self.spatial_layers[i](x)

            # 5. Permute back
            x = x.permute(0, 2, 1)

        # ---- Flatten + Classify ----
        x = rearrange(x, "b s f -> b (f s) 1")
        x = x.reshape(x.shape[0], -1)

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
