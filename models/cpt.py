"""Committed Position Transformer (CPT): LOB trading with DP-supervised trade filter.

Architecture:
  LOB → Encoder (TLOB backbone) → features
  features + spread_context + pos_embed → Direction Head → target {long, flat, short}
  features + spread_context + pos_embed → Trade Filter → trade/hold ∈ [0, 1]

  Inference: if filter > τ → adopt direction prediction
             else → hold current position

The trade filter is supervised by DP optimal trade/hold labels, learning when NOT
to trade based on market microstructure (spread, volatility, position state).
"""

import torch
import torch.nn as nn
from einops import rearrange

from models.bin import BiN
from models.tlob import TransformerLayer, sinusoidal_positional_embedding


class SpreadContext(nn.Module):
    """Extract and embed spread features from LOB data.

    Computes [z_half_spread, spread/vol_ratio, log(1 + |spread|)]
    from the raw LOB input, projects to an embedding.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, lob_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob_input: (B, seq_size, num_features) raw LOB window

        Returns:
            (B, embed_dim)
        """
        # Best ask (sell1) = col 0, best bid (buy1) = col 2
        ask1 = lob_input[:, :, 0]  # (B, S)
        bid1 = lob_input[:, :, 2]  # (B, S)
        mid = (ask1 + bid1) / 2

        z_half_spread = (ask1[:, -1] - bid1[:, -1]) / 2  # (B,)
        mid_diff = mid[:, 1:] - mid[:, :-1]
        rolling_vol = mid_diff.std(dim=1).clamp(min=1e-8)  # (B,)
        spread_vol_ratio = z_half_spread.abs() / rolling_vol
        log_spread = torch.log1p(z_half_spread.abs())

        features = torch.stack([z_half_spread, spread_vol_ratio, log_spread], dim=1)
        return self.proj(features)


class DirectionHead(nn.Module):
    """3-class logits for target position: {long, flat, short}."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # (B, 3)


class TradeFilter(nn.Module):
    """Trade/hold filter conditioned on features + position state + spread.

    Output σ ∈ [0, 1]: high = trade, low = hold current position.
    Supervised by DP optimal trade/hold decisions.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)  # (B,) raw logits


class CPT(nn.Module):
    """Committed Position Transformer.

    TLOB encoder backbone + direction head + DP-supervised trade filter.
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
        spread_embed_dim: int = 16,
        pos_embed_dim: int = 16,
        head_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.use_fast_attention = use_fast_attention
        self.num_horizons = num_horizons
        self.dropout = dropout

        # --- Encoder (TLOB backbone, identical) ---
        self.order_type_embedder = nn.Embedding(3, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            pos_emb = sinusoidal_positional_embedding(seq_size, hidden_dim).unsqueeze(0)
            self.register_buffer("pos_encoder", pos_emb)
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(
                    TransformerLayer(hidden_dim, num_heads, hidden_dim,
                                    use_fast_attention=use_fast_attention, dropout=dropout)
                )
                self.layers.append(
                    TransformerLayer(seq_size, num_heads, seq_size,
                                    use_fast_attention=use_fast_attention, dropout=dropout)
                )
            else:
                self.layers.append(
                    TransformerLayer(hidden_dim, num_heads, hidden_dim // 4,
                                    use_fast_attention=use_fast_attention, dropout=dropout)
                )
                self.layers.append(
                    TransformerLayer(seq_size, num_heads, seq_size // 4,
                                    use_fast_attention=use_fast_attention, dropout=dropout)
                )

        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.total_dim = total_dim

        # --- Context modules ---
        self.spread_context = SpreadContext(embed_dim=spread_embed_dim)
        self.pos_embed = nn.Embedding(3, pos_embed_dim)  # 0=short, 1=flat, 2=long

        aug_dim = total_dim + spread_embed_dim + pos_embed_dim

        # --- Decision heads (one set per horizon) ---
        self.direction_heads = nn.ModuleList([
            DirectionHead(aug_dim, head_hidden_dim, dropout=dropout)
            for _ in range(num_horizons)
        ])
        self.trade_filters = nn.ModuleList([
            TradeFilter(aug_dim, head_hidden_dim, dropout=dropout)
            for _ in range(num_horizons)
        ])

    def set_fast_attention(self, use_fast_attention: bool):
        self.use_fast_attention = use_fast_attention
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                layer.set_fast_attention(use_fast_attention)

    def _pos_to_embed_idx(self, positions: torch.Tensor) -> torch.Tensor:
        """Map {-1, 0, +1} → {0, 1, 2} for embedding lookup."""
        return (positions + 1).long()

    def _encode(self, input: torch.Tensor) -> torch.Tensor:
        """Shared encoder: input → flat representation (identical to TLOB)."""
        if self.dataset_type == "LOBSTER":
            continuous_features = torch.cat([input[:, :, :41], input[:, :, 42:]], dim=2)
            order_type = input[:, :, 41].long()
            order_type_emb = self.order_type_embedder(order_type).detach()
            x = torch.cat([continuous_features, order_type_emb], dim=2)
        else:
            x = input
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")
        x = self.emb_layer(x)
        x = x[:] + self.pos_encoder
        for i in range(len(self.layers)):
            x, _ = self.layers[i](x)
            x = x.permute(0, 2, 1)
        x = rearrange(x, "b s f -> b (f s) 1")
        x = x.reshape(x.shape[0], -1)
        return x

    def forward(
        self,
        input: torch.Tensor,
        current_positions: torch.Tensor | None = None,
    ) -> dict:
        """Full forward pass: encode LOB → direction + trade filter.

        Args:
            input: (B, seq_size, num_features) LOB snapshot
            current_positions: (B, num_horizons) current positions

        Returns:
            dict with per-horizon lists: "directions" and "filter_logits"
        """
        B = input.shape[0]
        if current_positions is None:
            current_positions = torch.zeros(B, self.num_horizons, device=input.device)

        features = self._encode(input)
        spread_ctx = self.spread_context(input)

        directions = []
        filter_logits = []

        for h_idx in range(self.num_horizons):
            cur_pos_h = current_positions[:, h_idx]
            pos_idx = self._pos_to_embed_idx(cur_pos_h)
            pos_emb = self.pos_embed(pos_idx)
            aug = torch.cat([features, spread_ctx, pos_emb], dim=1)

            directions.append(self.direction_heads[h_idx](aug))
            filter_logits.append(self.trade_filters[h_idx](aug))

        return {
            "directions": directions,
            "filter_logits": filter_logits,
        }
