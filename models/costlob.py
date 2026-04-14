"""CostLOB: Cost-Conditioned LOB Classifier with Learned Hysteresis.

Architecture:
  LOB -> Encoder (TLOB backbone) -> features
  features + cost_context(raw_input) -> TradingHead -> direction_logits + confidence

  Training: CE(direction) + lambda_conf * BCE(confidence, profitable_target)
  Inference: position = argmax(direction) filtered by confidence threshold
             or used as the confidence signal in Schmitt-trigger hysteresis.

The confidence head shares a trunk with the direction head, preventing the
divergence seen in CPT's separate filter head. Cost context conditions the
head on spread/volatility so confidence adapts to market conditions.
"""

import torch
import torch.nn as nn
from einops import rearrange

from models.bin import BiN
from models.tlob import TransformerLayer, sinusoidal_positional_embedding


class CostContext(nn.Module):
    """Extract and embed cost features from the raw (pre-BiN) LOB input.

    Computes 3 scalar features from the last timestep:
      - z_half_spread: (ask1 - bid1) / 2
      - spread/vol ratio: z_half_spread / rolling_vol(mid)
      - recent volatility: std(mid_diff[-16:])
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(3, embed_dim)

    def forward(self, raw_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_input: (B, seq_size, num_features) raw LOB window (pre-BiN)
        Returns:
            (B, embed_dim)
        """
        ask1 = raw_input[:, :, 0]  # (B, S)
        bid1 = raw_input[:, :, 2]  # (B, S)
        mid = (ask1 + bid1) / 2

        z_half_spread = (ask1[:, -1] - bid1[:, -1]) / 2  # (B,)
        mid_diff = mid[:, 1:] - mid[:, :-1]
        rolling_vol = mid_diff.std(dim=1).clamp(min=1e-8)  # (B,)
        spread_vol_ratio = z_half_spread.abs() / rolling_vol
        recent_vol = mid_diff[:, -16:].std(dim=1).clamp(min=1e-8)  # (B,)

        features = torch.stack([z_half_spread, spread_vol_ratio, recent_vol], dim=1)
        return self.proj(features)  # (B, embed_dim)


class TradingHead(nn.Module):
    """Joint direction + trade-confidence head with shared trunk.

    The shared trunk ensures confidence is computed from the same features
    that determine direction — they cannot diverge (unlike CPT's separate heads).
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        trunk_dim = input_dim // 4
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, trunk_dim),
            nn.GELU(),
        )
        self.direction = nn.Linear(trunk_dim, 3)
        self.confidence = nn.Linear(trunk_dim, 1)
        # Initialize confidence near 0 -> sigmoid(0) = 0.5 (neutral)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)  # (B, trunk_dim)
        direction = self.direction(shared)  # (B, 3)
        conf_logit = self.confidence(shared).squeeze(-1)  # (B,) raw logit
        return direction, conf_logit


class CostLOB(nn.Module):
    """TLOB encoder + cost context + joint direction-confidence head.

    Same encoder as TLOB. The head produces both direction logits (for CE)
    and a cost-conditioned confidence (for trading decisions).

    At inference: position = argmax(direction) * I[confidence > threshold]
    This is learned hysteresis: the model learns when its own predictions
    are profitable, conditioned on the spread.
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
        cost_embed_dim: int = 8,
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

        # --- Encoder (identical to TLOB) ---
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

        # --- Cost context ---
        self.cost_context = CostContext(embed_dim=cost_embed_dim)

        # --- Per-horizon trading heads ---
        aug_dim = total_dim + cost_embed_dim
        if num_horizons == 1:
            self.head = TradingHead(aug_dim, dropout=dropout)
            self.heads = None
        else:
            self.head = None
            self.heads = nn.ModuleList([
                TradingHead(aug_dim, dropout=dropout) for _ in range(num_horizons)
            ])

    def set_fast_attention(self, use_fast_attention: bool):
        self.use_fast_attention = use_fast_attention
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                layer.set_fast_attention(use_fast_attention)

    def _encode(self, input: torch.Tensor) -> torch.Tensor:
        """Shared encoder: input -> flat representation (identical to TLOB)."""
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

    def forward(self, input: torch.Tensor) -> dict:
        """Full forward pass: encode LOB -> direction + confidence.

        Args:
            input: (B, seq_size, num_features) LOB snapshot

        Returns:
            dict with "directions" and "confidences" (lists for multi-horizon,
            single tensors for single-horizon).
        """
        features = self._encode(input)          # (B, total_dim)
        cost_ctx = self.cost_context(input)      # (B, cost_embed_dim)
        aug = torch.cat([features, cost_ctx], dim=1)  # (B, aug_dim)

        if self.heads is None:
            direction, confidence = self.head(aug)
            return {"directions": [direction], "confidences": [confidence]}
        else:
            directions = []
            confidences = []
            for head in self.heads:
                d, c = head(aug)
                directions.append(d)
                confidences.append(c)
            return {"directions": directions, "confidences": confidences}
