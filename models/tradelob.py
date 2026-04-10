"""TradeLOB: End-to-end LOB trading with learned no-transaction bands.

Architecture:
  LOB → Encoder (TLOB backbone) → features
  features → Signal Head → target position s ∈ [-1, +1]
  features + position_embedding → Band Head → half-width w ∈ [0, max_w]
  Position update: if |s - current_pos| > w → trade, else hold

The signal head outputs the desired position (continuous, discretized at
execution). The band head determines whether the signal is strong enough
to justify trading (incurring spread costs). Everything is learned from
data — no hardcoded min_hold, confidence threshold, or spread parameters.

References:
  - NTBN: Imaki et al. 2021 (arXiv:2103.01775)
  - WW-NTBN: Arzel & Lehdili 2026 (arXiv:2603.29994)
  - Deep Hedging: Buehler et al. 2019 (arXiv:1802.03042)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.bin import BiN
from models.tlob import TransformerLayer, sinusoidal_positional_embedding


class SignalHead(nn.Module):
    """Maps encoder features to a target position in [-1, +1].

    Unlike the 3-class classification head, this outputs a continuous signal
    where magnitude encodes conviction:
      s ≈ +1.0: strong long conviction
      s ≈  0.0: no directional conviction (flat)
      s ≈ -1.0: strong short conviction
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features).squeeze(-1))  # (B,)


class BandHead(nn.Module):
    """NTBN-inspired no-transaction band head.

    Outputs band half-width that determines when position changes are allowed.
    Depends purely on encoder features (market conditions), not current position.

    Wide band = hold (don't trade), narrow band = responsive to signals.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        max_band_width: float = 1.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_band_width = max_band_width
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize narrow bands: sigmoid(-3) ≈ 0.05 → half_width ≈ 0.075
        # Forces initial trading so gradients flow. The band learns to widen
        # for unprofitable trades during training.
        nn.init.constant_(self.net[-1].bias, -3.0)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, features: torch.Tensor, current_pos: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            features: (B, D) encoder output
            current_pos: unused, kept for API compatibility

        Returns:
            half_width: (B,) band half-width ∈ [0, max_band_width]
        """
        return torch.sigmoid(self.net(features).squeeze(-1)) * self.max_band_width  # (B,)


def ntb_position_update(
    current_pos: torch.Tensor,
    target_signal: torch.Tensor,
    half_width: torch.Tensor,
    sharpness: float = 10.0,
) -> torch.Tensor:
    """NTBN-style differentiable position update.

    If |target - current_pos| > half_width → trade (move to target)
    If |target - current_pos| ≤ half_width → hold (keep current_pos)

    Uses sigmoid approximation for differentiability during training.
    At eval time, the soft decision approaches a hard threshold.

    Args:
        current_pos: (B,) current position
        target_signal: (B,) desired position from signal head
        half_width: (B,) band half-width from band head
        sharpness: controls sigmoid sharpness (higher = more binary)

    Returns:
        new_pos: (B,) updated position
    """
    distance = torch.abs(target_signal - current_pos)
    # Probability of trading: high when distance >> half_width
    should_trade = torch.sigmoid(sharpness * (distance - half_width))
    new_pos = should_trade * target_signal + (1.0 - should_trade) * current_pos
    return new_pos


def discretize_signal(signal: torch.Tensor, temperature: float = 1.0, training: bool = True) -> torch.Tensor:
    """Convert continuous signal [-1, +1] to discrete position {-1, 0, +1}.

    During training: Gumbel-Softmax for differentiable discrete samples.
    During eval: hard argmax.

    The signal is converted to 3-class logits:
      logit_long  = signal      (high when signal is positive)
      logit_flat  = 0           (neutral baseline — flat selected by Gumbel noise when |signal| is small)
      logit_short = -signal     (high when signal is negative)
    """
    # Logits: at signal=0 all logits are equal (1/3 each).
    # At |signal|>0: long/short compete, flat stays neutral.
    logits = torch.stack([signal, torch.zeros_like(signal), -signal], dim=-1)  # (B, 3)

    position_map = torch.tensor([1.0, 0.0, -1.0], device=signal.device)

    if training:
        one_hot = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
    else:
        one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=3).float()

    return one_hot @ position_map  # (B,) ∈ {-1, 0, +1}


class TradeLOB(nn.Module):
    """End-to-end LOB trading model with learned no-transaction bands.

    Uses TLOB's dual-path transformer encoder as backbone, replacing the
    classification head with:
      - Signal head: continuous target position ∈ [-1, +1]
      - Band head: NTBN-style half-width determining when to trade

    The model outputs (signal, half_width) per timestep. For sequential
    evaluation, maintain position state externally and call ntb_position_update.
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
        max_band_width: float = 1.5,
        pos_embed_dim: int = 8,
        band_hidden_dim: int = 64,
        signal_hidden_dim: int = 64,
        sharpness: float = 10.0,
        gumbel_temperature: float = 1.0,
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
        self.sharpness = sharpness
        self.gumbel_temperature = gumbel_temperature

        # --- Encoder (TLOB backbone) ---
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

        # --- Trading heads (one per horizon) ---
        self.signal_heads = nn.ModuleList([
            SignalHead(total_dim, signal_hidden_dim, dropout=dropout)
            for _ in range(num_horizons)
        ])
        self.band_heads = nn.ModuleList([
            BandHead(total_dim, band_hidden_dim, max_band_width, dropout=dropout)
            for _ in range(num_horizons)
        ])

        # Optional: 3-class head for CE regularization and F1 reporting
        self.classification_heads = None

    def enable_classification_heads(self):
        """Add optional 3-class heads for CE regularization."""
        from models.tlob import _build_head
        total_dim = (self.hidden_dim // 4) * (self.seq_size // 4)
        self.classification_heads = nn.ModuleList([
            nn.ModuleList(_build_head(total_dim, dropout=self.dropout))
            for _ in range(self.num_horizons)
        ])

    def set_fast_attention(self, use_fast_attention: bool):
        self.use_fast_attention = use_fast_attention
        for layer in self.layers:
            if isinstance(layer, TransformerLayer):
                layer.set_fast_attention(use_fast_attention)

    def _encode(self, input: torch.Tensor) -> torch.Tensor:
        """Shared encoder: input → flat representation (same as TLOB)."""
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

    def forward_heads_only(
        self,
        features: torch.Tensor,
        current_positions: torch.Tensor | None = None,
    ) -> dict:
        """Run only signal/band heads on pre-encoded features.

        Use this when the encoder has already been run (e.g., batched encoding
        of all timesteps), and only the lightweight heads + position update
        need sequential processing.

        Args:
            features: (B, total_dim) pre-computed encoder output
            current_positions: (B, num_horizons) current positions

        Returns:
            Same dict as forward() (without ce_logits).
        """
        B = features.shape[0]
        if current_positions is None:
            current_positions = torch.zeros(B, self.num_horizons, device=features.device)

        signals = []
        half_widths = []
        new_positions = []

        for h_idx in range(self.num_horizons):
            pos_h = current_positions[:, h_idx]
            signal = self.signal_heads[h_idx](features)
            half_width = self.band_heads[h_idx](features)
            # Continuous NTB: signal IS the target position, band decides when to trade
            new_pos = ntb_position_update(pos_h, signal, half_width, self.sharpness)
            signals.append(signal)
            half_widths.append(half_width)
            new_positions.append(new_pos)

        return {"signals": signals, "half_widths": half_widths, "new_positions": new_positions}

    def forward(
        self,
        input: torch.Tensor,
        current_positions: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input: (B, seq_size, num_features) LOB sequence
            current_positions: (B, num_horizons) current position per horizon,
                continuous ∈ [-1, +1]. If None, assumes flat (0).

        Returns:
            dict with:
                signals: list of (B,) signal tensors per horizon
                half_widths: list of (B,) band widths per horizon
                new_positions: list of (B,) updated positions per horizon
                ce_logits: list of (B, 3) logits if classification heads enabled
        """
        B = input.shape[0]
        features = self._encode(input)  # (B, total_dim)

        if current_positions is None:
            current_positions = torch.zeros(B, self.num_horizons, device=input.device)

        signals = []
        half_widths = []
        new_positions = []

        for h_idx in range(self.num_horizons):
            pos_h = current_positions[:, h_idx]

            signal = self.signal_heads[h_idx](features)  # (B,) ∈ [-1, +1]
            half_width = self.band_heads[h_idx](features)  # (B,) ∈ [0, max_w]

            # Continuous NTB: signal IS the target position, band decides when to trade
            new_pos = ntb_position_update(pos_h, signal, half_width, self.sharpness)

            signals.append(signal)
            half_widths.append(half_width)
            new_positions.append(new_pos)

        result = {
            "signals": signals,
            "half_widths": half_widths,
            "new_positions": new_positions,
        }

        # Optional classification logits for CE regularization
        if self.classification_heads is not None:
            ce_logits = []
            for head in self.classification_heads:
                h = features
                for layer in head:
                    h = layer(h)
                ce_logits.append(h)
            result["ce_logits"] = ce_logits

        return result
