import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_ordinal_smoothing_matrix(num_classes: int, smoothing: float) -> torch.Tensor:
    """Build a (num_classes, num_classes) soft-target matrix based on ordinal distance.

    For true class c, the soft target puts (1 - smoothing) mass on c and
    distributes the remaining smoothing mass to other classes inversely
    proportional to their ordinal distance from c.

    Example with num_classes=3, smoothing=0.15:
      True=0 (up):   [0.850, 0.100, 0.050]
      True=1 (stat): [0.075, 0.850, 0.075]
      True=2 (down): [0.050, 0.100, 0.850]
    """
    matrix = torch.zeros(num_classes, num_classes)
    for c in range(num_classes):
        # Inverse-distance weights for off-diagonal entries
        off_weights = torch.zeros(num_classes)
        for j in range(num_classes):
            if j != c:
                off_weights[j] = 1.0 / abs(j - c)
        off_sum = off_weights.sum()
        for j in range(num_classes):
            if j == c:
                matrix[c, j] = 1.0 - smoothing
            else:
                matrix[c, j] = smoothing * (off_weights[j] / off_sum)
    return matrix


class FocalLoss(nn.Module):
    """Focal loss with optional ordinal-aware label smoothing.

    For a 3-class ordinal problem (down / stationary / up), combines:
      - Focal modulation: down-weights easy (high p_t) samples via (1 - p_t)^gamma
      - Alpha weighting: per-class weights (same role as class_weights in CrossEntropyLoss)
      - Ordinal smoothing: soft targets that encode "stationary is closer to up than down is"

    With gamma=0 and ordinal_smoothing=0 this reduces exactly to weighted cross-entropy,
    making it a clean drop-in replacement for nn.CrossEntropyLoss.

    Args:
        gamma: Focal modulation exponent. gamma=0 → standard CE. Default: 2.0.
        alpha: Per-class weight tensor of shape (num_classes,). Equivalent to the
               `weight` parameter in nn.CrossEntropyLoss. Default: None.
        num_classes: Number of output classes. Default: 3.
        ordinal_smoothing: Fraction of probability mass redistributed to neighbouring
                           classes according to inverse ordinal distance. 0.0 = disabled.
                           Default: 0.0.
        reduction: "mean" | "sum" | "none". Default: "mean".
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        num_classes: int = 3,
        ordinal_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.ordinal_smoothing = ordinal_smoothing
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

        if ordinal_smoothing > 0.0:
            smoothing_matrix = _build_ordinal_smoothing_matrix(num_classes, ordinal_smoothing)
            self.register_buffer("smoothing_matrix", smoothing_matrix)
        else:
            self.smoothing_matrix = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs, shape (B, C). Must be logits (pre-softmax).
            targets: Integer class indices, shape (B,).

        Returns:
            Scalar loss (if reduction="mean"/"sum") or per-sample losses (B,).
        """
        log_probs = F.log_softmax(logits, dim=1)   # (B, C)
        probs = log_probs.exp()                     # (B, C) — derived from log_probs for stability

        # Cross-entropy term — with or without ordinal smoothing
        if self.smoothing_matrix is not None:
            soft_targets = self.smoothing_matrix[targets]           # (B, C)
            ce = -(soft_targets * log_probs).sum(dim=1)             # (B,)
        else:
            ce = F.nll_loss(log_probs, targets, reduction="none")   # (B,)

        # Focal modulator based on probability of the hard (true) class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)
        focal_weight = (1.0 - p_t) ** self.gamma                   # (B,)

        # Per-class alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha[targets]                           # (B,)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce                                    # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
