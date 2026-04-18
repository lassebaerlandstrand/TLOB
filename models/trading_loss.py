"""Trading losses for LOB models.

Variants:
  - DFLProxyLoss: Direction-alignment using classification labels as proxy.
  - DFLTradingLoss: Spread-aware PnL with raw price changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Label-to-position mapping: up(0)→+1, stationary(1)→0, down(2)→-1
_POSITION_MAP = torch.tensor([1.0, 0.0, -1.0])


class DFLProxyLoss(nn.Module):
    """Direction-alignment loss using classification labels as proxy.

    Computes soft positions from logits via softmax, then rewards alignment
    with the true price direction. Key difference from cross-entropy: CE
    penalizes equally for wrong predictions, while this loss penalizes
    proportionally to how WRONG the trading decision would be.

    position = softmax(logits) @ [+1, 0, -1]  ∈  [-1, +1]
    target_dir = label_map[y]                  ∈  {-1, 0, +1}
    proxy_pnl = position × target_dir          ∈  [-1, +1]

    Loss = -mean(proxy_pnl)  or  -mean(proxy_pnl) / std(proxy_pnl)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        objective: str = "pnl",
        lambda_turnover: float = 0.0,
        lambda_entropy: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.objective = objective
        self.lambda_turnover = lambda_turnover
        self.lambda_entropy = lambda_entropy
        self.register_buffer("position_map", _POSITION_MAP)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prev_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, 3) raw model output
            labels: (B,) integer class labels {0, 1, 2}
            prev_positions: (B,) previous soft positions for turnover penalty

        Returns:
            loss: scalar
            info: dict with positions, proxy_pnl for logging
        """
        if self.training:
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)
        else:
            one_hot = F.one_hot(logits.argmax(dim=1), num_classes=3).float()
        position = one_hot @ self.position_map  # (B,) ∈ {-1, 0, +1}
        # Soft probs for entropy regularization
        probs = F.softmax(logits / self.temperature, dim=1)

        target_direction = self.position_map[labels]  # (B,)
        proxy_pnl = position * target_direction  # (B,)

        if self.objective == "sharpe":
            loss = -proxy_pnl.mean() / (proxy_pnl.std().clamp(min=1e-4))
        elif self.objective == "sortino":
            downside = proxy_pnl.clamp(max=0)
            downside_std = (downside ** 2).mean().sqrt().clamp(min=1e-4)
            loss = -proxy_pnl.mean() / downside_std
        else:  # pnl
            loss = -proxy_pnl.mean()

        # Turnover penalty
        if self.lambda_turnover > 0 and prev_positions is not None:
            turnover = torch.abs(position - prev_positions).mean()
            loss = loss + self.lambda_turnover * turnover

        # Entropy regularization (prevent position collapse to always +1/-1)
        if self.lambda_entropy > 0:
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            loss = loss - self.lambda_entropy * entropy

        info = {
            "positions": position.detach(),
            "proxy_pnl": proxy_pnl.detach(),
            "mean_abs_position": position.abs().mean().detach(),
        }
        return loss, info


class DFLTradingLoss(nn.Module):
    """Spread-aware differentiable trading loss using raw price changes.

    Computes realistic PnL with actual bid-ask spread as transaction cost:
      - Buy at ask (sell1), sell at bid (buy1)
      - Cost per position change = |Δposition| × half_spread

    Requires raw delta_mid and half_spread in the batch data (from modified
    preprocessing pipeline).

    position(t) = softmax(logits / τ) @ [+1, 0, -1]
    gross_return(t) = position(t) × delta_mid(t)
    cost(t) = cost_mult × |position(t) - prev_position(t)| × half_spread(t)
    net_return(t) = gross_return(t) - cost(t)

    Loss = -Sharpe(net_return) or -mean(net_return)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        cost_multiplier: float = 1.0,
        objective: str = "sharpe",
        lambda_drawdown: float = 0.0,
        lambda_turnover: float = 0.0,
        lambda_entropy: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.cost_multiplier = cost_multiplier
        self.objective = objective
        self.lambda_drawdown = lambda_drawdown
        self.lambda_turnover = lambda_turnover
        self.lambda_entropy = lambda_entropy
        self.register_buffer("position_map", _POSITION_MAP)

    def forward(
        self,
        logits: torch.Tensor,
        delta_mid: torch.Tensor,
        half_spread: torch.Tensor,
        prev_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (B, 3) raw model output
            delta_mid: (B,) raw mid-price change at horizon h
            half_spread: (B,) half bid-ask spread = (ask1 - bid1) / 2
            prev_positions: (B,) previous soft positions

        Returns:
            loss: scalar, differentiable through logits
            info: dict with positions, returns, costs for logging
        """
        if self.training:
            one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)
        else:
            one_hot = F.one_hot(logits.argmax(dim=1), num_classes=3).float()
        position = one_hot @ self.position_map  # (B,) ∈ {-1, 0, +1}
        # Soft probs for entropy regularization
        probs = F.softmax(logits / self.temperature, dim=1)

        # Gross return from position
        gross_return = position * delta_mid  # (B,)

        # Transaction cost (spread traversal)
        if prev_positions is None:
            prev_positions = torch.zeros_like(position)
        # Smooth absolute value for better gradients
        pos_change = position - prev_positions
        smooth_abs_change = torch.sqrt(pos_change ** 2 + 1e-8)
        cost = self.cost_multiplier * smooth_abs_change * half_spread.abs()

        net_return = gross_return - cost  # (B,)

        # Objective (clamp std to 1e-4 to avoid gradient explosion on uniform batches)
        if self.objective == "sharpe":
            loss = -net_return.mean() / net_return.std().clamp(min=1e-4)
        elif self.objective == "sortino":
            downside = net_return.clamp(max=0)
            downside_std = (downside ** 2).mean().sqrt().clamp(min=1e-4)
            loss = -net_return.mean() / downside_std
        else:  # pnl
            loss = -net_return.mean()

        # Drawdown penalty
        if self.lambda_drawdown > 0:
            cum_returns = net_return.cumsum(dim=0)
            running_max = cum_returns.cummax(dim=0).values
            drawdown = (running_max - cum_returns).max()
            loss = loss + self.lambda_drawdown * drawdown

        # Turnover penalty
        if self.lambda_turnover > 0:
            loss = loss + self.lambda_turnover * smooth_abs_change.mean()

        # Entropy regularization
        if self.lambda_entropy > 0:
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            loss = loss - self.lambda_entropy * entropy

        info = {
            "positions": position.detach(),
            "gross_return": gross_return.detach(),
            "cost": cost.detach(),
            "net_return": net_return.detach(),
            "mean_abs_position": position.abs().mean().detach(),
            "trade_fraction": (smooth_abs_change > 0.01).float().mean().detach(),
        }
        return loss, info
