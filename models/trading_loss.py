"""Trading losses for LOB models.

Variants:
  - DFLProxyLoss: Direction-alignment using classification labels as proxy.
  - DFLTradingLoss: Spread-aware PnL with raw price changes.
  - NTBTradingLoss: Sequential no-transaction band loss for TradeLOB.
  - CPTTradingLoss: Committed Position Transformer loss (CE + Sharpe + hold supervision).
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


class NTBTradingLoss(nn.Module):
    """No-Transaction Band loss for sequential LOB trading.

    Processes T-step chunks of consecutive samples. At each step:
      1. TradeLOB model outputs (signal, band_width) for each horizon
      2. NTBN position update: trade only if signal exceeds band
      3. PnL computed with actual spread costs

    The loss is the negative Sharpe ratio of net returns over the chunk,
    encouraging the model to learn profitable, selective trading.

    Unlike per-sample DFL losses, this:
      - Maintains position state across the chunk (position carry-forward)
      - Naturally rewards holding (holding is free, trading costs spread)
      - The band structure architecturally prevents unnecessary trades
    """

    def __init__(
        self,
        objective: str = "sharpe",
        lambda_activity: float = 0.0,
        lambda_ce: float = 0.0,
    ):
        super().__init__()
        self.objective = objective
        self.lambda_activity = lambda_activity
        self.lambda_ce = lambda_ce

    def forward(
        self,
        new_positions: list[torch.Tensor],
        current_positions: list[torch.Tensor],
        delta_mids: list[torch.Tensor],
        half_spreads: torch.Tensor,
        ce_logits: list[torch.Tensor] | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            new_positions: list of (B,) new positions per horizon
            current_positions: list of (B,) previous positions per horizon
            delta_mids: list of (B,) raw mid-price changes per horizon
            half_spreads: (B,) half bid-ask spread
            ce_logits: optional list of (B, 3) for CE regularization
            labels: optional (B, H) labels for CE regularization
            class_weights: optional (H, 3) per-horizon class weights

        Returns:
            loss: scalar
            info: dict with trading metrics
        """
        total_loss = torch.tensor(0.0, device=half_spreads.device)
        total_gross = torch.tensor(0.0, device=half_spreads.device)
        total_cost = torch.tensor(0.0, device=half_spreads.device)
        total_trades = torch.tensor(0.0, device=half_spreads.device)
        n_horizons = len(new_positions)

        all_net_returns = []

        for h_idx in range(n_horizons):
            new_pos = new_positions[h_idx]
            prev_pos = current_positions[h_idx]
            delta_mid = delta_mids[h_idx]

            # Position change and trade intensity (soft indicator of whether a trade happened)
            pos_change = new_pos - prev_pos
            smooth_abs_change = torch.sqrt(pos_change ** 2 + 1e-8)
            # Trade weight: ≈1 when position changes, ≈0 when holding.
            # This prevents double-counting h-step returns during hold periods.
            trade_weight = torch.sigmoid(20.0 * (smooth_abs_change - 0.01))

            # PnL only at trade entry: the h-step return "belongs" to the step
            # where the position was established, not to every hold step.
            gross_return = trade_weight * new_pos * delta_mid

            # Transaction cost (only at position changes, naturally weighted by |Δpos|)
            cost = smooth_abs_change * half_spreads.abs()

            net_return = gross_return - cost
            all_net_returns.append(net_return)

            total_gross = total_gross + gross_return.sum()
            total_cost = total_cost + cost.sum()
            total_trades = total_trades + (smooth_abs_change > 0.01).float().sum()

        # Stack all horizon returns for loss computation
        stacked = torch.cat(all_net_returns)

        if self.objective == "sharpe":
            total_loss = -stacked.mean() / stacked.std().clamp(min=1e-4)
        elif self.objective == "sortino":
            downside = stacked.clamp(max=0)
            downside_std = (downside ** 2).mean().sqrt().clamp(min=1e-4)
            total_loss = -stacked.mean() / downside_std
        else:  # pnl
            total_loss = -stacked.mean()

        # Activity penalty: penalize always-hold collapse
        if self.lambda_activity > 0:
            activity_rate = sum(
                (torch.abs(new_positions[h] - current_positions[h]) > 0.01).float().mean()
                for h in range(n_horizons)
            ) / n_horizons
            # Penalize if activity rate drops below 5%
            activity_penalty = F.relu(0.05 - activity_rate)
            total_loss = total_loss + self.lambda_activity * activity_penalty

        # Optional CE regularization
        if self.lambda_ce > 0 and ce_logits is not None and labels is not None:
            ce_total = torch.tensor(0.0, device=half_spreads.device)
            for h_idx in range(n_horizons):
                weight = class_weights[h_idx] if class_weights is not None else None
                ce_h = F.cross_entropy(ce_logits[h_idx], labels[:, h_idx], weight=weight)
                ce_total = ce_total + ce_h
            total_loss = total_loss + self.lambda_ce * ce_total / n_horizons

        info = {
            "total_gross": total_gross.detach(),
            "total_cost": total_cost.detach(),
            "total_net": (total_gross - total_cost).detach(),
            "total_trades": total_trades.item(),
            "mean_net_return": stacked.mean().detach(),
            "std_net_return": stacked.std().detach(),
        }
        return total_loss, info


class CPTFilterLoss(nn.Module):
    """Trade filter loss for CPT: weighted BCE on DP optimal trade/hold labels.

    DP optimal trades are ~6% of timesteps (heavily imbalanced), so the
    positive class weight is computed from the trade rate to balance gradients.
    """

    def __init__(self, pos_weight: float = 16.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(
        self,
        filter_logits: list[torch.Tensor],
        dp_trade_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            filter_logits: [H] of (B,) — raw logits from trade filter (pre-sigmoid)
            dp_trade_labels: (B,) — binary {0=hold, 1=trade} from DP optimal

        Returns:
            loss: scalar BCE
            info: dict with diagnostics
        """
        n_horizons = len(filter_logits)
        total_loss = torch.tensor(0.0, device=dp_trade_labels.device)

        # Trade filter is horizon-independent (DP labels don't vary by horizon),
        # but we average across horizon heads for consistent gradient to each.
        for h_idx in range(n_horizons):
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                filter_logits[h_idx], dp_trade_labels, pos_weight=self.pos_weight,
            )
        loss = total_loss / n_horizons

        # Diagnostics
        with torch.no_grad():
            probs = torch.sigmoid(filter_logits[0])
            pred_trade = (probs > 0.5).float()
            actual_trade = dp_trade_labels
            tp = ((pred_trade == 1) & (actual_trade == 1)).sum().float()
            fp = ((pred_trade == 1) & (actual_trade == 0)).sum().float()
            fn = ((pred_trade == 0) & (actual_trade == 1)).sum().float()
            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            info = {
                "filter_loss": loss.detach(),
                "filter_precision": precision,
                "filter_recall": recall,
                "filter_trade_rate": pred_trade.mean(),
                "filter_mean_prob": probs.mean(),
            }

        return loss, info
