import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


CLASS_NAMES = {
    0: "up",
    1: "stat",
    2: "down",
}


def _to_numpy(array_like: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _resolve_labels(targets: np.ndarray, predictions: np.ndarray | None = None) -> np.ndarray:
    if predictions is None:
        merged = targets
    else:
        merged = np.concatenate([targets, predictions])

    if merged.size == 0:
        return np.array([0, 1, 2], dtype=np.int64)

    merged = merged.astype(np.int64, copy=False)
    if merged.min() >= 0 and merged.max() <= 2:
        return np.array([0, 1, 2], dtype=np.int64)
    return np.unique(merged)


def _format_lift(score: float, baseline: float) -> str:
    if baseline <= 0:
        return "   n/a   "
    lift_pct = ((score / baseline) - 1.0) * 100.0
    return f"{lift_pct:+7.1f}%"


def _format_int_space(value: int) -> str:
    return f"{int(value):,}"


def compute_metrics(targets, predictions) -> dict[str, Any]:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)
    labels = _resolve_labels(targets, predictions)

    conf = confusion_matrix(targets, predictions, labels=labels)
    per_class_precision = precision_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    try:
        mcc = float(matthews_corrcoef(targets, predictions))
    except ValueError:
        mcc = 0.0

    try:
        kappa = float(cohen_kappa_score(targets, predictions, labels=labels))
    except ValueError:
        kappa = 0.0

    return {
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        "mcc": mcc,
        "kappa": kappa,
        "accuracy": float(accuracy_score(targets, predictions)),
        "per_class_f1": {int(lbl): float(val) for lbl, val in zip(labels, per_class_f1)},
        "per_class_precision": {int(lbl): float(val) for lbl, val in zip(labels, per_class_precision)},
        "per_class_recall": {int(lbl): float(val) for lbl, val in zip(labels, per_class_recall)},
        "confusion_matrix": conf,
        "support": conf.sum(axis=1),
        "labels": labels,
    }


def compute_baselines(targets) -> dict[str, Any]:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    labels = _resolve_labels(targets)
    counts = np.array([(targets == lbl).sum() for lbl in labels], dtype=np.int64)
    total = max(int(counts.sum()), 1)
    distribution = counts / total

    majority_idx = int(np.argmax(counts))
    majority_class = int(labels[majority_idx])
    majority_predictions = np.full_like(targets, fill_value=majority_class)

    majority_f1 = float(
        f1_score(
            targets,
            majority_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    random_f1 = 1.0 / float(len(labels)) if len(labels) > 0 else 0.0

    return {
        "majority_class": majority_class,
        "majority_f1": majority_f1,
        "random_f1": random_f1,
        "class_distribution": {int(lbl): float(pct) for lbl, pct in zip(labels, distribution)},
    }


def format_horizon_table(metrics_list, horizons, baselines) -> str:
    header = (
        "Horizon | F1(mac) | F1(wtd) |   MCC   |  Kappa  |   Acc   |     N     | vs Random | vs Majority"
    )
    separator = (
        "--------|---------|---------|---------|---------|---------|-----------|-----------|------------"
    )
    lines = [header, separator]

    for idx, horizon in enumerate(horizons):
        metrics = metrics_list[idx]
        baseline = baselines[idx]
        n_samples = int(np.sum(metrics["support"]))
        vs_random = _format_lift(metrics["macro_f1"], baseline["random_f1"])
        vs_majority = _format_lift(metrics["macro_f1"], baseline["majority_f1"])

        lines.append(
            f"h{horizon:<6}|"
            f" {metrics['macro_f1']:>7.4f} |"
            f" {metrics['weighted_f1']:>7.4f} |"
            f" {metrics['mcc']:>7.4f} |"
            f" {metrics['kappa']:>7.4f} |"
            f" {metrics['accuracy']:>7.4f} |"
            f" {_format_int_space(n_samples):>9} |"
            f" {vs_random:>9} |"
            f" {vs_majority:>10}"
        )

    return "\n".join(lines)


def format_prediction_distribution(targets, predictions) -> str:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)
    labels = _resolve_labels(targets, predictions)

    target_total = max(int(targets.shape[0]), 1)
    pred_total = max(int(predictions.shape[0]), 1)

    lines = [
        "Class  | Actual % | Pred %  | Actual n | Pred n",
        "-------|----------|---------|----------|-------",
    ]

    for label in labels:
        name = CLASS_NAMES.get(int(label), f"class{int(label)}")
        actual_n = int((targets == label).sum())
        pred_n = int((predictions == label).sum())
        lines.append(
            f"{name:<6} |"
            f" {100.0 * actual_n / target_total:>7.2f}% |"
            f" {100.0 * pred_n / pred_total:>6.2f}% |"
            f" {_format_int_space(actual_n):>8} |"
            f" {_format_int_space(pred_n):>6}"
        )

    return "\n".join(lines)


def format_confidence_stats(probas, targets, predictions) -> str:
    probas = _to_numpy(probas)
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)

    if probas.ndim == 2:
        confidence = probas.max(axis=1)
    elif probas.ndim == 1:
        confidence = probas
    else:
        raise ValueError(f"Expected 1D or 2D probabilities, got shape={probas.shape}")

    n = min(int(confidence.shape[0]), int(targets.shape[0]), int(predictions.shape[0]))
    if n == 0:
        return "No confidence stats available (empty inputs)."

    confidence = confidence[:n]
    targets = targets[:n]
    predictions = predictions[:n]

    correct_mask = predictions == targets
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]

    mean_correct = float(correct_conf.mean()) if correct_conf.size > 0 else float("nan")
    mean_incorrect = float(incorrect_conf.mean()) if incorrect_conf.size > 0 else float("nan")
    overconfident_errors = (
        float((incorrect_conf >= 0.80).mean()) if incorrect_conf.size > 0 else float("nan")
    )

    def _fmt(value: float) -> str:
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"

    lines = [
        f"Samples: {_format_int_space(n)}",
        f"Mean confidence (correct):   {_fmt(mean_correct)}",
        f"Mean confidence (incorrect): {_fmt(mean_incorrect)}",
        f"Overconfident errors (>=0.80): {_fmt(overconfident_errors)}",
    ]

    return "\n".join(lines)


def plot_confusion_matrices(metrics_list, horizons, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    n = len(metrics_list)
    n_cols = 2 if n > 1 else 1
    n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows), dpi=140)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, horizon in enumerate(horizons):
        metrics = metrics_list[idx]
        cm = np.asarray(metrics["confusion_matrix"])
        labels = [int(v) for v in np.asarray(metrics["labels"]).tolist()]

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"h{horizon} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = int(cm[row, col])
                color = "white" if value > threshold else "black"
                ax.text(col, row, _format_int_space(value), ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        single_path = os.path.join(save_dir, f"confusion_matrix_h{horizon}.png")
        single_fig, single_ax = plt.subplots(figsize=(6, 5), dpi=140)
        single_im = single_ax.imshow(cm, cmap="Blues")
        single_ax.set_title(f"h{horizon} confusion matrix")
        single_ax.set_xlabel("Predicted")
        single_ax.set_ylabel("Actual")
        single_ax.set_xticks(range(len(labels)))
        single_ax.set_yticks(range(len(labels)))
        single_ax.set_xticklabels(labels)
        single_ax.set_yticklabels(labels)

        single_threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = int(cm[row, col])
                color = "white" if value > single_threshold else "black"
                single_ax.text(col, row, _format_int_space(value), ha="center", va="center", color=color, fontsize=10)
        single_fig.colorbar(single_im, ax=single_ax, fraction=0.046, pad=0.04)
        single_fig.tight_layout()
        single_fig.savefig(single_path, bbox_inches="tight")
        plt.close(single_fig)
        saved_paths.append(single_path)

    for idx in range(len(horizons), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    combined_path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(combined_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Directional trading simulation (Zhang et al., 2019 protocol)
# ---------------------------------------------------------------------------

_POSITION_MAP = np.array([1, 0, -1], dtype=np.float64)  # UP=+1, STAT=0, DOWN=-1


def compute_trading_metrics(
    mid_prices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    logits: np.ndarray | None = None,
    half_spreads: np.ndarray | None = None,
    z_half_spreads: np.ndarray | None = None,
    cost_per_trade: float = 0.0,
    confidence_threshold: float = 0.0,
    min_hold: int = 0,
    segment_boundaries: np.ndarray | None = None,
    use_soft_positions: bool = False,
    hysteresis_entry: float = 0.0,
    hysteresis_exit: float = 0.0,
) -> dict[str, Any]:
    """Simulate a directional trading strategy and compute performance metrics.

    Extends the Zhang et al. (2019) DeepLOB protocol with:
    - Spread-aware transaction costs using actual bid-ask spreads
    - Soft (continuous) positions from logits for DFL models
    - Position persistence with minimum hold period

    Parameters
    ----------
    mid_prices : (N,) array of sequential (z-score normalized) mid-prices.
    predictions : (N,) array with values in {0, 1, 2}.
    probabilities : (N, 3) softmax probabilities, optional.
        Used for confidence thresholding (hard positions only).
    logits : (N, 3) raw model logits, optional.
        Required when use_soft_positions=True.
    half_spreads : (N,) raw (unnormalized) half bid-ask spread, optional.
        Used with z_half_spreads for spread-aware transaction costs.
    z_half_spreads : (N,) z-score normalized half spread, optional.
        Computed from input tensor as (x[:,-1,0] - x[:,-1,2]) / 2.
        Used to estimate the price normalization scale factor.
    cost_per_trade : Legacy cost multiplier (x mean |Δmid|) per unit of
        position change. Only used when spread data is not available.
    confidence_threshold : Minimum max-class probability to act.
        Predictions below this threshold are treated as STATIONARY.
        Not applied when use_soft_positions=True.
    min_hold : Minimum number of steps to hold a position before allowing
        a change. When > 0, positions persist until min_hold steps have
        elapsed AND the new prediction differs. Segment boundaries reset
        the hold counter.
    segment_boundaries : 1-D array of cumulative boundary indices, optional.
        Positions are forced to zero at each boundary (e.g. EPEX product
        boundaries).  Values are exclusive end indices of each segment.
    use_soft_positions : If True and logits provided, use continuous
        positions in [-1, +1] via softmax(logits) @ [+1, 0, -1].
    hysteresis_entry : Minimum max-class probability to ENTER a new position
        (transition from flat, or change direction). Requires probabilities.
    hysteresis_exit : Minimum max-class probability to STAY in current
        position. If confidence drops below this, position goes flat.
        When both > 0, creates a Schmitt-trigger effect: strong signal to
        enter, weaker signal sufficient to hold.
    """
    mid_prices = _to_numpy(mid_prices).astype(np.float64, copy=False).ravel()
    predictions = _to_numpy(predictions).astype(np.int64, copy=False).ravel()
    n = min(len(mid_prices), len(predictions))
    if logits is not None:
        logits = _to_numpy(logits).astype(np.float64, copy=False)
        n = min(n, len(logits))
    if n < 2:
        return _empty_trading_metrics()

    mid_prices = mid_prices[:n]
    predictions = predictions[:n]
    if logits is not None:
        logits = logits[:n]
    if half_spreads is not None:
        half_spreads = _to_numpy(half_spreads).astype(np.float64, copy=False).ravel()[:n]
    if z_half_spreads is not None:
        z_half_spreads = _to_numpy(z_half_spreads).astype(np.float64, copy=False).ravel()[:n]

    # --- build raw signal series ---
    if use_soft_positions and logits is not None:
        # Continuous positions from softmax(logits) @ [+1, 0, -1]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        raw_positions = probs @ _POSITION_MAP  # continuous in [-1, +1]
    else:
        # Hard discrete positions from class predictions
        if probabilities is not None and confidence_threshold > 0.0:
            probabilities = _to_numpy(probabilities).astype(np.float64, copy=False)
            if probabilities.ndim == 2:
                max_conf = probabilities[:n].max(axis=1)
            else:
                max_conf = probabilities[:n]
            low_conf = max_conf < confidence_threshold
            predictions = predictions.copy()
            predictions[low_conf] = 1  # treat as STATIONARY
        raw_positions = _POSITION_MAP[predictions]  # (N,)

    # --- apply hysteresis (Schmitt trigger) ---
    if (hysteresis_entry > 0 or hysteresis_exit > 0) and probabilities is not None:
        probs_arr = _to_numpy(probabilities).astype(np.float64, copy=False)
        if probs_arr.ndim == 2:
            max_conf = probs_arr[:n].max(axis=1)
        else:
            max_conf = probs_arr[:n]
        boundary_set_h = set()
        if segment_boundaries is not None:
            boundary_set_h = set(np.asarray(segment_boundaries, dtype=np.int64).tolist())
        hyst_positions = np.empty(n, dtype=np.float64)
        current_pos = 0.0
        for t in range(n):
            if t in boundary_set_h:
                current_pos = 0.0
            target = raw_positions[t]
            if target != current_pos:
                # Entering or changing direction: need high confidence
                if max_conf[t] >= hysteresis_entry:
                    current_pos = target
                elif current_pos != 0.0 and max_conf[t] < hysteresis_exit:
                    # Confidence too low to even hold — go flat
                    current_pos = 0.0
            else:
                # Same direction: check if confidence enough to hold
                if current_pos != 0.0 and max_conf[t] < hysteresis_exit:
                    current_pos = 0.0
            hyst_positions[t] = current_pos
        raw_positions = hyst_positions

    # --- apply min_hold persistence ---
    if min_hold > 0:
        boundary_set = set()
        if segment_boundaries is not None:
            boundary_set = set(np.asarray(segment_boundaries, dtype=np.int64).tolist())
        positions = np.empty(n, dtype=np.float64)
        current_pos = 0.0
        steps_held = min_hold  # allow trading on first step
        for t in range(n):
            if t in boundary_set:
                # Segment boundary: force flat, reset hold counter
                current_pos = 0.0
                steps_held = min_hold
            target = raw_positions[t]
            if target != current_pos and steps_held >= min_hold:
                current_pos = target
                steps_held = 0
            positions[t] = current_pos
            steps_held += 1
    else:
        positions = raw_positions

    # --- force close at segment boundaries ---
    if segment_boundaries is not None:
        segment_boundaries = np.asarray(segment_boundaries, dtype=np.int64)
        for b in segment_boundaries:
            if 0 < b < n:
                positions[b - 1] = 0.0

    # force close at the very end
    positions[-1] = 0.0

    # --- step returns ---
    price_changes = np.diff(mid_prices)  # (N-1,)
    mean_abs_price_change = float(np.mean(np.abs(price_changes)))
    step_positions = positions[:-1]  # position held during [t, t+1)
    gross_returns = step_positions * price_changes  # (N-1,)

    # --- transaction costs ---
    # For soft positions: threshold for detecting a meaningful position/change
    _POS_EPS = 0.01

    has_spread_costs = False
    std_price = None
    total_spread_cost = 0.0

    if half_spreads is not None and z_half_spreads is not None:
        # Spread-aware costs: convert raw half_spread to z-score units
        valid = np.abs(z_half_spreads) > 1e-10
        if valid.sum() > 10:
            std_price = float(np.median(half_spreads[valid] / z_half_spreads[valid]))
        else:
            std_price = None

        if std_price is not None and std_price > 1e-10:
            has_spread_costs = True
            half_spreads_z = half_spreads / std_price

            # Cost = |position_change| × half_spread_z at each step
            first_change = np.abs(positions[0])
            pos_changes = np.abs(np.diff(positions))  # (N-1,)

            costs = np.empty(n - 1, dtype=np.float64)
            costs[0] = first_change * half_spreads_z[0]
            costs[1:] = pos_changes[:-1] * half_spreads_z[1:n - 1]
            # Closing cost (going to positions[-1]=0)
            if n > 2:
                costs[-1] += pos_changes[-1] * half_spreads_z[n - 1]

            total_spread_cost = float(costs.sum())
        else:
            # Fallback: z_half_spreads only (no raw half_spreads or bad scale)
            first_change = np.abs(positions[0])
            pos_changes = np.abs(np.diff(positions))
            z_hs = np.abs(z_half_spreads)

            has_spread_costs = True
            costs = np.empty(n - 1, dtype=np.float64)
            costs[0] = first_change * z_hs[0]
            costs[1:] = pos_changes[:-1] * z_hs[1:n - 1]
            if n > 2:
                costs[-1] += pos_changes[-1] * z_hs[n - 1]
            total_spread_cost = float(costs.sum())
    elif z_half_spreads is not None:
        # Only z-scored spread available (e.g., CE runs without DFL data)
        has_spread_costs = True
        z_hs = np.abs(z_half_spreads)
        first_change = np.abs(positions[0])
        pos_changes = np.abs(np.diff(positions))

        costs = np.empty(n - 1, dtype=np.float64)
        costs[0] = first_change * z_hs[0]
        costs[1:] = pos_changes[:-1] * z_hs[1:n - 1]
        if n > 2:
            costs[-1] += pos_changes[-1] * z_hs[n - 1]
        total_spread_cost = float(costs.sum())
    elif cost_per_trade > 0.0:
        # Legacy cost model (proportional to mean |Δmid|)
        effective_cost = cost_per_trade * mean_abs_price_change
        first_change = np.abs(positions[0])
        pos_changes = np.abs(np.diff(positions))
        costs = np.empty(n - 1, dtype=np.float64)
        costs[0] = effective_cost * first_change
        costs[1:] = effective_cost * pos_changes[:-1]
        if n > 2:
            costs[-1] += effective_cost * pos_changes[-1]
    else:
        costs = np.zeros(n - 1, dtype=np.float64)

    # Zero out gross returns at boundary crossings (artificial price change
    # spanning two different products) but keep costs (closing cost is real).
    if segment_boundaries is not None:
        for b in segment_boundaries:
            idx = b - 1
            if 0 <= idx < len(gross_returns):
                gross_returns[idx] = 0.0

    net_returns = gross_returns - costs

    # --- cumulative PnL ---
    cumulative_pnl = np.cumsum(net_returns)
    total_pnl = float(cumulative_pnl[-1]) if len(cumulative_pnl) > 0 else 0.0

    # --- active steps (non-zero position) ---
    if use_soft_positions:
        active_mask = np.abs(step_positions) > _POS_EPS
    else:
        active_mask = step_positions != 0
    n_active = int(active_mask.sum())
    active_returns = net_returns[active_mask]

    # --- Per-step Sharpe ratio (signal-to-noise of single-step return) ---
    std_r = float(np.std(net_returns))
    mean_r = float(np.mean(net_returns))
    sharpe = (mean_r / std_r) if std_r > 1e-12 else 0.0

    # --- Per-step Sortino (downside deviation over all N returns) ---
    downside_returns = np.minimum(net_returns, 0.0)
    downside_std = float(np.sqrt(np.mean(downside_returns**2)))
    sortino = (mean_r / downside_std) if downside_std > 1e-12 else 0.0

    # --- Max drawdown ---
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = cumulative_pnl - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # MaxDD% via equity curve with notional starting capital = 1.0
    equity = 1.0 + cumulative_pnl
    if np.any(equity <= 0):
        max_dd_pct = -100.0
    elif len(equity) > 0:
        running_max_eq = np.maximum.accumulate(equity)
        drawdowns_pct = (equity - running_max_eq) / running_max_eq
        max_dd_pct = float(drawdowns_pct.min()) * 100.0
    else:
        max_dd_pct = 0.0

    # --- Calmar ---
    calmar = (total_pnl / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    # --- Win rate ---
    if n_active > 0:
        win_rate = float((active_returns > 0).sum()) / n_active
    else:
        win_rate = 0.0

    # --- Profit factor ---
    gross_profit = float(net_returns[net_returns > 0].sum())
    gross_loss = float(np.abs(net_returns[net_returns < 0].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else float("inf") if gross_profit > 0 else 0.0

    # --- Trade count (number of position changes) ---
    if use_soft_positions:
        pos_changes_all = np.abs(np.diff(positions))
        n_trades = int((pos_changes_all > _POS_EPS).sum())
    else:
        position_changes = np.abs(np.diff(positions))
        n_trades = int(positions[0] != 0) + int((position_changes > 0).sum())

    # --- Average hold duration (steps between position changes) ---
    change_indices = np.where(position_changes > (_POS_EPS if use_soft_positions else 0))[0]
    if len(change_indices) > 1:
        hold_durations = np.diff(change_indices)
        avg_hold_duration = float(hold_durations.mean())
    elif n_trades > 0:
        avg_hold_duration = float(n - 1)  # single trade held for entire period
    else:
        avg_hold_duration = 0.0

    # --- Exposure ---
    exposure_pct = (n_active / len(step_positions) * 100.0) if len(step_positions) > 0 else 0.0

    # --- Statistical significance (t-test: H0 mean return = 0) ---
    if len(net_returns) > 1 and std_r > 1e-12:
        t_stat, p_value = scipy_stats.ttest_1samp(net_returns, 0.0)
        t_stat = float(t_stat)
        p_value = float(p_value)
    else:
        t_stat = 0.0
        p_value = 1.0

    return {
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_hold_duration": avg_hold_duration,
        "exposure_pct": exposure_pct,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_steps": len(net_returns),
        "mean_abs_price_change": mean_abs_price_change,
        "returns_series": net_returns,
        "cumulative_pnl": cumulative_pnl,
        "positions": positions,
        "has_spread_costs": has_spread_costs,
        "total_spread_cost": total_spread_cost,
        "std_price": std_price,
    }


def compute_dp_optimal(
    z_mid: np.ndarray,
    z_half_spread: np.ndarray,
    segment_boundaries: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the position sequence {-1, 0, +1} that maximizes net PnL via DP.

    This is the true theoretical ceiling — no strategy can beat it.
    Runs in O(9N) time where N is the number of timesteps.

    Returns dict with total_pnl, n_trades, positions.
    """

    def _dp_segment(mid: np.ndarray, hs: np.ndarray) -> tuple[float, int, np.ndarray]:
        n = len(mid)
        if n < 2:
            return 0.0, 0, np.zeros(n)
        returns = np.diff(mid)  # length n-1
        pos_vals = np.array([-1.0, 0.0, 1.0])
        # dp[pos_idx] = best cumulative PnL ending at this step with this position
        dp = np.full(3, -np.inf)
        dp[1] = 0.0  # start flat
        backtrack = np.empty((n - 1, 3), dtype=np.int8)
        for t in range(n - 1):
            new_dp = np.full(3, -np.inf)
            for j in range(3):  # new position
                best_val = -np.inf
                best_prev = 0
                for i in range(3):  # previous position
                    cost = abs(pos_vals[j] - pos_vals[i]) * hs[t]
                    val = dp[i] + pos_vals[j] * returns[t] - cost
                    if val > best_val:
                        best_val = val
                        best_prev = i
                new_dp[j] = best_val
                backtrack[t, j] = best_prev
            dp = new_dp
        # Force close: best ending position accounting for closing cost
        best_end = -1
        best_pnl = -np.inf
        for j in range(3):
            close_cost = abs(pos_vals[j]) * hs[-1]
            val = dp[j] - close_cost
            if val > best_pnl:
                best_pnl = val
                best_end = j
        # Backtrack: path[0]=initial flat, path[t+1]=position earning returns[t]
        path = [best_end]
        for t in range(n - 2, -1, -1):
            path.append(backtrack[t, path[-1]])
        path.reverse()
        # Align: positions[t] = position earning returns[t], positions[-1] = 0
        positions = np.zeros(n)
        for t in range(n - 1):
            positions[t] = pos_vals[path[t + 1]]
        n_trades = int(np.abs(positions[0]) > 0) + int(np.sum(np.abs(np.diff(positions)) > 0))
        return best_pnl, n_trades, positions

    z_mid = np.asarray(z_mid, dtype=np.float64)
    z_half_spread = np.asarray(z_half_spread, dtype=np.float64)

    if segment_boundaries is not None:
        boundaries = np.asarray(segment_boundaries, dtype=np.int64)
        segments = []
        prev = 0
        for b in boundaries:
            if b > prev:
                segments.append((prev, b))
            prev = b
        if prev < len(z_mid):
            segments.append((prev, len(z_mid)))
    else:
        segments = [(0, len(z_mid))]

    total_pnl = 0.0
    total_trades = 0
    all_positions = np.zeros(len(z_mid))
    for start, end in segments:
        pnl, trades, pos = _dp_segment(z_mid[start:end], z_half_spread[start:end])
        total_pnl += pnl
        total_trades += trades
        all_positions[start:end] = pos

    return {
        "total_pnl": total_pnl,
        "n_trades": total_trades,
        "positions": all_positions,
    }


def _empty_trading_metrics() -> dict[str, Any]:
    return {
        "total_pnl": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "n_trades": 0,
        "avg_hold_duration": 0.0,
        "exposure_pct": 0.0,
        "t_stat": 0.0,
        "p_value": 1.0,
        "n_steps": 0,
        "mean_abs_price_change": 0.0,
        "returns_series": np.array([], dtype=np.float64),
        "cumulative_pnl": np.array([], dtype=np.float64),
        "positions": np.array([], dtype=np.float64),
        "has_spread_costs": False,
        "total_spread_cost": 0.0,
        "std_price": None,
    }


def format_trading_table(trading_metrics_list: list[dict], horizons: list[int]) -> str:
    header = (
        "Horizon | PnL(norm) |   Sharpe/step |  Sortino/step | MaxDD%  | WinRate | ProfitF | Trades | AvgHold | Exposure |  p-value"
    )
    separator = (
        "--------|-----------|--------------|--------------|---------|---------|---------|--------|---------|----------|----------"
    )
    lines = [header, separator]

    for idx, horizon in enumerate(horizons):
        m = trading_metrics_list[idx]
        pf = f"{m['profit_factor']:>7.2f}" if m["profit_factor"] != float("inf") else "    inf"
        lines.append(
            f"h{horizon:<6}|"
            f" {m['total_pnl']:>9.4f} |"
            f" {m['sharpe']:>12.2e} |"
            f" {m['sortino']:>12.2e} |"
            f" {m['max_drawdown_pct']:>6.1f}% |"
            f" {m['win_rate'] * 100:>6.1f}% |"
            f" {pf} |"
            f" {_format_int_space(m['n_trades']):>6} |"
            f" {m['avg_hold_duration']:>7.1f} |"
            f" {m['exposure_pct']:>7.1f}% |"
            f" {m['p_value']:>9.4f}"
        )

    return "\n".join(lines)
