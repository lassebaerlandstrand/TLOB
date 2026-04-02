"""Directional trading simulation evaluation script.

Loads saved test outputs (mid-prices, predictions, probabilities) from a
checkpoint directory and runs the Zhang et al. (2019) directional trading
protocol with optional cost sensitivity and confidence threshold sweeps.

Usage
-----
    # Single or multi-horizon (auto-detected)
    python evaluate_trading.py --checkpoint_dir data/checkpoints/TLOB/BTC_seq_size_128_horizon_10_seed_1_multi_horizon

    # With cost sensitivity and confidence sweep
    python evaluate_trading.py --checkpoint_dir <path> \
        --costs 0.0 0.5 1.0 2.0 5.0 \
        --confidence_thresholds 0.0 0.5 0.7 0.9

Parameters
----------
    Cost               Transaction cost as a multiple of the mean absolute
                       per-step price change (mean |Δmid|). Cost=0 is free,
                       cost=1.0 means each trade costs one average price
                       movement. Auto-scales across datasets: cost=1.0 means
                       the same thing for BTC (250ms) and Battery (10s+).
                       Charged per unit of position change (long->short = 2).

    Confidence         Minimum softmax probability to act on a prediction.
    threshold          If max(p_up, p_stat, p_down) < threshold, the
                       prediction is overridden to STATIONARY (flat).
                       Higher thresholds = fewer but more confident trades.
                       At 0.0 (default), all predictions are acted on.

Output columns
--------------
    PnL(norm)      Cumulative profit/loss in z-score normalized price units
                   over the entire test period. Not dollars — measures
                   prediction quality. Scale-invariant across datasets.

    Sharpe/step    Per-step Sharpe ratio = mean(step_return) / std(step_return).
                   Signal-to-noise of a single step's return. Small number
                   (e.g. 0.13) but directly comparable across datasets
                   regardless of number of steps or sampling frequency.

    Sortino/step   Like Sharpe but only penalizes downside volatility.
                   Always >= Sharpe for profitable strategies.

    MaxDD%         Maximum drawdown as % of equity (starting capital = 1.0).
                   Largest peak-to-trough decline. -100% means the strategy
                   lost all notional capital at some point (even if it recovered).

    WinRate        % of active steps (non-zero position) with positive return.
                   Can be low (e.g. 19%) yet profitable if winning steps are
                   much larger than losing ones (see ProfitF). Low win rates
                   are typical at high sampling frequencies (250ms) where most
                   individual steps are noise.

    ProfitF        Profit factor = sum(positive returns) / |sum(negative returns)|.
                   >1 = profitable, 1 = breakeven, <1 = losing. A ProfitF of
                   5.0 means $5 earned for every $1 lost across all steps.

    Trades         Number of position changes (long->short counts as 1 trade).

    Exposure       % of steps where the model holds a non-zero position
                   (long or short). Higher exposure = more active strategy.

    p-value        Two-sided t-test: "could this mean return be zero?"
                   p < 0.05 = statistically significant profit (or loss).
                   Near-zero p-values are normal with hundreds of thousands
                   of steps — even tiny signals are significant at that scale.
                   Only becomes non-trivial when very few trades are made
                   (e.g. high confidence thresholds filtering out most signals).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from utils.metrics import compute_trading_metrics, format_trading_table

HORIZONS = [10, 20, 50, 100]


def _detect_multi_horizon(checkpoint_dir: str) -> bool:
    return os.path.exists(os.path.join(checkpoint_dir, "predictions_h10.npy"))


def _load_arrays(checkpoint_dir: str, multi_horizon: bool) -> dict:
    """Load saved test arrays from checkpoint directory."""
    data = {}

    mid_prices_path = os.path.join(checkpoint_dir, "mid_prices.npy")
    if not os.path.exists(mid_prices_path):
        print(f"Error: {mid_prices_path} not found.")
        print("Re-run training/testing to generate the required files.")
        sys.exit(1)

    data["mid_prices"] = np.load(mid_prices_path)

    boundaries_path = os.path.join(checkpoint_dir, "product_boundaries.npy")
    data["segment_boundaries"] = np.load(boundaries_path) if os.path.exists(boundaries_path) else None

    if multi_horizon:
        data["horizons"] = []
        for h in HORIZONS:
            h_data = {}
            pred_path = os.path.join(checkpoint_dir, f"predictions_h{h}.npy")
            if not os.path.exists(pred_path):
                continue
            h_data["predictions"] = np.load(pred_path)

            target_path = os.path.join(checkpoint_dir, f"targets_h{h}.npy")
            h_data["targets"] = np.load(target_path) if os.path.exists(target_path) else None

            proba_path = os.path.join(checkpoint_dir, f"probabilities_h{h}.npy")
            h_data["probabilities"] = np.load(proba_path) if os.path.exists(proba_path) else None

            data["horizons"].append((h, h_data))
    else:
        data["predictions"] = np.load(os.path.join(checkpoint_dir, "predictions.npy"))

        target_path = os.path.join(checkpoint_dir, "targets.npy")
        data["targets"] = np.load(target_path) if os.path.exists(target_path) else None

        proba_path = os.path.join(checkpoint_dir, "probabilities.npy")
        data["probabilities"] = np.load(proba_path) if os.path.exists(proba_path) else None

    return data


def _run_evaluation(
    data: dict,
    multi_horizon: bool,
    costs: list[float],
    confidence_thresholds: list[float],
) -> dict:
    """Run trading simulation for all parameter combinations."""
    results = {}
    mid_prices = data["mid_prices"]
    segment_boundaries = data["segment_boundaries"]

    if multi_horizon:
        for cost in costs:
            cost_key = f"cost_{cost}"
            results[cost_key] = {}

            for conf_thresh in confidence_thresholds:
                conf_key = f"conf_{conf_thresh}"
                results[cost_key][conf_key] = {}
                metrics_list = []
                horizons_found = []

                for h, h_data in data["horizons"]:
                    tm = compute_trading_metrics(
                        mid_prices,
                        h_data["predictions"],
                        probabilities=h_data.get("probabilities"),
                        cost_per_trade=cost,
                        confidence_threshold=conf_thresh,
                        segment_boundaries=segment_boundaries,
                    )
                    # Store serializable metrics (drop arrays)
                    results[cost_key][conf_key][f"h{h}"] = _serializable(tm)
                    metrics_list.append(tm)
                    horizons_found.append(h)

                results[cost_key][conf_key]["_metrics_list"] = metrics_list
                results[cost_key][conf_key]["_horizons"] = horizons_found
    else:
        for cost in costs:
            cost_key = f"cost_{cost}"
            results[cost_key] = {}

            for conf_thresh in confidence_thresholds:
                conf_key = f"conf_{conf_thresh}"
                tm = compute_trading_metrics(
                    mid_prices,
                    data["predictions"],
                    probabilities=data.get("probabilities"),
                    cost_per_trade=cost,
                    confidence_threshold=conf_thresh,
                    segment_boundaries=segment_boundaries,
                )
                results[cost_key][conf_key] = _serializable(tm)
                results[cost_key][conf_key]["_metrics"] = tm

    return results


def _serializable(tm: dict) -> dict:
    """Return a JSON-serializable version of trading metrics."""
    return {k: v for k, v in tm.items() if not isinstance(v, np.ndarray)}


def _print_results(results: dict, multi_horizon: bool, costs: list[float], confidence_thresholds: list[float]):
    """Print formatted results to terminal."""
    for cost in costs:
        cost_key = f"cost_{cost}"
        for conf_thresh in confidence_thresholds:
            conf_key = f"conf_{conf_thresh}"

            header = f"Cost={cost}"
            if conf_thresh > 0:
                header += f"  Confidence threshold={conf_thresh}"
            print(f"\n{'=' * 80}")
            print(f"  {header}")
            print(f"{'=' * 80}")

            if multi_horizon:
                entry = results[cost_key][conf_key]
                metrics_list = entry["_metrics_list"]
                horizons = entry["_horizons"]
                print(format_trading_table(metrics_list, horizons))
            else:
                tm = results[cost_key][conf_key]["_metrics"]
                pf = "inf" if tm["profit_factor"] == float("inf") else f"{tm['profit_factor']:.2f}"
                print(
                    f"PnL(norm)={tm['total_pnl']:.4f}  Sharpe/step={tm['sharpe']:.2e}  "
                    f"Sortino/step={tm['sortino']:.2e}  MaxDD={tm['max_drawdown_pct']:.1f}%  "
                    f"WinRate={tm['win_rate'] * 100:.1f}%  ProfitF={pf}  "
                    f"Trades={tm['n_trades']}  Exposure={tm['exposure_pct']:.1f}%  "
                    f"p-value={tm['p_value']:.4f}"
                )


def _plot_cumulative_pnl(results: dict, data: dict, multi_horizon: bool, save_dir: str):
    """Plot cumulative PnL curves (zero-cost, zero-confidence-threshold baseline)."""
    baseline_key = "cost_0.0"
    conf_key = "conf_0.0"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

    if multi_horizon:
        entry = results[baseline_key][conf_key]
        for tm, h in zip(entry["_metrics_list"], entry["_horizons"]):
            cum_pnl = tm["cumulative_pnl"]
            if len(cum_pnl) > 0:
                ax.plot(cum_pnl, label=f"h{h} (Sharpe={tm['sharpe']:.2f})", linewidth=1.2)
    else:
        tm = results[baseline_key][conf_key]["_metrics"]
        cum_pnl = tm["cumulative_pnl"]
        if len(cum_pnl) > 0:
            ax.plot(cum_pnl, label=f"Model (Sharpe={tm['sharpe']:.2f})", linewidth=1.2)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)

    # Mark product boundaries if they exist
    boundaries = data.get("segment_boundaries")
    if boundaries is not None and len(boundaries) > 0:
        for b in boundaries:
            if 0 < b < ax.get_xlim()[1]:
                ax.axvline(x=b, color="red", linestyle=":", linewidth=0.4, alpha=0.5)

    ax.set_xlabel("Test step")
    ax.set_ylabel("Cumulative PnL (normalized)")
    ax.set_title("Directional Trading Simulation — Cumulative PnL")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "cumulative_pnl.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_confidence_sweep(
    results: dict,
    multi_horizon: bool,
    costs: list[float],
    confidence_thresholds: list[float],
    save_dir: str,
):
    """Plot Sharpe ratio vs confidence threshold."""
    if len(confidence_thresholds) < 2:
        return

    # Use zero-cost for the sweep plot
    cost_key = "cost_0.0"
    if cost_key not in results:
        cost_key = f"cost_{costs[0]}"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    if multi_horizon:
        # Get horizons from the first confidence threshold entry
        first_conf = f"conf_{confidence_thresholds[0]}"
        horizons = results[cost_key][first_conf]["_horizons"]

        for h_idx, h in enumerate(horizons):
            sharpes = []
            for conf in confidence_thresholds:
                conf_key = f"conf_{conf}"
                entry = results[cost_key][conf_key]
                sharpes.append(entry["_metrics_list"][h_idx]["sharpe"])
            ax.plot(confidence_thresholds, sharpes, marker="o", label=f"h{h}", linewidth=1.5)
    else:
        sharpes = []
        for conf in confidence_thresholds:
            conf_key = f"conf_{conf}"
            sharpes.append(results[cost_key][conf_key]["_metrics"]["sharpe"])
        ax.plot(confidence_thresholds, sharpes, marker="o", label="Model", linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Sharpe ratio (per-step)")
    ax.set_title("Sharpe vs Confidence Threshold (zero cost)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "confidence_sweep.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_cost_sensitivity(
    results: dict,
    multi_horizon: bool,
    costs: list[float],
    save_dir: str,
):
    """Plot Sharpe ratio vs transaction cost."""
    if len(costs) < 2:
        return

    conf_key = "conf_0.0"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    if multi_horizon:
        horizons = results[f"cost_{costs[0]}"][conf_key]["_horizons"]
        for h_idx, h in enumerate(horizons):
            sharpes = []
            for cost in costs:
                cost_key = f"cost_{cost}"
                entry = results[cost_key][conf_key]
                sharpes.append(entry["_metrics_list"][h_idx]["sharpe"])
            ax.plot(costs, sharpes, marker="s", label=f"h{h}", linewidth=1.5)
    else:
        sharpes = []
        for cost in costs:
            cost_key = f"cost_{cost}"
            sharpes.append(results[cost_key][conf_key]["_metrics"]["sharpe"])
        ax.plot(costs, sharpes, marker="s", label="Model", linewidth=1.5)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Transaction cost multiplier (× mean |Δmid|)")
    ax.set_ylabel("Sharpe ratio (per-step)")
    ax.set_title("Sharpe vs Transaction Cost")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(save_dir, "cost_sensitivity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _save_json_results(results: dict, save_dir: str):
    """Save JSON-serializable results (no numpy arrays)."""
    clean = {}
    for cost_key, cost_val in results.items():
        clean[cost_key] = {}
        if isinstance(cost_val, dict):
            for conf_key, conf_val in cost_val.items():
                if conf_key.startswith("_"):
                    continue
                if isinstance(conf_val, dict):
                    clean[cost_key][conf_key] = {k: v for k, v in conf_val.items() if not k.startswith("_")}
                else:
                    clean[cost_key][conf_key] = conf_val

    path = os.path.join(save_dir, "trading_results.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Directional trading simulation evaluation (Zhang et al., 2019 protocol)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing saved test arrays",
    )
    parser.add_argument(
        "--costs",
        type=float,
        nargs="+",
        default=[0.0],
        help="Cost as multiple of mean |Δmid| per trade (0=free, 1=one avg price change per trade)",
    )
    parser.add_argument(
        "--confidence_thresholds",
        type=float,
        nargs="+",
        default=[0.0],
        help="Confidence thresholds to evaluate (default: 0.0)",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: {checkpoint_dir} is not a directory")
        sys.exit(1)

    # Ensure 0.0 is always in the sweep lists (baseline)
    costs = sorted(set([0.0] + args.costs))
    confidence_thresholds = sorted(set([0.0] + args.confidence_thresholds))

    multi_horizon = _detect_multi_horizon(checkpoint_dir)
    mode_str = "multi-horizon" if multi_horizon else "single-horizon"
    print(f"Detected {mode_str} mode")

    if multi_horizon:
        print(f"Loading per-horizon arrays for horizons: {HORIZONS}")
    print(f"Cost levels: {costs}")
    print(f"Confidence thresholds: {confidence_thresholds}")

    data = _load_arrays(checkpoint_dir, multi_horizon)
    print(f"Loaded {len(data['mid_prices']):,} mid-price steps")
    mid = data["mid_prices"]
    if len(mid) > 1:
        mean_abs_change = float(np.mean(np.abs(np.diff(mid))))
        print(f"Mean |Δmid| = {mean_abs_change:.6f} (cost multiplier base)")
    if data["segment_boundaries"] is not None:
        print(f"Loaded {len(data['segment_boundaries'])} product boundaries (EPEX per-product mode)")

    results = _run_evaluation(data, multi_horizon, costs, confidence_thresholds)
    _print_results(results, multi_horizon, costs, confidence_thresholds)

    print(f"\nGenerating plots...")
    _plot_cumulative_pnl(results, data, multi_horizon, checkpoint_dir)
    _plot_confidence_sweep(results, multi_horizon, costs, confidence_thresholds, checkpoint_dir)
    _plot_cost_sensitivity(results, multi_horizon, costs, checkpoint_dir)
    _save_json_results(results, checkpoint_dir)

    print(f"\nCheckpoint dir: {checkpoint_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
