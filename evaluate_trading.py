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

from utils.metrics import compute_dp_optimal, compute_trading_metrics, format_trading_table

HORIZONS = [10, 20, 50, 100]
SMA_WINDOWS = [10, 20]
PF_MIN_HOLDS = [0, 10, 20, 30, 50, 100]


def _detect_std_price(checkpoint_dir: str, data: dict) -> tuple[float | None, str]:
    """Detect dataset and compute std_price for dollar/EUR conversion.

    Returns (std_price, currency_symbol).
    """
    has_boundaries = data.get("segment_boundaries") is not None

    if has_boundaries:
        # Battery (EPEX) — compute from raw/z spread ratio
        raw_hs = data.get("half_spreads")
        z_hs = data.get("z_half_spreads")
        if raw_hs is not None and z_hs is not None:
            valid = np.abs(z_hs) > 1e-6
            if valid.sum() > 10:
                return float(np.median(raw_hs[valid] / z_hs[valid])), "EUR"
        return None, "EUR"
    else:
        # BTC — try loading norm_stats.npz from data/BTC/
        for path in ["data/BTC/norm_stats.npz"]:
            if os.path.exists(path):
                stats = np.load(path)
                if "std_prices" in stats:
                    return float(stats["std_prices"]), "$"
        return None, "$"


def _compute_baselines(
    mid_prices: np.ndarray,
    targets: np.ndarray | None,
    half_spreads: np.ndarray | None,
    z_half_spreads: np.ndarray | None,
    segment_boundaries: np.ndarray | None,
) -> list[tuple[str, dict]]:
    """Compute reference baselines: Buy & Hold, SMA, Perfect Foresight + min_hold."""
    baselines = []
    n = len(mid_prices)

    # DP optimal (true ceiling — horizon-independent)
    if z_half_spreads is not None:
        dp_result = compute_dp_optimal(mid_prices, z_half_spreads, segment_boundaries)
        dp_pos = dp_result["positions"]
        # Convert positions {-1, 0, +1} to predictions {2, 1, 0}
        dp_preds = np.ones(n, dtype=np.int64)
        dp_preds[dp_pos > 0.5] = 0   # up
        dp_preds[dp_pos < -0.5] = 2  # down
        dp_tm = compute_trading_metrics(
            mid_prices, dp_preds,
            z_half_spreads=z_half_spreads,
            segment_boundaries=segment_boundaries,
        )
        baselines.append(("DP optimal", dp_tm))

    # Buy & Hold (always long = always predict UP)
    bnh_preds = np.zeros(n, dtype=np.int64)  # label 0 = up → position +1
    bnh = compute_trading_metrics(
        mid_prices, bnh_preds,
        half_spreads=half_spreads, z_half_spreads=z_half_spreads,
        segment_boundaries=segment_boundaries,
    )
    baselines.append(("Buy & Hold", bnh))

    # SMA crossover baselines
    for w in SMA_WINDOWS:
        if n < w + 10:
            continue
        sma = np.convolve(mid_prices, np.ones(w) / w, mode="valid")
        offset = w - 1
        sma_preds = np.ones(n, dtype=np.int64)  # default stationary
        for t in range(offset, n):
            if mid_prices[t] > sma[t - offset]:
                sma_preds[t] = 0  # up → long
            elif mid_prices[t] < sma[t - offset]:
                sma_preds[t] = 2  # down → short
        sma_tm = compute_trading_metrics(
            mid_prices, sma_preds,
            half_spreads=half_spreads, z_half_spreads=z_half_spreads,
            segment_boundaries=segment_boundaries,
        )
        baselines.append((f"SMA-{w}", sma_tm))

    # Perfect foresight + optimal min_hold (requires targets)
    if targets is not None:
        best_pnl = -float("inf")
        best_mh = 0
        best_tm = None
        for mh in PF_MIN_HOLDS:
            pf_tm = compute_trading_metrics(
                mid_prices, targets,
                half_spreads=half_spreads, z_half_spreads=z_half_spreads,
                min_hold=mh,
                segment_boundaries=segment_boundaries,
            )
            if pf_tm["total_pnl"] > best_pnl:
                best_pnl = pf_tm["total_pnl"]
                best_mh = mh
                best_tm = pf_tm
        if best_tm is not None:
            baselines.append((f"PF+mh={best_mh}", best_tm))
        # Also show PF without min_hold if different
        if best_mh != 0:
            pf_raw = compute_trading_metrics(
                mid_prices, targets,
                half_spreads=half_spreads, z_half_spreads=z_half_spreads,
                segment_boundaries=segment_boundaries,
            )
            baselines.append(("PF (no mh)", pf_raw))

    return baselines


def _print_baselines(
    baselines_per_h: dict[int, list[tuple[str, dict]]],
    horizons: list[int],
    std_price: float | None,
    currency: str,
):
    """Print baseline reference table."""
    print(f"\n{'=' * 100}")
    print(f"  Reference Baselines")
    if std_price is not None:
        print(f"  (1 z-unit = {currency}{std_price:.2f})")
    print(f"{'=' * 100}")

    header = (
        f"{'Horizon':<8} | {'Strategy':<16} | {'PnL(z)':>10} |"
    )
    if std_price is not None:
        header += f" {'PnL('+currency+')':>12} |"
    header += f" {'Sharpe':>10} | {'Trades':>7} | {'AvgHold':>8} | {'Exposure':>9}"
    print(header)
    print("-" * len(header))

    for h in horizons:
        if h not in baselines_per_h:
            continue
        for name, tm in baselines_per_h[h]:
            pf = f"{tm['profit_factor']:.2f}" if tm["profit_factor"] != float("inf") else "inf"
            line = (
                f"h{h:<7} | {name:<16} | {tm['total_pnl']:>10.4f} |"
            )
            if std_price is not None:
                line += f" {tm['total_pnl'] * std_price:>12.2f} |"
            line += (
                f" {tm['sharpe']:>10.2e} |"
                f" {tm['n_trades']:>7} |"
                f" {tm['avg_hold_duration']:>8.1f} |"
                f" {tm['exposure_pct']:>8.1f}%"
            )
            print(line)
        print("-" * len(header))


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

    # Load spread data for spread-aware evaluation
    for name in ["half_spreads", "z_half_spreads"]:
        path = os.path.join(checkpoint_dir, f"{name}.npy")
        data[name] = np.load(path) if os.path.exists(path) else None

    # Load CPT trade filter probabilities (if present)
    filter_path = os.path.join(checkpoint_dir, "filter_probs.npy")
    data["filter_probs"] = np.load(filter_path) if os.path.exists(filter_path) else None

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

            logits_path = os.path.join(checkpoint_dir, f"logits_h{h}.npy")
            h_data["logits"] = np.load(logits_path) if os.path.exists(logits_path) else None

            # CostLOB learned confidence (if present)
            conf_path = os.path.join(checkpoint_dir, f"confidence_h{h}.npy")
            h_data["confidence"] = np.load(conf_path) if os.path.exists(conf_path) else None

            data["horizons"].append((h, h_data))
    else:
        data["predictions"] = np.load(os.path.join(checkpoint_dir, "predictions.npy"))

        target_path = os.path.join(checkpoint_dir, "targets.npy")
        data["targets"] = np.load(target_path) if os.path.exists(target_path) else None

        proba_path = os.path.join(checkpoint_dir, "probabilities.npy")
        data["probabilities"] = np.load(proba_path) if os.path.exists(proba_path) else None

        logits_path = os.path.join(checkpoint_dir, "logits.npy")
        data["logits"] = np.load(logits_path) if os.path.exists(logits_path) else None

    return data


def _run_evaluation(
    data: dict,
    multi_horizon: bool,
    costs: list[float],
    confidence_thresholds: list[float],
    min_holds: list[int] | None = None,
    use_soft_positions: bool = False,
    filter_threshold: float | None = None,
) -> dict:
    """Run trading simulation for all parameter combinations."""
    if min_holds is None:
        min_holds = [0]
    results = {}
    mid_prices = data["mid_prices"]
    segment_boundaries = data["segment_boundaries"]
    half_spreads = data.get("half_spreads")
    z_half_spreads = data.get("z_half_spreads")
    filter_probs = data.get("filter_probs")

    if multi_horizon:
        for cost in costs:
            cost_key = f"cost_{cost}"
            results[cost_key] = {}

            for conf_thresh in confidence_thresholds:
                conf_key = f"conf_{conf_thresh}"
                results[cost_key][conf_key] = {}

                for mh in min_holds:
                    mh_key = f"hold_{mh}"
                    metrics_list = []
                    horizons_found = []

                    for h, h_data in data["horizons"]:
                        tm = compute_trading_metrics(
                            mid_prices,
                            h_data["predictions"],
                            probabilities=h_data.get("probabilities"),
                            logits=h_data.get("logits"),
                            half_spreads=half_spreads,
                            z_half_spreads=z_half_spreads,
                            cost_per_trade=cost,
                            confidence_threshold=conf_thresh,
                            min_hold=mh,
                            segment_boundaries=segment_boundaries,
                            use_soft_positions=use_soft_positions,
                            filter_probs=filter_probs,
                            filter_threshold=filter_threshold,
                        )
                        metrics_list.append(tm)
                        horizons_found.append(h)

                    results[cost_key][conf_key][mh_key] = {
                        "_metrics_list": metrics_list,
                        "_horizons": horizons_found,
                    }
                    for tm, h in zip(metrics_list, horizons_found):
                        results[cost_key][conf_key][mh_key][f"h{h}"] = _serializable(tm)
    else:
        for cost in costs:
            cost_key = f"cost_{cost}"
            results[cost_key] = {}

            for conf_thresh in confidence_thresholds:
                conf_key = f"conf_{conf_thresh}"
                results[cost_key][conf_key] = {}

                for mh in min_holds:
                    mh_key = f"hold_{mh}"
                    tm = compute_trading_metrics(
                        mid_prices,
                        data["predictions"],
                        probabilities=data.get("probabilities"),
                        logits=data.get("logits"),
                        half_spreads=half_spreads,
                        z_half_spreads=z_half_spreads,
                        cost_per_trade=cost,
                        confidence_threshold=conf_thresh,
                        min_hold=mh,
                        segment_boundaries=segment_boundaries,
                        use_soft_positions=use_soft_positions,
                        filter_probs=filter_probs,
                        filter_threshold=filter_threshold,
                    )
                    results[cost_key][conf_key][mh_key] = _serializable(tm)
                    results[cost_key][conf_key][mh_key]["_metrics"] = tm

    return results


def _serializable(tm: dict) -> dict:
    """Return a JSON-serializable version of trading metrics."""
    return {k: v for k, v in tm.items() if not isinstance(v, np.ndarray)}


def _print_results(
    results: dict,
    multi_horizon: bool,
    costs: list[float],
    confidence_thresholds: list[float],
    min_holds: list[int] | None = None,
):
    """Print formatted results to terminal."""
    if min_holds is None:
        min_holds = [0]
    for cost in costs:
        cost_key = f"cost_{cost}"
        for conf_thresh in confidence_thresholds:
            conf_key = f"conf_{conf_thresh}"
            for mh in min_holds:
                mh_key = f"hold_{mh}"

                header = f"Cost={cost}"
                if conf_thresh > 0:
                    header += f"  Confidence={conf_thresh}"
                if mh > 0:
                    header += f"  MinHold={mh}"
                print(f"\n{'=' * 90}")
                print(f"  {header}")
                print(f"{'=' * 90}")

                if multi_horizon:
                    entry = results[cost_key][conf_key][mh_key]
                    metrics_list = entry["_metrics_list"]
                    horizons = entry["_horizons"]
                    print(format_trading_table(metrics_list, horizons))
                else:
                    tm = results[cost_key][conf_key][mh_key]["_metrics"]
                    pf = "inf" if tm["profit_factor"] == float("inf") else f"{tm['profit_factor']:.2f}"
                    print(
                        f"PnL(norm)={tm['total_pnl']:.4f}  Sharpe/step={tm['sharpe']:.2e}  "
                        f"Sortino/step={tm['sortino']:.2e}  MaxDD={tm['max_drawdown_pct']:.1f}%  "
                        f"WinRate={tm['win_rate'] * 100:.1f}%  ProfitF={pf}  "
                        f"Trades={tm['n_trades']}  AvgHold={tm['avg_hold_duration']:.1f}  "
                        f"Exposure={tm['exposure_pct']:.1f}%  p-value={tm['p_value']:.4f}"
                    )


def _print_sweep_summary(
    results: dict,
    confidence_thresholds: list[float],
    min_holds: list[int],
):
    """Print a compact summary of the threshold sweep with best operating points."""
    cost_key = "cost_0.0"
    print(f"\n{'=' * 90}")
    print("  SWEEP SUMMARY — Best operating points per horizon (by PnL)")
    print(f"{'=' * 90}")

    # Find all horizons from the first entry
    first_conf = f"conf_{confidence_thresholds[0]}"
    first_hold = f"hold_{min_holds[0]}"
    horizons = results[cost_key][first_conf][first_hold]["_horizons"]

    for h_idx, h in enumerate(horizons):
        best_pnl = -float("inf")
        best_params = {}
        best_tm = None

        for conf in confidence_thresholds:
            for mh in min_holds:
                conf_key = f"conf_{conf}"
                mh_key = f"hold_{mh}"
                tm = results[cost_key][conf_key][mh_key]["_metrics_list"][h_idx]
                if tm["total_pnl"] > best_pnl:
                    best_pnl = tm["total_pnl"]
                    best_params = {"conf": conf, "min_hold": mh}
                    best_tm = tm

        if best_tm is not None:
            pf = f"{best_tm['profit_factor']:.2f}" if best_tm["profit_factor"] != float("inf") else "inf"
            print(
                f"  h{h:<4} | Best: conf={best_params['conf']}, hold={best_params['min_hold']:<3} | "
                f"PnL={best_pnl:>9.4f} Sharpe={best_tm['sharpe']:.2e} "
                f"Trades={best_tm['n_trades']:>6} AvgHold={best_tm['avg_hold_duration']:>6.1f} "
                f"ProfitF={pf:>6} Exposure={best_tm['exposure_pct']:.1f}%"
            )

    # Also show baseline (no filtering)
    baseline = results[cost_key]["conf_0.0"]["hold_0"]
    print(f"\n  Baseline (no filtering):")
    for h_idx, h in enumerate(horizons):
        tm = baseline["_metrics_list"][h_idx]
        print(
            f"  h{h:<4} | PnL={tm['total_pnl']:>9.4f} Sharpe={tm['sharpe']:.2e} "
            f"Trades={tm['n_trades']:>6}"
        )


def _plot_cumulative_pnl(results: dict, data: dict, multi_horizon: bool, save_dir: str):
    """Plot cumulative PnL curves (zero-cost, zero-confidence-threshold, zero-hold baseline)."""
    baseline_key = "cost_0.0"
    conf_key = "conf_0.0"
    hold_key = "hold_0"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)

    if multi_horizon:
        entry = results[baseline_key][conf_key][hold_key]
        for tm, h in zip(entry["_metrics_list"], entry["_horizons"]):
            cum_pnl = tm["cumulative_pnl"]
            if len(cum_pnl) > 0:
                ax.plot(cum_pnl, label=f"h{h} (Sharpe={tm['sharpe']:.2f})", linewidth=1.2)
    else:
        tm = results[baseline_key][conf_key][hold_key]["_metrics"]
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

    # Use zero-cost, zero-hold for the sweep plot
    cost_key = "cost_0.0"
    if cost_key not in results:
        cost_key = f"cost_{costs[0]}"
    hold_key = "hold_0"

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    if multi_horizon:
        first_conf = f"conf_{confidence_thresholds[0]}"
        horizons = results[cost_key][first_conf][hold_key]["_horizons"]

        for h_idx, h in enumerate(horizons):
            sharpes = []
            for conf in confidence_thresholds:
                conf_key = f"conf_{conf}"
                entry = results[cost_key][conf_key][hold_key]
                sharpes.append(entry["_metrics_list"][h_idx]["sharpe"])
            ax.plot(confidence_thresholds, sharpes, marker="o", label=f"h{h}", linewidth=1.5)
    else:
        sharpes = []
        for conf in confidence_thresholds:
            conf_key = f"conf_{conf}"
            sharpes.append(results[cost_key][conf_key][hold_key]["_metrics"]["sharpe"])
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
    hold_key = "hold_0"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)

    if multi_horizon:
        horizons = results[f"cost_{costs[0]}"][conf_key][hold_key]["_horizons"]
        for h_idx, h in enumerate(horizons):
            sharpes = []
            for cost in costs:
                cost_key = f"cost_{cost}"
                entry = results[cost_key][conf_key][hold_key]
                sharpes.append(entry["_metrics_list"][h_idx]["sharpe"])
            ax.plot(costs, sharpes, marker="s", label=f"h{h}", linewidth=1.5)
    else:
        sharpes = []
        for cost in costs:
            cost_key = f"cost_{cost}"
            sharpes.append(results[cost_key][conf_key][hold_key]["_metrics"]["sharpe"])
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
    parser.add_argument(
        "--min-holds",
        type=int,
        nargs="+",
        default=[0],
        help="Minimum hold steps before allowing position change (default: 0)",
    )
    parser.add_argument(
        "--soft-positions",
        action="store_true",
        help="Use continuous positions from logits instead of hard argmax (for DFL models)",
    )
    parser.add_argument(
        "--filter-threshold",
        type=float,
        default=0.5,
        help="CPT trade filter threshold (default: 0.5). Only used when filter_probs.npy exists.",
    )
    parser.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Run a predefined grid search over min_hold and confidence_threshold",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip reference baselines (Buy & Hold, SMA, Perfect Foresight)",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        choices=["softmax", "learned"],
        default="softmax",
        help="Confidence source for hysteresis: 'softmax' (default, max softmax prob) "
             "or 'learned' (CostLOB trained confidence). When 'learned', replaces "
             "softmax probabilities with the model's confidence_h*.npy arrays.",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: {checkpoint_dir} is not a directory")
        sys.exit(1)

    if args.sweep_thresholds:
        # Predefined grid for Phase 1 validation
        costs = [0.0]
        confidence_thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
        min_holds = [0, 5, 10, 20, 50, 100]
    else:
        costs = sorted(set([0.0] + args.costs))
        confidence_thresholds = sorted(set([0.0] + args.confidence_thresholds))
        min_holds = sorted(set([0] + args.min_holds))

    multi_horizon = _detect_multi_horizon(checkpoint_dir)
    mode_str = "multi-horizon" if multi_horizon else "single-horizon"
    print(f"Detected {mode_str} mode")

    if multi_horizon:
        print(f"Loading per-horizon arrays for horizons: {HORIZONS}")
    print(f"Cost levels: {costs}")
    print(f"Confidence thresholds: {confidence_thresholds}")
    print(f"Min hold steps: {min_holds}")

    data = _load_arrays(checkpoint_dir, multi_horizon)
    print(f"Loaded {len(data['mid_prices']):,} mid-price steps")
    mid = data["mid_prices"]
    if len(mid) > 1:
        mean_abs_change = float(np.mean(np.abs(np.diff(mid))))
        print(f"Mean |Δmid| = {mean_abs_change:.6f} (cost multiplier base)")
    if data["segment_boundaries"] is not None:
        print(f"Loaded {len(data['segment_boundaries'])} product boundaries (EPEX per-product mode)")
    if data.get("half_spreads") is not None:
        print(f"Loaded raw half-spreads (spread-aware costs enabled)")
    if data.get("z_half_spreads") is not None:
        print(f"Loaded z-scored half-spreads (spread-aware costs enabled)")
    if args.soft_positions:
        has_logits = False
        if multi_horizon:
            has_logits = any(h_data.get("logits") is not None for _, h_data in data.get("horizons", []))
        else:
            has_logits = data.get("logits") is not None
        if has_logits:
            print("Using soft (continuous) positions from logits")
        else:
            print("Warning: --soft-positions requested but no logits found, falling back to hard positions")

    # CostLOB learned confidence: replace softmax probabilities with confidence scalars
    # so the existing hysteresis and confidence_threshold logic uses them automatically.
    if args.confidence_mode == "learned" and multi_horizon:
        n_replaced = 0
        for h, h_data in data["horizons"]:
            if h_data.get("confidence") is not None:
                # Expand scalar confidence to (N, 1) so max(axis=1) returns the confidence
                # The hysteresis code uses max_conf = probabilities.max(axis=1)
                conf = h_data["confidence"]
                h_data["probabilities"] = conf.reshape(-1, 1)
                n_replaced += 1
        if n_replaced > 0:
            print(f"CostLOB: Replaced softmax probabilities with learned confidence for {n_replaced} horizons")
        else:
            print("Warning: --confidence-mode=learned but no confidence_h*.npy found, using softmax")

    filter_threshold = None
    if data.get("filter_probs") is not None:
        filter_threshold = args.filter_threshold
        n_filter = len(data["filter_probs"])
        trade_rate = (data["filter_probs"] > filter_threshold).mean()
        print(f"CPT trade filter detected ({n_filter:,} probs, threshold={filter_threshold}, "
              f"trade_rate={trade_rate:.1%})")

    # Detect dataset and std_price for dollar/EUR conversion
    std_price, currency = _detect_std_price(checkpoint_dir, data)
    if std_price is not None:
        print(f"Price conversion: 1 z-unit = {currency}{std_price:.2f}")

    results = _run_evaluation(
        data, multi_horizon, costs, confidence_thresholds,
        min_holds=min_holds, use_soft_positions=args.soft_positions,
        filter_threshold=filter_threshold,
    )
    _print_results(results, multi_horizon, costs, confidence_thresholds, min_holds=min_holds)

    # Print sweep summary if running threshold sweep
    if args.sweep_thresholds and multi_horizon:
        _print_sweep_summary(results, confidence_thresholds, min_holds)

    # Reference baselines
    if not args.no_baselines:
        mid_prices = data["mid_prices"]
        half_spreads = data.get("half_spreads")
        z_half_spreads = data.get("z_half_spreads")
        segment_boundaries = data.get("segment_boundaries")

        if multi_horizon:
            baselines_per_h = {}
            for h, h_data in data["horizons"]:
                targets = h_data.get("targets")
                baselines_per_h[h] = _compute_baselines(
                    mid_prices, targets, half_spreads, z_half_spreads, segment_boundaries,
                )
            _print_baselines(baselines_per_h, [h for h, _ in data["horizons"]], std_price, currency)
        else:
            targets = data.get("targets")
            baselines = _compute_baselines(
                mid_prices, targets, half_spreads, z_half_spreads, segment_boundaries,
            )
            baselines_per_h = {10: baselines}  # single horizon
            _print_baselines(baselines_per_h, [10], std_price, currency)

    print("\nGenerating plots...")
    _plot_cumulative_pnl(results, data, multi_horizon, checkpoint_dir)
    _plot_confidence_sweep(results, multi_horizon, costs, confidence_thresholds, checkpoint_dir)
    _plot_cost_sensitivity(results, multi_horizon, costs, checkpoint_dir)
    _save_json_results(results, checkpoint_dir)

    print(f"\nCheckpoint dir: {checkpoint_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
