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
_DPVN_ACTIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float64)


def _spread_argmax_predictions(
    v_values: np.ndarray,
    z_half_spread: np.ndarray,
    segment_boundaries: np.ndarray | None = None,
) -> np.ndarray:
    """DPVN spread-aware decision rule. Returns predictions in label space.

    For each t, choose a* = argmax_a [V(t, a) - |a - pos_prev| * z_half_spread[t]],
    with pos_prev reset to 0 at segment boundaries (Battery per-product).

    Label mapping: position +1 -> 0 (up), 0 -> 1 (stat), -1 -> 2 (down).
    """
    n = v_values.shape[0]
    if z_half_spread is None:
        z_half_spread = np.zeros(n, dtype=np.float64)
    z_half_spread = np.abs(z_half_spread.astype(np.float64))
    seg_starts = set(int(b) for b in segment_boundaries) if segment_boundaries is not None else set()
    positions = np.zeros(n, dtype=np.float64)
    pos_prev = 0.0
    for t in range(n):
        if t in seg_starts:
            pos_prev = 0.0
        best_a_idx = 0
        best_score = -np.inf
        for a_idx in range(3):
            a = _DPVN_ACTIONS[a_idx]
            score = v_values[t, a_idx] - abs(a - pos_prev) * z_half_spread[t]
            if score > best_score:
                best_score = score
                best_a_idx = a_idx
        positions[t] = _DPVN_ACTIONS[best_a_idx]
        pos_prev = positions[t]
    predictions = np.ones(n, dtype=np.int64)
    predictions[positions > 0.5] = 0
    predictions[positions < -0.5] = 2
    return predictions


def _dpvn_audit(data: dict, raw_preds: np.ndarray, spread_preds: np.ndarray, seed: int = 0) -> None:
    """Falsifiability audit for DPVN spread_argmax results.

    Prints:
      - Prediction distribution (raw argmax vs spread_argmax)
      - Accuracy-by-|delta-mid| decile
      - Per-product PnL distribution
      - Shuffled-V sanity test (should be ~0)
    """
    mid_prices = data["mid_prices"]
    v_values = data.get("dpvn_values")
    z_hs = data.get("z_half_spreads")
    targets = data.get("targets")
    boundaries = data.get("segment_boundaries")

    print(f"\n{'=' * 90}")
    print("  DPVN Audit — F1 vs PnL Disconnect Diagnostics")
    print(f"{'=' * 90}")

    # D2. Prediction distribution
    def _dist(preds):
        n = len(preds)
        long_pct = (preds == 0).mean() * 100
        flat_pct = (preds == 1).mean() * 100
        short_pct = (preds == 2).mean() * 100
        changes = int((np.diff(preds) != 0).sum())
        return long_pct, flat_pct, short_pct, changes

    print("\n  [D2] Prediction distribution")
    print(f"  {'strategy':<16} | {'long%':>6} | {'flat%':>6} | {'short%':>6} | {'changes':>8}")
    print(f"  {'-' * 62}")
    for name, preds in (("argmax", raw_preds), ("spread_argmax", spread_preds)):
        lp, fp, sp, ch = _dist(preds)
        print(f"  {name:<16} | {lp:>5.1f}% | {fp:>5.1f}% | {sp:>5.1f}% | {ch:>8,}")

    # D1. Accuracy-by-|delta-mid| decile (needs targets + mid sequence)
    if targets is not None and len(mid_prices) > 1:
        dmid = np.abs(np.diff(mid_prices, prepend=mid_prices[0]))
        n_align = min(len(targets), len(spread_preds), len(dmid))
        dmid_a = dmid[:n_align]
        preds_a = spread_preds[:n_align]
        tgt_a = targets[:n_align]
        order = np.argsort(dmid_a)
        buckets = np.array_split(order, 10)
        print("\n  [D1] Accuracy by |Δmid| decile (spread_argmax predictions)")
        print(f"  {'decile':<8} | {'|Δmid| mean':>12} | {'acc%':>6} | {'n':>6}")
        print(f"  {'-' * 42}")
        for i, idx in enumerate(buckets):
            if len(idx) == 0:
                continue
            acc = (preds_a[idx] == tgt_a[idx]).mean() * 100
            dmid_mean = dmid_a[idx].mean()
            print(f"  d{i + 1:<7} | {dmid_mean:>12.5f} | {acc:>5.1f}% | {len(idx):>6,}")

    # D4. Per-product PnL
    if boundaries is not None and len(boundaries) > 1:
        starts = np.concatenate([[0], boundaries[:-1]])
        ends = boundaries
        per_product = []
        for s, e in zip(starts, ends):
            if e - s < 10:
                continue
            sub_mid = mid_prices[s:e]
            sub_preds = spread_preds[s:e]
            sub_z_hs = z_hs[s:e] if z_hs is not None else None
            tm = compute_trading_metrics(
                sub_mid, sub_preds,
                z_half_spreads=sub_z_hs,
            )
            per_product.append(tm["total_pnl"])
        if per_product:
            arr = np.asarray(per_product)
            print(f"\n  [D4] Per-product PnL (N={len(arr)} products, spread_argmax)")
            print(f"  mean={arr.mean():>+8.3f}  std={arr.std():>7.3f}  "
                  f"min={arr.min():>+8.3f}  max={arr.max():>+8.3f}  "
                  f"frac>0={float((arr > 0).mean()):.2f}")
            print(f"  top 3 products by PnL: {np.sort(arr)[::-1][:3].round(2).tolist()}")
            print(f"  bot 3 products by PnL: {np.sort(arr)[:3].round(2).tolist()}")

    # D3. Shuffled-V sanity test
    if v_values is not None and z_hs is not None:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(v_values.shape[0])
        v_shuf = v_values[perm]
        shuf_preds = _spread_argmax_predictions(v_shuf, z_hs, boundaries)
        tm = compute_trading_metrics(
            mid_prices, shuf_preds,
            z_half_spreads=z_hs,
            segment_boundaries=boundaries,
        )
        real_tm = compute_trading_metrics(
            mid_prices, spread_preds,
            z_half_spreads=z_hs,
            segment_boundaries=boundaries,
        )
        print("\n  [D3] Shuffled-V sanity test (time-permuted V, spread_argmax rule)")
        print(f"  {'version':<14} | {'PnL(z)':>10} | {'Trades':>7} | {'Sharpe':>10}")
        print(f"  {'-' * 50}")
        print(f"  {'real':<14} | {real_tm['total_pnl']:>+10.3f} | {real_tm['n_trades']:>7} | {real_tm['sharpe']:>10.2e}")
        print(f"  {'shuffled':<14} | {tm['total_pnl']:>+10.3f} | {tm['n_trades']:>7} | {tm['sharpe']:>10.2e}")
        print("  Expected: shuffled PnL near zero. Large positive shuffled PnL ⇒ decision rule alone")
        print("  is picking up a timing signal (leakage red flag).")
    print()


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

        dpvn_path = os.path.join(checkpoint_dir, "dpvn_values.npy")
        data["dpvn_values"] = np.load(dpvn_path) if os.path.exists(dpvn_path) else None

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
    parser.add_argument(
        "--audit_seed",
        type=int,
        default=0,
        help="Seed for the shuffled-V sanity test (D3) in the DPVN audit.",
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
        print("Loaded raw half-spreads (spread-aware costs enabled)")
    if data.get("z_half_spreads") is not None:
        print("Loaded z-scored half-spreads (spread-aware costs enabled)")
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

    # DPVN auto-detection: if dpvn_values.npy exists and run is single-horizon,
    # run the evaluation for both raw argmax and spread-aware argmax.
    is_dpvn = (not multi_horizon) and (data.get("dpvn_values") is not None)
    raw_argmax_preds = None
    spread_argmax_preds = None
    if is_dpvn:
        v_values = data["dpvn_values"]
        z_hs = data.get("z_half_spreads")
        if z_hs is None:
            print("Warning: dpvn_values.npy present but z_half_spreads.npy missing; "
                  "running raw argmax only.")
            is_dpvn = False
        else:
            n_v = v_values.shape[0]
            n_pred = len(data["predictions"])
            if n_v != n_pred:
                print(f"Warning: dpvn_values length ({n_v}) != predictions length ({n_pred}); "
                      f"truncating to min.")
                m = min(n_v, n_pred)
                v_values = v_values[:m]
                z_hs = z_hs[:m]
                data["predictions"] = data["predictions"][:m]
                if data.get("targets") is not None:
                    data["targets"] = data["targets"][:m]
                data["dpvn_values"] = v_values
                data["z_half_spreads"] = z_hs
            raw_argmax_preds = data["predictions"].copy()
            spread_argmax_preds = _spread_argmax_predictions(
                v_values, z_hs, data.get("segment_boundaries"),
            )
            n_changed = int(np.sum(spread_argmax_preds != raw_argmax_preds))
            print(f"DPVN detected: {n_changed:,} / {spread_argmax_preds.shape[0]:,} predictions "
                  f"differ between raw and spread-aware argmax "
                  f"({n_changed / max(spread_argmax_preds.shape[0], 1) * 100:.2f}%)")

    # Detect dataset and std_price for dollar/EUR conversion
    std_price, currency = _detect_std_price(checkpoint_dir, data)
    if std_price is not None:
        print(f"Price conversion: 1 z-unit = {currency}{std_price:.2f}")

    def _eval_and_print(data_in, header=None):
        if header is not None:
            print("\n" + "=" * 72)
            print(header)
            print("=" * 72)
        r = _run_evaluation(
            data_in, multi_horizon, costs, confidence_thresholds,
            min_holds=min_holds, use_soft_positions=args.soft_positions,
            filter_threshold=filter_threshold,
        )
        _print_results(r, multi_horizon, costs, confidence_thresholds, min_holds=min_holds)
        return r

    if is_dpvn:
        # Run raw-argmax against a shallow copy so the shared dict stays on the
        # spread-aware predictions (the designed rule) for plots, JSON, and baselines.
        _eval_and_print({**data, "predictions": raw_argmax_preds},
                        header="=== Decision rule: RAW ARGMAX ===")
        data["predictions"] = spread_argmax_preds
        results = _eval_and_print(data, header="=== Decision rule: SPREAD-AWARE ARGMAX ===")
    else:
        results = _eval_and_print(data)

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

    if is_dpvn:
        _dpvn_audit(data, raw_argmax_preds, spread_argmax_preds, seed=args.audit_seed)

    print("\nGenerating plots...")
    _plot_cumulative_pnl(results, data, multi_horizon, checkpoint_dir)
    _plot_confidence_sweep(results, multi_horizon, costs, confidence_thresholds, checkpoint_dir)
    _plot_cost_sensitivity(results, multi_horizon, costs, checkpoint_dir)
    _save_json_results(results, checkpoint_dir)

    print(f"\nCheckpoint dir: {checkpoint_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
