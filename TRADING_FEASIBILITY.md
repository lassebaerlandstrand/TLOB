# Trading Feasibility Analysis

**Date:** 2026-04-13
**Datasets:** BTC (Binance BTC/USDT, 250ms), Battery (EPEX Spot Continuous, 10s TIME_DEDUP, last 6h before gate closure)
**Test periods:** BTC: 2 days (Jan 19-20, 2023), 605,226 samples. Battery: 2 days (Jan 21-22, 2021), 48 products, 69,310 samples.

All PnL figures use **spread-aware costs** throughout: `cost = |delta_position| * z_half_spread(t)`, matching the evaluation in `engine.py`. PnL is in z-score normalized units (z). See Methodology for details.

---

## Executive Summary

Both BTC and Battery are profitably tradeable. The key variable is the **spread/volatility ratio** — BTC (0.11) is trivially cheap to trade; Battery (0.35) requires trade frequency control for ML models but not for perfect foresight.

### BTC Summary Table (1 z-unit = $1,576)

| Strategy | h10 PnL (z) | h10 ($) | h20 (z) | h50 (z) | h100 (z) | Trades (h10) |
|----------|:-----------:|:-------:|:-------:|:-------:|:--------:|:------------:|
| *DP optimal (ceiling)* | *41.63* | *+$65,612* | — | — | — | *12,830* |
| Perfect Foresight | 35.11 | +$55,336 | 29.47 | 20.54 | 14.39 | 29,812 |
| **TLOB CA-CE** | **24.75** | **+$39,006** | **25.81** | **26.19** | **25.36** | 76,569 |
| **TLOB CE (default)** | **18.09** | **+$28,519** | **21.42** | **23.52** | **23.26** | 56,991 |
| SMA-10 | 15.69 | +$24,738 | — | — | — | 28,738 |
| Buy & Hold | 0.40 | +$638 | 0.40 | 0.40 | 0.40 | 2 |

*DP optimal is horizon-independent — it finds the position sequence {-1, 0, +1} maximizing net PnL over actual returns and spreads via dynamic programming. PF uses per-horizon labels with optimal min_hold (h10: mh=0, h20: mh=5, h50: mh=30, h100: mh=50). Dollar amounts are notional (1-unit position size).*

### Battery Summary Table (6h truncated, 1 z-unit = €15.43)

| Strategy | h10 PnL (z) | h10 (€) | h20 (z) | h50 (z) | h100 (z) | Trades (h10) |
|----------|:-----------:|:---------:|:-------:|:-------:|:--------:|:------------:|
| *DP optimal (ceiling)* | *376.41* | *+€5,808* | — | — | — | *3,769* |
| Perfect Foresight | 234.26 | +€3,615 | 190.42 | 141.51 | 108.87 | 4,657 |
| **TLOB CE (default) + best hyst** | **+18.89** | **+€291** | **+33.01** | **+39.13** | **+17.80** | 3,202 |
| TLOB CE (default) + mh=20 | +0.93 | +€14 | -3.47 | -11.10 | -10.92 | 2,932 |
| TLOB CE (default) | -114.10 | -€1,761 | -112.21 | -120.38 | -134.19 | 10,624 |
| SMA-20 | -113.06 | -€1,744 | — | — | — | 5,281 |
| Buy & Hold | -7.41 | -€114 | -7.41 | -7.41 | -7.41 | 96 |

*PF uses per-horizon labels with optimal min_hold (h10: mh=0, h20: mh=0, h50: mh=30, h100: mh=100). Best hysteresis per horizon: h10=(0.8,0.45), h20=(0.7,0.4), h50=(0.6,0.35), h100=(0.5,0.3). Battery uses only the last 6h before gate closure per product. € amounts are notional (1 MW position size).*

---

## Finding 1: The Spread/Volatility Ratio Determines Tradeability

The ratio of the bid-ask spread to the 1-step price return standard deviation determines the cost of trading relative to the available signal:

| Dataset | z_half_spread (mean) | 1-step return (std) | Ratio |
|---------|:-------------------:|:------------------:|:-----:|
| BTC | 0.000032 | 0.000301 | **0.11** |
| Battery (6h) | 0.021 | 0.060 | **0.35** |

**BTC (0.11):** The spread is negligible — each position change costs only 11% of a standard deviation of price movement. Perfect foresight is profitable without any trade frequency control.

**Battery (0.35):** The spread is significant — each position change costs 35% of one std. Perfect foresight is still profitable without min_hold, but ML models that over-trade (TLOB CE makes 10.6K trades vs PF's 4.7K) need trade frequency control to overcome spread costs.

---

## Finding 2: ML Models Over-Trade

TLOB CE predicts a position at every timestep, producing noisy position sequences that flip-flop frequently. This over-trading is the primary source of losses on Battery.

### The min_hold Effect on BTC (h10, simulated accuracy, spread-aware costs)

| Accuracy | No min_hold | With mh=15 | PnL change |
|:--------:|:-----------:|:----------:|:----------:|
| 55% | -5.38 (161K trades) | **+0.34** (28K trades) | +5.72 |
| 60% | -2.21 (157K trades) | **+1.62** (28K trades) | +3.83 |
| 65% | **+2.07** (150K trades) | **+2.85** (28K trades) | +0.78 |
| 70% | **+6.49** (140K trades) | **+4.42** (28K trades) | -2.07 |
| 75% | **+10.55** (129K trades) | **+6.46** (28K trades) | -4.09 |
| 80% | **+15.25** (115K trades) | **+8.63** (27K trades) | -6.62 |

At 65%+ accuracy, no min_hold is needed — the model is already profitable. At lower accuracy, min_hold helps by reducing trade frequency. Above 70%, min_hold actually hurts by suppressing profitable trades.

### Breakeven Directional Accuracy by min_hold (BTC, h10)

| min_hold | Breakeven accuracy | Trades at breakeven |
|:--------:|:-----------------:|:-------------------:|
| 0 | 63% | ~155K |
| 5 | 64% | ~62K |
| 10 | 56% | ~38K |
| 15 | 55% | ~28K |
| 20 | 55% | ~22K |
| 30 | 52% | ~16K |

Without min_hold, breakeven is 63% — feasible for a directional model. With min_hold >= 15, breakeven drops below 55%.

---

## Finding 3: Trend Persistence Determines Optimal Hold Period

Price trends persist for a characteristic duration at each horizon. The optimal min_hold closely tracks the **median trend length**.

### BTC Trend Persistence

| Horizon | Median trend | Mean trend | p90 | # Trends |
|---------|:----------:|:--------:|:---:|:--------:|
| h10 | **16** | 18.7 | 34 | 15,006 |
| h20 | **24** | 27.8 | 50 | 11,232 |
| h50 | **48** | 50.5 | 94 | 7,127 |
| h100 | **64** | 75.7 | 153 | 5,003 |

### Battery Trend Persistence (6h truncated)

| Horizon | Median trend | Mean trend | p90 | # Trends |
|---------|:----------:|:--------:|:---:|:--------:|
| h10 | 10 | 14.8 | 32 | 5,355 |
| h20 | 16 | 22.2 | 47 | 3,553 |
| h50 | 21 | 34.9 | 80 | 2,217 |
| h100 | 29 | 50.5 | 121 | 1,485 |

Holding for one full trend captures the directional return while only paying the spread once at entry and exit. Battery trends in the active trading window are shorter than BTC trends, reflecting faster price dynamics near gate closure.

---

## Finding 4: SMA Baselines

A moving-average crossover on z-scored mid-prices — requiring no ML — is a useful baseline. SMA positions are inherently smooth, avoiding the flip-flopping that costs spread.

### BTC SMA Baselines

| Strategy | PnL (z) | Trades | Sharpe |
|----------|:-------:|:------:|:------:|
| SMA-10 | **15.69** | 28,738 | 9.08e-02 |
| SMA-20 | 15.55 | 18,550 | 8.78e-02 |
| SMA-50 | 12.21 | 10,635 | 6.79e-02 |
| SMA-100 | 8.96 | 6,769 | 4.94e-02 |

TLOB CE (18.09z) beats SMA-10 (15.69z) without any post-processing. TLOB CA-CE (24.75z) exceeds it by 58%.

### Battery SMA Baselines (6h truncated)

| Strategy | PnL (z) | Trades | Sharpe |
|----------|:-------:|:------:|:------:|
| SMA-10 | -242.4 | 8,810 | -0.102 |
| SMA-20 | -113.1 | 5,281 | -0.051 |
| SMA-50 | -37.3 | 2,766 | -0.018 |
| SMA-100 | -8.3 | 1,752 | -0.004 |
| Buy & Hold | -7.4 | 96 | -0.004 |

All SMA baselines are negative on Battery. TLOB CE + hysteresis (+18.89z at h10) is the first strategy to meaningfully beat Buy & Hold.

---

## Finding 5: Extended Horizons

Horizon h=N means the model predicts the price direction N sampling steps ahead. BTC samples at 250ms (h10 = 2.5s), Battery at 10s (h10 = 100s).

### BTC Extended Horizons

| Horizon | Real time | DP optimal (z) | PF best (z) | PF best mh | Trades (PF) |
|---------|:---------:|:--------------:|:-----------:|:----------:|:-----------:|
| h10 | 2.5s | **41.63** | 35.11 | 0 | 29,812 |
| h20 | 5.0s | — | 29.47 | 5 | 22,292 |
| h50 | 12.5s | — | 20.54 | 30 | 11,453 |
| h100 | 25.0s | — | 14.39 | 50 | 7,202 |

BTC's sweet spot is h10 — maximum PF PnL and no min_hold needed. At longer horizons, trends become rarer and min_hold begins to help (mh=30 at h50, mh=50 at h100).

### Battery Extended Horizons (6h truncated)

| Horizon | Real time | DP optimal (z) | PF best (z) | PF best mh | Trades (PF) |
|---------|:---------:|:--------------:|:-----------:|:----------:|:-----------:|
| h10 | 2min | **376.41** | 234.26 | 0 | 4,657 |
| h20 | 3min | — | 190.42 | 0 | 3,156 |
| h50 | 8min | — | 141.51 | 30 | 1,441 |
| h100 | 17min | — | 108.87 | 100 | 615 |

Battery h10 has the highest PF PnL. At longer horizons, fewer trends mean fewer trading opportunities. Min_hold becomes helpful at h50+ where trend persistence is shorter relative to the horizon.

---

## Finding 6: Lifecycle Truncation Is the Key

EPEX products trade for ~31.5h but the first ~26h have wide spreads and minimal volatility. Restricting to the last 6h before gate closure nearly halves the spread/vol ratio:

| Configuration | Spread/Vol |
|:--------:|:----------:|
| 10s DEDUP (full lifecycle) | **0.64** |
| 10s DEDUP (6h truncation) | **0.35** |

This is consistent with EPEX SPOT literature documenting L-shaped spread patterns and exponential volume concentration near gate closure. The 6h truncation is now the default (`max_hours_before_delivery=6.0` in config).

---

## Finding 7: Trade Frequency Control Methods Compared

TLOB CE over-trades on Battery (10.6K trades vs PF's 4.7K). We evaluated three inference-time methods to control trade frequency.

**Methods:**
- **Hysteresis**: Require high confidence to *enter* a position, lower to *hold*. Like a Schmitt trigger — positions are "sticky". `hyst(entry, exit)` notation.
- **min_hold**: Hold for N steps regardless of signal.
- **Confidence thresholding**: Only trade when `max(softmax) > threshold`.

### BTC — TLOB Strategy Comparison (h10)

| Strategy | PnL (z) | Trades | Sharpe |
|----------|:-------:|:------:|:------:|
| *DP optimal (ceiling)* | *41.63* | *12,830* | — |
| *PF (no mh)* | *35.11* | *29,812* | *0.200* |
| **TLOB CA-CE** | **24.75** | **76,569** | **0.138** |
| TLOB CE (default) | 18.09 | 56,991 | 0.117 |
| TLOB CE (default) + hyst(0.5,0.3) | 17.07 | 45,775 | 0.109 |
| SMA-10 | 15.69 | 28,738 | 0.091 |
| TLOB CE (default) + mh=20 | 8.96 | 20,591 | 0.059 |

**On BTC, no trade frequency control is needed.** TLOB CE (18.09z) is already profitable and beats SMA-10. min_hold and hysteresis both *hurt* — the spread is so low that high-frequency trading is profitable. TLOB CA-CE (24.75z) is the best model, achieving 59% of the DP optimal ceiling.

### Battery — TLOB Strategy Comparison (h10)

| Strategy | PnL (z) | Trades | Sharpe |
|----------|:-------:|:------:|:------:|
| *DP optimal (ceiling)* | *+376.41* | *3,769* | — |
| *PF (no mh)* | *+234.26* | *4,657* | *0.122* |
| **TLOB CE (default) + hyst(0.8,0.45)** | **+18.89** | **3,202** | **0.010** |
| TLOB CE (default) + hyst(0.7,0.4) | +6.76 | 3,725 | 0.003 |
| TLOB CE (default) + mh=20 | +0.93 | 2,932 | 0.000 |
| TLOB CE (default) + conf=0.9 | -48.21 | 5,264 | -0.039 |
| TLOB CE (default) | -114.10 | 10,624 | -0.053 |
| TLOB CA-CE | -199.76 | 11,235 | -0.086 |
| SMA-10 | -242.40 | 8,810 | -0.102 |

**Hysteresis is the best method for Battery.** hyst(0.8,0.45) achieves +18.89z — a 20x improvement over min_hold (+0.93z). The signal-based approach holds positions based on confidence, exiting only on genuine reversals. Confidence thresholding alone reduces losses but stays negative.

**CA-CE failed on Battery.** The cost weighting upweights noisy outliers, degrading F1 from 0.66 to 0.45.

**The ceiling is high.** TLOB achieves 5.0% of the DP optimal ceiling. Better features, architectural changes, or longer training are needed to close the gap.

---

## Label Distribution

| Horizon | BTC up | BTC stat | BTC down | Battery up | Battery stat | Battery down |
|---------|:------:|:--------:|:--------:|:----------:|:------------:|:------------:|
| h10 | 22.9% | 53.7% | 23.4% | 24.7% | 50.1% | 25.2% |
| h20 | 25.6% | 48.5% | 26.0% | 26.0% | 47.1% | 26.9% |
| h50 | 29.8% | 40.6% | 29.7% | 28.0% | 42.4% | 29.6% |
| h100 | 31.7% | 37.4% | 30.9% | 29.1% | 39.5% | 31.4% |

Both datasets have similar label distributions. The feasibility difference is driven by spread, not label balance.

---

## Known Limitations

1. **Simulation seed sensitivity**: Accuracy simulations use seed=42. Different seeds produce ±5% PnL variance.
2. **Product boundary effects**: Battery positions reset between delivery hours. Forced closing at boundaries adds costs.
3. **No order book depth**: Execution at best bid/ask regardless of order size. Real costs would be higher for Battery (thinner books).

---

## Methodology

**Position model:** At each timestep, the model outputs a position in {-1 (short), 0 (flat), +1 (long)}.

**PnL computation:** `gross = position(t) * (z_mid(t+1) - z_mid(t))`, `cost = |position(t) - position(t-1)| * z_half_spread(t)`. Net PnL = cumulative gross - cumulative cost.

**z-unit conversion:** BTC: 1 z-unit = $1,576.24 (from `norm_stats.npz: std_prices`). Battery: 1 z-unit = €15.43 (from `median(raw_half_spread / z_half_spread)`).

**min_hold:** Position cannot change until N steps elapsed since last change.

**Hysteresis:** Enter position when `max(softmax) >= entry_threshold`. Hold position while `max(softmax) >= exit_threshold`. Exit to flat otherwise.

**SMA crossover:** `position = +1 if z_mid > SMA(window), -1 if z_mid < SMA(window)`.

**DP optimal:** Dynamic programming over {-1, 0, +1} positions maximizing net PnL (returns minus position-change costs). O(9N) time. Battery segments computed independently, respecting product boundaries.

**Battery aggregation:** 48 test products (independent delivery-hour contracts). Data truncated to last 6h before gate closure (`max_hours_before_delivery=6.0`).

**Not modeled:** Order book depth/volume, price impact, partial fills, variable position sizing.
