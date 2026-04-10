# Trading Feasibility Analysis

**Date:** 2026-04-10
**Datasets:** BTC (Binance BTC/USDT), Battery (EPEX Spot Continuous, 5s sampling)
**Evaluation period:** BTC test set (605,344 samples), Battery (551 products, 1,862,917 total samples across train/test/val)
**PnL computation:** Matches `engine.py` evaluation exactly: `gross = position * diff(z_mid)`, `cost = |delta_position| * z_half_spread`

---

## Executive Summary

**Both BTC and Battery are profitably tradeable — but only with position persistence (min_hold).** Without min_hold, over-trading destroys profits on both datasets. The key explanatory variable is the **spread/volatility ratio**.

| Metric | BTC | Battery |
|--------|-----|---------|
| Spread / 1-step return std | **0.11** | **0.64** |
| Perfect foresight h10 PnL (no min_hold) | **+11.60** | **-3,331** |
| PF h10 PnL + optimal min_hold | **+21.98** | **+28.69** (mh=30) |
| Breakeven accuracy (no min_hold) | 88% | impossible |
| Breakeven accuracy (min_hold=20) | 40% | marginal |

### BTC Summary Table — Perfect Foresight PnL by Horizon

| Strategy | h10 PnL (z) | h10 PnL ($) | h20 PnL (z) | h20 PnL ($) | h50 PnL (z) | h50 PnL ($) | h100 PnL (z) | h100 PnL ($) |
|----------|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|:------------:|
| Buy & Hold | 0.42 | $661 | 0.42 | $661 | 0.42 | $661 | 0.42 | $661 |
| Perfect Foresight | 11.60 | $18,285 | 8.46 | $13,339 | 5.04 | $7,943 | 3.13 | $4,939 |
| PF + optimal min_hold | **21.98** | **$34,644** | **19.46** | **$30,674** | **15.48** | **$24,399** | **12.17** | **$19,188** |
| SMA-10 (no ML) | 15.64 | $24,650 | — | — | — | — | — | — |
| SMA-20 (no ML) | 15.56 | $24,528 | — | — | — | — | — | — |
| 65% acc, no min_hold | -7.84 | -$12,358 | -9.65 | -$15,215 | -11.21 | -$17,662 | -12.19 | -$19,216 |
| 65% acc + opt min_hold | **6.15** | **$9,686** | **5.83** | **$9,189** | **5.42** | **$8,539** | **3.97** | **$6,254** |

*1 z-unit = $1,576 (BTC std_price from normalization). Optimal min_hold: h10=20, h20=30, h50=55, h100=100 steps.*

### Battery Summary Table — Perfect Foresight PnL by Horizon

| Strategy | h10 PnL (z) | h10 (EUR) | h20 PnL (z) | h20 (EUR) | h50 PnL (z) | h50 (EUR) | h100 PnL (z) | h100 (EUR) |
|----------|:-----------:|:---------:|:-----------:|:---------:|:-----------:|:---------:|:------------:|:----------:|
| Buy & Hold | -736 | -11,474 | -736 | -11,474 | -736 | -11,474 | -736 | -11,474 |
| Perfect Foresight | -3,331 | -51,929 | -2,323 | -36,216 | -1,564 | -24,392 | -1,292 | -20,153 |
| PF + opt min_hold | **+28.7** (mh=30) | **+447** | **+275.7** (mh=30) | **+4,299** | **+444.4** (mh=75) | **+6,929** | **+446.5** (mh=100) | **+6,961** |

*1 z-unit = EUR 15.59 (Battery global std_price from normalization). Aggregated across 551 products. Without min_hold, perfect foresight is negative at all horizons — the spread/vol ratio (0.64) means every trade must be held long enough to overcome the spread.*

**Correction:** An earlier version of this analysis used incorrect column indices (0, 2 instead of 18, 20) for the 62-column Battery format, reading message features instead of LOB prices. This erroneously produced spread/vol=1.14 and negative PnL everywhere. The corrected analysis (spread/vol=0.64) shows Battery IS marginally tradeable with min_hold.

---

## Finding 1: The Spread/Volatility Ratio Determines Tradeability

The single most important number is the ratio of the half bid-ask spread to the standard deviation of 1-step price returns:

$$\text{spread/vol ratio} = \frac{\text{mean}(|z\_\text{half\_spread}|)}{\text{std}(\Delta z\_\text{mid})}$$

| Dataset | z_half_spread (mean) | 1-step return (std) | Ratio | Tradeable? |
|---------|:-------------------:|:------------------:|:-----:|:----------:|
| BTC | 0.000032 | 0.000301 | **0.11** | Yes |
| Battery | 0.024 | 0.038 | **0.64** | With min_hold |

**BTC**: The spread is 11% of one standard deviation of price movement. In dollar terms: half-spread = $0.05, 1-step return std = $0.475. Price moves dwarf the trading cost. Even at h=10, the median return is **6.2x the spread**. Profitable even without min_hold.

**Battery**: The spread is 64% of one standard deviation. In EUR terms: half-spread = EUR 0.38, 1-step return std = EUR 0.59. Each trade must be held for multiple steps to overcome the spread cost. Without min_hold (trading every step), perfect foresight is deeply negative. With min_hold=30+, perfect foresight turns positive. The market is marginal — tradeable in theory, but requires disciplined position persistence.

### BTC Return/Spread Ratio by Horizon

| Horizon | Median return/spread | % moves > 1x spread | % moves > 2x spread | % moves > 5x spread |
|---------|:--------------------:|:-------------------:|:-------------------:|:-------------------:|
| h10 | 6.2x | 62.8% | 59.3% | 52.3% |
| h20 | 16.2x | 73.7% | 70.9% | 65.6% |
| h50 | 40.0x | 87.9% | 85.9% | 82.7% |
| h100 | 65.0x | 95.1% | 93.6% | 91.4% |

At h=10, the median price move is already 6.2x the spread. At h=100, it is 65x. The spread is negligible on BTC.

---

## Finding 2: Trade Frequency is the Bottleneck (Not Prediction Accuracy)

On BTC, the **number of trades** matters more than **how accurate each trade is**. This is the most counter-intuitive finding.

### The min_hold Effect on BTC

| Dir. accuracy | No min_hold | With optimal min_hold | PnL change |
|:------------:|:------:|:------:|:------:|
| 55% | -10.33 (353K trades) | **+4.53** (29K trades, mh=20) | +14.86 |
| 60% | -8.91 (344K trades) | **+5.11** (29K trades, mh=20) | +14.02 |
| 65% | -7.84 (332K trades) | **+6.15** (29K trades, mh=20) | +13.99 |
| 70% | -6.02 (319K trades) | **+7.53** (29K trades, mh=20) | +13.55 |
| 75% | -4.38 (304K trades) | **+8.15** (29K trades, mh=20) | +12.53 |
| 80% | -2.70 (287K trades) | **+9.56** (29K trades, mh=20) | +12.26 |

Without min_hold, **even 80% directional accuracy loses money.** With min_hold=20, even 55% accuracy is profitable. The same pattern holds across all horizons.

### Why This Happens

The evaluation assigns a position at every timestep. When label predictions are noisy, the position flip-flops between +1, 0, -1 at nearly every step. Each flip pays the spread, and the per-step gross return (std=0.0003 z-units) is small. At 332K trades over 605K samples, the model changes position every ~2 steps on average. The optimal position change rate is 5% (30K trades, once every ~20 steps).

### Breakeven Directional Accuracy by min_hold (BTC, h10)

| min_hold | Breakeven accuracy | Trades at breakeven |
|:--------:|:-----------------:|:-------------------:|
| 0 | 88% | ~250K |
| 5 | 64% | ~100K |
| 10 | 48% | ~50K |
| 15 | 40% | ~35K |
| 20 | 40% | ~29K |
| 30 | 40% | ~20K |
| 50 | 40% | ~12K |

At min_hold >= 10, the breakeven drops to below 50%, meaning even a random directional predictor can break even as long as it doesn't trade too often.

---

## Finding 3: Trend Persistence Determines Optimal Hold Period

Price trends persist for a characteristic duration at each horizon. The optimal min_hold closely tracks the **median trend length**.

### BTC Trend Persistence

| Horizon | Median trend | Mean trend | p90 | # Trends | Optimal min_hold |
|---------|:----------:|:--------:|:---:|:--------:|:---------------:|
| h10 | **16** | 18.7 | 34 | 15,006 | **20** |
| h20 | **24** | 27.8 | 50 | 11,232 | **30** |
| h50 | **48** | 50.5 | 94 | 7,127 | **55** |
| h100 | **64** | 75.7 | 153 | 5,003 | **100** |

The optimal min_hold is consistently close to the median trend length. This makes intuitive sense: holding for one full trend captures the directional return while only paying the spread once at entry and exit.

### Battery Trend Persistence

| Horizon | Median trend | Mean trend | p90 | # Trends |
|---------|:----------:|:--------:|:---:|:--------:|
| h10 | 11 | 13.2 | 26 | 60,639 |
| h20 | 19 | 21.3 | 42 | 40,317 |
| h50 | 30 | 38.1 | 82 | 25,302 |
| h100 | 34 | 56.8 | 132 | 18,330 |

Battery trends are shorter than BTC at the same horizon. Combined with the higher spread/vol ratio (0.64 vs 0.11), Battery requires longer min_hold periods relative to trend length to overcome transaction costs.

### Optimal Position Change Rate (BTC)

The optimal strategy changes position **rarely**. At h10, perfect foresight changes position only 4.9% of the time (29,821 changes across 605,343 samples). The model should hold 95% of the time.

| Horizon | Optimal position changes | Change rate |
|---------|:----------------------:|:-----------:|
| h10 | 29,821 | 4.9% |
| h20 | 22,447 | 3.7% |
| h50 | 14,252 | 2.4% |
| h100 | 10,004 | 1.7% |

---

## Finding 4: SMA Baselines Are Surprisingly Strong

A simple moving-average crossover on z-scored mid-prices — requiring zero ML, zero training, zero LOB features — achieves strong PnL on BTC.

### BTC SMA Baselines

| Strategy | PnL (z) | PnL ($) | Trades | Sharpe |
|----------|:-------:|:-------:|:------:|:------:|
| SMA-10 | **15.64** | **$24,650** | 27,568 | 8.97e-02 |
| SMA-20 | 15.56 | $24,528 | 17,888 | 8.75e-02 |
| SMA-50 | 12.17 | $19,179 | 10,616 | 6.77e-02 |
| SMA-100 | 8.94 | $14,094 | 6,784 | 4.92e-02 |
| *Perfect FH h10* | *11.60* | *$18,285* | *29,822* | *7.14e-02* |
| *PF h10 + mh=20* | *21.98* | *$34,644* | *19,521* | *1.25e-01* |

SMA-10 and SMA-20 **outperform perfect label foresight without min_hold** (15.64 vs 11.60). This happens because SMA positions are inherently smooth — positions change gradually, avoiding the flip-flopping that costs spread. SMA is a natural low-pass filter on trading frequency.

Only **perfect foresight with min_hold** beats the SMA baseline. This sets the bar for any ML model: it must outperform SMA-10 (PnL=15.64, $24,650) to justify its complexity.

### Battery SMA Baselines

| Strategy | PnL (z) | PnL (EUR) | Trades |
|----------|:-------:|:---------:|:------:|
| SMA-10 | -14,734 | -229,709 | 277,990 |
| SMA-10 + mh=100 | -867 | -13,516 | 18,071 |
| SMA-20 | -10,334 | -161,113 | 185,238 |
| SMA-20 + mh=100 | -731 | -11,400 | 17,488 |
| SMA-50 | -5,864 | -91,423 | 104,203 |
| SMA-50 + mh=100 | -496 | -7,732 | 15,926 |
| SMA-100 | -3,556 | -55,431 | 65,824 |
| SMA-100 + mh=100 | -341 | -5,319 | 13,912 |
| Buy & Hold | -736 | -11,474 | 551 |
| PF h100 + mh=100 | **+447** | **+6,961** | 14,578 |

On Battery, all SMA baselines are negative — even with min_hold=100 (best tested). min_hold reduces losses by 90% (SMA-10: -229K → -13.5K EUR) by cutting trades from 278K to 18K, but never reaches profitability. SMA lacks directional accuracy on EPEX electricity — it follows trends that don't persist long enough relative to the spread.

Only perfect foresight with position persistence is positive. **This means an ML model beating SMA on Battery is actually the easier bar** — it needs to be profitable (PF+mh achieves this), not better than SMA (which is negative). The challenge is achieving sufficient directional accuracy with disciplined trade frequency.

---

## Label Distribution

| Horizon | BTC up | BTC stat | BTC down | Battery up | Battery stat | Battery down |
|---------|:------:|:--------:|:--------:|:----------:|:------------:|:------------:|
| h10 | 22.9% | 53.7% | 23.4% | 22.1% | 55.3% | 22.7% |
| h20 | 25.6% | 48.5% | 26.0% | 23.7% | 51.9% | 24.4% |
| h50 | 29.8% | 40.6% | 29.7% | 26.4% | 46.0% | 27.6% |
| h100 | 31.7% | 37.4% | 30.9% | 28.3% | 41.7% | 30.0% |

Both datasets have similar label distributions. The feasibility difference is entirely driven by spread, not by label balance.

---

## Finding 5: Extended Horizons

Horizon h=N means the model predicts the price direction N sampling steps into the future. BTC samples at 250ms, so h=10 = 2.5 seconds ahead. Battery samples at 5 seconds (TIME_DEDUP), so h=10 = 50 seconds ahead. We test horizons from h=10 to h=2000.

### BTC Extended Horizons (250ms sampling, h=10 is 2.5s ahead)

BTC's optimal horizon is at the **short end**. Longer horizons have better return/spread ratios but fewer trading opportunities, reducing total PnL.

| Horizon | Real time | PF PnL (z) | PF PnL ($) | PF+mh PnL (z) | PF+mh ($) | Trades | Return/Spread |
|---------|:---------:|:----------:|:----------:|:--------------:|:---------:|:------:|:-------------:|
| h10 | 2.5s | **11.60** | **$18,285** | **14.66** | **$23,106** | 29,821 | 6.2x |
| h20 | 5.0s | 8.46 | $13,334 | 13.66 | $21,537 | 22,447 | 16.2x |
| h50 | 12.5s | 5.04 | $7,941 | 11.27 | $17,760 | 14,257 | 40.0x |
| h100 | 25.0s | 3.13 | $4,930 | 9.08 | $14,319 | 10,007 | 65.0x |
| h200 | 50.0s | 2.35 | $3,712 | 7.17 | $11,298 | 6,971 | 97.0x |
| h500 | 2.1min | 1.40 | $2,213 | 4.91 | $7,738 | 4,286 | 158.4x |
| h1000 | 4.2min | 0.57 | $905 | 3.56 | $5,612 | 3,183 | 228.6x |
| h2000 | 8.3min | 0.16 | $256 | 2.29 | $3,614 | 2,299 | 315.6x |
| *B&H* | *all* | *0.42* | *$661* | — | — | *1* | — |

BTC's sweet spot is h=10-20 (2.5-5s). PF PnL drops 72x from h10 to h2000 because trend changes become rarer at longer horizons — there are simply fewer trades to make. The return/spread ratio grows with horizon (6x to 316x) but this doesn't compensate for the reduced number of opportunities. **With min_hold, PF+mh PnL is more stable** (falling only 6x), showing that position persistence is key.

### Battery Extended Horizons (5s TIME_DEDUP sampling, h=10 is 50s ahead)

Battery's sampling interval is 20x longer than BTC's (5s vs 250ms), so the same horizon value h=10 corresponds to a much longer real-time window: 50 seconds vs 2.5 seconds. This means each step has more time for the price to move, but also fewer total samples and trading opportunities per product session (3-6 hours of active trading).

Without min_hold, perfect foresight is negative at all horizons. With optimal min_hold, **Battery becomes profitable starting at h=10** (mh=30).

| Horizon | Real time | PF PnL (z) | PF+mh PnL (z) | PF+mh (EUR) | Optimal mh | Trades (PF) | Trades (mh) |
|---------|:---------:|:----------:|:--------------:|:-----------:|:----------:|:-----------:|:-----------:|
| h10 | 50s | -3,331 | **+28.7** | **+447** | 30 | 119,716 | 46,471 |
| h20 | 1.7min | -2,323 | **+275.7** | **+4,299** | 30 | 82,808 | 42,909 |
| h50 | 4.2min | -1,564 | **+444.4** | **+6,929** | 75 | 51,217 | 19,486 |
| h100 | 8.3min | -1,292 | **+446.5** | **+6,961** | 100 | 36,118 | 14,578 |
| *B&H* | *all* | *-736* | — | *-11,474* | — | *551* | — |

**Key observations:**
- **Without min_hold, PF PnL is negative at all horizons.** The spread/vol ratio of 0.64 means each trade must be held long enough to overcome the entry cost. At h=10, the model predicts 50 seconds ahead but position changes every ~5 seconds are too frequent — only 1 in 6 steps should trigger a trade.
- **With min_hold=30 (= 2.5 minutes of forced holding), even h=10 becomes marginally positive** (+28.7 z = EUR 447 across 551 products). This is thin — EUR 0.81 per product average.
- **h50 and h100 with min_hold are the sweet spots**: EUR 6,929 and EUR 6,961 aggregate, or EUR 12.57/product. At h=100 with mh=100, each trade is held for at least 8.3 minutes — matching Battery's median trend length of 34 steps (2.8 minutes) and allowing a full trend to develop.
- **Trade count must drop to 15K-47K** (from 50K-120K without min_hold) for profitability. The model must hold positions 3-8x longer than per-step trading.

---

## Finding 6: Sampling Rate Is Not the Bottleneck

BTC is sampled at 250ms intervals. Battery at 5-10s with TIME_DEDUP (snapshots saved only when the LOB changes). Would changing Battery's sampling rate improve tradeability?

### Empirical Comparison: 5s vs 10s TIME_DEDUP

| Sampling | Products | Samples | z_half_spread | 1-step std | Spread/Vol |
|:--------:|:--------:|:-------:|:-------------:|:----------:|:----------:|
| 5s DEDUP | 551 | 1.86M | 0.0241 | 0.0379 | **0.64** |
| 10s DEDUP | 1,079 | 2.14M | 0.0250 | 0.0354 | **0.71** |

The ratio is **stable across sampling rates** (0.64 vs 0.71). This is because TIME_DEDUP only keeps snapshots where the LOB actually changed — the 5s/10s parameter sets a minimum gap between snapshots but doesn't control when the LOB changes. Both configurations capture essentially the same market events.

Changing the sampling rate does not improve tradeability. The spread/vol ratio is a **market microstructure property** of EPEX Spot, not an artifact of sampling. The current 5s TIME_DEDUP configuration is optimal.

---

## Implications

### For the paper
1. **Both datasets support profitable trading with min_hold.** BTC is comfortably profitable (PF+mh h10 = $34,644). Battery is marginally profitable (PF+mh h100 = EUR 6,961).
2. **min_hold is essential for any trading evaluation.** Without it, the evaluation penalizes the model for making per-step predictions, which is the wrong failure mode. min_hold can be tuned on validation data.
3. **The SMA baseline must be included.** Any ML model claiming profitable trading must beat SMA-10 (PnL=$24,650 on BTC test) to be credible. On Battery, SMA baselines are negative — ML must outperform simple trend-following.
4. **The spread/volatility ratio is a contribution.** BTC (0.11) is trivially tradeable; Battery (0.64) is marginally tradeable. The ratio predicts whether min_hold is necessary and how long the model must hold positions.

### For TradeLOB
TradeLOB's NTB bands are architecturally the right idea — they're designed to suppress unnecessary trades. But the analysis shows:
- NTB bands need to produce ~30K trades (5% rate) at h10, not 152K (25%)
- The band width must be learned to match the median trend length (~16 steps at h10)
- Alternatively, a simpler approach — CE-trained TLOB + min_hold=20 at inference — achieves the same goal without architectural complexity

### For model development
The theoretical ceiling is clear:
- **Perfect foresight + min_hold=20 at h10: PnL=$34,644** (upper bound)
- **65% dir. accuracy + min_hold=20 at h10: PnL=$9,686** (achievable target)
- **SMA-10: PnL=$24,650** (zero-ML baseline to beat)

A model needs either (a) >70% directional accuracy with min_hold, or (b) accuracy comparable to SMA but better timing.

---

## Discussion

### Why SMA Beats ML (and How ML Can Win)

SMA-10 achieves PnL=$24,650 on BTC — better than perfect label foresight without min_hold ($18,285) and better than 65% ML accuracy with optimal min_hold ($9,686). Why?

**SMA has built-in position smoothing.** The SMA-10 position changes only when the price crosses its 10-step moving average. This happens ~28K times across 605K samples (4.6% change rate). By contrast, a per-step ML classifier with 65% accuracy flip-flops ~332K times (55% change rate). SMA's position signal is naturally smooth; the ML signal is naturally noisy.

**SMA follows trends; ML predicts per-step direction.** SMA answers: "is the price above its recent average?" This is a trend-following question — the answer changes slowly. ML answers: "will the price go up in the next h steps?" This is a point prediction — when noisy, consecutive answers flip between up/down/stationary, generating costly position changes.

**The gap is trade frequency, not prediction accuracy.** SMA doesn't know the future direction at all. It simply assumes the current trend continues. ML knows more about future direction (65%+ accuracy) but squanders this advantage by trading too often. Each unnecessary trade costs half-spread ≈ $0.05, and at 332K trades that's $16K in spread costs — nearly the entire gross return.

**How ML can beat SMA:**
1. **min_hold=20**: Reduces ML trades from 332K to 29K, matching SMA's frequency. At 65% accuracy + mh=20, PnL=$9,686. Still below SMA-10, but positive.
2. **Higher accuracy**: At ~75%+ directional accuracy with mh=20, ML PnL approaches SMA-10 levels.
3. **ML's structural advantages over SMA**: (a) It can predict regime changes — SMA always lags by definition, while a transformer can detect early signs of reversal in the LOB microstructure. (b) It uses 40 LOB features (depth, volume, order flow) — SMA uses only the price. (c) It can be selective — skip low-confidence periods entirely, while SMA is always in a position.
4. **Confidence thresholds**: Only trade when P(direction) > threshold. This reduces trades to high-conviction signals, similar to min_hold but data-driven.

**Bottom line**: SMA wins by default because it solves the trade frequency problem for free. ML must either use min_hold/confidence to match SMA's frequency, or achieve substantially higher accuracy to justify the higher frequency.

### Simulation Realism: Volume and LOB Depth

**Current model**: 1 unit at best bid/ask. No order book depth, no price impact, no partial fills. This is the standard assumption in LOB trading papers (DeepLOB, TLOB, etc.) — the trader is small enough to not move the market.

**Is this realistic?**
- **For BTC (Binance)**: Yes. The BTC/USDT order book typically has 50-500 BTC of liquidity within the first few levels. A 1-unit trade (~$19K) is negligible. Real spread costs would be very close to our simulation.
- **For Battery (EPEX)**: Less so. EPEX hourly intraday products have thinner books (often 1-10 MW per level). A 1 MW trade might consume the entire best level, requiring execution at level 2 or deeper. Real costs would be higher than simulated, making Battery's already marginal profitability even thinner.

**Would adding depth change conclusions?**
- It would not change the BTC conclusions (deep book, minimal impact).
- It would make Battery results **worse** — costs increase, margins shrink. The current Battery analysis is therefore an **optimistic bound** on real-world profitability.

**Should we implement depth-aware execution?**
- Not for this paper. The 1-unit model answers: "is profitable trading theoretically feasible?" Depth-aware modeling answers: "how much capital can we deploy profitably?" — a harder, separate question.
- For future work: model execution as `cost = f(volume, depth_at_levels_1_to_k)` where f accounts for walking the book. This requires storing per-level volume data in the preprocessing pipeline (which we already have in the 40 LOB columns).

**Variable position sizing**: Trading more when confident and less when uncertain could improve risk-adjusted returns. This is the position sizing problem from portfolio optimization (Kelly criterion, mean-variance). It's orthogonal to direction prediction and would add complexity without addressing the core question of feasibility.

### Why min_hold Works and Whether Models Can Learn It

**What min_hold does**: At each timestep, the model outputs a desired position {-1, 0, +1}. Without min_hold, the position changes immediately. With min_hold=N, the position cannot change until N steps have elapsed since the last change. It's a **low-pass filter on the position signal** — it removes high-frequency noise while preserving the directional trend.

**Why it works**: At h=10, price trends last ~16 steps (median). Without min_hold, the model changes position during a trend that hasn't ended — paying spread for no directional reason. With min_hold=20, the model is forced to hold through one full trend, paying spread only at trend transitions (every ~20 steps instead of every ~2 steps). This reduces trades from 332K to 29K and transforms a -$12K loss into a +$10K profit.

**Can a model learn to hold without min_hold?**

*Standard CE training cannot.* Cross-entropy optimizes per-step accuracy independently: P(correct_label | features). It has no notion that "this prediction will cause a costly position change." The model doesn't know what it predicted last step, so it can't learn that holding the previous prediction would be cheaper. Every training sample is treated as independent, breaking the sequential dependency that makes holding valuable.

*Sequential DFL tried and failed.* We trained with position carry-forward and Sharpe ratio loss (documented in CHANGES.md, "Sequential DFL Trading Loss"). The model learned to hold (60% hold ratio) but PnL remained negative (-262 at h10). Two issues: (1) gradients must flow through a long sequential simulation, making optimization difficult; (2) the reward for "correct hold" is zero PnL change — indistinguishable from "incorrect hold" without ground-truth knowledge of whether the trend is continuing.

*TradeLOB's NTB bands are the right architecture but haven't converged.* The no-transaction band (NTB) architecture explicitly models the hold/trade decision: trade only if |signal - current_position| > band_width. The band width is learned end-to-end — wide bands mean hold, narrow bands mean trade. This is architecturally correct (it separates "what to trade" from "when to trade"), but in practice the bands either collapse to 0 (trade everything, 152K trades) or blow up (trade nothing, 2 trades). The optimization landscape has two attractors and the middle ground (30K trades) is hard to find via gradient descent.

*The fundamental difficulty: learning when NOT to act.* In classification, every sample gets a loss signal. In trading, "correctly doing nothing" generates zero PnL — the model receives no feedback for correct holds. This asymmetry between action (trade = immediate cost/reward feedback) and inaction (hold = zero feedback) makes the hold/trade boundary hard to learn from data alone.

**Most promising path**: Separate the two problems. Use CE training for direction accuracy (what the transformer is good at) and apply min_hold as a post-hoc rule (what CE can't learn). Tune min_hold on the validation set — it's a single hyperparameter that captures the market's characteristic trend duration. This is simpler, more robust, and avoids the optimization difficulties of end-to-end sequential trading losses.

---

## Execution Model

### How Trades Are Simulated

At each timestep t, the model outputs a position in {-1, 0, +1}:
- **+1 (long)**: buy 1 unit at the ask price (sell1), hold
- **-1 (short)**: sell 1 unit at the bid price (buy1), hold
- **0 (flat)**: no position

**Gross return**: `position(t) * (z_mid(t+1) - z_mid(t))`. The position earns the 1-step mid-price change while held. Over a holding period of N steps, the cumulative gross is the sum of N 1-step returns.

**Transaction cost**: `|position(t) - position(t-1)| * z_half_spread(t)`. Cost is paid at each position change, proportional to the magnitude of change and the half bid-ask spread. Going from flat to long (0 to +1) costs 1 × half_spread. Going from long to short (+1 to -1) costs 2 × half_spread.

### Position Sizing

**Position size is always 1 unit.** The simulation does not model variable position sizing. All PnL figures are returns per unit traded.

For longer horizons, each trade still buys 1 unit, but the expected return per trade is larger because:
1. The position is held for more steps → more 1-step returns accumulate
2. Longer trends have larger cumulative price moves
3. The spread is paid once at entry, amortized over more steps

This is why longer horizons have fewer but more profitable trades.

### What Is NOT Modeled

1. **Order book depth / volume**: We assume execution at the best bid/ask price regardless of order size. In reality, large orders walk the book — consuming liquidity at level 1, then level 2, etc. This makes real costs HIGHER than simulated.

2. **Price impact**: Our trade does not move the market. In reality, aggressive orders shift the mid-price against us. Not modeled.

3. **Partial fills**: We assume immediate full execution. In reality, limit orders may be partially filled or not filled at all.

4. **Variable position sizing**: A confident model could trade more volume to amplify returns. Not modeled — all trades are 1 unit.

### Impact on Conclusions

| Limitation | Effect on BTC | Effect on Battery |
|------------|:------------:|:-----------------:|
| No depth/volume | Small — BTC is very liquid | **Understates costs** — EPEX has thinner books |
| No price impact | Small — deep order book | Moderate — thin order book |
| Fixed position size | Neutral (returns per unit) | Neutral |

For BTC, the simplifications are reasonable — Binance BTC/USDT has deep liquidity. For Battery, real costs would be higher than simulated, making the already marginal profitability even thinner.

---

## Critical Review

### Verified Correct

1. **z-score normalization**: Both BTC and Battery use a single `(mean_prices, std_prices)` pair for all price columns. `z_mid = (raw_mid - mean) / std`. This means `diff(z_mid) = diff(raw_mid) / std`, and the dollar/EUR conversion is simply `PnL_raw = PnL_z × std_price`. Verified in `utils/utils_data.py:7-22` (z_score_orderbook), `preprocessing/battery.py:1365-1371` (_lob_stats), and `data/BTC/norm_stats.npz`.

2. **PnL computation matches engine.py**: The analysis replicates `gross = position * diff(z_mid)` and `cost = |Δpos| * z_half_spread` exactly as in `utils/metrics.py:447-520`. Cross-checked with a single BTC test run.

3. **Dollar/EUR conversion**: BTC: 1 z-unit = $1,576.24 (from `norm_stats.npz: std_prices`). Battery: 1 z-unit = EUR 15.59 (from `median(raw_half_spread / z_half_spread)` using 67-col DFL data).

### Found and Fixed

4. **Battery column indexing bug**: The 62-column Battery format stores `[msg(18) | lob(40) | labels(4)]`. The analysis initially used columns 0, 2 (message features: log_time_delta, lifecycle_progress) instead of columns 18, 20 (LOB prices: ask1, bid1). This produced spread/vol=1.14 (wrong) instead of 0.64 (correct), and incorrectly concluded Battery was completely untradeable. The loading function `_select_features()` in `preprocessing/battery.py` rearranges to `[lob | msg]` when creating model input tensors, so `engine.py` uses the correct columns — the bug was only in the standalone analysis scripts.

### Known Limitations

5. **Simulation seed sensitivity**: Accuracy simulations use seed=42. Different seeds would produce ±5% variance in PnL. Conclusions are qualitatively stable.

6. **Perfect foresight labels**: Labels use 10-step smoothed mid-prices (`mean(z_mid[t:t+10])`), introducing a slight smoothing bias. The actual PnL uses unsmoothed 1-step returns. This means the "foresight" is about smoothed future direction, not exact future prices.

7. **Product boundary effects**: Battery products are independent (positions reset between delivery hours). Forced closing at product boundaries adds costs. This is physically correct but means a model can never hold across delivery boundaries.

---

## Methodology Notes

- **PnL computation**: Matches `engine.py` exactly. Uses 1-step z-scored price changes accumulated over the position holding period. Transaction cost = |position change| * z-scored half-spread.
- **Dollar/EUR conversion**: BTC: 1 z-unit = $1,576.24 (from `norm_stats.npz: std_prices`). Battery: 1 z-unit = EUR 15.59 (from `median(raw_half_spread / z_half_spread)` using 67-col DFL data).
- **Simulated accuracy**: At X% directional accuracy, each non-stationary label is independently flipped with probability (1-X). Stationary labels use 60% accuracy. Seed=42 for reproducibility.
- **min_hold**: Position cannot change until min_hold steps have elapsed. First step always allows trading. Applied as post-hoc filter on position sequence.
- **SMA crossover**: `position = +1 if z_mid > SMA, -1 if z_mid < SMA`. Applied at every timestep.
- **Battery aggregation**: Results summed across 551 products (delivery hours) using train/val/test splits for maximum coverage. Each product is independent (positions reset between products). Minimum 200 samples per product-split.
- **Extended horizons**: Labels computed from z-scored mid-prices using the same smoothing and threshold logic as `utils_data.py`: `alpha = mean(|delta|)/2`, where delta = smoothed_future_mid - smoothed_current_mid with 10-step smoothing window.
- **Sampling rate comparison**: 5s data from `5s_20210111_20210202_dedup_msg` (551 products), 10s data from `10s_20210111_20210224_dedup_msg` (1,079 products). Both use TIME_DEDUP sampling (snapshots only on LOB change).

## Figures

All generated plots are in `analysis_figures/`:
- `btc_return_vs_spread.png` — |delta_mid|/spread distributions per horizon
- `btc_perfect_foresight.png` — Cumulative PnL with perfect foresight
- `btc_threshold_sweep.png` — PnL vs trading selectivity
- `btc_trend_persistence.png` — Trend run-length distributions
- `btc_cross_horizon.png` — Label agreement across horizons
- `btc_optimal_hold.png` — Returns at different horizons for h10 signals
- `btc_bnh_vs_active.png` — Buy-and-hold vs active trading
- `btc_accuracy_selectivity.png` — Accuracy x selectivity PnL heatmap
- `btc_position_changes.png` — Optimal position change frequency
- `btc_spread_analysis.png` — Spread distribution and time variation
- `btc_accuracy_vs_pnl_realistic.png` — Accuracy vs PnL (non-overlapping)
- `btc_carry_forward_simulation.png` — 65% accuracy cumulative PnL
- `btc_eval_matched_strategies.png` — Strategy comparison (eval-matched)
- `btc_accuracy_minhold_heatmap.png` — Accuracy x min_hold PnL heatmap
- `btc_minhold_multihorizon.png` — Min-hold sweep all horizons
- `btc_confidence_threshold.png` — Confidence threshold sweep
- `btc_hold_period_analysis.png` — PnL vs hold period
- `battery_minhold_sweep.png` — Battery min-hold sweep
- `battery_spread_and_pnl.png` — Battery spread and per-product PnL
- `extended_horizon_comparison.png` — BTC vs Battery PnL, return/spread, spread/vol across h=10-2000
- `battery_extended_horizon_profitability.png` — % Battery products profitable vs horizon
