# Cost-Aware Cross-Entropy (CA-CE) — Ablation Idea

## Motivation

All DFL variants failed because PnL gradients are too weak for encoder training.
The TLOB encoder trained with standard CE achieves F1=0.76, proving it learns useful
LOB features. The question is: can we modify CE to produce features that are more
relevant for *profitable* trading, without abandoning the CE training framework?

## Idea

Weight each sample's CE loss by its economic relevance:

```
L_CA-CE = (1/N) * sum_i [ w_i * CE(logits_i, y_i) ]

w_i = class_weight[y_i] * clip(|delta_mid_i| / half_spread_i, min=epsilon)
```

- Samples where `|delta_mid| >> half_spread`: worth learning well — profitable trades
- Samples where `|delta_mid| << half_spread`: irrelevant for trading — getting them
  right or wrong doesn't matter because you shouldn't trade them anyway

This doesn't teach the model *when* to trade. It teaches the model to learn better
representations for the samples that *matter* for trading. The "when to trade"
decision is handled at inference time via confidence thresholds.

## Implementation

~15 lines in `models/engine.py` at `_multi_horizon_loss`:

1. Add `loss_type="cost_aware_ce"` option in config
2. In loss computation: use `reduction='none'` in CE, compute per-sample weights
   from `delta_mid` and `half_spread`, multiply, then mean-reduce
3. `delta_mid` and `half_spread` already available via DFL data in `MultiHorizonDataset`

## Inference

Use existing `evaluate_trading.py --sweep-thresholds` to find optimal
`(confidence_threshold, min_hold)` on validation set. No new inference code needed.

## Paper Story

"Same architecture (TLOB), different loss → profitable trading."
Supports the thesis: **loss > architecture**.

## Ablation Table

| Method                     | h10 PnL | h10 Trades | h20 PnL | ... |
|----------------------------|---------|------------|---------|-----|
| CE + no filter             |         |            |         |     |
| CE + optimal threshold     |         |            |         |     |
| CA-CE + no filter          |         |            |         |     |
| CA-CE + optimal threshold  |         |            |         |     |
| TradeLOB (NTB)             |         |            |         |     |
| DFL variants               |         |            |         |     |

## Risk

If `|delta_mid| / half_spread` distribution is narrow (most samples have similar
ratios), CA-CE degenerates to standard CE. Check distribution first.
Mitigation: use `w_i = max(0, |delta_mid_i| / half_spread_i - 1)` to only
upweight samples where expected return exceeds the spread.
