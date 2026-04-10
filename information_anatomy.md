# Plan: The Information Anatomy of Limit Order Books

## Context

The user has built a strong LOB prediction platform (TLOB, FuseLOB, NexusLOB) and documented two key findings: (1) all snapshot-based models hit the same performance ceiling regardless of architecture, (2) FuseLOB breaks this ceiling by adding raw events (+1.6% F1 h10, +55% PnL h100 — notably, the PnL improvement is 34x larger than the F1 improvement, showing classification metrics badly underestimate the value of event information). They want a genuinely novel research contribution for NeurIPS.

**Constraints:** Single RTX 4080 (16GB), ~5M samples across 3 markets (BTC, FI-2010, Battery/EPEX), events only for Battery. Can implement fast with coding agents. Can rent GPU if needed.

## Primary Direction

### "The Information Anatomy of Limit Order Books: What Events Know That Snapshots Forget"

**Core thesis:** Periodic LOB snapshots are a lossy compression of market dynamics. We quantify exactly what is lost, what it costs, and whether learned compressions can recover it.

**Why this works:** The contribution is a SCIENTIFIC FINDING + METHODOLOGY, not "our architecture is 2% better." The architecture (FuseLOB, Perceiver) is a TOOL for measuring the phenomenon, not the contribution itself.

**Key motivating observation:** FuseLOB's +1.6% F1 but +55% PnL disconnect shows that F1 is a poor proxy for the true value of event information. Events carry information that is disproportionately valuable for trading decisions — the standard metric (F1) almost completely masks this.

**Paper structure:**
1. **The Snapshot Ceiling** (motivation) — All snapshot-based architectures plateau (TLOB, DeepLOB, MLP, GBM, PatchLOB, DiffLOB, MambaLOB). Table showing this. The ceiling isn't an architecture problem — it's an INPUT REPRESENTATION problem.
2. **The F1-PnL Disconnect** — Events improve F1 by 1.6% but PnL by 55%. Why? Classification accuracy is nearly saturated, but the QUALITY of predictions (confidence, timing, directional precision) improves dramatically. This reframes what "information" means for LOB.
3. **The Compression-Performance Frontier** — Rate-distortion curve: Perceiver at query counts 1→32 vs snapshot at depths 1→10. Plot both F1 AND PnL on the frontier (two y-axes or separate plots). Does learned compression Pareto-dominate hand-designed snapshots?
4. **Information Decomposition** — WHICH event features drive the improvement? Mask event features by type (action codes, timing, quantity, aggressiveness, etc.). Isolate the dominant information source.
5. **Cross-Market Validation** — Add NASDAQ ITCH (free L3 events) as second event-equipped market. Show findings generalize beyond energy.
6. **Practical Implications** — The frontier tells practitioners: "at X compression, you lose Y% of trading value." Decision tool for event infrastructure investment.

**Core contributions (3 claims):**
1. "Standard 10-level snapshots capture X% of available predictive information" — falsifiable
2. "Learned compressions Pareto-dominate hand-designed snapshots above N queries" — empirically testable
3. "The F1-PnL disconnect is explained by [specific event feature], which provides trading-relevant information invisible to classification metrics" — decomposition

---

## Experiments

### Experiment 1: Perceiver Query Sweep (compression curve — learned side)
**Goal:** Map the learned compression frontier by varying Perceiver query count.
**Method:** Run FuseLOB with `n_perceiver_queries` ∈ {1, 2, 4, 8, 16, 32} on Battery. Measure F1 AND PnL at all horizons.
**What we learn:** How much compression can the Perceiver tolerate before trading performance degrades? Is there an "information cliff"?
**Infrastructure:** `n_perceiver_queries` already configurable in `config/config.py:172`. Just change the value and run.
**Files:** `config/config.py` (parameter), `models/fuselob.py:379` (already accepts n_queries)

### Experiment 2: Snapshot Depth Ablation (compression curve — hand-designed side)
**Goal:** Map the snapshot frontier by varying LOB depth.
**Method:** Truncate LOB to {1, 3, 5, 7, 10} levels. Each level = 4 features (sell_price, sell_vol, buy_price, buy_vol). Run TLOB on each. LOB layout is interleaved: `[sell1, vsell1, buy1, vbuy1, ..., sell10, vsell10, buy10, vbuy10]`, so truncating to N levels = first N*4 LOB columns.
**What we learn:** How much performance drops as LOB depth decreases. Compare to Perceiver curve at matched dimensionality.
**Files:** Need to add `n_lob_levels_used` parameter to config + modify data loading in `run.py` to truncate input features. Also adjust `feature_size` computation.

### Experiment 3: Event Feature Masking (information decomposition)
**Goal:** Identify WHICH event features drive FuseLOB's improvement.
**Method:** Run FuseLOB multiple times, each time zeroing out a specific event feature group:
- Mask action codes (indices 0, 6, 7, 8) → removes trade/cancel/iceberg/exec_restriction distinction
- Mask timing (indices 4, 9) → removes time_delta, order_age
- Mask price/quantity (indices 2, 3, 10) → removes price_relative, quantity_log, signed_quantity
- Mask side (index 1) → removes directional information
- Mask revision (index 5) → removes algo intensity signal
**What we learn:** Which information source is dominant. If masking action codes kills most of the gain, trade/cancel disambiguation is the key signal.
**Files:** Add `event_feature_mask` parameter to FuseLOB config. Modify `EventEmbedding` in `models/fuselob.py` to zero out masked features before embedding.

### Experiment 4: F1-vs-PnL Analysis
**Goal:** Rigorously document the F1-PnL disconnect across compression levels.
**Method:** For EVERY point on both compression curves (Experiments 1 & 2), compute BOTH F1 AND PnL. Plot F1 frontier and PnL frontier separately. Show that the curves have different shapes — compression that barely affects F1 can dramatically affect PnL.
**What we learn:** How much trading-relevant information is invisible to classification metrics.
**Files:** Existing `evaluate_trading.py` already computes PnL. Run it on each checkpoint.

### Experiment 5: Cross-Market Validation (NASDAQ ITCH)
**Goal:** Show findings generalize beyond energy markets.
**Method:** Download free NASDAQ ITCH sample (1 day, all tickers). Parse into event+snapshot format matching Battery pipeline. Run key experiments (Perceiver sweep, snapshot depth sweep) on ITCH data.
**What we learn:** Whether the information anatomy findings are universal or energy-specific.
**Files:** NEW `preprocessing/nasdaq_itch.py`. Adapt event tokenization from `preprocessing/events.py`.

### Experiment 6 (stretch): Information-Theoretic Bounds
**Goal:** Compute MI estimates to support the "X% information loss" claim.
**Method:** Variational MI estimation (InfoNCE bound) between representations and labels. Compare I(events; label) vs I(snapshot; label) vs I(compressed; label).
**What we learn:** Formal quantification of information loss.
**Risk:** MI estimation in high dimensions is noisy. May need to be supplementary rather than core contribution.
**Files:** NEW `analysis/information.py`.

## Implementation Steps

1. **Perceiver sweep** — Change config value, run 6 experiments. No code changes needed.
2. **Snapshot depth ablation** — Add `n_lob_levels_used` to config. Modify feature truncation in `run.py` data loading. Run 5 experiments.
3. **Event feature masking** — Add masking parameter to FuseLOB config. Small change to `EventEmbedding`. Run 5 experiments.
4. **PnL evaluation** — Run `evaluate_trading.py` on all checkpoints from steps 1-3.
5. **Plotting/analysis** — Create rate-distortion plots (F1 and PnL), decomposition charts.
6. **NASDAQ ITCH parser** — New preprocessing file. Download data, parse, run subset of experiments.
7. **MI estimation** (if time) — Implement InfoNCE, run on Battery.

## Key Files to Modify
- `config/config.py` — Add `n_lob_levels_used`, `event_feature_mask` parameters
- `run.py` — Feature truncation for snapshot depth ablation
- `models/fuselob.py` — Event feature masking in `EventEmbedding`
- NEW: `preprocessing/nasdaq_itch.py` — ITCH data parser
- NEW: `analysis/rate_distortion.py` — Frontier plotting and analysis

## Verification
- Run each experiment with `--epochs 2 --no-wandb` first to verify no errors
- Check that Perceiver with 8 queries matches existing FuseLOB results (sanity check)
- Check that TLOB with 10 levels matches existing TLOB results (sanity check)
- Verify PnL evaluation runs on all checkpoints
- Verify rate-distortion plot has meaningful curve shape (not flat)
