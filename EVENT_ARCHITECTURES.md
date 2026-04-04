# Event-Based LOB Architectures: Implementation Plan

## Why: The Snapshot Information Ceiling

All snapshot-based LOB models (TLOB, DeepLOB, MLP, GBM) hit the same performance ceiling regardless of architecture, sequence length, or feature engineering. The ceiling exists because periodic LOB snapshots are a **lossy compression** of market activity — between any two snapshots, dozens of order events occur (additions, cancellations, trades, modifications), and the snapshot only captures the end state. The dynamics are discarded.

We propose three architectures that use raw order events to break through this ceiling.

---

## Data Characteristics (Measured from EPEX Battery Dataset)

**Current baseline:** TLOB — 30s/epoch, 386,770 samples, 1.3M params, seq_size=128, hidden_dim=40.

**Measured event statistics (Jan 11, 2021 — one trading day, all 24 hourly products):**

| Metric | Value |
|--------|-------|
| Events per hourly product | ~63K (over ~24h trading window) |
| Events per 10s window | mean 21, median 10, 90th percentile 54, max 272 |
| Active orders at any time | mean 140, median 125, max 297 |
| Events per order lifecycle | mean 4.36 (78% are Add→Delete) |
| Time between events | median 0.09s, mean 0.85s |
| ActionCode distribution | A(39%), D(35%), C(17%), P(4%), M(3%), X/I/H(2%) |

**Intraday vs XBID products:** Both `Intraday_Hour_Power` and `XBID_Hour_Power` events contribute to the same LOB per delivery contract. XBID (Cross-Border Intraday) is the pan-European routing layer — these are orders from other European markets targeting the same German delivery slot. Bitepy merges both into a single order book, which is the correct behavior. Our event extraction does the same: all events for a delivery contract regardless of routing origin.

**What events capture that snapshots lose:**

1. **Event type disambiguation**: Volume decreased — was it a cancellation (D, informed retreat) or a trade (M/P, aggressive execution)? Snapshots cannot distinguish these.
2. **Event intensity**: 2 events vs 200 events between snapshots. High event rates signal volatility or information arrival.
3. **Event ordering**: Buy trade then sell cancellations ≠ sell cancellations then buy trade, even if the resulting snapshot is identical.
4. **Aggressive vs passive flow**: Market orders (M/P — aggressive, information-rich) vs limit order cancellations (D — passive) carry different predictive signals.
5. **Order lifecycle patterns**: How long an order rests before cancellation reveals trader patience and intent.

---

## Shared: Event Preprocessing Module

All three architectures need raw events extracted and aligned to snapshot windows. This is a new preprocessing stage that sits alongside (not replaces) the existing bitepy LOB reconstruction.

### New file: `preprocessing/events.py`

**Pipeline:**
1. Read raw EPEX zips for each day (same zips bitepy uses from `data/battery_markets/2021/`)
2. Filter to hourly power products (both `Intraday_Hour_Power` and `XBID_Hour_Power`)
3. Group events by delivery product (DeliveryStart → product key like `2021-01-11_H05`)
4. Sort events by TransactionTime within each product
5. Align events to time windows matching the configured `sampling_time` (default 10s, supports 5s, 30s, etc.)
6. Tokenize each event into a feature vector

**Event token features (7-dimensional per event):**

| Feature | Type | Encoding |
|---------|------|----------|
| `action_code` | categorical | integer 0–7 (A=0, D=1, M=2, P=3, X=4, C=5, I=6, H=7) |
| `side` | categorical | integer 0–1 (BUY=0, SELL=1) |
| `price_relative` | continuous | (price − current_mid) / current_mid |
| `quantity_log` | continuous | log1p(quantity in MW) |
| `time_delta` | continuous | log1p(seconds since previous event in this product) |
| `revision_flag` | binary | 1 if RevisionNo > 1 (order was previously modified) |
| `is_aggressive` | binary | 1 if ActionCode ∈ {M, P, C} (trade execution / partial fill) |

**Handling variable event counts per window:**

- Pad each window to `max_events_per_window` (configurable, default 64)
- **Windows with >64 events:** Subsample with priority ordering:
  1. Keep ALL trade events (M, P, C) — these are the most information-rich
  2. Keep ALL expiry/hibernate events (X, H) — rare but informative
  3. Fill remaining slots with most recent A and D events
  4. This ensures trade information is never lost, even in high-activity windows
- Windows with 0 events: all-zero padding with attention mask
- Attention mask tensor: `(B, T, max_events)` boolean, False for padded positions

**Output per product:**

Stored as `events.npz` alongside existing `.npy` files:
- `event_features`: `(N_windows, max_events_per_window, 7)` float32
- `event_mask`: `(N_windows, max_events_per_window)` bool (True = real event)
- `n_events`: `(N_windows,)` int — actual event count per window (before padding)

**Sampling time flexibility:** The event extraction reads `sampling_time` from the BATTERY config (same parameter that controls snapshot sampling). When using 5s windows, each window has fewer events (mean ~10); when using 30s windows, each has more (mean ~63). The `max_events_per_window` parameter should scale accordingly.

### Config changes

In `config/config.py`, add to BATTERY:
```python
extract_events: bool = False    # Set True for event-based models
max_events_per_window: int = 64 # Pad/subsample to this many events per window
```

When `extract_events=False` (default), the existing pipeline runs unchanged — no event preprocessing overhead for snapshot-only models like TLOB.

### Integration with battery.py

After existing Stage 3 (LOB snapshot extraction via bitepy), if `extract_events=True`:
- Run event extraction from the same raw zip files
- Align event windows to the same timestamps as the snapshot sequence
- Cache events in `per_product/{subdir}/products/{product_key}/events.npz`

**Estimated preprocessing time:** ~15–30 min for 287 products × 12 days. Cached after first run.

---

## FuseLOB (Dual-Stream: Events + Snapshots)

**Implementation priority: FIRST** — lowest risk, enables clean ablation, shares code with other proposals.

### Core Idea

Two parallel lightweight encoders — one for raw events, one for LOB snapshots — fused via a learned gate. The model discovers when event dynamics add value beyond what the snapshot state already captures.

### Architecture (`models/fuselob.py`)

```
Stream 1 (Events):    (B, T=128, E=64, 7) → EventEncoder → (B, T, d)
Stream 2 (Snapshots): (B, T=128, F=50)     → SnapEncoder  → (B, T, d)
                              ↓                    ↓
                         Gated Fusion → (B, T, d)
                              ↓
                    Temporal Transformer (4 layers) → (B, T, d')
                              ↓
                    Multi-Horizon Classification Head → (B, 3) × 4
```

### Component Details

**Event Encoder** (per-window, processes events within each time window independently):

1. **Event Embedding** (7 features → d_event=64):
   - Categorical: `action_code` → Embedding(8, 16), `side` → Embedding(2, 8), `revision_flag` → Embedding(2, 4), `is_aggressive` → Embedding(2, 4)
   - Continuous: `[price_relative, quantity_log, time_delta]` → Linear(3, 32)
   - Concatenate: 16 + 8 + 4 + 4 + 32 = 64
   - Positional encoding within window (max_pos = max_events_per_window)

2. **Causal Transformer** (2 layers, d=64, 4 heads, d_ff=128):
   - Each event attends only to earlier events in the window (causal mask)
   - Batched as (B × T, E, 64) — all windows processed independently in one GPU call
   - FlashAttention on length-64 sequences is extremely fast

3. **Perceiver Compression** (M=8 learnable queries, d=64):
   - 8 learnable query tokens cross-attend over event encoder outputs
   - Attention mask applied to ignore padded events
   - Output: (B × T, 8, 64) → mean pool → (B × T, 64) → reshape to (B, T, 64)

**Snapshot Encoder** (lightweight, NOT full TLOB):

1. **BiN normalization** (same as TLOB)
2. **Linear embedding**: (B, T, 50) → (B, T, d=64)
3. **2-layer temporal transformer** (d=64, 4 heads, d_ff=128):
   - Bidirectional attention over the feature dimension
   - RMSNorm + GELU + residual connections
   - Output: (B, T, 64)

Rationale for lightweight encoder: The snapshot stream doesn't need to be as powerful as standalone TLOB. Its job is to provide the accumulated book state as context; the fusion layer decides how to weight it. Using a lighter encoder also prevents the snapshot stream from dominating the gradient.

**Gated Fusion:**

```python
concat = torch.cat([event_repr, snap_repr], dim=-1)   # (B, T, 2d)
gate = torch.sigmoid(self.gate_proj(concat))            # (B, T, d)
fused = gate * event_repr + (1 - gate) * snap_repr      # (B, T, d)
```

The gate learns per-dimension which stream to trust more. Gate values are interpretable: `gate → 1` means events dominate, `gate → 0` means snapshots dominate. We log mean gate values during validation to understand when events matter most (e.g., near gate closure? during high-activity periods?).

**Temporal Transformer** (4 layers, d=64, 4 heads, d_ff=256):
- Bidirectional attention over T=128 timesteps
- RMSNorm, GELU, residual connections
- Last layer reduces: d → d//4 (dimensional compression, same as TLOB)

**Multi-Horizon Classification Head:**
- Same shrink-to-3 architecture as TLOB
- 4 independent heads for h=10, 20, 50, 100
- Homoscedastic uncertainty weighting (existing engine.py logic)

### Ablation Structure

FuseLOB's design directly enables the key ablation table:

| Variant | Event Stream | Snapshot Stream | Fusion | What it shows |
|---------|:---:|:---:|:---:|---|
| TLOB (existing) | ✗ | ✓ (full 1.3M) | ✗ | Snapshot ceiling baseline |
| SnapEncoder-only | ✗ | ✓ (2-layer) | ✗ | Lightweight encoder quality |
| EventEncoder-only | ✓ | ✗ | ✗ | Events alone vs snapshots |
| **FuseLOB** | **✓** | **✓** | **✓** | **Events + snapshots** |

The event-only variant is essentially PerceiverLOB (see below). Implementing FuseLOB first gives us 3 of the 4 ablation rows immediately.

### Estimates

| Metric | Value |
|--------|-------|
| Parameters | ~850K |
| Epoch time | ~50–75s (1.7–2.5× TLOB) |
| Preprocessing | Existing snapshots + ~20 min event extraction (cached) |

---

## PerceiverLOB (Learned Event Compression)

**Implementation priority: SECOND** — shares event encoder with FuseLOB, the novel contribution is replacing snapshots entirely.

### Core Idea

Replace the hand-coded LOB snapshot (top-N price levels at fixed intervals) with a **learned** state representation compressed from raw events via Perceiver cross-attention. The model learns what to extract from events end-to-end — no hand-designed aggregation.

### Architecture (`models/perceiverlob.py`)

```
Raw events per window (B, T=128, E=64, 7)
        ↓
Event Embedding → (B*T, E, 64)
        ↓
Causal Event Encoder (2 layers) → (B*T, E, 64)
        ↓
Perceiver Cross-Attention (M=8 queries) → (B*T, 8, 64)
        ↓
State Projection → (B, T, d_model)
        ↓
Temporal Transformer (4 layers) → (B, T, d_model')
        ↓
Classification Head → (B, 3) × 4 horizons
```

This is identical to FuseLOB's event stream, followed by the shared temporal transformer and classification head — but with **no snapshot input at all**. The learned state tokens ARE the LOB representation.

### What makes this novel

No existing work learns the LOB aggregation function. Every model (TLOB, DeepLOB, LiT, HLOB, LOBERT) takes either fixed snapshots or raw events as-is. PerceiverLOB is the first to:
- Learn a differentiable compression from events to state (Perceiver cross-attention)
- Let the model discover what information matters (not hand-specified price levels)
- Potentially extract information that no fixed snapshot format can capture (event intensity, type ratios, order flow momentum)

### Code sharing with FuseLOB

The event encoder, Perceiver compression, temporal transformer, and classification head are **identical** to FuseLOB's components. PerceiverLOB is literally FuseLOB with the snapshot stream disabled and the gate removed. Implementation: `FuseLOB(use_snapshot_stream=False)` or a standalone class that reuses the same modules.

### Estimates

| Metric | Value |
|--------|-------|
| Parameters | ~760K |
| Epoch time | ~40–60s (1.5–2× TLOB) |
| Preprocessing | Event extraction only (no snapshot needed, but keep for comparison) |

---

## FlowLOB (Order Lifecycle Transformer)

**Implementation priority: THIRD** — most novel but hardest to implement efficiently.

### Core Idea

Model the LOB as a **set of evolving entities** — each entity is an order tracked through its lifecycle from placement to removal/execution. The market state is derived by aggregating all active order embeddings at each prediction time.

### Architecture (`models/flowlob.py`)

```
Events with order ID linkage (B, T=128, E=64, features+InitialId)
        ↓
Event Embedding → (d=64 per event)
        ↓
Order State Update (GRU per order) → per-order embeddings evolve
        ↓
Market State Aggregation (split bid/ask attention pooling) → (B, T, d)
        ↓
Temporal Transformer (4 layers) → (B, T, d')
        ↓
Classification Head → (B, 3) × 4 horizons
```

### Order State Update (the novel component)

For each time window, events are processed sequentially. Each event updates the embedding of the order it belongs to (linked via `InitialId`):

- **Add (A)**: Initialize new order embedding from event features
- **Modify/Partial fill (C, P)**: Update existing order embedding via GRU cell: `h_new = GRU(h_old, event_embedding)`
- **Match/Delete/Expire (M, D, X)**: Mark order complete, optionally emit final embedding to "recently completed" buffer

The GRU captures how an order evolves: its resting time, number of modifications, whether it was price-chased. This is information no other model captures.

### Market State Aggregation

At each timestep, aggregate all ~140 active order embeddings into a fixed-size market state:

**Split Bid/Ask Attention Pooling (recommended):**
```python
bid_state = attention_pool(bid_order_embeddings)   # (d,) via learned query
ask_state = attention_pool(ask_order_embeddings)   # (d,)
market_state = Linear([bid_state || ask_state || bid_state - ask_state])  # (3d → d)
```

### Preprocessing requirements (beyond shared event module)

- Preserve `InitialId` in event features (for order linkage)
- Pre-compute active order sets at each window boundary during preprocessing:
  - Which orders are active, their side, current price/quantity, cumulative event count
  - This avoids slow runtime sequential processing
- Cap active orders at K=64 per side (nearest to mid-price)

### Practical fallback: Pre-aggregated lifecycle features

If full order tracking proves too slow or complex, compute per-window aggregate lifecycle statistics:
- `n_new_orders_bid/ask`, `n_cancellations_bid/ask`, `n_trades_bid/ask`
- `avg_order_resting_time`, `cancel_to_trade_ratio`, `n_modifications`
- These become additional input features alongside snapshots (hand-crafted version of what FlowLOB tries to learn)

### Estimates

| Metric | Value |
|--------|-------|
| Parameters | ~650K |
| Epoch time | ~120–200s (4–7× TLOB) — sequential order updates are the bottleneck |
| Preprocessing | Event extraction + order registry pre-computation |

---

## Comparison Summary

| Aspect | FuseLOB | PerceiverLOB | FlowLOB |
|--------|:---:|:---:|:---:|
| **Core idea** | Fuse events + snapshots | Learn LOB from events | Track order lifecycles |
| **Parameters** | ~850K | ~760K | ~650K |
| **Epoch time** | ~60s (2×) | ~50s (1.7×) | ~160s (5×) |
| **Novelty** | Medium-High | High | Very High |
| **Implementation effort** | Medium | Medium (shares code) | High |
| **GPU efficiency** | Good | Good | Poor (sequential) |
| **Ablation value** | Very High | Medium | Low |
| **Risk of failure** | Low | Medium | High |
| **Interpretability** | Gate analysis | Attention visualization | Order embedding analysis |

---

## Critical Review: Will These Architectures Actually Beat TLOB?

### FuseLOB — Likely yes (small-moderate improvement)

**Arguments for improvement:**
- The most directly actionable signal events add is **distinguishing cancellations from trades**. When volume at the best bid drops by 10 MW, was it a cancellation (bearish — someone pulled their bid) or a buy trade execution (also bearish — aggressive selling hit the bid)? Snapshots see the same state change but the implications differ. With 39% of events being additions and 35% deletions vs only 7% trades, this distinction is frequent and meaningful.
- **Event intensity** (how many events per window) is a direct volatility/information signal. A window with 54 events (90th percentile) vs 2 events (quiet period) indicates very different market conditions, but both produce one snapshot.
- The gated fusion lets the model learn WHEN events matter — likely more near gate closure (last few hours) when trading is most active and informed.

**Arguments against:**
- The existing 10 message features already capture some event-derived signals (spread changes, direction, imbalance). The incremental value of raw events over these hand-crafted summaries might be small.
- With only ~21 events per 10s window median, the event sequences are short. Much of the "dynamics" signal might already be captured by TLOB's temporal attention over 128 consecutive snapshots.

**Prediction: +1-4% F1 improvement at h10, diminishing at longer horizons.** The improvement should be more pronounced for directional trading simulation, where distinguishing aggressive from passive flow matters for entry/exit timing.

### PerceiverLOB — Uncertain, depends on learned representation quality

**Arguments for:**
- If the Perceiver discovers a representation that captures event composition (trade ratio, cancel intensity, flow imbalance) more efficiently than fixed snapshots, it could match or beat TLOB while being conceptually simpler.
- The end-to-end learning removes human bias about what a "good" LOB representation looks like.

**Arguments against:**
- The hand-coded LOB snapshot (top-10 levels, prices + volumes) is an extremely efficient summary of the order book. Millions of engineering hours went into LOB design. The learned representation might just converge to something snapshot-like, adding no benefit.
- With M=8 queries compressing ~21 events, the information bottleneck is tight. The model might not have enough capacity to discover anything novel.
- Loss of explicit price level structure (the snapshot's spatial organization) might hurt — TLOB's spatial attention exploits this structure.

**Prediction: Ties or slightly underperforms TLOB.** The learned representation is unlikely to significantly outperform the well-designed snapshot format on its own. The real value of PerceiverLOB is the ablation: if it ties TLOB, it proves snapshots are near-optimal (interesting finding). If it beats TLOB, that's the stronger paper.

### FlowLOB — High risk, likely modest improvement

**Arguments for:**
- Completely novel paradigm. If it works even modestly, the novelty carries the paper.
- Order resting time (how long before cancellation) is a genuine signal: iceberg orders that rest for hours behave differently than HFT orders that cancel in milliseconds.

**Arguments against:**
- 78% of order lifecycles are simple Add→Delete. Only 13% have >2 lifecycle events. The per-order signal is sparse.
- The sequential processing makes this 5× slower than TLOB. Slow iteration speed means fewer experiments, fewer hyperparameter searches.
- Active order sets (140 avg) require efficient set operations on GPU, which is non-trivial.

**Prediction: +0-2% F1 improvement, but at 5× the compute cost.** The sparse lifecycle signal likely doesn't justify the complexity. The pre-aggregated lifecycle features fallback (cancel ratio, resting time statistics) would likely capture 80% of the signal at 10% of the complexity.

### Overall recommendation for beating TLOB

**FuseLOB is the safest bet for actually beating TLOB.** It preserves everything TLOB already does well (snapshot encoding) and adds event information on top. The gated fusion means it can never do worse than the snapshot-only model (worst case: gate → 0, ignore events). The improvement will come primarily from:

1. **Trade/cancel disambiguation** — the single biggest information gain from events
2. **Event rate as volatility proxy** — contextualizes snapshot changes
3. **Aggressive flow direction** — market orders signal informed trading

For **directional trading simulation** specifically, events should help even more than for F1 scores, because:
- Trade execution events (M/P) directly indicate aggressive directional pressure
- Cancellation cascades on one side often precede price moves
- These signals matter for entry/exit timing, not just label classification

---

## Implementation Plan

### Phase 1: Event Preprocessing (shared foundation)

**Create `preprocessing/events.py`:**
- `extract_events_for_product(zip_path, product_key, sampling_seconds, max_events)` → events.npz
- Priority-based subsampling when events exceed max_events_per_window
- Support configurable `sampling_time` (5s, 10s, 30s, etc.)

**Modify `config/config.py`:**
- Add `extract_events: bool = False` and `max_events_per_window: int = 64` to BATTERY dataclass

**Modify `preprocessing/battery.py`:**
- After LOB snapshot extraction, if `extract_events=True`, run event extraction
- Use same `sampling_time` parameter for both snapshot and event window alignment

**Create `preprocessing/event_dataset.py`:**
- `EventDataset`: loads events.npz, returns `(events, mask, labels)` per window
- `EventSnapshotDataset`: loads both .npy and events.npz, returns `(snapshots, events, mask, labels)`

### Phase 2: FuseLOB (implement first)

**Create `models/fuselob.py`:**
- EventEncoder: embedding + 2-layer causal transformer + Perceiver compression
- SnapshotEncoder: BiN + embedding + 2-layer transformer
- GatedFusion: learned per-dimension gate
- TemporalTransformer: 4-layer bidirectional transformer
- MultiHorizonHead: 4 × shrink-to-3 heads

**Modify `constants.py`:** Add `ModelType.FUSELOB`
**Modify `config/config.py`:** Add FuseLOB config dataclass
**Modify `run.py`:** Add FuseLOB data loading path
**Modify `run_experiments.py`:** Add `fuselob` model option

**Verify:**
```bash
python run_experiments.py --mode multi-horizons --model fuselob --dataset battery --epochs 2 --no-wandb
```
- Check convergence (loss decreasing)
- Compare val F1/MCC against TLOB at h10
- Log and inspect gate values (should not be trivially 0 or 1)

### Phase 3: PerceiverLOB

Largely reuses FuseLOB's event encoder. Either:
- `FuseLOB(use_snapshot_stream=False)` mode flag, or
- Standalone class importing shared modules

### Phase 4: FlowLOB (if time permits)

Most complex implementation. Consider starting with pre-aggregated lifecycle features as a quick test of whether order lifecycle information helps at all before building the full GRU-based order tracking.

---

## Files Summary

### New files
| File | Purpose |
|------|---------|
| `preprocessing/events.py` | Event extraction and tokenization from raw EPEX zips |
| `preprocessing/event_dataset.py` | Dataset classes for event and event+snapshot data |
| `models/fuselob.py` | FuseLOB model (dual-stream with gated fusion) |
| `models/perceiverlob.py` | PerceiverLOB model (or mode within FuseLOB) |
| `models/flowlob.py` | FlowLOB model (order lifecycle tracking) |

### Modified files
| File | Changes |
|------|---------|
| `config/config.py` | Add `extract_events`, `max_events_per_window` to BATTERY; add FuseLOB/PerceiverLOB/FlowLOB model configs |
| `constants.py` | Add `ModelType.FUSELOB`, `PERCEIVERLOB`, `FLOWLOB` |
| `preprocessing/battery.py` | Trigger event extraction when `extract_events=True` |
| `run.py` | Add data loading paths for event-based models |
| `run_experiments.py` | Add new model CLI options |
| `models/engine.py` | Should work as-is (multi-horizon loss is model-agnostic) |
