# NeurIPS 2025 Research Directions

## Why We're Pivoting: The Snapshot Information Ceiling

Through extensive experimentation we've established that **all snapshot-based LOB models hit the same performance ceiling**, regardless of architecture (TLOB, DeepLOB, MLP, GBM), sequence length (1–128 snapshots), or feature engineering (microstructure features, different normalizations). Our BTC results:

| Model | h10 | h20 | h50 | h100 |
|-------|-----|-----|-----|------|
| GBM (1 snapshot) | 0.341 | — | — | — |
| GBM (10 snapshots) | 0.321 | — | — | — |
| TLOB (128 snapshots) | 0.761 | 0.644 | 0.520 | 0.454 |
| TLOB + microstructure | 0.759 | 0.641 | 0.517 | 0.452 |

The ceiling exists because periodic LOB snapshots are a **lossy compression** of market activity. Between any two snapshots, dozens or hundreds of order events occur — additions, cancellations, modifications, executions — and the snapshot only captures the end state. The *dynamics* (how the book changed, what types of events occurred, how fast) are discarded.

## Literature Gap (confirmed via survey, 2023–2025)

| Direction | Papers found | Gap? |
|-----------|-------------|------|
| Combining event stream + LOB snapshots | **None** | Yes |
| Learning LOB representations from raw events | **None** (LOBench/SimLOB learn from snapshots) | Yes |
| Perceiver-style compression for LOB/financial data | **None** | Yes |
| Order lifecycle modeling (tracking individual orders) | **None** (LOBDIF predicts event types but doesn't track orders) | Yes |

All competitive models (TLOB, DeepLOB, LiT, HLOB) operate on fixed-interval LOB snapshots as 2D tensors. The event stream is used only in preprocessing to reconstruct snapshots, never as a first-class model input.

LOBERT (NeurIPS 2025 Workshop) operates on events, but with BERT-style masked modeling for generation — not for discriminative prediction, and not combined with snapshots.

## What Information Do Events Contain That Snapshots Lose?

1. **Event type**: Volume decreased — was it a cancellation (informed retreat) or a trade (aggressive execution)? Snapshots can't distinguish.
2. **Event rate/intensity**: Between two snapshots, 2 events or 200 events could have occurred. High event intensity signals volatility or information arrival.
3. **Event ordering**: Buy trade then sell cancellations ≠ sell cancellations then buy trade, even if the resulting snapshot is identical.
4. **Aggressive vs passive flow**: Market orders (aggressive, information-rich) vs limit orders (passive, liquidity-providing) carry different predictive signals.
5. **Order size distribution**: One 100-lot order vs twenty 5-lot orders at the same price level — different trader behavior, same snapshot.
6. **Execution speed**: How quickly orders get filled reveals urgency and information asymmetry.

---

## Architecture Proposals

### Proposal A: Learned LOB State Compression (Perceiver-style)

**Core idea**: Replace the hand-coded LOB snapshot (top-N price levels at fixed intervals) with a *learned* state representation compressed from the raw event stream using Perceiver-style cross-attention.

**Architecture**:
```
Raw events → Event Encoder (causal transformer) → Learned Compression (Perceiver cross-attention) → State Tokens → Temporal Attention → Multi-horizon Prediction
```

**Detailed design**:

1. **Event tokenization**: Each event becomes a token:
   - Categorical: event_type (add/cancel/modify/execute), side (buy/sell)
   - Continuous: price (relative to mid), quantity, time_delta since previous event
   - Learned embedding per token combining categorical embeddings + linear projection of continuous features

2. **Event encoder**: Causal transformer (4–6 layers) processes events sequentially within a window. Causal masking ensures each event only attends to past events.

3. **Learned state compression**: At regular intervals (every T seconds or every K events), use Perceiver cross-attention:
   - M learnable query tokens (M << number of events, e.g., M=8–32)
   - Query tokens cross-attend over all event encoder outputs in the window
   - Output: M "state tokens" representing the learned LOB state at that time point
   - This is the KEY architectural contribution — a differentiable, end-to-end replacement for fixed LOB reconstruction

4. **Temporal model**: Stack of transformer layers with attention over the sequence of state token sets across time. Captures how the learned LOB state evolves.

5. **Prediction head**: Multi-horizon classification (up/stationary/down for h=10,20,50,100).

**Why this is novel**:
- No one learns the LOB aggregation — everyone takes fixed snapshots as given
- Perceiver cross-attention is proven (DeepMind) but never applied to LOB
- The model can learn to extract information that fixed snapshots discard (event intensity, type ratios, order flow momentum)
- Fully market-agnostic: works on any exchange with event-level data

**Risks**:
- The compressed tokens might just learn to reconstruct the standard snapshot representation, offering no benefit
- Variable event counts between compression points need careful handling (padding/masking)
- Longer effective sequences (events >> snapshots) increase compute

**Mitigation**: Ablation comparing learned vs fixed snapshots directly. If learned states converge to snapshot-like representations, the paper pivots to "snapshots are near-optimal" (also a finding). Visualization of what the learned queries attend to.

---

### Proposal B: Order Lifecycle Transformer

**Core idea**: Instead of treating the LOB as a time series (of snapshots or events), model it as a *set of evolving entities* — each entity is an order with a lifecycle from placement through modification to execution/cancellation.

**Architecture**:
```
Events → Order State Update (per-order embeddings evolve with each event) → Set Aggregation (current active orders → market state) → Temporal Attention → Prediction
```

**Detailed design**:

1. **Order registry**: Maintain a dictionary of active order embeddings, keyed by order ID.

2. **Order state update**: When an event arrives:
   - **Add**: Initialize a new order embedding from (side, price, quantity, time). Add to registry.
   - **Modify/Partial fill**: Update the existing order's embedding using a learned update function (e.g., GRU cell taking old embedding + event features → new embedding).
   - **Match/Delete/Expire**: Mark order as complete. Optionally emit its final embedding for "recently completed" context.

3. **Market state derivation**: At each prediction time, the market state is an aggregation over all active order embeddings:
   - Options: DeepSets-style (sum/mean), attention-based pooling, or separate bid/ask aggregation
   - Output: Fixed-size market state vector regardless of number of active orders

4. **Temporal model**: Attention over market states at successive time points.

5. **Prediction head**: Same as Proposal A.

**Why this is novel**:
- Completely new paradigm — no paper models individual order lifecycles
- Captures trader intent directly: resting time (patience), modification patterns (price chasing), cancellation cascades
- Theoretically grounded: the LOB IS a set of orders, so modeling it as such is natural
- The order ID linkage (InitialId/RevisionNo in EPEX, order_id in LOBSTER/ITCH) is available in all major datasets but unused by any model

**Risks**:
- Variable-size active order sets are hard to batch efficiently on GPU
- Number of active orders can be large (hundreds to thousands) — attention over all orders is expensive
- Not all datasets provide clean order ID tracking (crypto L2 data typically doesn't)
- Most complex to implement — 2-month timeline is tight

**Mitigation**: Start with top-K active orders by recency or proximity to mid-price. Use efficient set attention (like Set Transformer). Prototype on LOBSTER data which has clean order linkage.

---

### Proposal C: Dual-Stream (Events + Snapshots)

**Core idea**: Two parallel encoders — one for the raw event stream, one for LOB snapshots — fused via cross-attention. This tests whether events *complement* or *replace* snapshots.

**Architecture**:
```
Stream 1: Events → Event Encoder → Event representations
Stream 2: Snapshots → Snapshot Encoder (TLOB-style) → Snapshot representations
                      ↓                    ↓
                  Cross-Attention Fusion
                      ↓
              Joint Temporal Attention → Prediction
```

**Detailed design**:

1. **Event stream**: Between consecutive snapshots, process all events with a small causal transformer. Output: one aggregate representation per snapshot interval (via mean pooling, last hidden state, or learned pooling).

2. **Snapshot stream**: Standard TLOB architecture (spatial + temporal attention) on LOB snapshots.

3. **Fusion**: At each time step, cross-attention between the event aggregate and the snapshot representation. The snapshot can "query" the events for dynamics information; the events can "query" the snapshot for structural context.

4. **Joint temporal attention**: Over the fused representations.

5. **Prediction head**: Multi-horizon.

**Why this is interesting**:
- Clean ablation: event-only, snapshot-only, and dual → quantifies what events add
- Builds on TLOB (our existing strong baseline) — less risky than full redesign
- If events help: the paper shows what information snapshots miss
- If events don't help: the paper shows snapshots are near-sufficient (also publishable as a negative result with analysis)

**Why this might not be enough for NeurIPS**:
- Dual-stream architectures are common in vision/NLP — the novelty is in application, not architecture
- Risk of being perceived as "TLOB + event features" rather than a fundamentally new approach
- Does not propose a new *representation* — just combines two existing ones

---

## Comparison Matrix

| Aspect | A: Learned Compression | B: Order Lifecycle | C: Dual-Stream |
|--------|----------------------|-------------------|----------------|
| **Novelty** | High — new representation paradigm | Very high — new modeling paradigm | Medium — application of known technique |
| **Implementation effort** | Medium (2 months feasible) | High (tight for 2 months) | Low–Medium (extends TLOB) |
| **Risk** | Medium — might converge to snapshots | High — batching, scaling | Low — clean ablation either way |
| **NeurIPS fit** | Strong (method + finding) | Strongest if it works | Needs strong analysis to compensate |
| **Market-agnostic** | Yes | Yes (needs order IDs) | Yes |
| **Story clarity** | "Learn the aggregation" | "Model orders, not time series" | "Two views are better than one" |

## Recommended Approach

**Primary**: Proposal A (Learned Compression) — highest novelty-to-risk ratio for the timeline.

**Enhancement**: Incorporate elements of Proposal C as an ablation — also test dual-stream to show whether learned event representations *replace* or *complement* fixed snapshots.

**Stretch goal**: If time permits, add order-type-aware attention (from Proposal B) — the event encoder can condition on order linkage even without full lifecycle tracking.

Paper structure would be:
1. **Motivation**: Snapshot-based models plateau (brief, 1 paragraph + table)
2. **Method**: Learned LOB compression from events via Perceiver cross-attention
3. **Experiments**: Snapshot-only (TLOB) vs event-only (learned) vs dual-stream, across multiple markets
4. **Analysis**: What do the learned state tokens capture? Visualization of attention patterns.
5. **Results**: On public dataset (LOBSTER equities) + proprietary (EPEX energy)

## Practical Concerns

### Speed: How to handle orders of magnitude more events than snapshots

The battery dataset at 10s sampling has ~435K snapshots. Between snapshots, there could be 10–1000 events. Processing every event naively with a full transformer is too expensive. Solutions:

1. **Chunked processing**: Don't run one giant sequence. Process events in fixed-size chunks (e.g., 64–128 events per chunk) with a small, efficient event encoder. Then compress each chunk into a few state tokens. The temporal model only sees the compressed tokens — same sequence length as snapshot-based models.

2. **Hierarchical compute budget**: Event encoder = small (2–3 layers, narrow). Temporal model = larger (4–6 layers). Most parameters go into temporal reasoning, not event processing. The event encoder's job is just compression.

3. **FlashAttention + mixed precision**: Already used in TLOB. Extends naturally to the event encoder.

4. **Subsampling fallback**: If a window has 1000 events, subsample to top-K most informative (e.g., executions and large cancellations). Less elegant but practical.

**Rough compute comparison**: If we compress every 64 events into 8 state tokens, and use 128 compression windows (matching TLOB's seq_size=128), the event encoder processes 128×64 = 8192 events total, but in 128 independent chunks of 64. Each chunk is smaller than TLOB's full sequence. The temporal model sees 128 time steps × 8 tokens = same order as TLOB.

### Output: Still up/stationary/down classification

Yes — the prediction task stays identical: 3-class mid-price direction (up=0, stationary=1, down=2) at horizons h=10,20,50,100. The innovation is entirely on the **input representation** side. Same labeling, same metrics (weighted F1, MCC), same evaluation protocol. This means we can directly compare against TLOB, DeepLOB, etc.

### One dataset enough for NeurIPS?

With only EPEX energy (proprietary, can't release): **risky**. Reviewers will question generalizability. But mitigations exist:
- Show results on multiple products within EPEX (287 delivery contracts = effectively 287 mini-datasets)
- Show the snapshot ceiling analysis on BTC and FI-2010 (public, snapshot-only) as motivation
- Argue the architecture is market-agnostic by design

**Stronger**: Add a second dataset. See free options below.

---

## Available Datasets with Order-Level Events

### Free options (confirmed available)

| Dataset | Market | What's included | Size | Access |
|---------|--------|----------------|------|--------|
| **NASDAQ ITCH samples** | US equities | Full L3 event stream: every add, cancel, execute, replace, delete for ALL NASDAQ tickers. One full trading day. | ~5 GB compressed | Free download from `emi.nasdaq.com/ITCH/Nasdaq%20ITCH/`. NASDAQ rotates which days are available. |
| **LOBSTER samples** | US equities | Pre-parsed ITCH: clean message CSV + orderbook CSV per stock. 5 stocks (AAPL, AMZN, GOOG, INTC, MSFT), 1 day (2012-06-21). | 0.5–7 MB/file | Free at `data.lobsterdata.com/info/DataSamples.php`. No registration. |
| **Tardis.dev 1st-of-month** | Crypto | L2 incremental book updates + trades for Binance, Bybit, Deribit, etc. | Large | 1st of each month is free. |
| **Bybit 10ms snapshots** | Crypto futures | Order book snapshots every 10ms, 500 levels. Can diff consecutive snapshots to approximate events. | Large | Free at `bybit.com/derivatives/en/history-data`. |
| **Databento** | US equities | Full ITCH L3 via clean API. | Limited | $125 free credit on signup. |

### Proprietary (ours)

| Dataset | Market | What's included |
|---------|--------|----------------|
| **EPEX** | Energy (Germany) | Full event stream: Add, Delete, Change, Match, Expire. Order ID linkage (InitialId/RevisionNo). 287 delivery products. |

### Recommended dataset strategy

**Primary public benchmark**: NASDAQ ITCH sample (free, full L3 events, real equities). Parse with existing tooling ([stefan-jansen's ML for Trading toolkit](https://github.com/stefan-jansen/machine-learning-for-trading)). One day of data for all NASDAQ stocks — pick 3–5 liquid stocks (AAPL, MSFT, INTC, etc.). This gives us:
- Public, free, reproducible
- True L3 order events (add/cancel/execute)
- Same market as FI-2010/LOBSTER baselines — direct comparison

**Secondary**: EPEX energy (proprietary). Shows market-agnostic applicability across a structurally different market (energy vs equities).

**Concern**: One day of ITCH data may be limited. Mitigations:
- Multiple stocks = multiple experiments from one day
- LOBSTER samples add another day (2012-06-21) for 5 stocks
- Databento $125 credit could get a few more days if needed
- The EPEX dataset has 12 days × 287 products — substantial even alone

## NeurIPS Reproducibility (Without Releasing EPEX Data)

NeurIPS accepts papers with proprietary datasets. Standard approach:
- Release all code (model + training pipeline)
- Show primary results on a public benchmark (NASDAQ ITCH) for reproducibility
- Show additional results on proprietary data (EPEX) as supplementary evidence
- Provide detailed dataset description and statistics

Reviewers care about the *method* being reproducible, not the specific dataset.
