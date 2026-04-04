# Datasets

This file describes the fixed, inherent properties of each dataset used in this project.
Configurable choices (model dimensions, batch size, horizons, dropout, etc.) are in `config/config.py`.

## Overview

| Property | FI-2010 | BTC | Battery |
|---|---|---|---|
| **Asset class** | Equities | Crypto perpetual | Electricity |
| **Exchange** | NASDAQ Helsinki | Binance | EPEX SPOT (Germany) |
| **Instrument** | 5 Finnish stocks | BTCUSDT.P | Hourly delivery contracts |
| **Time period** | 10 days, 2010 | 12 days, 2023-01-09 to 01-20 | 12 days, 2021-01-11 to 01-22 |
| **Sampling** | Event-based (irregular) | 250ms uniform | 10s uniform, deduplicated |
| **LOB depth** | 10 levels | 10 levels | 10 levels |
| **Raw LOB features** | 40 | 40 | 40 |
| **Extra features** | 104 handcrafted (optional) | — | 10 synthesized message features |
| **Total rows** | ~150K per stock | 3,730,870 | Varies per product |
| **Negative prices** | No | No | Yes |
| **Pre-normalized** | Yes (z-score) | No | No |
| **Split** | 80/10/10 temporal | 9d train / 1d val / 2d test | 80/10/10 by delivery date |

---

## FI-2010

**5 Finnish stocks** (KESBV, OUT1V, SAMPO, RTRKS, WRT1V) traded on NASDAQ Helsinki over 10 trading days in 2010.

**Event-based sampling**: each row corresponds to an LOB update event (order submission, cancellation, or execution), NOT uniformly sampled in time. Row spacing varies with market activity. This means prediction horizons are k *events* ahead, not k time-units — an important distinction from the other datasets.

**Features**:
- 40 raw features: [ask_p, ask_v, bid_p, bid_v] × 10 depth levels
- 144 handcrafted features (optional): includes spreads, mid-price, price/volume derivatives, order arrival intensity, and cross-level statistics

**Labels**: pre-computed in the raw `.txt` files using a smoothed mid-price method. The label compares the mean of the next k mid-prices against the current mid-price, thresholded into 3 classes (up/stationary/down). Labels ship with the data — we extract them by column index.

**Pre-normalized**: the published data files have z-score normalization already applied.

**Split**: temporal split on columns — first 80% of training file for train, last 20% for validation. Three separate test files are concatenated.

**Citation**: Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods." *Journal of Forecasting*, 37(8), 852–866.

**Notes**: the most widely used LOB benchmark in the literature. Small size (10 days) and potential benchmark saturation are known limitations.

---

## BTC

**Binance Bitcoin perpetual swap** (BTCUSDT.P) over 12 consecutive days from 2023-01-09 to 2023-01-20, sourced from Kaggle (siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data).

**Time-sampled at 250ms**: uniform intervals, pre-sampled in the source data. Each row is a full 10-level LOB snapshot at that instant. Total: 3,730,870 rows.

**Features**:
- 40 raw LOB features: [sell_p, sell_v, buy_p, buy_v] × 10 depth levels
- Temporal diff features (40 additional) are computed during preprocessing when enabled

**Labels**: computed from mid-price changes over a smoothing window. An adaptive threshold (mean of absolute price changes / 2) determines the 3-class boundaries (up/stationary/down). Uses absolute change by default.

**Normalization**: z-score normalization fitted on the training split and applied to val/test. Prices and volumes are normalized separately.

**Split**: first 9 days for training, day 10 for validation, final 2 days for testing (~80/10/10 temporal split by day).

**Properties**: single continuous instrument (perpetual swap, no expiry), prices always positive, high-frequency crypto market with tight spreads and deep liquidity. 24/7 trading — no market open/close.

---

## Battery

**EPEX SPOT intraday continuous market** for electricity delivery in Germany/Luxembourg, over 12 days from 2021-01-11 to 2021-01-22.

### What is EPEX SPOT?

EPEX SPOT SE (European Power Exchange) operates short-term electricity markets across Europe. The **intraday continuous market** is a limit order book where participants trade electricity delivery contracts up until shortly before physical delivery. Unlike equity or crypto markets where a single ticker persists indefinitely, each electricity delivery period is a **separate, ephemeral product** with its own independent order book.

### Product structure

Each **product** corresponds to a delivery period — e.g., electricity delivered from 14:00 to 15:00 on January 12, 2021. Key properties:

- **One order book per product**: the 14:00–15:00 product and the 15:00–16:00 product are entirely separate books with independent price discovery
- **Trading opens** ~15:00 CET the day before delivery (D-1) for hourly products
- **Gate closure**: 5 minutes before delivery start (e.g., 13:55 for the 14:00–15:00 product)
- **Trading lifetime**: ~21–23 hours per product
- **Liquidity profile**: thin when trading opens, concentrated in the last few hours before gate closure

We primarily use **per-product mode** (one model evaluation per delivery period) rather than concatenating all products, since each product has its own independent price dynamics.

### Data pipeline

Raw EPEX order data (zips) → parsed CSV → bitepy binary format → LOB snapshots extracted via bitepy C++ simulation engine. The reconstruction handles order matching, cancellations, and book state at each sampling point.

**Time-sampled at 10s with deduplication**: snapshots taken every 10 seconds, but duplicate snapshots (unchanged LOB state) are dropped. Snapshots after gate closure (frozen book) are excluded.

### Features

- **40 raw LOB features**: [sell_p, sell_v, buy_p, buy_v] × 10 depth levels (same layout as BTC)
- **10 synthesized message features** (computed from LOB state):
  1. `log_time_delta` — log(seconds since last snapshot)
  2. `time_to_delivery_hrs` — hours until delivery start
  3. `lifecycle_progress` — normalized position in trading window [0, 1]
  4. `direction` — sign of mid-price change since last snapshot
  5. `spread_bps` — bid-ask spread in basis points
  6. `book_imbalance` — (total bid volume − total ask volume) / total depth
  7. `top_imbalance` — (best bid volume − best ask volume) / (best bid + best ask volume)
  8. `weighted_mid_dev` — deviation of volume-weighted mid from simple mid
  9. `log_total_volume` — log of total depth across all levels
  10. `price_range_ratio` — (level 10 spread) / (level 1 spread), clipped
- Temporal diff features (40 additional) are computed during preprocessing when enabled

### Key differences from equity/crypto LOBs

| Property | Equity/Crypto | Energy (EPEX) |
|---|---|---|
| **Asset lifetime** | Perpetual | Ephemeral (each delivery period expires) |
| **Order books** | One per ticker | 24+ simultaneous per day (one per delivery hour) |
| **Negative prices** | No (typically) | Yes — renewable oversupply can push prices negative |
| **Update frequency** | Milliseconds | Seconds to minutes |
| **Liquidity** | Relatively stable | Highly time-dependent, concentrates near gate closure |
| **Price drivers** | Earnings, sentiment, macro | Weather forecasts (wind/solar), demand, plant outages |
| **Seasonality** | Weak intraday | Strong time-of-day and day-of-week patterns |

**Split**: 80/10/10 by delivery date (observation day), applied per-product in per-product mode.

**Normalization**: z-score fitted on training split, applied globally across products. Prices/volumes normalized separately. Message features normalized per-column.
