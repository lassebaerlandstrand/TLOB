"""
Event extraction from raw EPEX order data for event-based LOB models.

Reads the same raw zip files that bitepy uses and produces per-product event
tensors aligned to the existing snapshot timestamps.  Events within each
snapshot window are tokenised into 7-dimensional feature vectors.

Output per product (saved as events.npz alongside existing .npy files):
    event_features : float32 (N_windows, max_events, 7)
    event_mask     : bool    (N_windows, max_events)
    n_events       : int32   (N_windows,)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

# ─── Action code mapping ─────────────────────────────────────────────────────
ACTION_CODE_MAP = {"A": 0, "D": 1, "M": 2, "P": 3, "X": 4, "C": 5, "I": 6, "H": 7}
SIDE_MAP = {"BUY": 0, "SELL": 1}
N_ACTION_CODES = len(ACTION_CODE_MAP)
N_EVENT_FEATURES = 7

# Trade/execution action codes — highest priority during subsampling
_TRADE_CODES = {"M", "P", "C"}
_TRADE_CODES_LIST = sorted(_TRADE_CODES)
# Expiry/hibernate — second priority
_EXPIRY_CODES = {"X", "H"}
_EXPIRY_CODES_LIST = sorted(_EXPIRY_CODES)

# Columns to read from raw CSV
_RAW_COLS = [
    "Product",
    "DeliveryStart",
    "TransactionTime",
    "ActionCode",
    "Side",
    "Price",
    "Quantity",
    "RevisionNo",
    "InitialId",
]

# Hourly power product types (both local Intraday and cross-border XBID)
_HOURLY_PRODUCTS = {"Intraday_Hour_Power", "XBID_Hour_Power"}


# ─── Snapshot index (shared by all products) ─────────────────────────────────

class SnapshotIndex:
    """Snapshot timestamps and LOB mid-prices per product, loaded from snap_cache."""

    __slots__ = ("snap_times", "mid_prices")

    def __init__(
        self,
        snap_times: dict[str, np.ndarray],
        mid_prices: dict[str, np.ndarray],
    ):
        self.snap_times = snap_times  # product_key → int64 ns array
        self.mid_prices = mid_prices  # product_key → float64 array (same length)

    def __contains__(self, key: str) -> bool:
        return key in self.snap_times

    def __len__(self) -> int:
        return len(self.snap_times)


def _load_snapshot_index(snap_cache_path: Path) -> SnapshotIndex:
    """Load snapshot timestamps and LOB mid-prices from snap_cache .npz files.

    Returns a SnapshotIndex with per-product arrays of timestamps (int64 ns)
    and mid-prices (float64, (best_ask + best_bid) / 2).
    """
    snap_times_dict: dict[str, list[int]] = {}
    mid_prices_dict: dict[str, list[float]] = {}

    for npz_path in sorted(snap_cache_path.glob("*.npz")):
        data = np.load(npz_path)
        snap_times = data["snap_times"]   # int64 ns
        deliv_times = data["deliv_times"]  # int64 ns
        lobs = data["lobs"]                # float32 (N, 40)

        # LOB layout: [sell1, vsell1, buy1, vbuy1, ...] per level
        mid = ((lobs[:, 0] + lobs[:, 2]) / 2.0).astype(np.float64)

        for dt_ns in np.unique(deliv_times):
            delivery_time = pd.Timestamp(int(dt_ns), unit="ns", tz="UTC")
            key = f"{delivery_time.date()}_H{delivery_time.hour:02d}"

            mask = deliv_times == dt_ns

            if key not in snap_times_dict:
                snap_times_dict[key] = []
                mid_prices_dict[key] = []
            snap_times_dict[key].extend(snap_times[mask].tolist())
            mid_prices_dict[key].extend(mid[mask].tolist())

    # Sort by timestamp and deduplicate
    result_times: dict[str, np.ndarray] = {}
    result_mids: dict[str, np.ndarray] = {}
    for key in snap_times_dict:
        times = np.array(snap_times_dict[key], dtype=np.int64)
        mids = np.array(mid_prices_dict[key], dtype=np.float64)
        order = np.argsort(times, kind="stable")
        times = times[order]
        mids = mids[order]
        # Deduplicate (should already be unique, but be safe)
        uniq_mask = np.concatenate([[True], np.diff(times) > 0])
        result_times[key] = times[uniq_mask]
        result_mids[key] = mids[uniq_mask]

    return SnapshotIndex(result_times, result_mids)


# ─── Zip file discovery ──────────────────────────────────────────────────────

def _find_zips_for_range(
    zip_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    """Find raw EPEX zip files covering the delivery date range.

    The zip for observation date D contains events for delivery products on D.
    Include end_date + 1 day because the last delivery hour (H23 on end_date)
    may have events placed on the following observation day.
    """
    zips: list[Path] = []
    end_plus1 = end.normalize() + pd.Timedelta(days=1)

    for month_dir in sorted(zip_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for zf in sorted(month_dir.glob("Continuous_Orders-DE-*.csv.zip")):
            name_parts = zf.stem.split("-")
            date_str = name_parts[2]
            try:
                file_date = pd.Timestamp(date_str, tz="UTC")
            except ValueError:
                continue

            if start.normalize() <= file_date <= end_plus1:
                zips.append(zf)

    return zips


# ─── CSV reading with parquet cache ──────────────────────────────────────────

def _read_hourly_events(
    zip_path: Path,
    parquet_cache_dir: Path | None,
) -> pd.DataFrame:
    """Read filtered hourly events, using parquet cache when available.

    On first read, parses the raw EPEX zip (slow: ~5s per file) and caches
    the filtered result as a compressed parquet file.  Subsequent reads load
    from parquet (~0.3s).

    Parameters
    ----------
    zip_path : path to the raw EPEX zip
    parquet_cache_dir : directory for parquet cache files, or None to disable
    """
    # Try parquet cache first
    if parquet_cache_dir is not None:
        date_str = zip_path.stem.split("-")[2]  # YYYYMMDD
        parquet_path = parquet_cache_dir / f"{date_str}.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            # Ensure UTC timezone (parquet may strip it)
            for col in ("TransactionTime", "DeliveryStart"):
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize("UTC")
            return df

    # Parse from raw zip
    df = _read_hourly_events_from_zip(zip_path)

    # Cache as parquet for next time
    if parquet_cache_dir is not None and not df.empty:
        parquet_cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, compression="zstd", index=False)

    return df


def _read_hourly_events_from_zip(zip_path: Path) -> pd.DataFrame:
    """Read raw events from a single EPEX zip, filtered to hourly products."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=1, usecols=_RAW_COLS)

    # Filter to hourly power products
    df = df[df["Product"].isin(_HOURLY_PRODUCTS)].copy()
    if df.empty:
        return df

    # Parse timestamps
    df["TransactionTime"] = pd.to_datetime(df["TransactionTime"], utc=True)
    df["DeliveryStart"] = pd.to_datetime(df["DeliveryStart"], utc=True)

    # Drop iceberg replenishments (I) — not visible in LOB
    df = df[df["ActionCode"] != "I"].copy()

    return df


# ─── Timestamp conversion ────────────────────────────────────────────────────

def _to_nanoseconds(ts_array: np.ndarray) -> np.ndarray:
    """Convert a datetime64 array to int64 nanoseconds, regardless of resolution.

    pandas 2.x defaults to datetime64[us]; earlier versions use datetime64[ns].
    This function handles both by converting through datetime64[ns].
    """
    return ts_array.astype("datetime64[ns]").view("int64")


# ─── Vectorized tokenization ─────────────────────────────────────────────────

def _tokenize_events_for_product(
    events: pd.DataFrame,
    snap_times_ns: np.ndarray,
    mid_prices: np.ndarray,
    max_events: int,
) -> dict[str, np.ndarray]:
    """Tokenize events for one product, aligned to snapshot windows.

    Events are tokenized into 7 features:
        0: action_code (int 0-7)
        1: side (int 0-1)
        2: price_relative (float, event price relative to LOB mid-price of its window)
        3: quantity_log (float, log1p of quantity)
        4: time_delta (float, log1p of seconds since previous event)
        5: revision_flag (int 0-1, whether this is a modified order)
        6: is_aggressive (int 0-1, whether this is a trade execution)

    Windows with more than max_events are subsampled with priority:
        1. Trade events (M, P, C) — always kept
        2. Expiry/hibernate events (X, H) — kept if space allows
        3. Remaining events (A, D) — most recent kept
    """
    n_windows = len(snap_times_ns)
    out_features = np.zeros((n_windows, max_events, N_EVENT_FEATURES), dtype=np.float32)
    out_mask = np.zeros((n_windows, max_events), dtype=bool)
    out_n_events = np.zeros(n_windows, dtype=np.int32)

    if events.empty:
        return {
            "event_features": out_features,
            "event_mask": out_mask,
            "n_events": out_n_events,
        }

    # ── Vectorized feature computation for ALL events at once ────────────

    event_times_ns = _to_nanoseconds(events["TransactionTime"].values)
    action_codes_str = events["ActionCode"].values
    sides_str = events["Side"].values
    prices = events["Price"].values.astype(np.float64)
    quantities = events["Quantity"].values.astype(np.float64)
    revision_nos = events["RevisionNo"].values.astype(np.int64)

    n_events_total = len(events)

    # Feature 0: action_code (vectorized map)
    ac_map_vec = np.zeros(n_events_total, dtype=np.float32)
    for code, idx in ACTION_CODE_MAP.items():
        ac_map_vec[action_codes_str == code] = idx

    # Feature 1: side (vectorized)
    side_vec = np.where(sides_str == "SELL", 1.0, 0.0).astype(np.float32)

    # Feature 2: price_relative — computed per-window below (needs mid_prices)

    # Feature 3: quantity_log (vectorized)
    qty_vec = np.log1p(np.maximum(quantities, 0.0)).astype(np.float32)

    # Feature 4: time_delta (vectorized)
    time_deltas_ns = np.diff(event_times_ns, prepend=event_times_ns[0])
    time_deltas_s = np.maximum(time_deltas_ns / 1e9, 0.0)
    td_vec = np.log1p(time_deltas_s).astype(np.float32)

    # Feature 5: revision_flag (vectorized)
    rev_vec = (revision_nos > 1).astype(np.float32)

    # Feature 6: is_aggressive (vectorized)
    agg_vec = np.isin(action_codes_str, _TRADE_CODES_LIST).astype(np.float32)

    # ── Assign events to windows ─────────────────────────────────────────

    window_indices = np.searchsorted(snap_times_ns, event_times_ns, side="right") - 1
    window_indices = np.clip(window_indices, 0, n_windows - 1)

    # ── Fill output arrays per window ────────────────────────────────────

    for w in range(n_windows):
        w_mask = window_indices == w
        w_count = w_mask.sum()
        if w_count == 0:
            continue

        w_idx = np.where(w_mask)[0]

        # Priority-based subsampling if needed
        if w_count > max_events:
            w_idx = _priority_subsample(w_idx, action_codes_str, max_events)
            w_count = len(w_idx)

        out_n_events[w] = w_count

        # Compute price_relative using actual LOB mid-price for this window
        mid = mid_prices[w]
        w_prices = prices[w_idx]
        if abs(mid) > 0.01:
            price_rel = (w_prices - mid) / abs(mid)
        else:
            price_rel = np.zeros(w_count, dtype=np.float64)
        # Clip extreme limit orders (safety net — most values are well-behaved
        # with actual LOB mid, but orders at ±9999 EUR/MWh still exist)
        price_rel = np.clip(price_rel, -10.0, 10.0).astype(np.float32)

        # Copy pre-computed features into output
        out_features[w, :w_count, 0] = ac_map_vec[w_idx]
        out_features[w, :w_count, 1] = side_vec[w_idx]
        out_features[w, :w_count, 2] = price_rel
        out_features[w, :w_count, 3] = qty_vec[w_idx]
        out_features[w, :w_count, 4] = td_vec[w_idx]
        out_features[w, :w_count, 5] = rev_vec[w_idx]
        out_features[w, :w_count, 6] = agg_vec[w_idx]
        out_mask[w, :w_count] = True

    return {
        "event_features": out_features,
        "event_mask": out_mask,
        "n_events": out_n_events,
    }


def _priority_subsample(
    indices: np.ndarray,
    action_codes: np.ndarray,
    max_events: int,
) -> np.ndarray:
    """Subsample events with priority ordering.

    Priority:
        1. Trade events (M, P, C) — always kept
        2. Expiry/hibernate events (X, H) — kept if space
        3. Remaining (A, D) — most recent kept

    Returns array of selected indices (in original chronological order).
    """
    codes = action_codes[indices]

    trade_mask = np.isin(codes, _TRADE_CODES_LIST)
    expiry_mask = np.isin(codes, _EXPIRY_CODES_LIST)
    other_mask = ~(trade_mask | expiry_mask)

    trade_idx = indices[trade_mask]
    expiry_idx = indices[expiry_mask]
    other_idx = indices[other_mask]

    selected: list[np.ndarray] = []
    remaining = max_events

    # Priority 1: trades
    if len(trade_idx) <= remaining:
        selected.append(trade_idx)
        remaining -= len(trade_idx)
    else:
        selected.append(trade_idx[-remaining:])
        remaining = 0

    # Priority 2: expiry/hibernate
    if remaining > 0 and len(expiry_idx) > 0:
        n_take = min(len(expiry_idx), remaining)
        selected.append(expiry_idx[-n_take:])
        remaining -= n_take

    # Priority 3: other events (A, D) — most recent
    if remaining > 0 and len(other_idx) > 0:
        n_take = min(len(other_idx), remaining)
        selected.append(other_idx[-n_take:])

    result = np.concatenate(selected)
    result.sort()  # Maintain chronological order
    return result


# ─── Public entry point ──────────────────────────────────────────────────────

def extract_events_for_date_range(
    raw_data_path: str | Path,
    start_date: str,
    end_date: str,
    snap_cache_path: Path,
    output_root: Path,
    sampling_seconds: int = 10,
    max_events_per_window: int = 64,
    parsed_data_path: str | Path | None = None,
) -> list[str]:
    """Extract event features for all hourly products in a date range.

    For each delivery product, reads raw events from EPEX zips, aligns them
    to existing snapshot timestamps (from snap_cache), tokenises, and saves
    as events.npz in the corresponding product directory.

    Parameters
    ----------
    raw_data_path : path to data/battery_markets containing 2021/ zip dir
    start_date, end_date : date strings like "2021-01-11"
    snap_cache_path : path to snap_cache/{hash}/ containing per-day .npz files
    output_root : path to per_product/{subdir}/products/ directory
    sampling_seconds : snapshot interval (for reference, alignment uses actual timestamps)
    max_events_per_window : pad/subsample to this many events per window
    parsed_data_path : path to parsed/ dir for parquet event cache (None to disable)

    Returns
    -------
    List of product keys that had events extracted.
    """
    raw_data_path = Path(raw_data_path)
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")

    # Parquet cache directory (alongside CSVs and bins in parsed/)
    parquet_cache_dir: Path | None = None
    if parsed_data_path is not None:
        parquet_cache_dir = Path(parsed_data_path) / "event_parquet"

    # Load snapshot timestamps and LOB mid-prices from snap_cache
    snap_index = _load_snapshot_index(snap_cache_path)
    if len(snap_index) == 0:
        log.warning("No snapshot index loaded — skipping event extraction")
        return []

    # Find raw zip files covering our date range
    zip_dir = raw_data_path / "2021"
    zip_files = _find_zips_for_range(zip_dir, start, end)
    if not zip_files:
        log.warning(f"No raw zip files found in {zip_dir} for {start_date}→{end_date}")
        return []

    print(f"\n[EVENTS] Extracting events from {len(zip_files)} zip files")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  Max events/window: {max_events_per_window}")
    if parquet_cache_dir is not None:
        print(f"  Parquet cache: {parquet_cache_dir}")

    extracted_keys: list[str] = []

    for zip_path in tqdm(zip_files, desc="Processing event zips", unit="zip"):
        events_df = _read_hourly_events(zip_path, parquet_cache_dir)
        if events_df.empty:
            continue

        # Group by delivery product
        events_df["product_key"] = events_df["DeliveryStart"].dt.strftime(
            "%Y-%m-%d_H%H"
        )

        for product_key, product_events in events_df.groupby(
            "product_key", sort=False
        ):
            if product_key not in snap_index:
                continue

            snap_times = snap_index.snap_times[product_key]
            mid_prices = snap_index.mid_prices[product_key]
            if len(snap_times) == 0:
                continue

            prod_dir = output_root / product_key
            if not prod_dir.exists():
                continue

            result = _tokenize_events_for_product(
                product_events.sort_values("TransactionTime"),
                snap_times,
                mid_prices,
                max_events_per_window,
            )

            np.savez_compressed(
                prod_dir / "events.npz",
                event_features=result["event_features"],
                event_mask=result["event_mask"],
                n_events=result["n_events"],
            )
            extracted_keys.append(product_key)

    print(f"[EVENTS] Extracted events for {len(extracted_keys)} products")
    return extracted_keys
