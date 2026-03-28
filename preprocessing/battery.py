"""
Battery (EPEX intraday energy market) preprocessing and data loading.

Uses bitepy to reconstruct limit order books from raw EPEX Spot Continuous
Orders data.  The EPEX intraday continuous market has one independent LOB per
delivery hour (e.g., 2021-01-05 14:00-15:00).  Trading opens ~3pm the previous
day and closes ~5 minutes before delivery.

.npy format
-----------
  all_features=True  → 50 columns: [6 msg | 40 LOB | 4 labels]
  all_features=False → 44 columns: [40 LOB | 4 labels]

Message columns (synthesised from LOB state at sampling time):
  [time_delta_s, event_type, total_volume, mid_price, direction, spread_ticks]

LOB columns (10 levels, interleaved ask/bid):
  [sell1, vsell1, buy1, vbuy1, sell2, vsell2, buy2, vbuy2, ..., sell10, vsell10, buy10, vbuy10]

Label columns: label_h10, label_h20, label_h50, label_h100  (0=up, 1=stat, 2=down, nan=invalid)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import constants as cst
from constants import ProductMode
from utils.utils_data import labeling, normalize_messages, z_score_orderbook

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from cst
# ─────────────────────────────────────────────────────────────────────────────
_N_LOB = cst.N_LOB_LEVELS * cst.LEN_LEVEL   # 40
_N_LABELS = len(cst.LOBSTER_HORIZONS)         # 4
_NCOLS_FULL = cst.LEN_ORDER + _N_LOB + _N_LABELS   # 50
_NCOLS_LOB  = _N_LOB + _N_LABELS                    # 44
_HORIZON_IDX = {h: i for i, h in enumerate(cst.LOBSTER_HORIZONS)}  # {10:0, 20:1, 50:2, 100:3}


# ─────────────────────────────────────────────────────────────────────────────
# Data-loading functions (called by run.py at training time)
# ─────────────────────────────────────────────────────────────────────────────

def battery_load(path: str, all_features: bool, len_smooth: int, h: int, seq_size: int):
    """Load a single-horizon battery dataset from a .npy file.

    The .npy format is auto-detected from the column count:
      50 cols → full format (6 msg + 40 LOB + 4 labels)
      44 cols → LOB-only format (40 LOB + 4 labels)

    Parameters
    ----------
    path : str
        Path to the .npy file.
    all_features : bool
        If True, return LOB + message columns (46 features).
        If False, return LOB columns only (40 features).
    len_smooth : int
        Label smoothing window (cst.LEN_SMOOTH = 10).
    h : int
        Prediction horizon; one of {10, 20, 50, 100}.
    seq_size : int
        Model sequence length; labels start at index seq_size - len_smooth.

    Returns
    -------
    input_ : FloatTensor (N, num_features)
    labels  : LongTensor  (N_valid,)   values in {0, 1, 2}
    """
    if h not in _HORIZON_IDX:
        raise ValueError(f"Unsupported horizon: {h}. Must be one of {list(_HORIZON_IDX)}")

    arr = np.load(path)
    msg_cols, lob_cols, label_cols = _split_columns(arr, path)

    labels = label_cols[seq_size - len_smooth:, _HORIZON_IDX[h]]
    labels = labels[np.isfinite(labels)].astype(np.int64)
    if labels.size == 0:
        raise ValueError(f"No valid labels in {path} for horizon {h}.")
    if labels.min() < 0 or labels.max() > 2:
        raise ValueError(f"Invalid labels in {path}: {np.unique(labels)[:20]}")

    input_ = _select_features(msg_cols, lob_cols, all_features, path)
    return torch.from_numpy(input_).float(), torch.from_numpy(labels).long()


def battery_load_multi(path: str, all_features: bool, len_smooth: int, seq_size: int):
    """Load all 4 horizon labels stacked from a battery .npy file.

    Returns
    -------
    input_ : FloatTensor (N, num_features)
    labels  : LongTensor  (N_valid, 4)   columns: h10, h20, h50, h100
    """
    arr = np.load(path)
    msg_cols, lob_cols, label_cols = _split_columns(arr, path)

    label_start = seq_size - len_smooth
    all_labels = label_cols[label_start:]            # (N_valid, 4)
    finite_mask = np.all(np.isfinite(all_labels), axis=1)
    all_labels = all_labels[finite_mask].astype(np.int64)

    input_ = _select_features(msg_cols, lob_cols, all_features, path)
    return torch.from_numpy(input_).float(), torch.from_numpy(all_labels).long()


def _split_columns(arr: np.ndarray, path: str):
    """Parse .npy into (msg_cols, lob_cols, label_cols) based on column count."""
    ncols = arr.shape[1]
    if ncols == _NCOLS_FULL:        # 50: full format
        msg_cols   = arr[:, :cst.LEN_ORDER]
        lob_cols   = arr[:, cst.LEN_ORDER : cst.LEN_ORDER + _N_LOB]
        label_cols = arr[:, cst.LEN_ORDER + _N_LOB :]
    elif ncols == _NCOLS_LOB:       # 44: LOB-only format
        msg_cols   = None
        lob_cols   = arr[:, :_N_LOB]
        label_cols = arr[:, _N_LOB:]
    else:
        raise ValueError(
            f"Unexpected column count {ncols} in {path}. "
            f"Expected {_NCOLS_FULL} (full) or {_NCOLS_LOB} (LOB-only)."
        )
    return msg_cols, lob_cols, label_cols


def _select_features(msg_cols, lob_cols, all_features: bool, path: str) -> np.ndarray:
    """Return the feature array based on all_features flag."""
    if all_features:
        if msg_cols is None:
            raise ValueError(
                f"all_features=True but {path} was preprocessed with all_features=False "
                f"(LOB-only format, 44 columns). Re-preprocess with all_features=True."
            )
        return np.concatenate([lob_cols, msg_cols], axis=1).astype(np.float32)  # (N, 46)
    return lob_cols.astype(np.float32)  # (N, 40)


def battery_cache_subdir(sampling_time: str, dates: list[str]) -> str:
    """Return a human-readable subdirectory name encoding sampling_time and dates.

    Example: battery_cache_subdir("10s", ["2021-01-11", "2021-01-22"]) -> "10s_20210111_20210122"
    """
    start = dates[0].replace("-", "")
    end = dates[1].replace("-", "")
    return f"{sampling_time}_{start}_{end}"


# ─────────────────────────────────────────────────────────────────────────────
# BatteryDataBuilder — preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class BatteryDataBuilder:
    """Preprocess raw EPEX Spot Continuous Orders data into TLOB-compatible .npy files.

    Uses the bitepy library (C++ backend) for reliable LOB reconstruction.

    Parameters
    ----------
    data_dir : str
        Root data directory (cst.DATA_DIR = "data").
    date_trading_days : list[str]
        [start_date, end_date] strings, e.g. ["2021-01-01", "2021-01-12"].
    split_rates : list[float]
        [train_frac, val_frac, test_frac] summing to 1.
    sampling_type : SamplingType
        Must be TIME for battery; controls sampling mode.
    sampling_time : str
        Snapshot interval as pandas offset string, e.g. "5s", "10s".
    sampling_quantity : int
        Unused for battery (kept for API compatibility).
    product_mode : str
        "concat" — all products interleaved chronologically (single train/val/test.npy).
        "per_product" — one .npy per unique delivery contract (date+hour).
    market_type : str
        Market identifier passed to bitepy, default "EPEX".
    raw_data_path : str
        Path to directory containing the raw 2021/ zip files.
    parsed_data_path : str
        Path where intermediate CSVs and bins are cached.
    max_lob_depth : float
        Maximum cumulative volume (MWh) to query per LOB side.
    n_days : int
        Maximum number of days to process (truncates date range if longer).
    all_features : bool
        If True, synthesise and save 6 message columns (50-col format).
        If False, save LOB columns only (44-col format).
    """

    def __init__(
        self,
        data_dir: str,
        date_trading_days: list,
        split_rates: list,
        sampling_type,
        sampling_time: str,
        sampling_quantity: int,
        product_mode: str = "concat",
        market_type: str = "EPEX",
        raw_data_path: str = "data/battery_markets",
        parsed_data_path: str = "data/battery_markets/parsed",
        max_lob_depth: float = 1000.0,
        all_features: bool = True,
        force_rebuild: bool = False,
    ):
        self.data_dir = data_dir
        self.sampling_time_str = sampling_time
        self.date_strs = list(date_trading_days)
        self.start_date = pd.Timestamp(date_trading_days[0], tz="UTC")
        self.end_date   = pd.Timestamp(date_trading_days[1], tz="UTC")
        self.split_rates = split_rates
        self.sampling_interval = pd.tseries.frequencies.to_offset(sampling_time)
        self.sampling_seconds = int(pd.Timedelta(sampling_time).total_seconds())
        if isinstance(product_mode, str):
            self.product_mode = ProductMode(product_mode)
        else:
            self.product_mode = product_mode
        self.market_type = market_type
        self.raw_data_path = Path(raw_data_path)
        self.parsed_data_path = Path(parsed_data_path)
        self.max_lob_depth = max_lob_depth
        self.all_features = all_features
        self.force_rebuild = force_rebuild
        self.n_lob_levels = cst.N_LOB_LEVELS  # 10

    # ── Public entry point ────────────────────────────────────────────────────

    def prepare_save_datasets(self):
        """Run the full preprocessing pipeline and save .npy files."""
        print(f"\n[BATTERY] Starting preprocessing pipeline")
        print(f"  Date range : {self.start_date.date()} → {self.end_date.date()}")
        print(f"  Sampling   : {self.sampling_seconds}s")
        print(f"  Mode       : {self.product_mode}")
        print(f"  Features   : {'LOB + messages (50 cols)' if self.all_features else 'LOB only (44 cols)'}")

        # ── Stage 1: Parse raw EPEX zips to CSV ──────────────────────────────
        print(f"\n[BATTERY] Stage 1/4: Parsing raw EPEX data to CSV...")
        csv_path = self._parse_raw_to_csv()

        # ── Stage 2: Convert CSVs to binary ──────────────────────────────────
        print(f"\n[BATTERY] Stage 2/4: Converting CSVs to binary format...")
        bin_path = self._convert_csv_to_bins(csv_path)

        # ── Stage 3: Extract LOB snapshots via simulation (per-day cached) ──
        print(f"\n[BATTERY] Stage 3/4: Extracting LOB snapshots...")
        cache_hash = self._snapshot_cache_hash()
        days = list(pd.date_range(self.start_date, self.end_date, freq="D"))
        day_paths = [self._day_cache_path(bin_path, d.date(), cache_hash) for d in days]

        if self.force_rebuild:
            snap_dir = bin_path.parent / "snap_cache" / cache_hash
            if snap_dir.exists():
                shutil.rmtree(snap_dir)
                print(f"  [BATTERY] force_rebuild: deleted cache dir {snap_dir}")

        all_cached = all(p.exists() for p in day_paths)
        if all_cached:
            print(f"  [BATTERY] Using cached per-day snapshots (hash={cache_hash})")
            snapshots = self._load_day_caches(day_paths)
        else:
            snapshots = self._extract_all_snapshots(bin_path, cache_hash)

        if snapshots["lobs"].shape[0] == 0:
            raise ValueError("No LOB snapshots extracted. Check raw data and date range.")

        # ── Stage 4: Split, label, normalise, save ────────────────────────────
        print(f"\n[BATTERY] Stage 4/4: Building datasets...")
        if self.product_mode == ProductMode.CONCAT:
            self._build_concat_datasets(snapshots)
        elif self.product_mode == ProductMode.PER_PRODUCT:
            self._build_per_product_datasets(snapshots)
        else:
            raise ValueError(f"Unknown product_mode: {self.product_mode!r}.")

        print(f"\n[BATTERY] Preprocessing complete.\n")

    # ── Stage 1: Parse raw data ───────────────────────────────────────────────

    def _parse_raw_to_csv(self) -> Path:
        """Call bitepy.Data.parse_market_data(); skip if CSVs already exist."""
        import bitepy
        csv_path = self.parsed_data_path / "csvs"
        csv_path.mkdir(parents=True, exist_ok=True)

        # Determine how many days we need: from start - 1 day (for trading window) to end
        parse_start = (self.start_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        parse_end   = self.end_date.strftime("%Y-%m-%d")

        # Check cache: need CSVs for start-1 through end
        expected_dates = pd.date_range(
            self.start_date - pd.Timedelta(days=1), self.end_date, freq="D"
        )
        existing = set(csv_path.glob("orderbook_*.csv.zip"))
        existing_dates = {
            pd.Timestamp(f.name.replace("orderbook_", "").replace(".csv.zip", ""))
            for f in existing
        }
        missing = [d for d in expected_dates if pd.Timestamp(d.date()) not in existing_dates]

        if not missing:
            print(f"  [BATTERY] Using cached CSVs from {csv_path}")
        else:
            print(f"  Parsing {len(expected_dates)} days ({parse_start} → {parse_end})...")
            data_handler = bitepy.Data()
            data_handler.parse_market_data(
                start_date_str=parse_start,
                end_date_str=parse_end,
                marketdatapath=str(self.raw_data_path),
                savepath=str(csv_path) + "/",
                market_type=self.market_type,
                verbose=True,
            )

        return csv_path

    # ── Stage 2: Convert to binary ────────────────────────────────────────────

    def _convert_csv_to_bins(self, csv_path: Path) -> Path:
        """Call bitepy.Data.create_bins_from_csv(); skip if bins already exist."""
        import bitepy
        bin_path = self.parsed_data_path / "bins"
        bin_path.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(csv_path.glob("orderbook_*.csv.zip"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {csv_path}. Did Stage 1 succeed?")

        # Check cache: compare number of bins vs csvs
        existing_bins = set(bin_path.glob("*.bin"))
        missing_bins = [f for f in csv_files if not (bin_path / (f.stem.replace(".csv", "") + ".bin")).exists()]

        if not missing_bins:
            print(f"  [BATTERY] Using cached bins from {bin_path}")
        else:
            print(f"  Converting {len(csv_files)} CSV files to binary...")
            data_handler = bitepy.Data()
            data_handler.create_bins_from_csv(csv_files, str(bin_path) + "/", verbose=True)

        return bin_path

    # ── Stage 3: Extract LOB snapshots ───────────────────────────────────────

    def _snapshot_cache_hash(self) -> str:
        """Return deterministic hash for Stage 3 per-day cache keying."""
        import hashlib
        key = (
            f"{self.start_date}_{self.end_date}"
            f"_{self.sampling_seconds}_{self.max_lob_depth}_{self.all_features}"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _day_cache_path(self, bin_path: Path, day_date, cache_hash: str) -> Path:
        """Return path for a single observation day's snapshot cache."""
        snap_dir = bin_path.parent / "snap_cache" / cache_hash
        snap_dir.mkdir(parents=True, exist_ok=True)
        return snap_dir / f"{day_date}.npz"

    def _flush_day_buffer(self, day_buffer: list[dict], path: Path) -> None:
        """Convert a day's snapshot dicts to numpy arrays and save to .npz."""
        if not day_buffer:
            return
        lobs = np.array([s["lob"] for s in day_buffer], dtype=np.float32)
        snap_times = np.array([s["snapshot_time"].value for s in day_buffer], dtype=np.int64)
        deliv_times = np.array([s["delivery_time"].value for s in day_buffer], dtype=np.int64)
        deliv_dates = np.array(
            [s["delivery_date"].toordinal() for s in day_buffer], dtype=np.int32
        )
        save_kw: dict[str, np.ndarray] = dict(
            lobs=lobs, snap_times=snap_times,
            deliv_times=deliv_times, deliv_dates=deliv_dates,
        )
        if day_buffer[0].get("msg") is not None:
            save_kw["msgs"] = np.array(
                [s["msg"] for s in day_buffer], dtype=np.float32
            )
        np.savez(path, **save_kw)

    def _load_day_caches(self, day_paths: list[Path]) -> dict[str, np.ndarray]:
        """Load and concatenate all per-day .npz cache files."""
        all_lobs, all_snap, all_deliv, all_dd, all_msgs = [], [], [], [], []
        has_msgs = False
        for p in day_paths:
            data = np.load(p)
            all_lobs.append(data["lobs"])
            all_snap.append(data["snap_times"])
            all_deliv.append(data["deliv_times"])
            all_dd.append(data["deliv_dates"])
            if "msgs" in data:
                all_msgs.append(data["msgs"])
                has_msgs = True
        result: dict[str, np.ndarray] = {
            "lobs": np.concatenate(all_lobs) if all_lobs else np.empty((0, self.n_lob_levels * 4), dtype=np.float32),
            "snap_times": np.concatenate(all_snap) if all_snap else np.empty(0, dtype=np.int64),
            "deliv_times": np.concatenate(all_deliv) if all_deliv else np.empty(0, dtype=np.int64),
            "deliv_dates": np.concatenate(all_dd) if all_dd else np.empty(0, dtype=np.int32),
        }
        if has_msgs:
            result["msgs"] = np.concatenate(all_msgs)
        n = result["lobs"].shape[0]
        print(f"  Loaded {n:,} cached (snapshot, product) pairs from {len(day_paths)} day files")
        return result

    def _extract_all_snapshots(self, bin_path: Path, cache_hash: str) -> dict[str, np.ndarray]:
        """Run bitepy Simulation, flushing snapshots to per-day .npz files.

        Each observation day's snapshots are converted to numpy arrays and saved
        to disk immediately, keeping peak memory bounded to one day's data.

        Returns
        -------
        dict with keys:
            snap_times  : int64 (N,) — nanoseconds since epoch
            deliv_times : int64 (N,)
            deliv_dates : int32 (N,) — date ordinals
            lobs        : float32 (N, 40)
            msgs        : float32 (N, 6) — only present when all_features=True
        """
        import bitepy

        days = list(pd.date_range(self.start_date, self.end_date, freq="D"))
        actual_end = self.end_date + pd.Timedelta(days=1)  # exclusive

        print(f"  Processing {len(days)} days: {days[0].date()} → {days[-1].date()}")

        sim = bitepy.Simulation(
            start_date=self.start_date,
            end_date=actual_end,
            only_traverse_lob=True,
        )

        # Build bin list day-by-day: add_bin_to_orderqueue REPLACES the queue rather than
        # appending, so bins must be loaded one at a time. Each bin covers one delivery date;
        # when the current bin is exhausted we load the next. The LOB state persists across
        # bin loads, so products from earlier bins remain visible while new ones open.
        # Include start_date - 1 bin so products delivering on start_date are pre-loaded.
        bin_start = self.start_date - pd.Timedelta(days=1)
        all_bins = sorted([
            str(p) for p in bin_path.glob("*.bin")
            if bin_start.date() <= pd.Timestamp(p.stem.split("_", 1)[1]).date() <= self.end_date.date()
        ])
        if not all_bins:
            raise FileNotFoundError(f"No bin files found in {bin_path} for {bin_start.date()} → {self.end_date.date()}")

        # Load first bin; subsequent bins are loaded when the previous one is exhausted
        bin_idx = 0
        sim.add_bin_to_orderqueue(all_bins[bin_idx])
        bin_idx += 1

        current_time = self.start_date
        interval = pd.Timedelta(seconds=self.sampling_seconds)
        _GATE_CLOSURE = pd.Timedelta(minutes=5)  # EPEX gate closure before delivery

        # Per-day buffer — flushed to disk at each day boundary to bound memory
        day_buffer: list[dict] = []
        day_paths = [self._day_cache_path(bin_path, d.date(), cache_hash) for d in days]
        total_flushed = 0

        # Outer tqdm: one step per calendar day
        day_bar = tqdm(days, desc="Extracting LOB snapshots", unit="day")
        day_idx = 0

        # Pre-track prev_lob per delivery product for message feature synthesis
        prev_lob_per_product: dict[pd.Timestamp, np.ndarray] = {}
        prev_time_per_product: dict[pd.Timestamp, pd.Timestamp] = {}

        while current_time < actual_end:
            stop_at = current_time + interval
            sim.set_stop_time(stop_at)
            is_last_bin = bin_idx >= len(all_bins)
            sim.run_one_day(is_last=is_last_bin)

            if sim.has_stopped_at_stop_time():
                lob_dict = sim.get_limit_order_book_state(
                    max_action=self.max_lob_depth, return_dict=True
                )

                if lob_dict:
                    for dt_key in sorted(lob_dict.keys()):
                        # Keys from return_dict=True are Unix ms timestamps
                        delivery_time = pd.Timestamp(dt_key, unit="ms", tz="UTC")

                        # Skip post-gate-closure snapshots immediately to avoid
                        # accumulating frozen LOB state in memory (~3x RAM reduction).
                        if current_time >= delivery_time - _GATE_CLOSURE:
                            continue

                        product_data = lob_dict[dt_key]
                        lob_row = self._aggregate_to_levels_numpy(product_data)
                        if lob_row is None:
                            continue

                        msg_row = None
                        if self.all_features:
                            prev_lob = prev_lob_per_product.get(delivery_time)
                            prev_ts  = prev_time_per_product.get(delivery_time, current_time)
                            msg_row  = self._synthesize_message_features(lob_row, prev_lob, prev_ts, current_time)
                            prev_lob_per_product[delivery_time] = lob_row
                            prev_time_per_product[delivery_time] = current_time

                        day_buffer.append({
                            "snapshot_time": current_time,
                            "delivery_time": delivery_time,
                            "delivery_date": delivery_time.date(),
                            "lob": lob_row,
                            "msg": msg_row,
                        })

            # When the current bin is exhausted, load the next one (or stop if done)
            if not sim.has_orders_remaining():
                if bin_idx < len(all_bins):
                    sim.add_bin_to_orderqueue(all_bins[bin_idx])
                    bin_idx += 1
                else:
                    break  # All bins exhausted; end early

            # Advance day bar when we cross into a new calendar day;
            # flush the day buffer to disk at each boundary to bound memory.
            next_day_boundary = self.start_date + pd.Timedelta(days=day_idx + 1)
            while day_idx < len(days) and stop_at >= next_day_boundary:
                if day_buffer:
                    self._flush_day_buffer(day_buffer, day_paths[day_idx])
                    total_flushed += len(day_buffer)
                    day_buffer = []
                day_bar.update(1)
                day_idx += 1
                next_day_boundary = self.start_date + pd.Timedelta(days=day_idx + 1)

            current_time = stop_at

        # Flush any remaining snapshots (last day)
        if day_buffer and day_idx < len(day_paths):
            self._flush_day_buffer(day_buffer, day_paths[min(day_idx, len(day_paths) - 1)])
            total_flushed += len(day_buffer)
            day_buffer = []

        # Flush remaining day bar ticks
        while day_idx < len(days):
            day_bar.update(1)
            day_idx += 1
        day_bar.close()

        print(f"  Extracted and saved {total_flushed:,} snapshots across {len(days)} day files")

        # Load all per-day files and return concatenated arrays
        existing_paths = [p for p in day_paths if p.exists()]
        return self._load_day_caches(existing_paths)

    # ── Helpers: LOB aggregation ──────────────────────────────────────────────

    def _aggregate_to_levels_numpy(self, product_data: dict) -> np.ndarray | None:
        """Aggregate individual orders into top-N price levels using numpy.

        Accepts the per-product dict from get_limit_order_book_state(return_dict=True).
        bitepy pre-sorts orders: sells ascending by price, buys descending by price.
        We use np.add.reduceat to aggregate consecutive same-price entries in O(N),
        avoiding pandas groupby overhead.

        Returns float32 ndarray [sell1, vsell1, buy1, vbuy1, ...] of length 40,
        or None if either side is empty at level 1.
        """
        sell_prices  = np.asarray(product_data["sell_prices"],  dtype=np.float64)
        sell_volumes = np.asarray(product_data["sell_volumes"], dtype=np.float64)
        buy_prices   = np.asarray(product_data["buy_prices"],   dtype=np.float64)
        buy_volumes  = np.asarray(product_data["buy_volumes"],  dtype=np.float64)

        if len(sell_prices) == 0 or len(buy_prices) == 0:
            return None

        sell_p, sell_v = self._reduceat_aggregate(sell_prices, sell_volumes)
        buy_p,  buy_v  = self._reduceat_aggregate(buy_prices,  buy_volumes)

        if len(sell_p) == 0 or len(buy_p) == 0:
            return None

        n = self.n_lob_levels
        row: list[float] = []
        for level in range(n):
            # Forward-fill prices beyond the available depth: repeat the last known
            # price level with zero volume so z-scoring doesn't create an artificial
            # cluster at a single pad value.
            if level < len(sell_p):
                ask_p, ask_v = float(sell_p[level]), float(sell_v[level])
            else:
                ask_p, ask_v = float(sell_p[-1]), 0.0
            if level < len(buy_p):
                bid_p, bid_v = float(buy_p[level]), float(buy_v[level])
            else:
                bid_p, bid_v = float(buy_p[-1]), 0.0
            row.extend([ask_p, ask_v, bid_p, bid_v])

        # Skip crossed book or zero volume at best level
        if row[1] == 0.0 or row[3] == 0.0:  # vsell1 or vbuy1 == 0
            return None

        return np.array(row, dtype=np.float32)

    @staticmethod
    def _reduceat_aggregate(
        prices: np.ndarray, volumes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate volumes at consecutive same-price entries.

        Parameters are assumed pre-sorted (ascending for sells, descending for buys).
        Returns (unique_prices, aggregated_volumes) — same ordering as input.
        """
        if len(prices) == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        # Find indices where the price changes
        changes = np.where(np.diff(prices.astype(np.float64)) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes])
        agg_volumes = np.add.reduceat(volumes.astype(np.float64), boundaries)
        agg_prices  = prices.astype(np.float64)[boundaries]
        return agg_prices, agg_volumes

    # ── Helpers: message feature synthesis ───────────────────────────────────

    def _synthesize_message_features(
        self,
        lob_row: list[float],
        prev_lob: list[float] | None,
        prev_time: pd.Timestamp,
        curr_time: pd.Timestamp,
    ) -> list[float]:
        """Synthesise 6 message columns from LOB state.

        Columns (matching LOBSTER convention):
          [time_delta_s, event_type, total_volume, mid_price, direction, spread_ticks]

        direction: sign of mid-price change since last snapshot (0 if first).
        spread_ticks: (sell1 - buy1) / 0.1 (EPEX tick = 0.01 EUR/MWh, but we use 0.1 as
                      the normalisation unit consistent with existing battery code).
        """
        sell1, vsell1, buy1, vbuy1 = lob_row[0], lob_row[1], lob_row[2], lob_row[3]
        mid = (sell1 + buy1) / 2.0
        spread_ticks = max((sell1 - buy1) / 0.1, 0.0)
        total_vol = sum(lob_row[1::2])  # sum of all volume columns (odd indices in lob_row)
        time_delta = max((curr_time - prev_time).total_seconds(), 0.0)

        if prev_lob is not None:
            prev_mid = (prev_lob[0] + prev_lob[2]) / 2.0
            direction = float(np.sign(mid - prev_mid))
        else:
            direction = 0.0

        event_type = 1.0  # time-sampled snapshot → treat as limit-order event

        return [time_delta, event_type, total_vol, mid, direction, spread_ticks]

    # ── Stage 4a: Concat mode ─────────────────────────────────────────────────

    def _build_concat_datasets(self, snapshots: dict[str, np.ndarray]):
        """Build train/val/test.npy with all products interleaved chronologically."""
        import datetime as _dt

        snap_times  = snapshots["snap_times"]
        deliv_times = snapshots["deliv_times"]
        lobs        = snapshots["lobs"]
        msgs        = snapshots.get("msgs")

        # Sort by (snapshot_time, delivery_time)
        sort_idx = np.lexsort((deliv_times, snap_times))
        snap_times  = snap_times[sort_idx]
        deliv_times = deliv_times[sort_idx]
        lobs        = lobs[sort_idx]
        if msgs is not None:
            msgs = msgs[sort_idx]

        # Derive observation day ordinals from snap_times (int64 ns since epoch)
        _EPOCH_ORD = _dt.date(1970, 1, 1).toordinal()
        _NS_PER_DAY = 86400 * 10**9
        obs_date_ord = (snap_times // _NS_PER_DAY + _EPOCH_ORD).astype(np.int32)

        # Determine split by observation day
        unique_obs_ord = np.unique(obs_date_ord)
        obs_days = sorted([_dt.date.fromordinal(int(o)) for o in unique_obs_ord])
        train_days, val_days, test_days = self._split_day_list(obs_days)

        print(f"  Concat split — train: {len(train_days)} days, val: {len(val_days)} days, test: {len(test_days)} days")

        train_ord = np.array([d.toordinal() for d in train_days], dtype=np.int32)
        val_ord   = np.array([d.toordinal() for d in val_days], dtype=np.int32)
        test_ord  = np.array([d.toordinal() for d in test_days], dtype=np.int32)

        train_mask = np.isin(obs_date_ord, train_ord)
        val_mask   = np.isin(obs_date_ord, val_ord)
        test_mask  = np.isin(obs_date_ord, test_ord)

        print(f"  train: {train_mask.sum():,} snapshots")
        print(f"  val: {val_mask.sum():,} snapshots")
        print(f"  test: {test_mask.sum():,} snapshots")

        subdir = battery_cache_subdir(self.sampling_time_str, self.date_strs)
        out_dir = Path(self.data_dir) / "battery_markets" / "concat" / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build feature arrays directly from numpy slices
        def _make_features(mask):
            raw_lob = lobs[mask].astype(np.float32)
            if self.all_features and msgs is not None:
                m = msgs[mask].astype(np.float32)
                return np.concatenate([m, raw_lob], axis=1), raw_lob
            return raw_lob, raw_lob

        train_features, train_raw_lob = _make_features(train_mask)
        val_features,   val_raw_lob   = _make_features(val_mask)
        test_features,  test_raw_lob  = _make_features(test_mask)

        # Compute labels on raw LOB (before normalisation)
        print("  Computing labels...")
        train_labels = self._build_labels(train_raw_lob)
        val_labels   = self._build_labels(val_raw_lob)
        test_labels  = self._build_labels(test_raw_lob)

        # Normalise
        print("  Normalising features...")
        train_norm, val_norm, test_norm = self._normalise_splits(
            train_features, val_features, test_features
        )

        # Save
        self._save_split(train_norm, train_labels, out_dir / "train.npy")
        self._save_split(val_norm,   val_labels,   out_dir / "val.npy")
        self._save_split(test_norm,  test_labels,  out_dir / "test.npy")

        self._validate_and_report(out_dir)

    # ── Stage 4b: Per-product mode ────────────────────────────────────────────

    def _build_per_product_datasets(self, snapshots: dict[str, np.ndarray]):
        """Build one .npy per unique delivery contract from numpy arrays."""
        import datetime as _dt

        snap_times  = snapshots["snap_times"]     # int64 (N,)
        deliv_times = snapshots["deliv_times"]    # int64 (N,)
        deliv_dates = snapshots["deliv_dates"]    # int32 (N,) ordinals
        lobs        = snapshots["lobs"]           # float32 (N, 40)
        msgs        = snapshots.get("msgs")       # float32 (N, 6) or None

        # Group by unique delivery contract (one per delivery_time)
        unique_dt, inverse = np.unique(deliv_times, return_inverse=True)

        # Determine split assignment by delivery date
        unique_dd_ord = np.unique(deliv_dates)
        unique_dd = sorted([_dt.date.fromordinal(int(o)) for o in unique_dd_ord])
        train_days, val_days, test_days = self._split_day_list(unique_dd)
        train_ord_set = {d.toordinal() for d in train_days}
        val_ord_set   = {d.toordinal() for d in val_days}
        test_ord_set  = {d.toordinal() for d in test_days}

        print(f"  Per-product split — train: {len(train_days)} delivery days, "
              f"val: {len(val_days)}, test: {len(test_days)}")
        print(f"  Total unique delivery contracts: {len(unique_dt)}")

        subdir = battery_cache_subdir(self.sampling_time_str, self.date_strs)
        out_root = Path(self.data_dir) / "battery_markets" / "per_product" / subdir
        out_root.mkdir(parents=True, exist_ok=True)
        products_dir = out_root / "products"

        # Remove stale outputs from previous runs
        if products_dir.exists():
            shutil.rmtree(products_dir)
            print("  Cleaned stale products directory")
        for stale in out_root.glob("*.npy"):
            stale.unlink()
        if (out_root / "products.json").exists():
            (out_root / "products.json").unlink()

        products_dir.mkdir(parents=True, exist_ok=True)

        # Compute global train normalization stats using a single numpy mask
        print("  Computing global train normalization stats...")
        train_ord_arr = np.array(sorted(train_ord_set), dtype=np.int32)
        train_mask = np.isin(deliv_dates, train_ord_arr)
        train_lobs_global = lobs[train_mask]  # direct slice — no dict copies
        if self.all_features and msgs is not None:
            train_msgs_global = msgs[train_mask]
            train_features_global = np.concatenate(
                [train_msgs_global.astype(np.float32), train_lobs_global.astype(np.float32)], axis=1
            )
        else:
            train_features_global = train_lobs_global.astype(np.float32)
        _, global_stats = self._normalise_features(train_features_global, stats=None)
        del train_features_global  # free memory

        # Save per-product files
        product_keys_saved: list[str] = []
        for i in tqdm(range(len(unique_dt)), desc="Saving products", unit="contract"):
            mask = inverse == i
            prod_snap_t = snap_times[mask]
            prod_lobs   = lobs[mask]
            prod_msgs   = msgs[mask] if msgs is not None else None
            dd_ord      = int(deliv_dates[mask][0])

            # Determine split
            if dd_ord in train_ord_set:
                split_name = "train"
            elif dd_ord in val_ord_set:
                split_name = "val"
            elif dd_ord in test_ord_set:
                split_name = "test"
            else:
                continue

            # Sort by snapshot time (should be in order, but ensure)
            sort_idx = np.argsort(prod_snap_t, kind="stable")
            prod_lobs = prod_lobs[sort_idx]
            if prod_msgs is not None:
                prod_msgs = prod_msgs[sort_idx]

            delivery_time = pd.Timestamp(int(unique_dt[i]), unit="ns", tz="UTC")
            key = self._delivery_key(delivery_time)

            prod_dir = products_dir / key
            prod_dir.mkdir(exist_ok=True)
            self._save_product_arrays(split_name, prod_lobs, prod_msgs, prod_dir, global_stats)
            product_keys_saved.append(key)

        # Save manifest at root level (read by run.py)
        manifest = sorted(product_keys_saved)
        with open(out_root / "products.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Saved {len(product_keys_saved)} product directories under {products_dir}")

    def _save_product_arrays(
        self,
        split_name: str,
        lobs: np.ndarray,
        msgs: np.ndarray | None,
        prod_dir: Path,
        global_stats: dict | None = None,
    ):
        """Build and save train/val/test.npy for one delivery contract from arrays."""
        if lobs.shape[0] == 0:
            return

        if self.all_features and msgs is not None:
            features = np.concatenate(
                [msgs.astype(np.float32), lobs.astype(np.float32)], axis=1
            )
        else:
            features = lobs.astype(np.float32)

        labels = self._build_labels(lobs.astype(np.float32))
        if global_stats:
            norm_features, _ = self._normalise_features(features, stats=global_stats)
        else:
            norm_features, _, _ = self._normalise_single(features)
        self._save_split(norm_features, labels, prod_dir / f"{split_name}.npy")

        # Save empty placeholders for the other two splits
        ncols = _NCOLS_FULL if self.all_features else _NCOLS_LOB
        for sname in ("train", "val", "test"):
            path = prod_dir / f"{sname}.npy"
            if not path.exists():
                np.save(path, np.zeros((0, ncols), dtype=np.float32))

    # ── Feature / label / normalisation helpers ───────────────────────────────

    def _delivery_key(self, delivery_time: pd.Timestamp) -> str:
        """Return a filesystem-safe key like '2021-01-05_H14'."""
        dt = delivery_time.tz_convert("UTC") if delivery_time.tzinfo else delivery_time
        return f"{dt.date()}_H{dt.hour:02d}"

    def _split_day_list(self, days: list) -> tuple[list, list, list]:
        """Split a sorted list of dates into (train, val, test) using split_rates."""
        n = len(days)
        if n < 3:
            raise ValueError(f"Need at least 3 days for train/val/test split, got {n}.")
        n_train = max(1, int(n * self.split_rates[0]))
        n_val   = max(1, int(n * self.split_rates[1]))
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_val  = max(1, n_val - 1)
            n_test = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test  = n - n_train - n_val
        return days[:n_train], days[n_train:n_train + n_val], days[n_train + n_val:]

    def _build_labels(self, raw_lob: np.ndarray) -> np.ndarray:
        """Compute multi-horizon labels from raw (un-normalised) LOB array.

        Returns (N, 4) float array with np.inf for invalid positions.
        """
        if raw_lob.shape[0] == 0:
            return np.full((0, _N_LABELS), np.inf, dtype=np.float32)

        label_cols = []
        for h in tqdm(cst.LOBSTER_HORIZONS, desc="  Computing labels", leave=False):
            lbls = labeling(raw_lob, cst.LEN_SMOOTH, h)  # (N - h - LEN_SMOOTH + 1,)
            pad = np.full(raw_lob.shape[0] - lbls.shape[0], np.inf, dtype=np.float64)
            label_cols.append(np.concatenate([lbls.astype(np.float64), pad]))

        return np.stack(label_cols, axis=1).astype(np.float32)  # (N, 4)

    def _normalise_splits(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        test_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z-score normalise all splits, fitting stats on train only."""
        train_norm, stats = self._normalise_features(train_features, stats=None)
        val_norm,   _     = self._normalise_features(val_features,   stats=stats)
        test_norm,  _     = self._normalise_features(test_features,  stats=stats)
        return train_norm, val_norm, test_norm

    def _normalise_single(self, features: np.ndarray) -> tuple[np.ndarray, dict, dict]:
        """Normalise a single split using its own stats (used for per-product)."""
        norm, stats = self._normalise_features(features, stats=None)
        return norm, stats, {}

    def _normalise_features(
        self, features: np.ndarray, stats: dict | None
    ) -> tuple[np.ndarray, dict]:
        """Z-score normalise feature columns.

        For all_features=True:  features is (N, 46) = [6 msg | 40 LOB]
        For all_features=False: features is (N, 40) = [40 LOB]
        """
        if features.shape[0] == 0:
            return features, stats or {}

        if self.all_features:
            msg_df = pd.DataFrame(
                features[:, :cst.LEN_ORDER],
                columns=["time", "event_type", "size", "price", "direction", "depth"]
            )
            lob_df = pd.DataFrame(features[:, cst.LEN_ORDER:])

            if stats is None:
                ms, mp, ss, sp = self._lob_stats(lob_df)
                lob_df, _, _, _, _ = z_score_orderbook(lob_df, ms, mp, ss, sp)
                msg_df, ms2, mp2, ss2, sp2, mt, st, md, sd = self._normalise_messages_safe(msg_df)
                stats = dict(
                    lob_mean_size=ms, lob_mean_price=mp, lob_std_size=ss, lob_std_price=sp,
                    msg_mean_size=ms2, msg_mean_price=mp2, msg_std_size=ss2, msg_std_price=sp2,
                    msg_mean_time=mt, msg_std_time=st, msg_mean_depth=md, msg_std_depth=sd,
                )
            else:
                lob_df, _, _, _, _ = z_score_orderbook(
                    lob_df,
                    stats["lob_mean_size"], stats["lob_mean_price"],
                    stats["lob_std_size"],  stats["lob_std_price"],
                )
                msg_df, *_ = self._normalise_messages_safe(
                    msg_df,
                    stats["msg_mean_size"], stats["msg_mean_price"],
                    stats["msg_std_size"],  stats["msg_std_price"],
                    stats["msg_mean_time"], stats["msg_std_time"],
                    stats["msg_mean_depth"], stats["msg_std_depth"],
                )

            norm = np.concatenate([msg_df.values, lob_df.values], axis=1).astype(np.float32)
        else:
            lob_df = pd.DataFrame(features)
            if stats is None:
                ms, mp, ss, sp = self._lob_stats(lob_df)
                lob_df, _, _, _, _ = z_score_orderbook(lob_df, ms, mp, ss, sp)
                stats = dict(lob_mean_size=ms, lob_mean_price=mp, lob_std_size=ss, lob_std_price=sp)
            else:
                lob_df, _, _, _, _ = z_score_orderbook(
                    lob_df,
                    stats["lob_mean_size"], stats["lob_mean_price"],
                    stats["lob_std_size"],  stats["lob_std_price"],
                )
            norm = lob_df.values.astype(np.float32)

        return norm, stats

    @staticmethod
    def _lob_stats(lob_df: pd.DataFrame, eps: float = 1e-8) -> tuple:
        """Compute LOB normalisation stats with a zero-std floor."""
        ms = float(lob_df.iloc[:, 1::2].stack().mean())
        ss = max(float(lob_df.iloc[:, 1::2].stack().std()), eps)
        mp = float(lob_df.iloc[:, 0::2].stack().mean())
        sp = max(float(lob_df.iloc[:, 0::2].stack().std()), eps)
        return ms, mp, ss, sp

    def _normalise_messages_safe(self, data: pd.DataFrame, *args):
        """Call normalize_messages() with a zero-std guard.

        normalize_messages raises ValueError when any column has std=0 (produces NaN).
        We pre-compute stats with a floor of 1e-8 to prevent this.
        """
        _EPS = 1e-8
        if not args:
            # Compute stats — guard against zero std (e.g. constant time_delta, direction)
            mean_size   = float(data["size"].mean())
            std_size    = max(float(data["size"].std()), _EPS)
            mean_prices = float(data["price"].mean())
            std_prices  = max(float(data["price"].std()), _EPS)
            mean_time   = float(data["time"].mean())
            std_time    = max(float(data["time"].std()), _EPS)
            mean_depth  = float(data["depth"].mean())
            std_depth   = max(float(data["depth"].std()), _EPS)
            result = normalize_messages(
                data, mean_size, mean_prices, std_size, std_prices,
                mean_time, std_time, mean_depth, std_depth,
            )
        else:
            # args already carry safe stats from a prior call
            result = normalize_messages(data, *args)
        # Belt-and-suspenders: clean any residual NaN/Inf
        df_out = result[0]
        df_out.replace([np.inf, -np.inf], 0.0, inplace=True)
        df_out.fillna(0.0, inplace=True)
        return (df_out,) + result[1:]

    def _save_split(self, features: np.ndarray, labels: np.ndarray, path: Path):
        """Concatenate features and labels and save as .npy."""
        if features.shape[0] == 0:
            ncols = _NCOLS_FULL if self.all_features else _NCOLS_LOB
            np.save(path, np.zeros((0, ncols), dtype=np.float32))
            return
        arr = np.concatenate([features, labels], axis=1).astype(np.float32)
        np.save(path, arr)
        print(f"  Saved {path.name}: shape {arr.shape}")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_and_report(self, out_dir: Path):
        """Load saved .npy files and run shape, value, and LOB structure checks."""
        expected_ncols = _NCOLS_FULL if self.all_features else _NCOLS_LOB
        print(f"\n[BATTERY] Validating output in {out_dir}...")

        for split in ("train", "val", "test"):
            path = out_dir / f"{split}.npy"
            if not path.exists():
                print(f"  WARNING: {path} not found, skipping.")
                continue
            arr = np.load(path)
            if arr.shape[0] == 0:
                print(f"  {split}.npy: EMPTY (0 rows)")
                continue

            # Column count
            assert arr.shape[1] == expected_ncols, (
                f"{split}.npy has {arr.shape[1]} columns, expected {expected_ncols}"
            )

            # Features finite
            feat = arr[:, :-_N_LABELS]
            non_finite_pct = (~np.isfinite(feat)).mean() * 100
            if non_finite_pct > 0:
                print(f"  WARNING {split}.npy: {non_finite_pct:.2f}% non-finite feature values")

            # Labels
            label_arr = arr[:, -_N_LABELS:]
            for i, h in enumerate(cst.LOBSTER_HORIZONS):
                col = label_arr[:, i]
                valid = col[np.isfinite(col)]
                if valid.size > 0:
                    unique, counts = np.unique(valid.astype(int), return_counts=True)
                    dist = {int(u): f"{c/valid.size*100:.1f}%" for u, c in zip(unique, counts)}
                    print(f"  {split}.npy h{h} labels: {dist} ({valid.size:,} valid)")
                else:
                    print(f"  WARNING {split}.npy h{h}: no valid labels")

            # LOB structure (on first 1000 rows for speed)
            lob_start = cst.LEN_ORDER if self.all_features else 0
            self._validate_lob_structure(arr[:1000, lob_start : lob_start + _N_LOB], split)

        print(f"[BATTERY] Validation complete.\n")

    def _validate_lob_structure(self, lob: np.ndarray, name: str = ""):
        """Check ask/bid price ordering after z-score normalisation.

        Note: volumes are checked for ordering consistency only — z-score normalization
        makes raw volumes span negative values, so a non-negativity check is not applied.
        """
        if lob.shape[0] == 0:
            return
        sell_prices = lob[:, 0::4]  # columns 0, 4, 8, ..., 36
        buy_prices  = lob[:, 2::4]  # columns 2, 6, 10, ..., 38

        # Asks should be ascending across levels (z-score preserves relative order)
        for lvl in range(self.n_lob_levels - 1):
            bad = (sell_prices[:, lvl] > sell_prices[:, lvl + 1]).mean()
            if bad > 0.05:
                print(f"  WARNING {name}: sell prices not ascending at level {lvl+1}→{lvl+2}: {bad*100:.1f}% rows")

        # Bids should be descending across levels
        for lvl in range(self.n_lob_levels - 1):
            bad = (buy_prices[:, lvl] < buy_prices[:, lvl + 1]).mean()
            if bad > 0.05:
                print(f"  WARNING {name}: buy prices not descending at level {lvl+1}→{lvl+2}: {bad*100:.1f}% rows")

        # Best ask should be above best bid (after z-score: sell1 > buy1)
        crossed = (sell_prices[:, 0] <= buy_prices[:, 0]).mean()
        if crossed > 0.05:
            print(f"  WARNING {name}: {crossed*100:.1f}% snapshots have crossed book (sell1 <= buy1)")
