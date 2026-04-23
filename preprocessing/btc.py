import os

import kagglehub
import numpy as np
import pandas as pd
import torch

import constants as cst
from constants import SamplingType
from utils.utils_data import labeling, z_score_orderbook


def btc_load_tc_columns(path: str):
    """Return (raw_hs, raw_tc) aligned to the LOB input (length N, not mask-filtered).

    Helper for the single-horizon DPVN/DAVN path that needs raw half-spread and
    transaction cost alongside the z-scored LOB to compute tc-aware Q-targets.
    Raises if the .npy lacks the expected 50-column layout.
    """
    arr = np.load(path)
    if arr.shape[1] != 50:
        raise RuntimeError(
            f"{path}: {arr.shape[1]}-col layout cannot supply transaction_cost; "
            "run with --rebuild-data."
        )
    n_lob = cst.N_LOB_LEVELS * 4
    raw_hs = arr[:, n_lob + 8].astype(np.float32)
    raw_tc = arr[:, n_lob + 9].astype(np.float32)
    return raw_hs, raw_tc


def btc_load(path, len_smooth, h, seq_size):
    arr = np.load(path)
    n_lob = cst.N_LOB_LEVELS * 4  # 40
    # Labels always occupy the 4 columns immediately after the LOB block,
    # in canonical order [h10, h20, h50, h100]. Older .npy files were 44 cols
    # (labels at the tail); newer files append delta_mids + half_spread (+ tc),
    # which pushed the labels away from the tail and broke -tmp indexing.
    horizon_to_idx = {10: 0, 20: 1, 50: 2, 100: 3}
    label_col = n_lob + horizon_to_idx[h]
    labels = arr[seq_size - len_smooth :, label_col]
    labels = labels[np.isfinite(labels)]
    labels = torch.from_numpy(labels).long()
    input = torch.from_numpy(arr[:, :n_lob]).float()
    return input, labels


def btc_load_multi(path, len_smooth, seq_size):
    """Load BTC data returning all 4 horizon labels stacked.

    Expected .npy format (post-DFL + post-transaction-cost):
      - 50 cols: [40 LOB | 4 labels | 4 delta_mids | 1 half_spread | 1 tc_raw]

    Older layouts (49 cols = no tc; 44 cols = legacy labels-only) raise a
    rebuild prompt so training never silently runs with stale cost inputs.

    Returns:
        input:    FloatTensor (N, 40)
        labels:   LongTensor  (N_valid, 4)
        dfl_data: tuple(delta_mids (N_valid, 4), half_spreads (N_valid,),
                  transaction_costs (N_valid,)) or None
    """
    arr = np.load(path)
    ncols = arr.shape[1]
    n_lob = cst.N_LOB_LEVELS * 4  # 40

    if ncols == 50:
        # [40 LOB | 4 labels | 4 delta_mids | 1 half_spread | 1 tc_raw]
        lob = arr[:, :n_lob]
        label_arr = arr[:, n_lob:n_lob + 4]
        delta_mids_arr = arr[:, n_lob + 4:n_lob + 8]
        half_spreads_arr = arr[:, n_lob + 8].ravel()
        tc_arr = arr[:, n_lob + 9].ravel()
    elif ncols == 49 or ncols == 44:
        raise RuntimeError(
            f"{path}: {ncols}-col layout is stale (missing transaction_cost column). "
            "Run with --rebuild-data to regenerate the .npy files."
        )
    else:
        raise RuntimeError(f"{path}: unexpected {ncols}-col layout (expected 50).")

    label_start = seq_size - len_smooth
    all_labels = label_arr[label_start:]
    finite_mask = np.all(np.isfinite(all_labels), axis=1)
    all_labels = all_labels[finite_mask].astype(np.int64)
    labels = torch.from_numpy(all_labels).long()
    input = torch.from_numpy(lob).float()

    dm = delta_mids_arr[label_start:][finite_mask]
    hs = half_spreads_arr[label_start:][finite_mask]
    tc = tc_arr[label_start:][finite_mask]
    dfl_data = (
        torch.from_numpy(dm.astype(np.float32)),
        torch.from_numpy(hs.astype(np.float32)),
        torch.from_numpy(tc.astype(np.float32)),
    )

    return input, labels, dfl_data


class BTCDataBuilder:
    def __init__(
        self,
        data_dir,
        date_trading_days,
        split_rates,
        sampling_type,
        sampling_time,
        sampling_quantity,
        label_mode: str = "absolute_change",
        tc_abs: float = 0.0,
        tc_bps: float = 0.0,
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.split_rates = split_rates

        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity
        self.label_mode = label_mode
        self.tc_abs = float(tc_abs)
        self.tc_bps = float(tc_bps)

    def prepare_save_datasets(self):

        # Create directory if it doesn't exist
        # Continue with the existing code
        save_dir = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            "BTC",
            "BTC",
            self.date_trading_days[0],
            self.date_trading_days[1],
        )
        os.makedirs(save_dir, exist_ok=True)
        # check if the directory is empty
        if len(os.listdir(save_dir)) == 0:
            print("Downloading BTC dataset from Kaggle...")
            # Download the dataset from Kaggle
            path = kagglehub.dataset_download(
                "siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data"
            )

            # Get all CSV files in the downloaded directory
            file = os.listdir(path)[0]
            file_path = os.path.join(path, file)
            print(f"Processing {file}...")

            # Load the CSV file
            df = pd.read_csv(
                filepath_or_buffer=file_path, index_col="Unnamed: 0", parse_dates=True
            )
            df.columns = np.arange(42)

            # Select specific columns for the order book and
            # order in such a way that we have sell, vsell, buy, vbuy
            df = df.loc[
                :,
                [
                    1,
                    22,
                    23,
                    2,
                    3,
                    24,
                    25,
                    4,
                    5,
                    26,
                    27,
                    6,
                    7,
                    28,
                    29,
                    8,
                    9,
                    30,
                    31,
                    10,
                    11,
                    32,
                    33,
                    12,
                    13,
                    34,
                    35,
                    14,
                    15,
                    36,
                    37,
                    16,
                    17,
                    38,
                    39,
                    18,
                    19,
                    40,
                    41,
                    20,
                    21,
                ],
            ]
            # Rename the columns for better readability
            df.columns = [
                "timestamp",
                "sell1",
                "vsell1",
                "buy1",
                "vbuy1",
                "sell2",
                "vsell2",
                "buy2",
                "vbuy2",
                "sell3",
                "vsell3",
                "buy3",
                "vbuy3",
                "sell4",
                "vsell4",
                "buy4",
                "vbuy4",
                "sell5",
                "vsell5",
                "buy5",
                "vbuy5",
                "sell6",
                "vsell6",
                "buy6",
                "vbuy6",
                "sell7",
                "vsell7",
                "buy7",
                "vbuy7",
                "sell8",
                "vsell8",
                "buy8",
                "vbuy8",
                "sell9",
                "vsell9",
                "buy9",
                "vbuy9",
                "sell10",
                "vsell10",
                "buy10",
                "vbuy10",
            ]

            # transform string into timestamp
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], errors="coerce"
            )  # Let pandas infer format

            print("Splitting data by day and saving CSV files...")
            unique_dates = df["timestamp"].apply(lambda x: x.date()).unique()

            for date in unique_dates:
                # Convert date to string format YYYY-MM-DD
                date_str = date.strftime("%Y-%m-%d")
                # Filter data for the current date
                day_data = df[df["timestamp"].apply(lambda x: x.date()) == date]
                # day_data = day_data.drop(columns=["timestamp"])

                # Create the filename in the specified format
                filename = f"BTC_{date_str}_34200000_57600000_orderbook_10.csv"
                file_path = os.path.join(save_dir, filename)

                # Save the data to CSV without header and index
                day_data.to_csv(file_path, index=False, header=False)
                print(f"Saved {filename} with {len(day_data)} records")

        self.dataframes = []
        self._prepare_dataframes(save_dir)

        path_where_to_save = "{}/{}".format(
            self.data_dir,
            "BTC",
        )
        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values

        parts_train = [train_input, self.train_labels_horizons.values]
        parts_val = [val_input, self.val_labels_horizons.values]
        parts_test = [test_input, self.test_labels_horizons.values]

        # Append DFL columns (delta_mids + half_spread + transaction_cost) if available.
        # transaction_cost is in raw USDT per unit |Δpos|:
        #   tc_raw[t] = tc_abs + (tc_bps / 10_000) * |mid_raw[t]|
        if hasattr(self, "train_delta_mids"):
            parts_train.extend([
                self.train_delta_mids,
                self.train_half_spreads.reshape(-1, 1),
                self.train_transaction_costs.reshape(-1, 1),
            ])
            parts_val.extend([
                self.val_delta_mids,
                self.val_half_spreads.reshape(-1, 1),
                self.val_transaction_costs.reshape(-1, 1),
            ])
            parts_test.extend([
                self.test_delta_mids,
                self.test_half_spreads.reshape(-1, 1),
                self.test_transaction_costs.reshape(-1, 1),
            ])

        self.train_set = np.concatenate(parts_train, axis=1)
        self.val_set = np.concatenate(parts_val, axis=1)
        self.test_set = np.concatenate(parts_test, axis=1)
        self._save(path_where_to_save)

    def _prepare_dataframes(self, path):
        COLUMNS_NAMES = {
            "orderbook": [
                "timestamp",
                "sell1",
                "vsell1",
                "buy1",
                "vbuy1",
                "sell2",
                "vsell2",
                "buy2",
                "vbuy2",
                "sell3",
                "vsell3",
                "buy3",
                "vbuy3",
                "sell4",
                "vsell4",
                "buy4",
                "vbuy4",
                "sell5",
                "vsell5",
                "buy5",
                "vbuy5",
                "sell6",
                "vsell6",
                "buy6",
                "vbuy6",
                "sell7",
                "vsell7",
                "buy7",
                "vbuy7",
                "sell8",
                "vsell8",
                "buy8",
                "vbuy8",
                "sell9",
                "vsell9",
                "buy9",
                "vbuy9",
                "sell10",
                "vsell10",
                "buy10",
                "vbuy10",
            ]
        }
        self.num_trading_days = len(os.listdir(path))
        split_days = self._split_days()
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)

        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values

        # Extract DFL data (delta_mids, half_spreads) from raw prices before normalization
        # LOB layout: [sell_p(0), sell_v(1), buy_p(2), buy_v(3)] × 10 levels
        self.train_half_spreads = (train_input[:, 0] - train_input[:, 2]).astype(np.float32) / 2
        self.val_half_spreads = (val_input[:, 0] - val_input[:, 2]).astype(np.float32) / 2
        self.test_half_spreads = (test_input[:, 0] - test_input[:, 2]).astype(np.float32) / 2

        # Transaction cost per unit |Δpos| in raw USDT:
        #   tc_raw[t] = tc_abs + (tc_bps / 10_000) * |mid_raw[t]|
        # For BTC the bps term dominates (e.g. 4 bps × $20k ≈ $8/unit) vs half-spread ≈ $0.05.
        def _tc_raw(raw_lob: np.ndarray) -> np.ndarray:
            mid = (raw_lob[:, 0] + raw_lob[:, 2]) / 2.0
            return (self.tc_abs + (self.tc_bps / 10_000.0) * np.abs(mid)).astype(np.float32)

        self.train_transaction_costs = _tc_raw(train_input)
        self.val_transaction_costs = _tc_raw(val_input)
        self.test_transaction_costs = _tc_raw(test_input)

        train_deltas, val_deltas, test_deltas = [], [], []

        # create a dataframe for the labels
        for i in range(len(cst.LOBSTER_HORIZONS)):
            if i == 0:
                train_labels, train_pc = labeling(
                    train_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                val_labels, val_pc = labeling(
                    val_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                test_labels, test_pc = labeling(
                    test_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                # Pad delta_mids to full length (invalid positions = 0.0)
                train_deltas.append(np.concatenate([train_pc, np.zeros(train_input.shape[0] - len(train_pc))]).astype(np.float32))
                val_deltas.append(np.concatenate([val_pc, np.zeros(val_input.shape[0] - len(val_pc))]).astype(np.float32))
                test_deltas.append(np.concatenate([test_pc, np.zeros(test_input.shape[0] - len(test_pc))]).astype(np.float32))

                train_labels = np.concatenate(
                    [
                        train_labels,
                        np.full(
                            shape=(train_input.shape[0] - train_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                val_labels = np.concatenate(
                    [
                        val_labels,
                        np.full(
                            shape=(val_input.shape[0] - val_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                test_labels = np.concatenate(
                    [
                        test_labels,
                        np.full(
                            shape=(test_input.shape[0] - test_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                self.train_labels_horizons = pd.DataFrame(
                    train_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])]
                )
                self.val_labels_horizons = pd.DataFrame(
                    val_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])]
                )
                self.test_labels_horizons = pd.DataFrame(
                    test_labels, columns=["label_h{}".format(cst.LOBSTER_HORIZONS[i])]
                )
            else:
                train_labels, train_pc = labeling(
                    train_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                val_labels, val_pc = labeling(
                    val_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                test_labels, test_pc = labeling(
                    test_input,
                    cst.LEN_SMOOTH,
                    cst.LOBSTER_HORIZONS[i],
                    label_mode=self.label_mode,
                    return_price_change=True,
                )
                # Pad delta_mids to full length
                train_deltas.append(np.concatenate([train_pc, np.zeros(train_input.shape[0] - len(train_pc))]).astype(np.float32))
                val_deltas.append(np.concatenate([val_pc, np.zeros(val_input.shape[0] - len(val_pc))]).astype(np.float32))
                test_deltas.append(np.concatenate([test_pc, np.zeros(test_input.shape[0] - len(test_pc))]).astype(np.float32))

                train_labels = np.concatenate(
                    [
                        train_labels,
                        np.full(
                            shape=(train_input.shape[0] - train_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                val_labels = np.concatenate(
                    [
                        val_labels,
                        np.full(
                            shape=(val_input.shape[0] - val_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                test_labels = np.concatenate(
                    [
                        test_labels,
                        np.full(
                            shape=(test_input.shape[0] - test_labels.shape[0]),
                            fill_value=np.inf,
                        ),
                    ]
                )
                self.train_labels_horizons[
                    "label_h{}".format(cst.LOBSTER_HORIZONS[i])
                ] = train_labels
                self.val_labels_horizons[
                    "label_h{}".format(cst.LOBSTER_HORIZONS[i])
                ] = val_labels
                self.test_labels_horizons[
                    "label_h{}".format(cst.LOBSTER_HORIZONS[i])
                ] = test_labels

        # Stack delta_mids: (N, 4) for horizons [10, 20, 50, 100]
        self.train_delta_mids = np.stack(train_deltas, axis=1)
        self.val_delta_mids = np.stack(val_deltas, axis=1)
        self.test_delta_mids = np.stack(test_deltas, axis=1)

        # to conclude the preprocessing we normalize the dataframes
        self._normalize_dataframes()

    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        # Initialize empty dataframes for each split
        train_orderbooks = None
        val_orderbooks = None
        test_orderbooks = None
        for i, filename in enumerate(sorted(os.listdir(path))):
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                df_ob = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                # sample the dataframes according to the sampling type
                if self.sampling_type == SamplingType.TIME:
                    df_ob = self._sampling_time(df_ob, self.sampling_time)
                if i < split_days[0]:
                    train_orderbooks = (
                        df_ob
                        if train_orderbooks is None
                        else pd.concat([train_orderbooks, df_ob], axis=0)
                    )
                elif split_days[0] <= i < split_days[1]:
                    val_orderbooks = (
                        df_ob
                        if val_orderbooks is None
                        else pd.concat([val_orderbooks, df_ob], axis=0)
                    )
                else:
                    test_orderbooks = (
                        df_ob
                        if test_orderbooks is None
                        else pd.concat([test_orderbooks, df_ob], axis=0)
                    )
            else:
                raise ValueError(f"File {f} is not a file")
        # Save the splitted dataframes
        train_orderbooks = train_orderbooks.drop(columns=["timestamp"])
        val_orderbooks = val_orderbooks.drop(columns=["timestamp"])
        test_orderbooks = test_orderbooks.drop(columns=["timestamp"])
        self.dataframes = [train_orderbooks, val_orderbooks, test_orderbooks]

    def _normalize_dataframes(self):
        # apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if i == 0:
                self.dataframes[i], mean_size, mean_prices, std_size, std_prices = (
                    z_score_orderbook(self.dataframes[i])
                )
            else:
                self.dataframes[i], _, _, _, _ = z_score_orderbook(
                    self.dataframes[i], mean_size, mean_prices, std_size, std_prices
                )

    def _save(self, path_where_to_save):
        np.save(path_where_to_save + "/train.npy", self.train_set)
        np.save(path_where_to_save + "/val.npy", self.val_set)
        np.save(path_where_to_save + "/test.npy", self.test_set)

    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(
            f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing"
        )
        return [train, val, test]

    def _sampling_time(self, dataframe, time):
        # Convert the time column to datetime format if it's not already
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
        # Resample the messages dataframe to get data at every second
        dataframe = (
            dataframe.set_index("timestamp")
            .resample(time)
            .first()
            .dropna()
            .reset_index()
        )
        return dataframe
