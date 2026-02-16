import heapq
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

import constants as cst
from utils.utils_data import labeling, normalize_messages, z_score_orderbook


def battery_load(path, all_features, len_smooth, h, seq_size):
    set_ = np.load(path)
    if h == 10:
        tmp = 4
    elif h == 20:
        tmp = 3
    elif h == 50:
        tmp = 2
    elif h == 100:
        tmp = 1
    else:
        raise ValueError(f"Unsupported horizon: {h}")

    labels = set_[seq_size - len_smooth :, -tmp]
    labels = labels[np.isfinite(labels)]
    labels = pd.Series(labels).astype("int64").to_numpy()
    if labels.size == 0:
        raise ValueError(f"No valid labels found in {path} for horizon {h}.")
    if labels.min() < 0 or labels.max() > 2:
        unique_labels = np.unique(labels)
        raise ValueError(
            f"Invalid battery labels in {path}: expected classes in [0, 2], got {unique_labels[:20]}"
        )

    if all_features:
        orderbook = set_[:, cst.LEN_ORDER : cst.LEN_ORDER + cst.N_LOB_LEVELS * cst.LEN_LEVEL]
        messages = set_[:, : cst.LEN_ORDER]
        input_ = np.concatenate([orderbook, messages], axis=1)
    else:
        input_ = set_[:, cst.LEN_ORDER : cst.LEN_ORDER + cst.N_LOB_LEVELS * cst.LEN_LEVEL]

    input_ = torch.from_numpy(input_).float()
    labels = torch.from_numpy(labels).long()

    return input_, labels


class BatteryDataBuilder:
    def __init__(
        self,
        data_dir,
        date_trading_days,
        split_rates,
        sampling_type,
        sampling_time,
        sampling_quantity,
    ):
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.split_rates = split_rates
        self.sampling_type = sampling_type
        self.sampling_time = sampling_time
        self.sampling_quantity = sampling_quantity
        self.n_lob_levels = cst.N_LOB_LEVELS

    def prepare_save_datasets(self):
        raw_base = Path(self.data_dir) / "battery_markets"
        if not raw_base.exists():
            raise FileNotFoundError(
                f"Expected raw battery data under {raw_base}. Please move 2021 into data/battery_markets/2021 first."
            )

        start_date = pd.Timestamp(self.date_trading_days[0])
        end_date = pd.Timestamp(self.date_trading_days[1])
        dates = pd.date_range(start_date, end_date, freq="D")

        daily_frames = []
        valid_dates = []
        for dt in dates:
            day_df = self._build_day_frame(dt, raw_base)
            if day_df is None or day_df.empty:
                continue
            daily_frames.append(day_df)
            valid_dates.append(dt)

        if not daily_frames:
            raise ValueError("No valid battery market days were parsed.")

        self.num_trading_days = len(daily_frames)
        split_days = self._split_days()

        train_df = pd.concat(daily_frames[: split_days[0]], axis=0, ignore_index=True)
        val_df = pd.concat(daily_frames[split_days[0] : split_days[1]], axis=0, ignore_index=True)
        test_df = pd.concat(daily_frames[split_days[1] :], axis=0, ignore_index=True)

        self.dataframes = [train_df, val_df, test_df]
        self._build_labels_from_orderbook()
        self._normalize_dataframes()

        self.train_set = np.concatenate(
            [self.dataframes[0].values, self.train_labels_horizons.values], axis=1
        )
        self.val_set = np.concatenate(
            [self.dataframes[1].values, self.val_labels_horizons.values], axis=1
        )
        self.test_set = np.concatenate(
            [self.dataframes[2].values, self.test_labels_horizons.values], axis=1
        )

        out_dir = Path(self.data_dir) / "battery_markets"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "train.npy", self.train_set)
        np.save(out_dir / "val.npy", self.val_set)
        np.save(out_dir / "test.npy", self.test_set)

        self._validate_output_shapes(out_dir)

    def _split_days(self):
        if self.num_trading_days < 3:
            raise ValueError(
                f"Need at least 3 parsed days for train/val/test split, found {self.num_trading_days}."
            )
        n_days = self.num_trading_days
        train = max(1, int(n_days * self.split_rates[0]))
        val = train + max(1, int(n_days * self.split_rates[1]))
        if val >= n_days:
            val = n_days - 1
        if train >= val:
            train = max(1, val - 1)
        test = n_days
        print(
            f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing"
        )
        return [train, val, test]

    def _read_id_table_2021(self, timestamp, datapath: Path):
        year = timestamp.strftime("%Y")
        month = timestamp.strftime("%m")
        datestr = "Continuous_Orders-DE-" + timestamp.strftime("%Y%m%d")

        month_path = datapath / year / month
        if not month_path.exists():
            return pd.DataFrame()

        zip_candidates = sorted([f for f in os.listdir(month_path) if datestr in f and f.endswith(".zip")])
        if not zip_candidates:
            return pd.DataFrame()

        zip_path = month_path / zip_candidates[0]
        with ZipFile(zip_path) as zip_file:
            csv_members = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                return pd.DataFrame()
            csv_member = csv_members[0]
            df = pd.read_csv(zip_file.open(csv_member), sep=",", decimal=".", skiprows=1)

        required = {
            "OrderId",
            "InitialId",
            "Side",
            "Product",
            "DeliveryStart",
            "UserDefinedBlock",
            "ActionCode",
            "TransactionTime",
            "ValidityTime",
            "Price",
            "Quantity",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {zip_path}: {sorted(missing)}")

        df = (
            df.drop_duplicates(
                subset=["OrderId", "InitialId", "ActionCode", "ValidityTime", "Price", "Quantity"]
            )
            .loc[lambda x: x["UserDefinedBlock"] == "N"]
            .loc[
                lambda x: (x["Product"] == "Intraday_Hour_Power")
                | (x["Product"] == "XBID_Hour_Power")
            ]
            .loc[lambda x: x["ActionCode"].isin(["A", "D", "C", "I"])]
            .rename(
                {
                    "OrderId": "order",
                    "InitialId": "initial",
                    "DeliveryStart": "start",
                    "Side": "side",
                    "Price": "price",
                    "ValidityTime": "validity",
                    "ActionCode": "action",
                    "TransactionTime": "transaction",
                    "Quantity": "quantity",
                },
                axis=1,
            )
        )

        df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
        df["validity"] = pd.to_datetime(df["validity"], errors="coerce", utc=True)
        df["transaction"] = pd.to_datetime(df["transaction"], errors="coerce", utc=True)

        df["start"] = df["start"].astype("datetime64[ns, UTC]")
        df["validity"] = df["validity"].astype("datetime64[ns, UTC]")
        df["transaction"] = df["transaction"].astype("datetime64[ns, UTC]")
        df = df.dropna(subset=["start", "transaction", "price", "quantity", "initial", "side"])

        iceberg_ids = df.loc[df["action"] == "I", "initial"].unique()
        df = df.loc[~df["initial"].isin(iceberg_ids)].copy()

        change_messages = df[df["action"] == "C"].drop_duplicates(subset=["order"], keep="first")
        not_added = change_messages[~(change_messages["order"].isin(df.loc[df["action"] == "A", "order"]))]
        change_messages = change_messages[~(change_messages["order"].isin(not_added["order"]))]

        while change_messages.shape[0] > 0:
            indexer_mess_a_with_change = (
                df[(df["order"].isin(change_messages["order"])) & (df["action"] == "A")]
                .sort_values("transaction")
                .groupby("order")
                .tail(1)
                .index
            )

            df["df_index_copy"] = df.index
            merged = pd.merge(change_messages, df.loc[indexer_mess_a_with_change], on="order")
            df.loc[merged["df_index_copy"].to_numpy(), "validity"] = pd.to_datetime(
                merged["transaction_x"], errors="coerce", utc=True
            ).astype("datetime64[ns, UTC]").to_numpy()
            df.loc[df.index.isin(change_messages.index), "action"] = "A"
            df.drop("df_index_copy", axis=1, inplace=True)

            change_messages = df[df["action"] == "C"].drop_duplicates(subset=["order"], keep="first")
            not_added = change_messages[
                ~(change_messages["order"].isin(df.loc[df["action"] == "A", "order"]))
            ]
            change_messages = change_messages[~(change_messages["order"].isin(not_added["order"]))]

        cancel_messages = df[df["action"] == "D"]
        not_added = cancel_messages[~(cancel_messages["order"].isin(df.loc[df["action"] == "A", "order"]))]
        cancel_messages = cancel_messages[~(cancel_messages["order"].isin(not_added["order"]))]

        if not cancel_messages.empty:
            indexer_mess_a_with_cancel = (
                df[(df["order"].isin(cancel_messages["order"])) & (df["action"] == "A")]
                .sort_values("transaction")
                .groupby("order")
                .tail(1)
                .index
            )
            df["df_index_copy"] = df.index
            merged = pd.merge(cancel_messages, df.loc[indexer_mess_a_with_cancel], on="order")
            df.loc[merged["df_index_copy"].to_numpy(), "validity"] = pd.to_datetime(
                merged["transaction_x"], errors="coerce", utc=True
            ).astype("datetime64[ns, UTC]").to_numpy()
            df.drop("df_index_copy", axis=1, inplace=True)

        df = df.loc[lambda x: ~(x["action"] == "D")]
        df = df.drop(["order", "action"], axis=1)

        df["side"] = df["side"].str.upper()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df = df.dropna(subset=["price", "quantity"])

        # Open validity orders are capped later during day-level processing.
        return df[["initial", "side", "start", "transaction", "validity", "price", "quantity"]]

    def _build_day_frame(self, dt: pd.Timestamp, raw_base: Path):
        df1 = self._read_id_table_2021(dt, raw_base)
        df2 = self._read_id_table_2021(dt + pd.Timedelta(days=1), raw_base)
        df = pd.concat([df1, df2], ignore_index=True)
        if df.empty:
            return None

        df = df.sort_values(by=["transaction", "initial"]).reset_index(drop=True)
        day = dt.date()
        df = df.loc[df["transaction"].dt.date == day].copy()
        if df.empty:
            return None

        day_end = pd.Timestamp(dt.date(), tz="UTC") + pd.Timedelta(days=1)
        df["validity"] = df["validity"].fillna(day_end)
        df.loc[df["validity"] <= df["transaction"], "validity"] = df["transaction"] + pd.Timedelta(seconds=1)

        df["price"] = df["price"].round(2)
        df["quantity"] = df["quantity"].round(1)

        contract_frames = []
        for _, contract_df in df.groupby("start", sort=True):
            frame = self._replay_contract(contract_df)
            if frame is not None and not frame.empty:
                contract_frames.append(frame)

        if not contract_frames:
            return None

        return pd.concat(contract_frames, axis=0, ignore_index=True)

    def _match_order(self, active, side, price, quantity):
        """Simulate order matching against resting orders on the opposite side.

        When an aggressive order crosses the book (buy price >= best ask, or
        sell price <= best bid), it should be matched against resting orders
        rather than sitting in the book and creating a crossed state.

        Returns the remaining (unfilled) quantity after matching.
        """
        remaining = quantity

        if side == "BUY":
            # Match against asks (sells) from lowest price up
            # Collect ask-side orders sorted by price (price-time priority)
            ask_orders = [
                (o["price"], oid)
                for oid, o in active.items()
                if o["side"] == "SELL" and o["quantity"] > 0
            ]
            ask_orders.sort()  # lowest price first

            for ask_price, oid in ask_orders:
                if remaining <= 0 or price < ask_price:
                    break
                fill = min(remaining, active[oid]["quantity"])
                active[oid]["quantity"] -= fill
                remaining -= fill
                if active[oid]["quantity"] <= 0:
                    del active[oid]

        elif side == "SELL":
            # Match against bids (buys) from highest price down
            bid_orders = [
                (o["price"], oid)
                for oid, o in active.items()
                if o["side"] == "BUY" and o["quantity"] > 0
            ]
            bid_orders.sort(reverse=True)  # highest price first

            for bid_price, oid in bid_orders:
                if remaining <= 0 or price > bid_price:
                    break
                fill = min(remaining, active[oid]["quantity"])
                active[oid]["quantity"] -= fill
                remaining -= fill
                if active[oid]["quantity"] <= 0:
                    del active[oid]

        return remaining

    def _replay_contract(self, df_contract: pd.DataFrame):
        df_contract = df_contract.sort_values(["transaction", "initial"]).reset_index(drop=True)
        active = {}
        versions = {}
        expiry_heap = []

        rows = []
        previous_ts = None
        book_ready = False

        for row in df_contract.itertuples(index=False):
            transaction = row.transaction
            validity = row.validity
            initial = int(row.initial)
            side = row.side
            price = float(row.price)
            quantity = float(row.quantity)

            # 1. Expire orders whose validity has been reached
            while expiry_heap and expiry_heap[0][0] <= transaction:
                _, expired_initial, expired_version = heapq.heappop(expiry_heap)
                if expired_initial in active and versions.get(expired_initial, -1) == expired_version:
                    del active[expired_initial]

            # 2. Simulate order matching: aggressive orders cross the book
            remaining = self._match_order(active, side, price, quantity)

            # 3. Add residual quantity as a resting order (if any)
            if remaining > 0:
                version = versions.get(initial, 0) + 1
                versions[initial] = version
                active[initial] = {
                    "side": side,
                    "price": price,
                    "quantity": remaining,
                    "validity": validity,
                    "version": version,
                }
                heapq.heappush(expiry_heap, (validity, initial, version))
            else:
                # Fully matched: bump version so stale expiry entries are ignored
                versions[initial] = versions.get(initial, 0) + 1

            # 4. Build LOB snapshot
            orderbook = self._top_levels(active)

            # Wait until the book is established before recording snapshots.
            # The C++ engine queries LOB state at specific times when the book
            # is mature; our event-by-event approach must skip the warmup
            # period at contract start where the book is still building up.
            # Once full depth is reached, we keep recording for this contract.
            # Use quantity > 0 (not price > 0) because energy prices can be
            # zero or negative.
            if not book_ready:
                n_ask = sum(1 for lvl in range(self.n_lob_levels) if orderbook[lvl * 4 + 1] > 0)
                n_bid = sum(1 for lvl in range(self.n_lob_levels) if orderbook[lvl * 4 + 3] > 0)
                if n_ask < self.n_lob_levels or n_bid < self.n_lob_levels:
                    continue
                book_ready = True

            # After the book is established, orders can still expire and drain
            # a side completely.  Skip snapshots where either side is empty
            # (qty == 0 at level 1) because the mid-price from a one-sided
            # book is not meaningful.
            if orderbook[1] == 0.0 or orderbook[3] == 0.0:
                continue

            time_delta = 0.0 if previous_ts is None else max((transaction - previous_ts).total_seconds(), 0.0)
            previous_ts = transaction

            direction = 1 if side == "BUY" else -1
            depth = self._compute_depth(price, direction, orderbook)

            # event_type: 1 = resting (limit), 3 = aggressive (trade)
            event_type = 1.0 if remaining > 0 else 3.0

            message = [
                time_delta,
                event_type,
                quantity,
                price,
                float(direction),
                float(depth),
            ]
            rows.append(message + orderbook)

        if not rows:
            return None

        message_cols = ["time", "event_type", "size", "price", "direction", "depth"]
        orderbook_cols = []
        for level in range(1, self.n_lob_levels + 1):
            orderbook_cols.extend([f"sell{level}", f"vsell{level}", f"buy{level}", f"vbuy{level}"])

        return pd.DataFrame(rows, columns=message_cols + orderbook_cols)

    def _compute_depth(self, price: float, direction: int, orderbook: list):
        tick = 0.1
        if direction == 1:
            best_bid = orderbook[2]
            return max(int((best_bid - price) / tick), 0)
        best_ask = orderbook[0]
        return max(int((price - best_ask) / tick), 0)

    def _top_levels(self, active):
        bid_levels = {}
        ask_levels = {}
        for order in active.values():
            if order["quantity"] <= 0:
                continue
            if order["side"] == "BUY":
                bid_levels[order["price"]] = bid_levels.get(order["price"], 0.0) + order["quantity"]
            elif order["side"] == "SELL":
                ask_levels[order["price"]] = ask_levels.get(order["price"], 0.0) + order["quantity"]

        asks = sorted(ask_levels.items(), key=lambda x: x[0])[: self.n_lob_levels]
        bids = sorted(bid_levels.items(), key=lambda x: x[0], reverse=True)[: self.n_lob_levels]

        row = []
        for level in range(self.n_lob_levels):
            if level < len(asks):
                ask_p, ask_q = asks[level]
            else:
                ask_p, ask_q = 0.0, 0.0
            if level < len(bids):
                bid_p, bid_q = bids[level]
            else:
                bid_p, bid_q = 0.0, 0.0
            row.extend([float(ask_p), float(ask_q), float(bid_p), float(bid_q)])
        return row

    def _build_labels_from_orderbook(self):
        train_orderbook = self.dataframes[0].iloc[:, cst.LEN_ORDER : cst.LEN_ORDER + 40].values
        val_orderbook = self.dataframes[1].iloc[:, cst.LEN_ORDER : cst.LEN_ORDER + 40].values
        test_orderbook = self.dataframes[2].iloc[:, cst.LEN_ORDER : cst.LEN_ORDER + 40].values

        for i, horizon in enumerate(cst.LOBSTER_HORIZONS):
            train_labels = labeling(train_orderbook, cst.LEN_SMOOTH, horizon)
            val_labels = labeling(val_orderbook, cst.LEN_SMOOTH, horizon)
            test_labels = labeling(test_orderbook, cst.LEN_SMOOTH, horizon)

            train_labels = np.concatenate(
                [train_labels, np.full(shape=(train_orderbook.shape[0] - train_labels.shape[0]), fill_value=np.inf)]
            )
            val_labels = np.concatenate(
                [val_labels, np.full(shape=(val_orderbook.shape[0] - val_labels.shape[0]), fill_value=np.inf)]
            )
            test_labels = np.concatenate(
                [test_labels, np.full(shape=(test_orderbook.shape[0] - test_labels.shape[0]), fill_value=np.inf)]
            )

            if i == 0:
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=[f"label_h{horizon}"])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=[f"label_h{horizon}"])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=[f"label_h{horizon}"])
            else:
                self.train_labels_horizons[f"label_h{horizon}"] = train_labels
                self.val_labels_horizons[f"label_h{horizon}"] = val_labels
                self.test_labels_horizons[f"label_h{horizon}"] = test_labels

    def _normalize_dataframes(self):
        orderbooks = []
        messages = []
        for df in self.dataframes:
            messages.append(df.iloc[:, : cst.LEN_ORDER].copy())
            orderbooks.append(df.iloc[:, cst.LEN_ORDER : cst.LEN_ORDER + 40].copy())

        for i in range(len(orderbooks)):
            if i == 0:
                orderbooks[i], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(orderbooks[i])
            else:
                orderbooks[i], _, _, _, _ = z_score_orderbook(
                    orderbooks[i], mean_size, mean_prices, std_size, std_prices
                )

        for i in range(len(messages)):
            if i == 0:
                (
                    messages[i],
                    mean_size,
                    mean_prices,
                    std_size,
                    std_prices,
                    mean_time,
                    std_time,
                    mean_depth,
                    std_depth,
                ) = normalize_messages(messages[i])
            else:
                messages[i], _, _, _, _, _, _, _, _ = normalize_messages(
                    messages[i],
                    mean_size,
                    mean_prices,
                    std_size,
                    std_prices,
                    mean_time,
                    std_time,
                    mean_depth,
                    std_depth,
                )

        self.dataframes = [
            pd.concat([messages[i], orderbooks[i]], axis=1) for i in range(len(self.dataframes))
        ]

    def _validate_output_shapes(self, out_dir: Path):
        train = np.load(out_dir / "train.npy")
        val = np.load(out_dir / "val.npy")
        test = np.load(out_dir / "test.npy")

        expected_cols = cst.LEN_ORDER + cst.N_LOB_LEVELS * cst.LEN_LEVEL + len(cst.LOBSTER_HORIZONS)
        for name, arr in [("train", train), ("val", val), ("test", test)]:
            if arr.shape[1] != expected_cols:
                raise ValueError(f"{name}.npy has {arr.shape[1]} cols, expected {expected_cols}")
            if not np.isfinite(arr[:, : cst.LEN_ORDER + cst.N_LOB_LEVELS * cst.LEN_LEVEL]).all():
                raise ValueError(f"{name}.npy contains non-finite feature values")

        print(
            f"Saved battery tensors at {out_dir} with shapes: train={train.shape}, val={val.shape}, test={test.shape}"
        )
