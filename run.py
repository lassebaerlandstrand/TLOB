import json
import os
from datetime import datetime

import lightning as L
import numpy as np
import omegaconf
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

import constants as cst
import wandb
from config.config import Config
from constants import ProductMode, SamplingType
from models.engine import Engine
from models.original.engine import Engine as OriginalEngine
from preprocessing.battery import (
    battery_cache_subdir,
    battery_load,
    battery_load_multi,
    battery_load_tc_columns,
)
from preprocessing.btc import btc_load, btc_load_multi, btc_load_tc_columns
from preprocessing.dataset import DataModule, Dataset, MultiHorizonDataset
from preprocessing.fi_2010 import fi_2010_load, fi_2010_load_multi
from preprocessing.lobster import lobster_load, lobster_load_multi
from utils.utils_data import compute_lob_diffs

torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])


def _fmt_int_space(value: int) -> str:
    return f"{int(value):,}"


def _fmt_float_space(value: float, decimals: int = 1) -> str:
    return f"{float(value):,.{decimals}f}"


# Number of message features in Battery preprocessing (before LOB columns)
_N_MSG_FEATURES = 18


def _compute_dpvn_q_targets(
    inp: torch.Tensor,
    seq_size: int,
    horizon: int,
    gamma: float = 1.0,
    raw_hs: np.ndarray | None = None,
    raw_tc: np.ndarray | None = None,
):
    """DP-distilled Q targets for DPVN, aligned to Dataset indexing.

    Returns (N_valid, 3) float32 array where row i corresponds to the window
    [i : i + seq_size] whose last step is at raw time (i + seq_size - 1).

    When both ``raw_hs`` and ``raw_tc`` are provided (length N, raw price units),
    the per-unit cost used for Q-target generation becomes ``z_hs + z_tc`` where
    ``z_tc = raw_tc / std_price`` and ``std_price`` is inferred from the
    ``raw_hs`` / ``z_hs`` ratio. Omit either argument to fall back to spread-only.
    """
    from models.dp_targets import compute_q_targets, extract_z_mid_z_half_spread
    from utils.metrics import _infer_std_price

    z_mid, z_hs = extract_z_mid_z_half_spread(inp.numpy() if isinstance(inp, torch.Tensor) else inp)
    if len(z_mid) < seq_size + horizon:
        return None

    z_tc = None
    if raw_hs is not None and raw_tc is not None:
        raw_hs = np.asarray(raw_hs).astype(np.float64).ravel()
        raw_tc = np.asarray(raw_tc).astype(np.float64).ravel()
        if len(raw_hs) == len(z_mid) and len(raw_tc) == len(z_mid):
            std_price = _infer_std_price(raw_hs, z_hs)
            if std_price is not None:
                z_tc = raw_tc / std_price

    q_raw = compute_q_targets(
        z_mid, z_hs, horizon=horizon, gamma=gamma, z_transaction_cost=z_tc
    )
    return q_raw[seq_size - 1:]


def _dataset_labels(dataset):
    if hasattr(dataset, "y_multi") and dataset.y_multi is not None:
        return dataset.y_multi[: len(dataset)]
    if hasattr(dataset, "labels"):
        return dataset.labels[: len(dataset)]
    return dataset.y[: len(dataset)]


def _aggregate_split_counts(datasets, multi_horizon: bool):
    if multi_horizon:
        counts = torch.zeros((len(cst.LOBSTER_HORIZONS), 3), dtype=torch.long)
        for ds in datasets:
            labels = _dataset_labels(ds)
            for h_idx in range(min(labels.shape[1], len(cst.LOBSTER_HORIZONS))):
                counts[h_idx] += torch.bincount(labels[:, h_idx], minlength=3)[:3]
        return counts

    counts = torch.zeros((1, 3), dtype=torch.long)
    for ds in datasets:
        labels = _dataset_labels(ds)
        counts[0] += torch.bincount(labels, minlength=3)[:3]
    return counts


def _inverse_sqrt_class_weights(class_counts: torch.Tensor) -> torch.Tensor:
    num_classes = int(class_counts.numel())
    total_count = class_counts.sum()
    if total_count <= 0:
        return torch.ones(num_classes, dtype=torch.float32)

    frequencies = class_counts.float() / total_count.float()
    frequencies = torch.clamp(frequencies, min=1e-8)
    weights = 1.0 / torch.sqrt(frequencies)
    return weights * (float(num_classes) / weights.sum())


def _print_per_product_split_diagnostics(split_name: str, datasets, multi_horizon: bool, horizon: int):
    sizes = np.asarray([len(ds) for ds in datasets], dtype=np.float64)
    print(
        f"{split_name} product sizes: min={_fmt_int_space(int(sizes.min()))} "
        f"max={_fmt_int_space(int(sizes.max()))} "
        f"mean={_fmt_float_space(sizes.mean(), 1)} std={_fmt_float_space(sizes.std(), 1)}"
    )

    counts = _aggregate_split_counts(datasets, multi_horizon)
    horizons = cst.LOBSTER_HORIZONS if multi_horizon else [horizon]

    for h_idx, h in enumerate(horizons):
        class_counts = counts[h_idx].cpu().numpy()
        total = max(int(class_counts.sum()), 1)
        up = class_counts[0] / total
        stat = class_counts[1] / total
        down = class_counts[2] / total

        non_zero = class_counts[class_counts > 0]
        imbalance_ratio = float(class_counts.max() / non_zero.min()) if non_zero.size else float("inf")
        imbalance_text = f"{imbalance_ratio:.2f}x" if np.isfinite(imbalance_ratio) else "inf"

        print(
            f"  h{h}: up={up:.3f} stat={stat:.3f} down={down:.3f} "
            f"(N={_fmt_int_space(total)}, imbalance max/min={imbalance_text})"
        )


def run(config: Config, accelerator):
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    dataset = config.dataset.type.value
    horizon = config.experiment.horizon
    multi_horizon = config.experiment.multi_horizon
    mh_suffix = "_multi_horizon" if multi_horizon else ""
    if dataset == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        config.experiment.dir_ckpt = f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}{mh_suffix}"
    else:
        config.experiment.dir_ckpt = (
            f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{config.experiment.seed}{mh_suffix}"
        )

    trainer = L.Trainer(
        accelerator=accelerator,
        precision=config.experiment.precision,
        max_epochs=config.experiment.max_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=2,
                verbose=True,
                min_delta=0.0005,
            ),
            TQDMProgressBar(refresh_rate=100),
        ],
        num_sanity_val_steps=0,
        detect_anomaly=False,
        profiler=None,
        check_val_every_n_epoch=1,
    )
    train(config, trainer)


def train(config: Config, trainer: L.Trainer, run=None):
    print_setup(config)
    dataset_type = config.dataset.type.value
    seq_size = config.model.hyperparameters_fixed["seq_size"]
    horizon = config.experiment.horizon
    multi_horizon = config.experiment.multi_horizon
    use_diff_features = config.experiment.use_diff_features
    n_lob_features = cst.N_LOB_LEVELS * cst.LEN_LEVEL

    def maybe_add_diff_features(input_tensor: torch.Tensor) -> torch.Tensor:
        if not use_diff_features:
            return input_tensor
        return compute_lob_diffs(input_tensor, n_lob=n_lob_features)

    model_type = config.model.type
    checkpoint_ref = config.experiment.checkpoint_reference
    checkpoint_path = os.path.join(cst.DIR_SAVED_MODEL, model_type.value, checkpoint_ref)
    dataset_type = config.dataset.type.value

    if multi_horizon and model_type in {
        cst.ModelType.MLPLOB_ORIGINAL,
        cst.ModelType.TLOB_ORIGINAL,
    }:
        raise ValueError("Multi-horizon mode is not supported for *_ORIGINAL baseline models.")
    if (
        model_type in {cst.ModelType.MLPLOB_ORIGINAL, cst.ModelType.TLOB_ORIGINAL}
        and config.experiment.loss_type != "cross_entropy"
    ):
        raise ValueError("Original baseline models support only loss_type='cross_entropy'.")

    if dataset_type == "FI_2010":
        path = cst.DATA_DIR + "/FI_2010"
        if multi_horizon:
            (
                train_input,
                train_labels,
                val_input,
                val_labels,
                test_input,
                test_labels,
            ) = fi_2010_load_multi(path, seq_size, config.model.hyperparameters_fixed["all_features"])
            train_input = maybe_add_diff_features(train_input)
            val_input = maybe_add_diff_features(val_input)
            test_input = maybe_add_diff_features(test_input)
            train_set = MultiHorizonDataset(train_input, train_labels, seq_size)
            val_set = MultiHorizonDataset(val_input, val_labels, seq_size)
            test_set = MultiHorizonDataset(test_input, test_labels, seq_size)
        else:
            (
                train_input,
                train_labels,
                val_input,
                val_labels,
                test_input,
                test_labels,
            ) = fi_2010_load(
                path,
                seq_size,
                horizon,
                config.model.hyperparameters_fixed["all_features"],
            )
            train_input = maybe_add_diff_features(train_input)
            val_input = maybe_add_diff_features(val_input)
            test_input = maybe_add_diff_features(test_input)
            train_set = Dataset(train_input, train_labels, seq_size)
            val_set = Dataset(val_input, val_labels, seq_size)
            test_set = Dataset(test_input, test_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size * 4,
            num_workers=4,
        )
        test_loaders = [data_module.test_dataloader()]

    elif dataset_type == "BTC":
        if multi_horizon:
            train_input, train_labels, train_dfl = btc_load_multi(
                cst.DATA_DIR + "/BTC/train.npy", cst.LEN_SMOOTH, seq_size
            )
            val_input, val_labels, val_dfl = btc_load_multi(cst.DATA_DIR + "/BTC/val.npy", cst.LEN_SMOOTH, seq_size)
            test_input, test_labels, test_dfl = btc_load_multi(cst.DATA_DIR + "/BTC/test.npy", cst.LEN_SMOOTH, seq_size)
            train_input = maybe_add_diff_features(train_input)
            val_input = maybe_add_diff_features(val_input)
            test_input = maybe_add_diff_features(test_input)
            train_set = MultiHorizonDataset(train_input, train_labels, seq_size, dfl_data=train_dfl)
            val_set = MultiHorizonDataset(val_input, val_labels, seq_size, dfl_data=val_dfl)
            test_set = MultiHorizonDataset(test_input, test_labels, seq_size, dfl_data=test_dfl)
        else:
            train_input, train_labels = btc_load(cst.DATA_DIR + "/BTC/train.npy", cst.LEN_SMOOTH, horizon, seq_size)
            val_input, val_labels = btc_load(cst.DATA_DIR + "/BTC/val.npy", cst.LEN_SMOOTH, horizon, seq_size)
            test_input, test_labels = btc_load(cst.DATA_DIR + "/BTC/test.npy", cst.LEN_SMOOTH, horizon, seq_size)
            train_q = val_q = test_q = None
            train_dfl = val_dfl = test_dfl = None
            if model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN):
                train_hs, train_tc = btc_load_tc_columns(cst.DATA_DIR + "/BTC/train.npy")
                val_hs, val_tc = btc_load_tc_columns(cst.DATA_DIR + "/BTC/val.npy")
                test_hs, test_tc = btc_load_tc_columns(cst.DATA_DIR + "/BTC/test.npy")
                train_q = _compute_dpvn_q_targets(train_input, seq_size, horizon, raw_hs=train_hs, raw_tc=train_tc)
                val_q = _compute_dpvn_q_targets(val_input, seq_size, horizon, raw_hs=val_hs, raw_tc=val_tc)
                test_q = _compute_dpvn_q_targets(test_input, seq_size, horizon, raw_hs=test_hs, raw_tc=test_tc)
                # Align per-sample cost arrays to Dataset indexing (last-step of each window).
                train_dfl = (train_hs[seq_size - 1:], train_tc[seq_size - 1:])
                val_dfl = (val_hs[seq_size - 1:], val_tc[seq_size - 1:])
                test_dfl = (test_hs[seq_size - 1:], test_tc[seq_size - 1:])
            train_input = maybe_add_diff_features(train_input)
            val_input = maybe_add_diff_features(val_input)
            test_input = maybe_add_diff_features(test_input)
            train_set = Dataset(train_input, train_labels, seq_size, q_targets=train_q, dfl_data=train_dfl)
            val_set = Dataset(val_input, val_labels, seq_size, q_targets=val_q, dfl_data=val_dfl)
            test_set = Dataset(test_input, test_labels, seq_size, q_targets=test_q, dfl_data=test_dfl)
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size * 4,
            num_workers=4,
        )

        test_loaders = [data_module.test_dataloader()]

    elif dataset_type == "BATTERY":
        stock = config.dataset.training_stocks[0]
        # Model-level flag controls input features passed to the model.
        all_features = config.model.hyperparameters_fixed["all_features"]
        # Cache subdir tracks how the data was *preprocessed*, which is a
        # dataset-level concern. Decoupling from the model flag lets e.g. DPVN
        # (model all_features=False) read a `_msg` cache and downselect to LOB
        # at load time, avoiding a cache rebuild per model variant.
        cache_all_features = getattr(config.dataset, "all_features", all_features)
        base_dir = cst.DATA_DIR + f"/{stock}"
        _pm = getattr(config.dataset, "product_mode", "concat")
        product_mode = ProductMode(_pm) if isinstance(_pm, str) else _pm
        cache_sub = battery_cache_subdir(
            config.dataset.sampling_time,
            config.dataset.dates,
            config.dataset.sampling_type.value,
            cache_all_features,
            getattr(config.dataset, "max_hours_before_delivery", 0.0),
        )

        if product_mode == ProductMode.PER_PRODUCT:
            pp_dir = os.path.join(base_dir, "per_product", cache_sub)
            if not os.path.isdir(pp_dir):
                raise FileNotFoundError(
                    f"[BATTERY] No preprocessed data for sampling_time={config.dataset.sampling_time}, "
                    f"dates={config.dataset.dates} at {pp_dir}. "
                    f"Run with is_data_preprocessed=False to build it."
                )
            with open(os.path.join(pp_dir, "products.json")) as f:
                products = json.load(f)

            train_datasets = []
            val_datasets = []
            test_datasets = []

            for product in products:
                product_dir = os.path.join(pp_dir, "products", product)

                for split in ("train", "val", "test"):
                    path = os.path.join(product_dir, f"{split}.npy")
                    if not os.path.exists(path):
                        continue
                    try:
                        if multi_horizon:
                            inp, lab, prod_dfl = battery_load_multi(path, all_features, cst.LEN_SMOOTH, seq_size)
                        else:
                            inp, lab = battery_load(path, all_features, cst.LEN_SMOOTH, horizon, seq_size)
                            prod_dfl = None
                    except ValueError:
                        continue  # product too small for seq_size / horizon
                    inp = maybe_add_diff_features(inp)
                    if inp.shape[0] < seq_size:
                        continue

                    if multi_horizon:
                        ds = MultiHorizonDataset(inp, lab, seq_size, dfl_data=prod_dfl)
                    else:
                        q_targets = None
                        ds_dfl = None
                        if model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN):
                            prod_hs, prod_tc = battery_load_tc_columns(path)
                            q_targets = _compute_dpvn_q_targets(
                                inp, seq_size, horizon, raw_hs=prod_hs, raw_tc=prod_tc
                            )
                            if q_targets is None:
                                continue  # product too small for DP targets
                            ds_dfl = (prod_hs[seq_size - 1:], prod_tc[seq_size - 1:])
                        ds = Dataset(inp, lab, seq_size, q_targets=q_targets, dfl_data=ds_dfl)

                    if split == "train":
                        train_datasets.append(ds)
                    elif split == "val":
                        val_datasets.append(ds)
                    else:
                        test_datasets.append(ds)

            if not train_datasets:
                raise RuntimeError("[BATTERY] No training data found in per_product mode")
            if not val_datasets:
                raise RuntimeError("[BATTERY] No validation data found in per_product mode")
            if not test_datasets:
                raise RuntimeError("[BATTERY] No test data found in per_product mode")

            test_concat = ConcatDataset(test_datasets)
            _test_mult = 4
            test_loaders = [
                DataLoader(
                    dataset=test_concat,
                    batch_size=config.dataset.batch_size * _test_mult,
                    shuffle=False,
                    pin_memory=True,
                    drop_last=False,
                    num_workers=4,
                    persistent_workers=True,
                    multiprocessing_context="spawn",
                )
            ]
            # Save product boundary indices for directional trading evaluation.
            # ConcatDataset.cumulative_sizes gives the exclusive end index of
            # each sub-dataset, which is where one product ends and the next
            # begins — the trading simulation must close positions at these
            # boundaries.
            _boundaries = np.array(test_concat.cumulative_sizes, dtype=np.int64)
            _boundaries_dir = os.path.join(cst.DIR_SAVED_MODEL, config.model.type.value, config.experiment.dir_ckpt)
            os.makedirs(_boundaries_dir, exist_ok=True)
            np.save(os.path.join(_boundaries_dir, "product_boundaries"), _boundaries)

            train_set = ConcatDataset(train_datasets)
            val_concat = ConcatDataset(val_datasets)
            _val_boundaries = np.array(val_concat.cumulative_sizes, dtype=np.int64)
            np.save(os.path.join(_boundaries_dir, "val_product_boundaries"), _val_boundaries)
            val_set = val_concat
            # Expose train_input for num_features used by model instantiation
            first_train = train_datasets[0]
            train_input = first_train.x if hasattr(first_train, "x") else first_train.data
            data_module = DataModule(
                train_set=train_set,
                val_set=val_set,
                batch_size=config.dataset.batch_size,
                test_batch_size=config.dataset.batch_size * _test_mult,
                num_workers=4,
            )

        else:
            # concat mode (default)
            concat_dir = os.path.join(base_dir, "concat", cache_sub)
            if not os.path.isdir(concat_dir):
                raise FileNotFoundError(
                    f"[BATTERY] No preprocessed data for sampling_time={config.dataset.sampling_time}, "
                    f"dates={config.dataset.dates} at {concat_dir}. "
                    f"Run with is_data_preprocessed=False to build it."
                )
            if multi_horizon:
                train_input, train_labels, train_dfl = battery_load_multi(
                    concat_dir + "/train.npy", all_features, cst.LEN_SMOOTH, seq_size
                )
                val_input, val_labels, val_dfl = battery_load_multi(
                    concat_dir + "/val.npy", all_features, cst.LEN_SMOOTH, seq_size
                )
                test_input, test_labels, test_dfl = battery_load_multi(
                    concat_dir + "/test.npy", all_features, cst.LEN_SMOOTH, seq_size
                )
                train_input = maybe_add_diff_features(train_input)
                val_input = maybe_add_diff_features(val_input)
                test_input = maybe_add_diff_features(test_input)
                train_set = MultiHorizonDataset(train_input, train_labels, seq_size, dfl_data=train_dfl)
                val_set = MultiHorizonDataset(val_input, val_labels, seq_size, dfl_data=val_dfl)
                test_set = MultiHorizonDataset(test_input, test_labels, seq_size, dfl_data=test_dfl)
            else:
                train_input, train_labels = battery_load(
                    concat_dir + "/train.npy",
                    all_features,
                    cst.LEN_SMOOTH,
                    horizon,
                    seq_size,
                )
                val_input, val_labels = battery_load(
                    concat_dir + "/val.npy",
                    all_features,
                    cst.LEN_SMOOTH,
                    horizon,
                    seq_size,
                )
                test_input, test_labels = battery_load(
                    concat_dir + "/test.npy",
                    all_features,
                    cst.LEN_SMOOTH,
                    horizon,
                    seq_size,
                )
                train_q = val_q = test_q = None
                train_dfl = val_dfl = test_dfl = None
                if model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN):
                    train_hs, train_tc = battery_load_tc_columns(concat_dir + "/train.npy")
                    val_hs, val_tc = battery_load_tc_columns(concat_dir + "/val.npy")
                    test_hs, test_tc = battery_load_tc_columns(concat_dir + "/test.npy")
                    train_q = _compute_dpvn_q_targets(train_input, seq_size, horizon, raw_hs=train_hs, raw_tc=train_tc)
                    val_q = _compute_dpvn_q_targets(val_input, seq_size, horizon, raw_hs=val_hs, raw_tc=val_tc)
                    test_q = _compute_dpvn_q_targets(test_input, seq_size, horizon, raw_hs=test_hs, raw_tc=test_tc)
                    train_dfl = (train_hs[seq_size - 1:], train_tc[seq_size - 1:])
                    val_dfl = (val_hs[seq_size - 1:], val_tc[seq_size - 1:])
                    test_dfl = (test_hs[seq_size - 1:], test_tc[seq_size - 1:])
                train_input = maybe_add_diff_features(train_input)
                val_input = maybe_add_diff_features(val_input)
                test_input = maybe_add_diff_features(test_input)
                train_set = Dataset(train_input, train_labels, seq_size, q_targets=train_q, dfl_data=train_dfl)
                val_set = Dataset(val_input, val_labels, seq_size, q_targets=val_q, dfl_data=val_dfl)
                test_set = Dataset(test_input, test_labels, seq_size, q_targets=test_q, dfl_data=test_dfl)
            if config.experiment.is_debug:
                train_set.length = 1000
                val_set.length = 1000
                test_set.length = 10000
            data_module = DataModule(
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                batch_size=config.dataset.batch_size,
                test_batch_size=config.dataset.batch_size * 4,
                num_workers=4,
            )
            test_loaders = [data_module.test_dataloader()]

    elif dataset_type == "LOBSTER":
        training_stocks = config.dataset.training_stocks
        testing_stocks = config.dataset.testing_stocks
        for i in range(len(training_stocks)):
            if i == 0:
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        if multi_horizon:
                            train_input, train_labels, train_dfl = lobster_load_multi(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                seq_size,
                            )
                        else:
                            train_input, train_labels = lobster_load(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                horizon,
                                seq_size,
                            )
                        train_input = maybe_add_diff_features(train_input)
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        if multi_horizon:
                            val_input, val_labels, val_dfl = lobster_load_multi(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                seq_size,
                            )
                        else:
                            val_input, val_labels = lobster_load(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                horizon,
                                seq_size,
                            )
                        val_input = maybe_add_diff_features(val_input)
            else:
                for j in range(2):
                    if j == 0:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/train.npy"
                        pad = (
                            torch.zeros(
                                seq_size + horizon - 1,
                                len(cst.LOBSTER_HORIZONS) if multi_horizon else 1,
                                dtype=torch.long,
                            )
                            if multi_horizon
                            else torch.zeros(seq_size + horizon - 1, dtype=torch.long)
                        )
                        train_labels = torch.cat((train_labels, pad), 0)
                        if multi_horizon:
                            train_input_tmp, train_labels_tmp, _ = lobster_load_multi(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                seq_size,
                            )
                        else:
                            train_input_tmp, train_labels_tmp = lobster_load(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                horizon,
                                seq_size,
                            )
                        train_input_tmp = maybe_add_diff_features(train_input_tmp)
                        train_input = torch.cat((train_input, train_input_tmp), 0)
                        train_labels = torch.cat((train_labels, train_labels_tmp), 0)
                    if j == 1:
                        path = cst.DATA_DIR + "/" + training_stocks[i] + "/val.npy"
                        pad = (
                            torch.zeros(
                                seq_size + horizon - 1,
                                len(cst.LOBSTER_HORIZONS) if multi_horizon else 1,
                                dtype=torch.long,
                            )
                            if multi_horizon
                            else torch.zeros(seq_size + horizon - 1, dtype=torch.long)
                        )
                        val_labels = torch.cat((val_labels, pad), 0)
                        if multi_horizon:
                            val_input_tmp, val_labels_tmp, _ = lobster_load_multi(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                seq_size,
                            )
                        else:
                            val_input_tmp, val_labels_tmp = lobster_load(
                                path,
                                config.model.hyperparameters_fixed["all_features"],
                                cst.LEN_SMOOTH,
                                horizon,
                                seq_size,
                            )
                        val_input_tmp = maybe_add_diff_features(val_input_tmp)
                        val_input = torch.cat((val_input, val_input_tmp), 0)
                        val_labels = torch.cat((val_labels, val_labels_tmp), 0)
        test_loaders = []
        for i in range(len(testing_stocks)):
            path = cst.DATA_DIR + "/" + testing_stocks[i] + "/test.npy"
            if multi_horizon:
                test_input, test_labels, test_dfl = lobster_load_multi(
                    path,
                    config.model.hyperparameters_fixed["all_features"],
                    cst.LEN_SMOOTH,
                    seq_size,
                )
                test_input = maybe_add_diff_features(test_input)
                test_set = MultiHorizonDataset(test_input, test_labels, seq_size, dfl_data=test_dfl)
            else:
                test_input, test_labels = lobster_load(
                    path,
                    config.model.hyperparameters_fixed["all_features"],
                    cst.LEN_SMOOTH,
                    horizon,
                    seq_size,
                )
                test_input = maybe_add_diff_features(test_input)
                test_set = Dataset(test_input, test_labels, seq_size)
            test_dataloader = DataLoader(
                dataset=test_set,
                batch_size=config.dataset.batch_size * 4,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                num_workers=4,
                persistent_workers=True,
                multiprocessing_context="spawn",
            )
            test_loaders.append(test_dataloader)

        train_set = Dataset(train_input, train_labels, seq_size)
        val_set = Dataset(val_input, val_labels, seq_size)
        if config.experiment.is_debug:
            train_set.length = 1000
            val_set.length = 1000
            test_set.length = 10000
        data_module = DataModule(
            train_set=train_set,
            val_set=val_set,
            batch_size=config.dataset.batch_size,
            test_batch_size=config.dataset.batch_size * 4,
            num_workers=4,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if isinstance(train_set, ConcatDataset):
        test_total_samples = sum(len(loader.dataset) for loader in test_loaders)
        test_total_products = 0
        for loader in test_loaders:
            if isinstance(loader.dataset, ConcatDataset):
                test_total_products += len(loader.dataset.datasets)
            else:
                test_total_products += 1
        product_label = "product" if test_total_products == 1 else "products"
        loader_label = "DataLoader" if len(test_loaders) == 1 else "DataLoaders"

        print(f"\nTrain set: {_fmt_int_space(len(train_set))} samples ({len(train_set.datasets)} products)")
        print(f"Val set:   {_fmt_int_space(len(val_set))} samples ({len(val_set.datasets)} products)")
        print(
            f"Test:      {_fmt_int_space(test_total_samples)} samples "
            f"({test_total_products} {product_label}, {len(test_loaders)} {loader_label})\n"
        )

        if dataset_type == "BATTERY":
            print("Per-product aggregated diagnostics")
            _print_per_product_split_diagnostics("train", train_set.datasets, multi_horizon, horizon)
            _print_per_product_split_diagnostics("val", val_set.datasets, multi_horizon, horizon)

            test_concat = test_loaders[0].dataset
            if isinstance(test_concat, ConcatDataset):
                _print_per_product_split_diagnostics("test", test_concat.datasets, multi_horizon, horizon)
            print()
    else:
        counts_train = torch.unique(train_labels, return_counts=True)
        counts_val = torch.unique(val_labels, return_counts=True)
        counts_test = torch.unique(test_labels, return_counts=True)
        print()
        print("Train set shape: ", train_input.shape)
        print("Val set shape: ", val_input.shape)
        print("Test set shape: ", test_input.shape)
        print(
            f"Classes distribution in train set: up {(counts_train[1][0].item() / train_labels.shape[0]):.2f} stat {(counts_train[1][1].item() / train_labels.shape[0]):.2f} down {(counts_train[1][2].item() / train_labels.shape[0]):.2f} ",
        )
        print(
            f"Classes distribution in val set: up {(counts_val[1][0].item() / val_labels.shape[0]):.2f} stat {(counts_val[1][1].item() / val_labels.shape[0]):.2f} down {(counts_val[1][2].item() / val_labels.shape[0]):.2f} ",
        )
        print(
            f"Classes distribution in test set: up {(counts_test[1][0].item() / test_labels.shape[0]):.2f} stat {(counts_test[1][1].item() / test_labels.shape[0]):.2f} down {(counts_test[1][2].item() / test_labels.shape[0]):.2f} ",
        )
        print()

    counts_source = train_set.datasets if isinstance(train_set, ConcatDataset) else [train_set]
    train_counts = _aggregate_split_counts(counts_source, multi_horizon).float()
    class_weights = None
    if config.experiment.use_class_weights:
        if multi_horizon:
            per_horizon_weights = []
            for horizon_index, horizon_value in enumerate(cst.LOBSTER_HORIZONS):
                horizon_counts = train_counts[horizon_index]
                horizon_weights = _inverse_sqrt_class_weights(horizon_counts)
                per_horizon_weights.append(horizon_weights)
                print(
                    f"Using class weights (h{horizon_value}): "
                    f"counts={horizon_counts.to(torch.long).tolist()} "
                    f"weights={[round(weight, 4) for weight in horizon_weights.tolist()]}"
                )

            class_weights = torch.stack(per_horizon_weights, dim=0)
        else:
            horizon_counts = train_counts[0]
            if horizon_counts.sum() > 0:
                class_weights = _inverse_sqrt_class_weights(horizon_counts)
                print(
                    "Using class weights (h10): "
                    f"counts={horizon_counts.to(torch.long).tolist()} "
                    f"weights={[round(weight, 4) for weight in class_weights.tolist()]}"
                )
            else:
                print("Skipping class weights: unable to compute non-empty h10 class counts.")
    else:
        print("Class weights disabled via experiment.use_class_weights=False")

    # Log dataset stats to wandb
    if run is not None:
        n_train = len(train_set)
        run.log({"n_train_rows": n_train}, commit=False)

        # Gather all test labels
        if isinstance(train_set, ConcatDataset):
            test_concat = test_loaders[0].dataset
            all_test_labels = []
            for ds in test_concat.datasets:
                y_m = getattr(ds, "y_multi", None)
                y = y_m if y_m is not None else ds.y
                all_test_labels.append(y[: len(ds)])
            all_test_labels = torch.cat(all_test_labels, dim=0)
        else:
            all_test_labels = test_labels

        # Compute average label distribution across horizons
        if all_test_labels.dim() == 2:
            up_pcts, stat_pcts, down_pcts = [], [], []
            for h_idx in range(all_test_labels.shape[1]):
                col = all_test_labels[:, h_idx].float()
                n = col.shape[0]
                up_pcts.append((col == 0).sum().item() / n)
                stat_pcts.append((col == 1).sum().item() / n)
                down_pcts.append((col == 2).sum().item() / n)
            test_up = sum(up_pcts) / len(up_pcts)
            test_stat = sum(stat_pcts) / len(stat_pcts)
            test_down = sum(down_pcts) / len(down_pcts)
        else:
            n = all_test_labels.shape[0]
            test_up = (all_test_labels == 0).sum().item() / n
            test_stat = (all_test_labels == 1).sum().item() / n
            test_down = (all_test_labels == 2).sum().item() / n

        run.log({"test_label_up_pct": test_up}, commit=False)
        run.log({"test_label_stat_pct": test_stat}, commit=False)
        run.log({"test_label_down_pct": test_down}, commit=False)
        print(
            f"Logged to wandb: n_train_rows={n_train}, test_labels: up={test_up:.3f} stat={test_stat:.3f} down={test_down:.3f}"
        )

    experiment_type = config.experiment.type
    if "FINETUNING" in experiment_type or "EVALUATION" in experiment_type:
        if checkpoint_ref != "":
            checkpoint = torch.load(checkpoint_path, map_location=cst.DEVICE, weights_only=False)

        print("Loading model from checkpoint: ", config.experiment.checkpoint_reference)
        lr = checkpoint["hyper_parameters"]["lr"]
        dir_ckpt = checkpoint["hyper_parameters"]["dir_ckpt"]
        hidden_dim = checkpoint["hyper_parameters"]["hidden_dim"]
        num_layers = checkpoint["hyper_parameters"]["num_layers"]
        optimizer = checkpoint["hyper_parameters"]["optimizer"]
        model_type = checkpoint["hyper_parameters"]["model_type"]
        max_epochs = checkpoint["hyper_parameters"]["max_epochs"]
        horizon = checkpoint["hyper_parameters"]["horizon"]
        seq_size = checkpoint["hyper_parameters"]["seq_size"]
        if model_type == "MLPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
                map_location=cst.DEVICE,
                weights_only=False,
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
            )
        elif model_type == "MLPLOB_ORIGINAL":
            model = OriginalEngine.load_from_checkpoint(
                checkpoint_path,
                map_location=cst.DEVICE,
                weights_only=False,
            )
        elif model_type == "TLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=checkpoint["hyper_parameters"]["num_heads"],
                is_sin_emb=checkpoint["hyper_parameters"]["is_sin_emb"],
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
                map_location=cst.DEVICE,
                weights_only=False,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
            )
        elif model_type == "TLOB_ORIGINAL":
            model = OriginalEngine.load_from_checkpoint(
                checkpoint_path,
                map_location=cst.DEVICE,
                weights_only=False,
            )
        elif model_type == "BINCTABL":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
                map_location=cst.DEVICE,
                weights_only=False,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
            )
        elif model_type == "DEEPLOB":
            model = Engine.load_from_checkpoint(
                checkpoint_path,
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=max_epochs,
                model_type=model_type,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=lr,
                optimizer=optimizer,
                dir_ckpt=dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
                map_location=cst.DEVICE,
                weights_only=False,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
            )

    else:
        if model_type == cst.ModelType.MLPLOB_ORIGINAL:
            optimizer_name = "Adam" if config.experiment.optimizer == "AdamW" else config.experiment.optimizer
            model = OriginalEngine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=optimizer_name,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
            )
        elif model_type == cst.ModelType.TLOB_ORIGINAL:
            optimizer_name = "Adam" if config.experiment.optimizer == "AdamW" else config.experiment.optimizer
            model = OriginalEngine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=optimizer_name,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                len_test_dataloader=len(test_loaders[0]),
            )
        elif model_type == cst.ModelType.MLPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
                weight_decay=config.model.hyperparameters_fixed["weight_decay"],
                multi_horizon=multi_horizon,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
            )
        elif model_type == cst.ModelType.TLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=config.model.hyperparameters_fixed["hidden_dim"],
                num_layers=config.model.hyperparameters_fixed["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=config.model.hyperparameters_fixed["num_heads"],
                is_sin_emb=config.model.hyperparameters_fixed["is_sin_emb"],
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
                weight_decay=config.model.hyperparameters_fixed["weight_decay"],
                dropout=config.model.hyperparameters_fixed.get("dropout", 0.0),
                multi_horizon=multi_horizon,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
            )
        elif model_type == cst.ModelType.BINCTABL:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
            )
        elif model_type == cst.ModelType.DEEPLOB:
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=config.model.hyperparameters_fixed["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=config.experiment.use_torch_compile,
                torch_compile_mode=config.experiment.torch_compile_mode,
                torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                torch_compile_backend=config.experiment.torch_compile_backend,
                use_fast_attention=config.experiment.use_fast_attention,
                class_weights=class_weights,
                loss_type=config.experiment.loss_type,
                focal_gamma=config.experiment.focal_gamma,
                ordinal_smoothing=config.experiment.ordinal_smoothing,
                dfl_temperature=config.experiment.dfl_temperature,
                dfl_temperature_final=config.experiment.dfl_temperature_final,
                dfl_objective=config.experiment.dfl_objective,
                dfl_lambda_turnover=config.experiment.dfl_lambda_turnover,
                dfl_lambda_entropy=config.experiment.dfl_lambda_entropy,
                dfl_cost_multiplier=config.experiment.dfl_cost_multiplier,
                dfl_lambda_drawdown=config.experiment.dfl_lambda_drawdown,
            )
        elif model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN):
            hp = config.model.hyperparameters_fixed
            extra_model_kwargs = {
                "all_features": hp.get("all_features", False),
            }
            if model_type == cst.ModelType.DAVN:
                extra_model_kwargs["davn_dual_axis"] = hp.get("davn_dual_axis", True)
                extra_model_kwargs["davn_attn_pool"] = hp.get("davn_attn_pool", True)
            model = Engine(
                seq_size=seq_size,
                horizon=horizon,
                max_epochs=config.experiment.max_epochs,
                model_type=config.model.type.value,
                is_wandb=config.experiment.is_wandb,
                experiment_type=experiment_type,
                lr=hp["lr"],
                optimizer=config.experiment.optimizer,
                dir_ckpt=config.experiment.dir_ckpt,
                hidden_dim=hp["hidden_dim"],
                num_layers=hp["num_layers"],
                num_features=train_input.shape[1],
                dataset_type=dataset_type,
                num_heads=hp["num_heads"],
                is_sin_emb=False,
                len_test_dataloader=len(test_loaders[0]),
                use_torch_compile=False,
                use_fast_attention=config.experiment.use_fast_attention,
                weight_decay=hp.get("weight_decay", 0.01),
                dropout=hp.get("dropout", 0.0),
                multi_horizon=False,
                class_weights=None,
                loss_type=config.experiment.loss_type,
                dpvn_gamma=hp.get("dpvn_gamma", 1.0),
                dpvn_huber_delta=hp.get("dpvn_huber_delta", 1.0),
                dpvn_label_normalize=hp.get("dpvn_label_normalize", True),
                **extra_model_kwargs,
            )

    print("total number of parameters: ", sum(p.numel() for p in model.parameters()))

    # Log per-timestep channel breakdown for DPVN-family models. num_features
    # reported above reflects only the input tensor on disk; DPVN-F and DAVN
    # additionally compute spread + engineered features in-model, so the
    # effective channel count exceeds num_features.
    if model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN):
        inner = model.model
        if hasattr(inner, "input_breakdown"):
            breakdown = inner.input_breakdown()
            parts = [
                f"mode={breakdown['mode']}",
                f"input_tensor_cols={breakdown['input_tensor_cols']}",
                f"lob={breakdown['lob_cols']}",
                f"aux_extra={breakdown['aux_extra_cols']}",
                f"spread_in_model={breakdown['spread_in_model']}",
                f"engineered_in_model={breakdown['engineered_in_model']}",
                f"aux_proj_in={breakdown['aux_proj_in_dim']}",
                f"effective_channels={breakdown['effective_channels']}",
            ]
            if breakdown["ignored_input_cols"]:
                parts.append(f"ignored_input_cols={breakdown['ignored_input_cols']}")
            print("Model input breakdown: " + ", ".join(parts), flush=True)
            if run is not None:
                # Strings go to config/metadata; numeric fields to run.log so
                # they show up alongside other per-run scalars.
                run.config.update({"input_mode": breakdown["mode"]}, allow_val_change=True)
                for key, value in breakdown.items():
                    if key == "mode":
                        continue
                    run.log({f"input/{key}": int(value)}, commit=False)

    train_dataloader, val_dataloader = (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
    )

    if "TRAINING" in experiment_type or "FINETUNING" in experiment_type:
        trainer.fit(model, train_dataloader, val_dataloader)
        best_model_path = model.last_path_ckpt
        print("Best model path: ", best_model_path)
        try:
            if model_type in {
                cst.ModelType.MLPLOB_ORIGINAL,
                cst.ModelType.TLOB_ORIGINAL,
            }:
                best_model = OriginalEngine.load_from_checkpoint(
                    best_model_path,
                    map_location=cst.DEVICE,
                    weights_only=False,
                )
            else:
                _is_value_net = model_type in (cst.ModelType.DPVN, cst.ModelType.DAVN)
                _load_class_weights = None if _is_value_net else class_weights
                best_model = Engine.load_from_checkpoint(
                    best_model_path,
                    map_location=cst.DEVICE,
                    weights_only=False,
                    use_torch_compile=config.experiment.use_torch_compile if not _is_value_net else False,
                    torch_compile_mode=config.experiment.torch_compile_mode,
                    torch_compile_dynamic=config.experiment.torch_compile_dynamic,
                    torch_compile_backend=config.experiment.torch_compile_backend,
                    use_fast_attention=config.experiment.use_fast_attention,
                    class_weights=_load_class_weights,
                    loss_type=config.experiment.loss_type,
                    focal_gamma=config.experiment.focal_gamma,
                    ordinal_smoothing=config.experiment.ordinal_smoothing,
                )
        except Exception as checkpoint_error:
            print(f"failed to load best checkpoint ({best_model_path}): {checkpoint_error}")
            print("selecting the last model")
            best_model = model
        best_model.experiment_type = "EVALUATION"
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(best_model, test_dataloader)
            f1 = output[0].get("f1_score", output[0].get("f1_score_h10"))
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": f1}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010 ": f1}, commit=False)
    else:
        for i in range(len(test_loaders)):
            test_dataloader = test_loaders[i]
            output = trainer.test(model, test_dataloader)
            f1 = output[0].get("f1_score", output[0].get("f1_score_h10"))
            if run is not None and dataset_type == "LOBSTER":
                run.log({f"f1 {testing_stocks[i]} best": f1}, commit=False)
            elif run is not None and dataset_type == "FI_2010":
                run.log({f"f1 FI_2010 ": f1}, commit=False)


def run_wandb(config: Config, accelerator):
    def wandb_sweep_callback():
        wandb_logger = WandbLogger(project=cst.PROJECT_NAME, log_model=False, save_dir=cst.DIR_SAVED_MODEL)
        run_name = None
        if not config.experiment.is_sweep:
            run_name = ""
            for param in config.model.keys():
                value = config.model[param]
                if param == "hyperparameters_sweep":
                    continue
                if type(value) == omegaconf.dictconfig.DictConfig:
                    for key in value.keys():
                        run_name += str(key[:2]) + "_" + str(value[key]) + "_"
                else:
                    run_name += str(param[:2]) + "_" + str(value.value) + "_"

        run = wandb.init(project=cst.PROJECT_NAME, name=run_name, entity="")  # set entity to your wandb username

        if config.experiment.is_sweep:
            model_params = run.config
        else:
            model_params = config.model.hyperparameters_fixed
        wandb_instance_name = ""
        for param in config.model.hyperparameters_fixed.keys():
            if param in model_params:
                config.model.hyperparameters_fixed[param] = model_params[param]
                wandb_instance_name += str(param) + "_" + str(model_params[param]) + "_"

        run.name = wandb_instance_name
        seq_size = config.model.hyperparameters_fixed["seq_size"]
        horizon = config.experiment.horizon
        dataset = config.dataset.type.value
        seed = config.experiment.seed
        mh_suffix = "_multi_horizon" if config.experiment.multi_horizon else ""
        if dataset == "LOBSTER":
            training_stocks = config.dataset.training_stocks
            config.experiment.dir_ckpt = (
                f"{dataset}_{training_stocks}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}{mh_suffix}"
            )
        else:
            config.experiment.dir_ckpt = f"{dataset}_seq_size_{seq_size}_horizon_{horizon}_seed_{seed}{mh_suffix}"
        wandb_instance_name = config.experiment.dir_ckpt

        trainer = L.Trainer(
            accelerator=accelerator,
            precision=config.experiment.precision,
            max_epochs=config.experiment.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=2,
                    verbose=True,
                    min_delta=0.0005,
                ),
                TQDMProgressBar(refresh_rate=100),
            ],
            num_sanity_val_steps=0,
            logger=wandb_logger,
            detect_anomaly=False,
            check_val_every_n_epoch=1,
        )

        # log simulation details in WANDB console
        run.log({"model": config.model.type.value}, commit=False)
        run.log({"dataset": config.dataset.type.value}, commit=False)
        run.log({"seed": config.experiment.seed}, commit=False)
        run.log(
            {"all_features": config.model.hyperparameters_fixed["all_features"]},
            commit=False,
        )
        run.log({"multi_horizon": config.experiment.multi_horizon}, commit=False)
        run.log({"use_diff_features": config.experiment.use_diff_features}, commit=False)
        dates = config.dataset.dates
        num_days = (datetime.strptime(dates[1], "%Y-%m-%d") - datetime.strptime(dates[0], "%Y-%m-%d")).days
        run.log({"num_dates": f"{num_days} ({dates[0]} - {dates[1]})"}, commit=False)
        if hasattr(config.dataset, "sampling_type"):
            run.log({"sampling_type": config.dataset.sampling_type.value}, commit=False)
            if config.dataset.sampling_type in (
                SamplingType.TIME,
                SamplingType.TIME_DEDUP,
            ):
                run.log({"sampling_time": config.dataset.sampling_time}, commit=False)
            elif config.dataset.sampling_type == SamplingType.QUANTITY:
                run.log(
                    {"sampling_quantity": config.dataset.sampling_quantity},
                    commit=False,
                )
        if config.dataset.type == cst.DatasetType.LOBSTER:
            for i in range(len(config.dataset.training_stocks)):
                run.log(
                    {f"training stock{i}": config.dataset.training_stocks[i]},
                    commit=False,
                )
            for i in range(len(config.dataset.testing_stocks)):
                run.log(
                    {f"testing stock{i}": config.dataset.testing_stocks[i]},
                    commit=False,
                )
        if config.dataset.type == cst.DatasetType.BATTERY:
            run.log({"product_mode": config.dataset.product_mode.value}, commit=False)
        train(config, trainer, run)
        run.finish()

    return wandb_sweep_callback


def sweep_init(config: Config):
    # put your wandb key here
    wandb.login("")
    parameters = {}
    for key in config.model.hyperparameters_sweep.keys():
        parameters[key] = {"values": list(config.model.hyperparameters_sweep[key])}
    sweep_config = {
        "method": "grid",
        "metric": {"goal": "minimize", "name": "val_loss"},
        "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 1.5},
        "run_cap": 100,
        "parameters": {**parameters},
    }
    return sweep_config


def print_setup(config: Config):
    print("Model type: ", config.model.type)
    print("Dataset: ", config.dataset.type)
    print("Seed: ", config.experiment.seed)
    print("Sequence size: ", config.model.hyperparameters_fixed["seq_size"])
    print("Horizon: ", config.experiment.horizon)
    print("All features: ", config.model.hyperparameters_fixed["all_features"])
    print("Is data preprocessed: ", config.experiment.is_data_preprocessed)
    print("Is wandb: ", config.experiment.is_wandb)
    print("Is sweep: ", config.experiment.is_sweep)
    print("Use torch.compile: ", config.experiment.use_torch_compile)
    print("torch.compile mode: ", config.experiment.torch_compile_mode)
    print("torch.compile dynamic: ", config.experiment.torch_compile_dynamic)
    print("torch.compile backend: ", config.experiment.torch_compile_backend)
    print("Precision: ", config.experiment.precision)
    print("Use fast attention: ", config.experiment.use_fast_attention)
    print("Use diff features: ", config.experiment.use_diff_features)
    print(config.experiment.type)
    print("Is debug: ", config.experiment.is_debug)
    if config.dataset.type == cst.DatasetType.LOBSTER:
        print("Training stocks: ", config.dataset.training_stocks)
        print("Testing stocks: ", config.dataset.testing_stocks)
