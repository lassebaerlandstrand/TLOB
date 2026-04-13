import os
import random
import warnings
import zipfile

warnings.filterwarnings("ignore")
import hydra
import numpy as np
import torch

import constants as cst
import wandb
from config.config import Config
from preprocessing.battery import BatteryDataBuilder
from preprocessing.btc import BTCDataBuilder
from preprocessing.lobster import LOBSTERDataBuilder
from run import run, run_wandb, sweep_init


@hydra.main(config_path="config", config_name="config")
def hydra_app(config: Config):
    set_reproducibility(config.experiment.seed)
    print("Using device: ", cst.DEVICE)
    if cst.DEVICE == "cpu":
        accelerator = "cpu"
    else:
        accelerator = "gpu"

    # Apply dataset-specific model overrides from config.
    # Supports per-model overrides: {"_default": {...}, "FUSELOB": {...}}
    # or flat dict for backward compatibility: {"hidden_dim": 50, ...}
    if hasattr(config.dataset, "model_overrides") and config.dataset.model_overrides:
        overrides = config.dataset.model_overrides
        model_name = config.model.type.value
        if "_default" in overrides:
            # Per-model override format
            resolved = dict(overrides.get("_default", {}))
            resolved.update(overrides.get(model_name, {}))
        else:
            # Flat dict (backward compatible)
            resolved = dict(overrides)
        for key, value in resolved.items():
            config.model.hyperparameters_fixed[key] = value

    # Keep model-level all_features aligned with dataset setting when available.
    if hasattr(config.dataset, "all_features"):
        config.model.hyperparameters_fixed["all_features"] = config.dataset.all_features

    if (
        config.dataset.type.value == "LOBSTER"
        and not config.experiment.is_data_preprocessed
    ):
        # prepare the datasets, this will save train.npy, val.npy and test.npy in the data directory
        data_builder = LOBSTERDataBuilder(
            stocks=config.dataset.training_stocks,
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
            label_mode=config.experiment.label_mode,
        )
        data_builder.prepare_save_datasets()

    elif (
        config.dataset.type.value == "FI_2010"
        and not config.experiment.is_data_preprocessed
    ):
        try:
            # take the .zip files name in data/FI_2010
            dir = cst.DATA_DIR + "/FI_2010/"
            for filename in os.listdir(dir):
                if filename.endswith(".zip"):
                    filename = dir + filename
                    with zipfile.ZipFile(filename, "r") as zip_ref:
                        zip_ref.extractall(dir)  # Extracts to the current directory
            print("Data extracted.")
        except Exception as e:
            raise (f"Error downloading or extracting data: {e}")

    elif (
        config.dataset.type == cst.DatasetType.BTC
        and not config.experiment.is_data_preprocessed
    ):
        data_builder = BTCDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
            label_mode=config.experiment.label_mode,
        )
        data_builder.prepare_save_datasets()

    elif (
        config.dataset.type == cst.DatasetType.BATTERY
        and not config.experiment.is_data_preprocessed
    ):
        data_builder = BatteryDataBuilder(
            data_dir=cst.DATA_DIR,
            date_trading_days=config.dataset.dates,
            split_rates=cst.SPLIT_RATES,
            sampling_type=config.dataset.sampling_type,
            sampling_time=config.dataset.sampling_time,
            sampling_quantity=config.dataset.sampling_quantity,
            product_mode=config.dataset.product_mode,
            market_type=config.dataset.market_type,
            raw_data_path=config.dataset.raw_data_path,
            parsed_data_path=config.dataset.parsed_data_path,
            max_lob_depth=config.dataset.max_lob_depth,
            all_features=config.dataset.all_features,
            force_rebuild=True,
            label_mode=config.experiment.label_mode,
            extract_events=getattr(config.dataset, "extract_events", False),
            max_events_per_window=getattr(config.dataset, "max_events_per_window", 64),
            max_hours_before_delivery=getattr(config.dataset, "max_hours_before_delivery", 0.0),
        )
        data_builder.prepare_save_datasets()

    if config.experiment.is_wandb:
        if config.experiment.is_sweep:
            sweep_config = sweep_init(config)
            sweep_id = wandb.sweep(sweep_config, project=cst.PROJECT_NAME, entity="")
            wandb.agent(
                sweep_id, run_wandb(config, accelerator), count=sweep_config["run_cap"]
            )
        else:
            start_wandb = run_wandb(config, accelerator)
            start_wandb()

    # training without using wandb
    else:
        run(config, accelerator)


def set_reproducibility(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_torch():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    set_torch()
    hydra_app()
