from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from constants import DatasetType, ModelType, ProductMode, SamplingType


@dataclass
class Model:
    hyperparameters_fixed: dict = MISSING
    hyperparameters_sweep: dict = MISSING
    type: ModelType = MISSING


@dataclass
class MLPLOB(Model):
    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 3,
            "hidden_dim": 40,
            "lr": 0.0003,
            "seq_size": 384,
            "all_features": True,
            "weight_decay": 0.01,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [3, 6],
            "hidden_dim": [128],
            "lr": [0.0003],
            "seq_size": [384],
        }
    )
    type: ModelType = ModelType.MLPLOB


@dataclass
class TLOB(Model):
    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 4,
            "hidden_dim": 40,
            "num_heads": 1,
            "is_sin_emb": True,
            "lr": 0.0001,
            "seq_size": 128,
            "all_features": True,
            "weight_decay": 0.01,
            "dropout": 0.0,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [4, 6],
            "hidden_dim": [128, 256],
            "num_heads": [1],
            "is_sin_emb": [True],
            "lr": [0.0001],
            "seq_size": [128],
        }
    )
    type: ModelType = ModelType.TLOB


@dataclass
class MLPLOBOriginal(Model):
    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 3,
            "hidden_dim": 40,
            "lr": 0.0003,
            "seq_size": 384,
            "all_features": True,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [3, 6],
            "hidden_dim": [128],
            "lr": [0.0003],
            "seq_size": [384],
        }
    )
    type: ModelType = ModelType.MLPLOB_ORIGINAL


@dataclass
class TLOBOriginal(Model):
    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 4,
            "hidden_dim": 40,
            "num_heads": 1,
            "is_sin_emb": True,
            "lr": 0.0001,
            "seq_size": 128,
            "all_features": True,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [4, 6],
            "hidden_dim": [128, 256],
            "num_heads": [1],
            "is_sin_emb": [True],
            "lr": [0.0001],
            "seq_size": [128],
        }
    )
    type: ModelType = ModelType.TLOB_ORIGINAL


@dataclass
class BiNCTABL(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.001, "seq_size": 10, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.001], "seq_size": [10]})
    type: ModelType = ModelType.BINCTABL


@dataclass
class DeepLOB(Model):
    hyperparameters_fixed: dict = field(default_factory=lambda: {"lr": 0.01, "seq_size": 100, "all_features": False})
    hyperparameters_sweep: dict = field(default_factory=lambda: {"lr": [0.01], "seq_size": [100]})
    type: ModelType = ModelType.DEEPLOB


@dataclass
class Dataset:
    type: DatasetType = MISSING
    dates: list = MISSING
    batch_size: int = MISSING
    model_overrides: dict = field(default_factory=dict)


@dataclass
class FI_2010(Dataset):
    type: DatasetType = DatasetType.FI_2010
    dates: list = field(default_factory=lambda: ["2010-01-01", "2010-12-31"])
    batch_size: int = 32
    model_overrides: dict = field(default_factory=lambda: {"hidden_dim": 144})


@dataclass
class LOBSTER(Dataset):
    type: DatasetType = DatasetType.LOBSTER
    dates: list = field(default_factory=lambda: ["2015-01-02", "2015-01-30"])
    sampling_type: SamplingType = SamplingType.QUANTITY
    sampling_time: str = "1s"
    sampling_quantity: int = 500
    training_stocks: list = field(default_factory=lambda: ["INTC"])
    testing_stocks: list = field(default_factory=lambda: ["INTC"])
    batch_size: int = 128
    model_overrides: dict = field(default_factory=lambda: {"hidden_dim": 46})


@dataclass
class BTC(Dataset):
    type: DatasetType = DatasetType.BTC
    dates: list = field(default_factory=lambda: ["2023-01-09", "2023-01-20"])
    sampling_type: SamplingType = SamplingType.NONE
    sampling_time: str = "100ms"
    sampling_quantity: int = 0
    batch_size: int = 128
    training_stocks: list = field(default_factory=lambda: ["BTC"])
    testing_stocks: list = field(default_factory=lambda: ["BTC"])
    model_overrides: dict = field(
        default_factory=lambda: {
            "_default": {"hidden_dim": 40},
            "DPVN": {
                "hidden_dim": 96,
                "num_heads": 4,
                "dropout": 0.1,
                "all_features": True,
            },
            "DAVN": {
                "hidden_dim": 64,
                "num_heads": 4,
                "dropout": 0.1,
            },
        }
    )


@dataclass
class BATTERY(Dataset):
    type: DatasetType = DatasetType.BATTERY
    dates: list = field(default_factory=lambda: ["2021-01-11", "2021-01-22"])
    sampling_type: SamplingType = SamplingType.TIME_DEDUP
    sampling_time: str = "10s"
    sampling_quantity: int = 0
    batch_size: int = 128
    training_stocks: list = field(default_factory=lambda: ["battery_markets"])
    testing_stocks: list = field(default_factory=lambda: ["battery_markets"])
    product_mode: ProductMode = ProductMode.PER_PRODUCT
    market_type: str = "EPEX"
    raw_data_path: str = "data/battery_markets"
    parsed_data_path: str = "data/battery_markets/parsed"
    max_lob_depth: float = 1000.0
    all_features: bool = True
    extract_events: bool = True
    max_events_per_window: int = 64
    max_hours_before_delivery: float = 6.0
    model_overrides: dict = field(
        default_factory=lambda: {
            "_default": {"hidden_dim": 50, "num_heads": 1, "dropout": 0.1},
            "DPVN": {"hidden_dim": 64, "num_heads": 4, "dropout": 0.1, "all_features": True},
            "DAVN": {"hidden_dim": 96, "num_heads": 4, "dropout": 0.1, "all_features": True},
        }
    )


@dataclass
class DPVNConfig(Model):
    """DP-Distilled Value Network.

    Predicts per-action value V(s, a) for a in {-1, 0, +1} and decides via a
    spread-aware argmax at inference. Supervised against truncated-horizon Q
    targets bootstrapped from the DP-optimal trajectory.
    """

    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 4,
            "hidden_dim": 64,
            "num_heads": 4,
            "lr": 0.0003,
            "seq_size": 128,
            "all_features": False,
            "weight_decay": 0.01,
            "dropout": 0.0,
            "dpvn_gamma": 1.0,
            "dpvn_huber_delta": 1.0,
            "dpvn_label_normalize": True,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [4],
            "hidden_dim": [64, 96],
            "num_heads": [4],
            "lr": [0.0003],
            "seq_size": [128],
        }
    )
    type: ModelType = ModelType.DPVN


@dataclass
class DAVNConfig(Model):
    """Dual-Axis Value Network.

    Extends DPVN's DP-distilled value paradigm with (1) a fused input path
    (LOB + aux engineered features + in-model spread features), (2) a dual-axis
    transformer trunk alternating temporal and feature-axis attention, and
    (3) an attention-pool readout over all timesteps. Single-horizon only;
    same Huber loss on DP Q-targets as DPVN.
    """

    hyperparameters_fixed: dict = field(
        default_factory=lambda: {
            "num_layers": 2,  # 2 DualBlocks = 4 attention ops (2 temporal + 2 feature-axis)
            "hidden_dim": 64,
            "num_heads": 4,
            "lr": 0.0003,
            "seq_size": 128,
            "all_features": True,
            "weight_decay": 0.01,
            "dropout": 0.0,
            "dpvn_gamma": 1.0,
            "dpvn_huber_delta": 1.0,
            "dpvn_label_normalize": True,
            "davn_dual_axis": True,
            "davn_attn_pool": True,
        }
    )
    hyperparameters_sweep: dict = field(
        default_factory=lambda: {
            "num_layers": [2],
            "hidden_dim": [64, 96],
            "num_heads": [4],
            "lr": [0.0003],
            "seq_size": [128],
        }
    )
    type: ModelType = ModelType.DAVN


@dataclass
class Experiment:
    is_data_preprocessed: bool = False
    is_wandb: bool = True
    is_sweep: bool = False
    type: list = field(default_factory=lambda: ["TRAINING"])
    is_debug: bool = False
    checkpoint_reference: str = ""
    seed: int = 1
    horizon: int = 10
    max_epochs: int = 50
    dir_ckpt: str = "model.ckpt"
    optimizer: str = "AdamW"
    use_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_dynamic: bool = False
    torch_compile_backend: str = "inductor"
    precision: str = "bf16-mixed"
    use_fast_attention: bool = True
    use_diff_features: bool = False
    use_class_weights: bool = True
    label_mode: str = "absolute_change"  # "absolute_change" | "percent_change"
    multi_horizon: bool = False
    loss_type: str = (
        "cross_entropy"  # "cross_entropy" | "cost_aware_ce" | "focal" | "focal_ordinal" | "dfl_proxy" | "dfl_trading"
    )
    focal_gamma: float = 2.0  # Unused if loss_type is not "focal" or "focal_ordinal"
    ordinal_smoothing: float = 0.15  # Unused if loss_type is not "focal_ordinal"
    trading_cost: float = 0.0  # Cost multiplier (x mean |Δmid|) for trading simulation
    # DFL parameters (used when loss_type starts with "dfl_")
    dfl_temperature: float = 1.0  # Gumbel-Softmax initial temperature
    dfl_temperature_final: float = 0.1  # Final temperature after annealing
    dfl_cost_multiplier: float = 1.0  # Transaction cost multiplier for spread-aware DFL
    dfl_objective: str = "sharpe"  # "pnl" | "sharpe" | "sortino"
    dfl_lambda_drawdown: float = 0.0  # Drawdown penalty weight
    dfl_lambda_turnover: float = 0.0  # Turnover penalty weight
    dfl_lambda_entropy: float = 0.01  # Entropy regularization (prevents position collapse)


defaults = [Model, Experiment, Dataset]


@dataclass
class Config:
    model: Model
    dataset: Dataset
    experiment: Experiment = field(default_factory=Experiment)
    defaults: List = field(
        default_factory=lambda: [
            {"hydra/job_logging": "disabled"},
            {"hydra/hydra_logging": "disabled"},
            "_self_",
        ]
    )


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="mlplob", node=MLPLOB)
cs.store(group="model", name="tlob", node=TLOB)
cs.store(group="model", name="mlplob_original", node=MLPLOBOriginal)
cs.store(group="model", name="tlob_original", node=TLOBOriginal)
cs.store(group="model", name="binctabl", node=BiNCTABL)
cs.store(group="model", name="deeplob", node=DeepLOB)
cs.store(group="model", name="dpvn", node=DPVNConfig)
cs.store(group="model", name="davn", node=DAVNConfig)
cs.store(group="dataset", name="lobster", node=LOBSTER)
cs.store(group="dataset", name="fi_2010", node=FI_2010)
cs.store(group="dataset", name="btc", node=BTC)
cs.store(group="dataset", name="battery", node=BATTERY)
