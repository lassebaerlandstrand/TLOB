from lightning import LightningModule
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import numpy as np
import time
import warnings
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import wandb
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from utils.utils_model import pick_model
from utils.metrics import (
    _infer_std_price,
    compute_baselines,
    compute_metrics,
    compute_trading_metrics,
    format_confidence_stats,
    format_horizon_table,
    format_prediction_distribution,
    format_trading_table,
    plot_confusion_matrices,
)
import constants as cst
from scipy.stats import mode
from models.losses import FocalLoss
from models.trading_loss import DFLProxyLoss, DFLTradingLoss

# Horizons in canonical order (h10, h20, h50, h100)
HORIZONS = [10, 20, 50, 100]


class Engine(LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
        use_torch_compile=False,
        torch_compile_mode="default",
        torch_compile_dynamic=False,
        torch_compile_backend="inductor",
        use_fast_attention=True,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        multi_horizon: bool = False,
        class_weights: torch.Tensor | None = None,
        loss_type: str = "focal_ordinal",
        focal_gamma: float = 2.0,
        ordinal_smoothing: float = 0.15,
        dfl_temperature: float = 1.0,
        dfl_temperature_final: float = 0.1,
        dfl_objective: str = "sharpe",
        dfl_lambda_turnover: float = 0.0,
        dfl_lambda_entropy: float = 0.01,
        dfl_cost_multiplier: float = 1.0,
        dfl_lambda_drawdown: float = 0.0,
        **model_kwargs,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        self.use_torch_compile = use_torch_compile
        self.torch_compile_mode = torch_compile_mode
        self.torch_compile_dynamic = torch_compile_dynamic
        self.torch_compile_backend = torch_compile_backend
        self.use_fast_attention = use_fast_attention
        self.multi_horizon = multi_horizon
        # DeepLOB and BiNCTABL are unmodified reference implementations that output
        # probabilities (they apply softmax internally). Focal loss requires logits,
        # so fall back to cross_entropy for those models.
        _PROB_OUTPUT_MODELS = ("DEEPLOB", "BINCTABL")
        if loss_type != "cross_entropy" and str(model_type).upper() in _PROB_OUTPUT_MODELS:
            print(
                f"[Engine] loss_type='{loss_type}' is not supported for {model_type} "
                "(model outputs probabilities, not logits). Falling back to cross_entropy."
            )
            loss_type = "cross_entropy"
        self.loss_type = loss_type
        if loss_type != "cross_entropy":
            self.focal_gamma = focal_gamma
            self.ordinal_smoothing = ordinal_smoothing

        num_horizons = len(HORIZONS) if multi_horizon else 1
        self.model = pick_model(
            model_type,
            hidden_dim,
            num_layers,
            seq_size,
            num_features,
            num_heads,
            is_sin_emb,
            dataset_type,
            use_fast_attention=use_fast_attention,
            num_horizons=num_horizons,
            dropout=dropout,
            **model_kwargs,
        )
        self._compile_model()
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        if class_weights is not None:
            class_weights = class_weights.detach().float()
            if self.multi_horizon:
                if class_weights.ndim == 1:
                    class_weights = class_weights.unsqueeze(0).repeat(len(HORIZONS), 1)
                if class_weights.ndim != 2:
                    raise ValueError("class_weights must have shape (H, C) or (C,) in multi_horizon mode")
                if class_weights.shape[0] != len(HORIZONS):
                    raise ValueError(
                        f"Expected class_weights for {len(HORIZONS)} horizons, got {class_weights.shape[0]}"
                    )
                self.register_buffer("class_weights", class_weights)
            else:
                if class_weights.ndim == 2:
                    class_weights = class_weights[0]
                if class_weights.ndim != 1:
                    raise ValueError("class_weights must have shape (C,) in single-horizon mode")
                self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.is_dfl = loss_type.startswith("dfl_")
        if self.loss_type in ("cross_entropy", "cost_aware_ce"):
            if self.multi_horizon:
                # Per-horizon weights applied manually in _multi_horizon_loss
                self.loss_function = nn.CrossEntropyLoss()
            else:
                self.loss_function = nn.CrossEntropyLoss(
                    weight=self.class_weights if self.class_weights is not None else None
                )
        elif self.loss_type == "dfl_proxy":
            self.dfl_loss = DFLProxyLoss(
                temperature=dfl_temperature,
                objective=dfl_objective,
                lambda_turnover=dfl_lambda_turnover,
                lambda_entropy=dfl_lambda_entropy,
            )
            self.loss_function = None
        elif self.loss_type == "dfl_trading":
            self.dfl_loss = DFLTradingLoss(
                temperature=dfl_temperature,
                cost_multiplier=dfl_cost_multiplier,
                objective=dfl_objective,
                lambda_drawdown=dfl_lambda_drawdown,
                lambda_turnover=dfl_lambda_turnover,
                lambda_entropy=dfl_lambda_entropy,
            )
            self.loss_function = None
        else:
            # focal or focal_ordinal
            smoothing = ordinal_smoothing if self.loss_type == "focal_ordinal" else 0.0
            if self.multi_horizon:
                # One FocalLoss per horizon, each with its own per-horizon alpha
                self.horizon_losses = nn.ModuleList()
                for h_idx in range(len(HORIZONS)):
                    alpha_h = self.class_weights[h_idx] if self.class_weights is not None else None
                    self.horizon_losses.append(
                        FocalLoss(gamma=focal_gamma, alpha=alpha_h, ordinal_smoothing=smoothing)
                    )
                self.loss_function = None  # not used in multi-horizon; _multi_horizon_loss handles it
            else:
                self.loss_function = FocalLoss(
                    gamma=focal_gamma,
                    alpha=self.class_weights,
                    ordinal_smoothing=smoothing,
                )

        # --- DPVN family (DPVN + DAVN): DP-distilled value networks ---
        # Both share the same training paradigm (Huber on DP Q-targets), inference
        # rule (spread_argmax), and output shape (B, 3) — only the model
        # architecture differs. Kept under the `is_dpvn` flag name for minimal
        # disruption of downstream checks.
        self.is_dpvn = str(model_type).upper() in ("DPVN", "DAVN")
        if self.is_dpvn:
            self.dpvn_gamma = model_kwargs.get("dpvn_gamma", 1.0)
            self.dpvn_huber_delta = model_kwargs.get("dpvn_huber_delta", 1.0)
            self.dpvn_label_normalize = model_kwargs.get("dpvn_label_normalize", True)
            self.loss_type = "dpvn_huber"
            self.is_dfl = False
            self.loss_function = None
            self.register_buffer("_dpvn_target_std", torch.tensor(1.0))
            self._dpvn_target_std_init = False
            self._batch_q_target = None

        # Learnable log-variance parameters for homoscedastic uncertainty weighting.
        # One scalar per horizon; initialised to 0 (σ_h = 1, equal initial weights).
        # Stored in Engine (not the backbone) so ONNX export is unaffected.
        # Unsure if this makes it so ONNX can be used
        if multi_horizon:
            self.log_vars = nn.Parameter(torch.zeros(len(HORIZONS)))

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.test_proba_full = []
        self.test_v_values = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.val_v_values = []
        self.val_mid_prices = []
        self.val_z_half_spreads = []
        self.min_loss = np.inf
        # Save hyperparameters, ignoring unused ones for cleaner checkpoints
        ignore_params = []
        if not self.loss_type.startswith("focal"):
            ignore_params.extend(["focal_gamma", "ordinal_smoothing"])
        if not self.is_dfl:
            ignore_params.extend(["dfl_temperature", "dfl_temperature_final", "dfl_objective",
                                  "dfl_lambda_turnover", "dfl_lambda_entropy",
                                  "dfl_cost_multiplier", "dfl_lambda_drawdown"])
        self.save_hyperparameters(ignore=ignore_params if ignore_params else None)
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        self.test_half_spreads = []
        self.test_z_half_spreads = []
        self.test_transaction_costs = []
        self.train_epoch_start_time = None
        self.epoch_sample_count = 0
        self.epoch_iteration_count = 0
        self.total_train_time_s = 0.0
        self.total_train_samples = 0
        self.total_train_steps = 0

        # Multi-horizon per-horizon accumulators
        if multi_horizon:
            self.train_losses_per_h = [[] for _ in HORIZONS]
            self.val_targets_per_h = [[] for _ in HORIZONS]
            self.val_predictions_per_h = [[] for _ in HORIZONS]
            self.test_targets_per_h = [[] for _ in HORIZONS]
            self.test_predictions_per_h = [[] for _ in HORIZONS]
            self.test_proba_per_h = [[] for _ in HORIZONS]
            self.test_losses_per_h = [[] for _ in HORIZONS]
            self.test_logits_per_h = [[] for _ in HORIZONS]

    def _console(self, *message) -> None:
        # Before trainer attachment (e.g., in __init__), LightningModule.print raises.
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            print(*message)
            return
        try:
            # LightningModule.print integrates better with tqdm/progress bars.
            self.print(*message)
        except RuntimeError:
            print(*message)

    def _section(self, title: str) -> None:
        rule = "=" * 92
        self._console("")
        self._console(rule)
        self._console(title)
        self._console(rule)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def _compile_model(self):
        if not self.use_torch_compile:
            return
        if self.model_type not in {"TLOB", "MLPLOB"}:
            return
        try:
            self.model = torch.compile(
                self.model,
                mode=self.torch_compile_mode,
                dynamic=self.torch_compile_dynamic,
                backend=self.torch_compile_backend,
            )
            self._console(
                "torch.compile enabled for",
                self.model_type,
                f"(backend={self.torch_compile_backend}, mode={self.torch_compile_mode}, dynamic={self.torch_compile_dynamic})",
            )
        except Exception as compile_error:
            self._console(f"torch.compile failed, continuing without compilation: {compile_error}")

    # ------------------------------------------------------------------
    # Forward / Loss
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch):
        """Unpack batch into (x, y, dfl_data).

        Supported shapes:
            (x, y)                                         — standard classification
            (x, y, half_spread, transaction_cost)          — single-horizon + cost accumulators
            (x, y, q_target)                               — DPVN / DAVN, no cost
            (x, y, q_target, half_spread, transaction_cost) — DPVN / DAVN with cost
            (x, y, delta_mids, half_spread, transaction_cost) — multi-horizon DFL
        """
        # DPVN with q_target + dfl: 5-element batch (DPVN flag disambiguates from multi-horizon DFL)
        if self.is_dpvn and len(batch) == 5:
            x, y, q_target, half_spread, transaction_cost = batch
            self._batch_q_target = q_target
            # delta_mids placeholder — single-horizon DPVN doesn't use DFL loss, we only need
            # tc/hs accumulation in test_step. Store as (None, hs, tc) for the downstream unpacker.
            return x, y, (None, half_spread, transaction_cost)

        # DPVN with q_target only: 3-element batch
        if self.is_dpvn and len(batch) == 3:
            x, y, q_target = batch
            self._batch_q_target = q_target
            return x, y, None

        if len(batch) == 5:
            # Multi-horizon DFL data: (x, y_multi, delta_mids, half_spread, transaction_cost)
            x, y, delta_mids, half_spread, transaction_cost = batch
            return x, y, (delta_mids, half_spread, transaction_cost)

        if len(batch) == 4:
            # Single-horizon classification with cost accumulators: (x, y, hs, tc)
            x, y, half_spread, transaction_cost = batch
            return x, y, (None, half_spread, transaction_cost)

        x, y = batch
        return x, y, None

    def forward(self, x, batch_idx=None):
        return self.model(x)

    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)

    def _multi_horizon_loss(self, y_hat_list, y_multi, dfl_data=None):
        """Homoscedastic uncertainty-weighted multi-task loss (Kendall et al. 2018).

        L = Σ_h [ CE(ŷ_h, y_h) / (2σ_h²) + log σ_h ]
        where log σ_h² = self.log_vars[h] (log-variance parameterisation).

        For DFL: replaces CE_h with DFL loss per horizon.
        """
        horizon_losses = []
        for horizon_index, y_hat_h in enumerate(y_hat_list):
            if self.is_dfl:
                if self.loss_type == "dfl_trading" and dfl_data is not None:
                    delta_mids, half_spreads, transaction_costs = dfl_data
                    dfl_loss_h, _ = self.dfl_loss(
                        y_hat_h,
                        delta_mids[:, horizon_index],
                        half_spreads,
                        transaction_cost=transaction_costs,
                    )
                else:
                    if self.loss_type == "dfl_trading" and dfl_data is None:
                        if not hasattr(self, "_warned_dfl_fallback"):
                            print("WARNING: loss_type=dfl_trading but dfl_data is None. "
                                  "Falling back to proxy loss. Rebuild data with --rebuild-data.")
                            self._warned_dfl_fallback = True
                    # dfl_proxy (or dfl_trading fallback): uses labels as direction signal
                    dfl_loss_h, _ = self.dfl_loss(y_hat_h, y_multi[:, horizon_index])
                horizon_losses.append(dfl_loss_h)
            elif self.loss_type == "cost_aware_ce" and dfl_data is not None:
                # Cost-aware CE: weight each sample by |delta_mid| / (half_spread + tc)
                delta_mids, half_spreads, transaction_costs = dfl_data
                dm_h = delta_mids[:, horizon_index].abs()
                hs = (half_spreads + transaction_costs.abs()).clamp(min=1e-8)
                sample_w = (dm_h / hs).clamp(min=0.01, max=10.0)
                horizon_weight = None
                if self.class_weights is not None:
                    if self.class_weights.ndim == 2:
                        horizon_weight = self.class_weights[horizon_index]
                    else:
                        horizon_weight = self.class_weights
                ce_unreduced = torch.nn.functional.cross_entropy(
                    y_hat_h,
                    y_multi[:, horizon_index],
                    weight=horizon_weight,
                    reduction="none",
                )
                ce_h = (ce_unreduced * sample_w).mean()
                horizon_losses.append(ce_h)
            elif self.loss_type in ("cross_entropy", "cost_aware_ce"):
                # Standard CE (also fallback if cost_aware_ce has no dfl_data)
                horizon_weight = None
                if self.class_weights is not None:
                    if self.class_weights.ndim == 2:
                        horizon_weight = self.class_weights[horizon_index]
                    else:
                        horizon_weight = self.class_weights
                ce_h = torch.nn.functional.cross_entropy(
                    y_hat_h,
                    y_multi[:, horizon_index],
                    weight=horizon_weight,
                    reduction="mean",
                )
                horizon_losses.append(ce_h)
            else:
                ce_h = self.horizon_losses[horizon_index](y_hat_h, y_multi[:, horizon_index])
                horizon_losses.append(ce_h)

        loss_per_h = torch.stack(horizon_losses)  # (H,)

        if self.is_dfl:
            # DFL losses are negative (-Sharpe/-PnL). Kendall uncertainty weighting
            # requires positive losses — with negative losses, σ_h collapses toward 0,
            # amplifying the negative loss without bound (same failure as focal loss,
            # see CHANGES.md). Use simple equal-weight sum instead.
            total_loss = loss_per_h.sum()
        else:
            # Uncertainty weighting (vectorised) — works for CE/focal (positive losses)
            sigma2 = torch.exp(self.log_vars)  # (H,)
            total_loss = (loss_per_h / (2.0 * sigma2) + 0.5 * self.log_vars).sum()

        return total_loss, loss_per_h.detach()  # scalar, (H,) tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y, dfl_data = self._unpack_batch(batch)

        # --- DPVN: DP-distilled value regression ---
        if self.is_dpvn:
            return self._dpvn_training_step(x, y)

        if self.multi_horizon:
            y_hat_list = self.forward(x)
            batch_loss, per_h_loss = self._multi_horizon_loss(y_hat_list, y, dfl_data)
            for i, loss_val in enumerate(per_h_loss.unbind()):
                self.train_losses_per_h[i].append(loss_val)
        elif self.is_dfl:
            y_hat = self.forward(x)
            if self.loss_type == "dfl_trading" and dfl_data is not None:
                delta_mids, half_spreads, transaction_costs = dfl_data
                # Single-horizon DFL: delta_mids is the h-specific column already.
                # For multi-horizon 2D tensors, pick column 0 (single-horizon path asserts this upstream).
                dm = delta_mids[:, 0] if delta_mids.dim() == 2 else delta_mids
                batch_loss, _ = self.dfl_loss(
                    y_hat, dm, half_spreads, transaction_cost=transaction_costs
                )
            else:
                if self.loss_type == "dfl_trading" and dfl_data is None:
                    if not hasattr(self, "_warned_dfl_fallback"):
                        print("WARNING: loss_type=dfl_trading but dfl_data is None. "
                              "Falling back to proxy loss. Rebuild data with --rebuild-data.")
                        self._warned_dfl_fallback = True
                batch_loss, _ = self.dfl_loss(y_hat, y)
        else:
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)

        batch_loss_mean = batch_loss if (self.multi_horizon or self.is_dfl) else torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.detach())
        self.epoch_iteration_count += 1
        self.epoch_sample_count += int(y.shape[0])
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.perf_counter()
        self.epoch_sample_count = 0
        self.epoch_iteration_count = 0

        # Temperature annealing for DFL Gumbel-Softmax
        if self.is_dfl:
            t0 = self.hparams.get("dfl_temperature", 1.0)
            t_final = self.hparams.get("dfl_temperature_final", 0.1)
            progress = self.current_epoch / max(self.max_epochs - 1, 1)
            new_tau = t0 * (t_final / t0) ** progress
            self.dfl_loss.temperature = new_tau

    def on_train_epoch_end(self) -> None:
        if self.train_epoch_start_time is None:
            return
        epoch_duration = max(time.perf_counter() - self.train_epoch_start_time, 1e-12)
        samples_per_sec = self.epoch_sample_count / epoch_duration
        it_per_sec = self.epoch_iteration_count / epoch_duration
        self.total_train_time_s += epoch_duration
        self.total_train_samples += self.epoch_sample_count
        self.total_train_steps += self.epoch_iteration_count
        self._console(
            f"Epoch {self.current_epoch} throughput - {epoch_duration:.1f}s, samples/s: {samples_per_sec:.2f}, it/s: {it_per_sec:.2f}"
        )

    # ------------------------------------------------------------------
    # DPVN steps
    # ------------------------------------------------------------------
    def _dpvn_training_step(self, x, y):
        """Huber regression on per-action DP-distilled Q targets."""
        import torch.nn.functional as F

        q_target = self._batch_q_target
        v_pred = self.model(x)  # (B, 3) raw values

        if self.dpvn_label_normalize and not self._dpvn_target_std_init:
            std = q_target.detach().std().clamp(min=1e-4)
            self._dpvn_target_std.copy_(std)
            self._dpvn_target_std_init = True

        if self.dpvn_label_normalize:
            scale = self._dpvn_target_std.clamp(min=1e-4)
            loss = F.huber_loss(v_pred / scale, q_target / scale, delta=self.dpvn_huber_delta)
        else:
            loss = F.huber_loss(v_pred, q_target, delta=self.dpvn_huber_delta)

        self.train_losses.append(loss.detach())
        self.epoch_iteration_count += 1
        self.epoch_sample_count += int(y.shape[0])
        self.ema.update()
        return loss

    def _dpvn_validation_step(self, x, y):
        import torch.nn.functional as F

        q_target = self._batch_q_target
        v_pred = self.model(x)  # (B, 3)

        if self.dpvn_label_normalize:
            scale = self._dpvn_target_std.clamp(min=1e-4)
            loss = F.huber_loss(v_pred / scale, q_target / scale, delta=self.dpvn_huber_delta)
        else:
            loss = F.huber_loss(v_pred, q_target, delta=self.dpvn_huber_delta)

        # Map argmax(V) (action index 0/1/2 = -1/0/+1) to label (down=2, stat=1, up=0)
        action_idx = v_pred.argmax(dim=1)
        _IDX_TO_LABEL = torch.tensor([2, 1, 0], device=x.device, dtype=torch.long)
        pred = _IDX_TO_LABEL[action_idx]

        self.val_targets.append(y)
        self.val_predictions.append(pred)
        self.val_v_values.append(v_pred.detach().cpu())
        self.val_losses.append(loss.detach())
        return loss

    def _dpvn_test_step(self, x, y):
        import torch.nn.functional as F
        from contextlib import nullcontext

        if self.experiment_type == "TRAINING":
            ctx = self.ema.average_parameters()
        else:
            ctx = nullcontext()
        with ctx:
            v_pred = self.model(x)

        # Classification-style predictions (raw argmax, no spread rule)
        action_idx = v_pred.argmax(dim=1)
        _IDX_TO_LABEL = torch.tensor([2, 1, 0], device=x.device, dtype=torch.long)
        pred = _IDX_TO_LABEL[action_idx]

        # Batch loss for logging (Huber on normalized targets if available)
        q_target = self._batch_q_target
        if q_target is not None:
            if self.dpvn_label_normalize:
                scale = self._dpvn_target_std.clamp(min=1e-4)
                batch_loss = F.huber_loss(v_pred / scale, q_target / scale, delta=self.dpvn_huber_delta)
            else:
                batch_loss = F.huber_loss(v_pred, q_target, delta=self.dpvn_huber_delta)
        else:
            batch_loss = v_pred.abs().mean()  # fallback

        self.test_targets.append(y)
        self.test_predictions.append(pred)
        self.test_v_values.append(v_pred.detach().cpu())
        self.test_losses.append(batch_loss.detach())
        return batch_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y, dfl_data = self._unpack_batch(batch)
        # Accumulate mid-prices for trading evaluation (same as test_step)
        self.val_mid_prices.append(((x[:, -1, 0] + x[:, -1, 2]) / 2).cpu().numpy().flatten())
        self.val_z_half_spreads.append(((x[:, -1, 0] - x[:, -1, 2]) / 2).cpu().numpy().flatten())
        with self.ema.average_parameters():
            if self.is_dpvn:
                return self._dpvn_validation_step(x, y)
            if self.multi_horizon:
                y_hat_list = self.forward(x)
                batch_loss, _ = self._multi_horizon_loss(y_hat_list, y, dfl_data)
                for i, y_hat in enumerate(y_hat_list):
                    self.val_targets_per_h[i].append(y[:, i])
                    self.val_predictions_per_h[i].append(y_hat.argmax(dim=1))
                # Keep primary horizon tracking for early-stopping compatibility.
                self.val_targets.append(y[:, 0])
                self.val_predictions.append(y_hat_list[0].argmax(dim=1))
                batch_loss_mean = batch_loss
            else:
                y_hat = self.forward(x)
                if self.is_dfl:
                    if self.loss_type == "dfl_trading" and dfl_data is not None:
                        delta_mids, half_spreads, transaction_costs = dfl_data
                        dm = delta_mids[:, 0] if delta_mids.dim() == 2 else delta_mids
                        batch_loss, _ = self.dfl_loss(
                            y_hat, dm, half_spreads, transaction_cost=transaction_costs
                        )
                    else:
                        batch_loss, _ = self.dfl_loss(y_hat, y)
                    batch_loss_mean = batch_loss
                else:
                    batch_loss = self.loss(y_hat, y)
                    batch_loss_mean = torch.mean(batch_loss)
                self.val_targets.append(y)
                self.val_predictions.append(y_hat.argmax(dim=1))
            self.val_losses.append(batch_loss_mean.detach())
        return batch_loss_mean

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y, dfl_data = self._unpack_batch(batch)
        mid_prices = ((x[:, -1, 0] + x[:, -1, 2]) / 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)

        # Accumulate z-scored half-spread for spread-aware trading evaluation
        z_half_spread = ((x[:, -1, 0] - x[:, -1, 2]) / 2).cpu().numpy().flatten()
        self.test_z_half_spreads.append(z_half_spread)

        # Accumulate raw half-spread and transaction cost from DFL data (if available)
        if dfl_data is not None:
            _, hs, tc = dfl_data
            self.test_half_spreads.append(hs.cpu().numpy().flatten())
            self.test_transaction_costs.append(tc.cpu().numpy().flatten())

        if self.is_dpvn:
            return self._dpvn_test_step(x, y)

        if self.multi_horizon:
            if self.experiment_type == "TRAINING":
                with self.ema.average_parameters():
                    y_hat_list = self.forward(x, batch_idx)
                    batch_loss, per_h_ce = self._multi_horizon_loss(y_hat_list, y, dfl_data)
            else:
                y_hat_list = self.forward(x, batch_idx)
                batch_loss, per_h_ce = self._multi_horizon_loss(y_hat_list, y, dfl_data)

            for i, y_hat in enumerate(y_hat_list):
                self.test_targets_per_h[i].append(y[:, i])
                self.test_predictions_per_h[i].append(y_hat.argmax(dim=1))
                self.test_proba_per_h[i].append(torch.softmax(y_hat, dim=1))
                self.test_logits_per_h[i].append(y_hat.detach().cpu())
                self.test_losses_per_h[i].append(per_h_ce[i])

            # Keep h10 in primary accumulators for compatibility with downstream logic.
            self.test_targets.append(y[:, 0])
            self.test_predictions.append(y_hat_list[0].argmax(dim=1))
            batch_loss_mean = batch_loss
        else:
            def _compute_loss(y_hat_, y_):
                if self.is_dfl:
                    if self.loss_type == "dfl_trading" and dfl_data is not None:
                        dm, hs, tc = dfl_data
                        dm0 = dm[:, 0] if dm.dim() == 2 else dm
                        loss_, _ = self.dfl_loss(y_hat_, dm0, hs, transaction_cost=tc)
                    else:
                        loss_, _ = self.dfl_loss(y_hat_, y_)
                    return loss_
                return self.loss(y_hat_, y_)

            if self.experiment_type == "TRAINING":
                with self.ema.average_parameters():
                    y_hat = self.forward(x, batch_idx)
                    batch_loss = _compute_loss(y_hat, y)
                    softmax_proba = torch.softmax(y_hat, dim=1)
                    self.test_targets.append(y)
                    self.test_predictions.append(y_hat.argmax(dim=1))
                    self.test_proba.append(softmax_proba[:, 1])
                    self.test_proba_full.append(softmax_proba)
                    batch_loss_mean = batch_loss if self.is_dfl else torch.mean(batch_loss)
            else:
                y_hat = self.forward(x, batch_idx)
                batch_loss = _compute_loss(y_hat, y)
                softmax_proba = torch.softmax(y_hat, dim=1)
                self.test_targets.append(y)
                self.test_predictions.append(y_hat.argmax(dim=1))
                self.test_proba.append(softmax_proba[:, 1])
                self.test_proba_full.append(softmax_proba)
                batch_loss_mean = torch.mean(batch_loss)

        self.test_losses.append(
            batch_loss_mean.item() if isinstance(batch_loss_mean, torch.Tensor) else float(batch_loss_mean)
        )
        return batch_loss_mean

    # ------------------------------------------------------------------
    # Epoch-end callbacks
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        self.current_train_loss = loss
        self._section(f"Epoch {self.current_epoch} - Train/Validation Diagnostics")
        self._console(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        self._console(f"Train loss on epoch {self.current_epoch}: {loss}")
        if self.multi_horizon:
            ce_chunks = []
            sigma_chunks = []
            for i, h in enumerate(HORIZONS):
                if self.train_losses_per_h[i]:
                    mean_ce = float(torch.stack(self.train_losses_per_h[i]).mean())
                    ce_chunks.append(f"h{h}={mean_ce:.3f}")
                else:
                    ce_chunks.append(f"h{h}=n/a")
                sigma_h = float(torch.exp(0.5 * self.log_vars[i]).detach().cpu())
                sigma_chunks.append(f"h{h}={sigma_h:.2f}")
            self._console(f"Per-horizon train CE:  {'  '.join(ce_chunks)}")
            self._console(f"Uncertainty sigma_h:   {'  '.join(sigma_chunks)}")

    def on_validation_epoch_end(self) -> None:
        self.val_loss = float(sum(self.val_losses) / len(self.val_losses))
        self.val_losses = []

        # Model checkpointing only; LR scheduling is handled by Lightning scheduler integration.
        if self.val_loss < self.min_loss:
            self.min_loss = self.val_loss
            self.model_checkpointing(self.val_loss)

        # W&B logging: combined losses (train/val) + optional per-horizon train CE + σ_h
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)

        # Lightning monitoring (EarlyStopping watches val_loss)
        self.log("val_loss", self.val_loss)
        self._console(f"Validation loss on epoch {self.current_epoch}: {self.val_loss}")
        self._console("")

        # Classification metrics (compact — no verbose per-class report)
        targets = torch.cat(self.val_targets).cpu().numpy()
        predictions = torch.cat(self.val_predictions).cpu().numpy()
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])

        if self.multi_horizon and all(len(self.val_targets_per_h[i]) > 0 for i in range(len(HORIZONS))):
            val_metrics = []
            val_baselines = []
            for i in range(len(HORIZONS)):
                h_targets = torch.cat(self.val_targets_per_h[i]).cpu().numpy()
                h_predictions = torch.cat(self.val_predictions_per_h[i]).cpu().numpy()
                val_metrics.append(compute_metrics(h_targets, h_predictions))
                val_baselines.append(compute_baselines(h_targets))

            self._console("Validation summary by horizon")
            self._console(format_horizon_table(val_metrics, HORIZONS, val_baselines))
            self._console("")

            # Trading simulation per horizon
            val_mid = np.concatenate(self.val_mid_prices)
            val_z_hs = np.concatenate(self.val_z_half_spreads)
            trading_per_h = []
            for i in range(len(HORIZONS)):
                h_preds = torch.cat(self.val_predictions_per_h[i]).cpu().numpy()
                tm = compute_trading_metrics(
                    val_mid, h_preds, z_half_spreads=val_z_hs,
                )
                trading_per_h.append(tm)
            self._console("Validation trading simulation")
            self._console(format_trading_table(trading_per_h, HORIZONS))

            low_mcc_horizons = [f"h{HORIZONS[i]}" for i, metrics in enumerate(val_metrics) if metrics["mcc"] < 0.15]
            if low_mcc_horizons:
                self._console(
                    "Warning: near-random validation behaviour (MCC < 0.15) at " + ", ".join(low_mcc_horizons)
                )
        else:
            # Single-horizon: compact classification + trading line
            self._console(f"F1(macro)={class_report['macro avg']['f1-score']:.4f}, "
                          f"Acc={class_report['accuracy']:.4f}")
            val_mid = np.concatenate(self.val_mid_prices)
            val_z_hs = np.concatenate(self.val_z_half_spreads)
            val_save_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt)
            val_segments_path = os.path.join(val_save_dir, "val_product_boundaries.npy")
            val_segments = np.load(val_segments_path) if os.path.exists(val_segments_path) else None
            tm = compute_trading_metrics(
                val_mid, predictions, z_half_spreads=val_z_hs,
                segment_boundaries=val_segments,
            )
            if self.is_dpvn and self.val_v_values:
                v_values = torch.cat(self.val_v_values).float().numpy()
                preds_spread = _dpvn_spread_argmax_predictions(v_values, val_z_hs, val_segments)
                tm_spread = compute_trading_metrics(
                    val_mid, preds_spread, z_half_spreads=val_z_hs,
                    segment_boundaries=val_segments,
                )
                self._console(
                    f"[argmax]        PnL={tm['total_pnl']:.4f}  Sharpe={tm['sharpe']:.2e}  "
                    f"Trades={tm['n_trades']}"
                )
                self._console(
                    f"[spread_argmax] PnL={tm_spread['total_pnl']:.4f}  Sharpe={tm_spread['sharpe']:.2e}  "
                    f"Sortino={tm_spread['sortino']:.2e}  MaxDD={tm_spread['max_drawdown_pct']:.1f}%  "
                    f"WinRate={tm_spread['win_rate'] * 100:.1f}%  Trades={tm_spread['n_trades']}"
                )
                self.log("val_trading/pnl", tm["total_pnl"])
                self.log("val_trading/sharpe", tm["sharpe"])
                self.log("val_trading/pnl_spread", tm_spread["total_pnl"])
                self.log("val_trading/sharpe_spread", tm_spread["sharpe"])
            else:
                self._console(f"Trading: PnL={tm['total_pnl']:.4f}, Sharpe={tm['sharpe']:.2e}, "
                              f"Trades={tm['n_trades']}")

        self.val_targets = []
        self.val_predictions = []
        self.val_v_values = []
        self.val_mid_prices = []
        self.val_z_half_spreads = []
        if self.multi_horizon:
            self.val_targets_per_h = [[] for _ in HORIZONS]
            self.val_predictions_per_h = [[] for _ in HORIZONS]

        # Reset per-horizon train accumulators
        if self.multi_horizon:
            self.train_losses_per_h = [[] for _ in HORIZONS]

    def on_test_epoch_end(self) -> None:
        targets = torch.cat(self.test_targets).cpu().numpy()
        predictions = torch.cat(self.test_predictions).cpu().numpy()
        save_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "predictions"), predictions)
        np.save(os.path.join(save_dir, "targets"), targets)

        # Save mid-prices and spread data for trading evaluation
        mid_prices = np.concatenate(self.test_mid_prices) if self.test_mid_prices else np.array([])
        np.save(os.path.join(save_dir, "mid_prices"), mid_prices)

        z_half_spreads_all = np.concatenate(self.test_z_half_spreads) if self.test_z_half_spreads else None
        half_spreads_all = np.concatenate(self.test_half_spreads) if self.test_half_spreads else None
        transaction_costs_all = (
            np.concatenate(self.test_transaction_costs) if self.test_transaction_costs else None
        )
        if z_half_spreads_all is not None:
            np.save(os.path.join(save_dir, "z_half_spreads"), z_half_spreads_all)
        if half_spreads_all is not None:
            np.save(os.path.join(save_dir, "half_spreads"), half_spreads_all)
        if transaction_costs_all is not None:
            np.save(os.path.join(save_dir, "transaction_costs"), transaction_costs_all)

        # Save per-horizon logits for soft-position trading evaluation
        if self.multi_horizon:
            for i, h in enumerate(HORIZONS):
                if self.test_logits_per_h[i]:
                    np.save(os.path.join(save_dir, f"logits_h{h}"),
                            torch.cat(self.test_logits_per_h[i]).float().numpy())

        self._section("Test Diagnostics")

        if self.multi_horizon:
            metrics_per_h = []
            baselines_per_h = []
            targets_per_h = []
            predictions_per_h = []
            proba_per_h = []

            for i, h in enumerate(HORIZONS):
                h_targets = torch.cat(self.test_targets_per_h[i]).cpu().numpy()
                h_preds = torch.cat(self.test_predictions_per_h[i]).cpu().numpy()
                h_proba = torch.cat(self.test_proba_per_h[i]).cpu().numpy()
                h_metrics = compute_metrics(h_targets, h_preds)
                h_baselines = compute_baselines(h_targets)

                metrics_per_h.append(h_metrics)
                baselines_per_h.append(h_baselines)
                targets_per_h.append(h_targets)
                predictions_per_h.append(h_preds)
                proba_per_h.append(h_proba)

                h_report = classification_report(h_targets, h_preds, digits=4, output_dict=True)
                self.log(f"f1_score_h{h}", h_report["macro avg"]["f1-score"])

                # Save per-horizon arrays for trading evaluation
                np.save(os.path.join(save_dir, f"predictions_h{h}"), h_preds)
                np.save(os.path.join(save_dir, f"targets_h{h}"), h_targets)
                np.save(os.path.join(save_dir, f"probabilities_h{h}"), h_proba)

            self._console("Test summary by horizon")
            self._console(format_horizon_table(metrics_per_h, HORIZONS, baselines_per_h))

            self._console("")
            self._console("Prediction distribution by horizon")
            for i, h in enumerate(HORIZONS):
                self._console(f"[h{h}]")
                self._console(format_prediction_distribution(targets_per_h[i], predictions_per_h[i]))

            self._console("")
            self._console("Confidence diagnostics by horizon")
            for i, h in enumerate(HORIZONS):
                self._console(f"[h{h}]")
                self._console(format_confidence_stats(proba_per_h[i], targets_per_h[i], predictions_per_h[i]))

            saved_paths = plot_confusion_matrices(metrics_per_h, HORIZONS, save_dir)
            self._console("")
            self._console("Saved confusion matrix plots:")
            for path in saved_paths:
                self._console(path)

            macro_f1_values = [metrics["macro_f1"] for metrics in metrics_per_h]
            mcc_values = [metrics["mcc"] for metrics in metrics_per_h]
            best_idx = int(np.argmax(macro_f1_values))
            worst_idx = int(np.argmin(macro_f1_values))
            near_random = [f"h{HORIZONS[i]}" for i, value in enumerate(mcc_values) if value < 0.15]

            self._console("")
            self._console("Final summary")
            self._console(
                f"Best horizon: h{HORIZONS[best_idx]} "
                f"(F1(mac)={macro_f1_values[best_idx]:.4f}, MCC={mcc_values[best_idx]:.4f})"
            )
            self._console(
                f"Worst horizon: h{HORIZONS[worst_idx]} "
                f"(F1(mac)={macro_f1_values[worst_idx]:.4f}, MCC={mcc_values[worst_idx]:.4f})"
            )
            if near_random:
                self._console("Warning: near-random test performance (MCC < 0.15) at " + ", ".join(near_random))

            self.log("test_loss", sum(self.test_losses) / len(self.test_losses))

            # Reset per-horizon accumulators
            self.test_targets_per_h = [[] for _ in HORIZONS]
            self.test_predictions_per_h = [[] for _ in HORIZONS]
            self.test_proba_per_h = [[] for _ in HORIZONS]
            self.test_losses_per_h = [[] for _ in HORIZONS]
        else:
            class_report = classification_report(targets, predictions, digits=4, output_dict=True)
            self._console(f"F1(macro)={class_report['macro avg']['f1-score']:.4f}, "
                          f"Acc={class_report['accuracy']:.4f}")
            self.log("test_loss", sum(self.test_losses) / len(self.test_losses))
            self.log("f1_score", class_report["macro avg"]["f1-score"])
            self.log("accuracy", class_report["accuracy"])
            self.log("precision", class_report["macro avg"]["precision"])
            self.log("recall", class_report["macro avg"]["recall"])

        # --- Directional trading simulation summary ---
        self._section("Directional Trading Simulation")
        boundaries_path = os.path.join(save_dir, "product_boundaries.npy")
        segment_boundaries = np.load(boundaries_path) if os.path.exists(boundaries_path) else None

        if half_spreads_all is not None or z_half_spreads_all is not None:
            self._console("(spread-aware costs enabled)")
        if transaction_costs_all is not None:
            self._console("(transaction-cost fee enabled — layered on top of spread)")
        if self.is_dfl:
            self._console("(DFL: Gumbel-Softmax training, hard argmax evaluation)")

        # Derive z-space transaction cost once (shares std_price with the spread path)
        z_transaction_cost_all = None
        if (
            transaction_costs_all is not None
            and half_spreads_all is not None
            and z_half_spreads_all is not None
        ):
            std_price_for_tc = _infer_std_price(half_spreads_all, z_half_spreads_all)
            if std_price_for_tc is not None:
                z_transaction_cost_all = transaction_costs_all / std_price_for_tc

        if self.multi_horizon:
            trading_metrics_per_h = []
            for i, h in enumerate(HORIZONS):
                h_logits = (torch.cat(self.test_logits_per_h[i]).float().numpy()
                            if self.test_logits_per_h[i] else None)
                tm = compute_trading_metrics(
                    mid_prices, predictions_per_h[i],
                    probabilities=proba_per_h[i],
                    logits=h_logits,
                    half_spreads=half_spreads_all,
                    z_half_spreads=z_half_spreads_all,
                    transaction_costs=transaction_costs_all,
                    segment_boundaries=segment_boundaries,
                    use_soft_positions=False,
                )
                trading_metrics_per_h.append(tm)
                self.log(f"trading/sharpe_h{h}", tm["sharpe"])
                self.log(f"trading/pnl_h{h}", tm["total_pnl"])
            self._console(format_trading_table(trading_metrics_per_h, HORIZONS))

            # Per-product breakdown (if segment boundaries exist)
            if segment_boundaries is not None and len(segment_boundaries) > 1:
                starts = np.concatenate([[0], segment_boundaries[:-1]])
                ends = segment_boundaries
                for i, h in enumerate(HORIZONS):
                    h_logits = (torch.cat(self.test_logits_per_h[i]).float().numpy()
                                if self.test_logits_per_h[i] else None)
                    product_sharpes = []
                    for s, e in zip(starts, ends):
                        if e - s < 10:
                            continue
                        prod_tm = compute_trading_metrics(
                            mid_prices[s:e], predictions_per_h[i][s:e],
                            logits=h_logits[s:e] if h_logits is not None else None,
                            half_spreads=half_spreads_all[s:e] if half_spreads_all is not None else None,
                            z_half_spreads=z_half_spreads_all[s:e] if z_half_spreads_all is not None else None,
                            transaction_costs=transaction_costs_all[s:e] if transaction_costs_all is not None else None,
                            use_soft_positions=False,
                        )
                        product_sharpes.append(prod_tm["sharpe"])
                    if product_sharpes:
                        self._console(
                            f"h{h} per-product Sharpe: "
                            f"mean={np.mean(product_sharpes):.4f} "
                            f"+/- {np.std(product_sharpes):.4f} "
                            f"(n={len(product_sharpes)})"
                        )
        else:
            preds_for_trading = predictions
            decision_label = "argmax"
            if self.is_dpvn and self.test_v_values and z_half_spreads_all is not None:
                v_values = torch.cat(self.test_v_values).float().numpy()
                np.save(os.path.join(save_dir, "dpvn_values"), v_values)
                self._console(f"DPVN values saved: {v_values.shape} to {save_dir}/dpvn_values.npy")
                self.test_v_values = []
                preds_for_trading = _dpvn_spread_argmax_predictions(
                    v_values,
                    z_half_spreads_all,
                    segment_boundaries,
                    z_transaction_cost=z_transaction_cost_all,
                )
                np.save(os.path.join(save_dir, "predictions_spread_argmax"), preds_for_trading)
                decision_label = "spread_argmax"
                tm_argmax = compute_trading_metrics(
                    mid_prices, predictions,
                    half_spreads=half_spreads_all,
                    z_half_spreads=z_half_spreads_all,
                    transaction_costs=transaction_costs_all,
                    segment_boundaries=segment_boundaries,
                )
                self._console(
                    f"[argmax]       PnL(norm)={tm_argmax['total_pnl']:.4f}  "
                    f"Sharpe/step={tm_argmax['sharpe']:.2e}  Trades={tm_argmax['n_trades']}"
                )
            tm = compute_trading_metrics(
                mid_prices, preds_for_trading,
                half_spreads=half_spreads_all,
                z_half_spreads=z_half_spreads_all,
                transaction_costs=transaction_costs_all,
                segment_boundaries=segment_boundaries,
            )
            self._console(
                f"[{decision_label}] PnL(norm)={tm['total_pnl']:.4f}  Sharpe/step={tm['sharpe']:.2e}  "
                f"Sortino/step={tm['sortino']:.2e}  MaxDD={tm['max_drawdown_pct']:.1f}%  "
                f"WinRate={tm['win_rate'] * 100:.1f}%  Trades={tm['n_trades']}  "
                f"p-value={tm['p_value']:.4f}"
            )
            self.log("trading/sharpe", tm["sharpe"])
            self.log("trading/pnl", tm["total_pnl"])

        self._console("")
        self._console(f"Checkpoint dir: {save_dir}")
        self._console(f"  evaluate: python evaluate_trading.py --checkpoint_dir {save_dir}")

        # Reset accumulators
        self.test_targets = []
        self.test_half_spreads = []
        self.test_z_half_spreads = []
        self.test_transaction_costs = []
        if self.multi_horizon:
            self.test_logits_per_h = [[] for _ in HORIZONS]
        self.test_predictions = []
        self.test_losses = []
        self.test_mid_prices = []
        self.first_test = False

        if not self.multi_horizon and self.test_proba:
            test_proba = torch.cat(self.test_proba).cpu().numpy()
            precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
            self.plot_pr_curves(recall, precision, self.is_wandb)

        # Save full (N, 3) probabilities for confidence thresholding in trading eval
        if not self.multi_horizon and self.test_proba_full:
            full_proba = torch.cat(self.test_proba_full).cpu().numpy()
            np.save(os.path.join(save_dir, "probabilities"), full_proba)

        # DPVN values are saved earlier (alongside the spread_argmax trading log).
        if self.is_dpvn:
            self.test_v_values = []

        self.test_proba = []
        self.test_proba_full = []

    def on_fit_end(self) -> None:
        if self.total_train_time_s <= 0:
            return
        throughput_samples_per_s = self.total_train_samples / self.total_train_time_s
        throughput_steps_per_s = self.total_train_steps / self.total_train_time_s
        metrics = {
            "train/throughput_samples_per_s": float(throughput_samples_per_s),
            "train/throughput_steps_per_s": float(throughput_steps_per_s),
            "fit/total_train_time_s": float(self.total_train_time_s),
        }
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=self.global_step)
        if self.is_wandb and wandb.run is not None:
            wandb.log(metrics)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=eps, weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == "Lion":
            self.optimizer = Lion(self.parameters(), lr=self.lr)

        # TLOB benefits from validation-aware LR drops when the larger model plateaus early.
        if self.model_type == "TLOB":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=1,
                threshold=0.001,
                threshold_mode="abs",
                min_lr=1e-6,
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                },
            }

        if self.model_type == "DPVN":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=1,
                threshold=0.005,
                threshold_mode="rel",
                min_lr=1e-6,
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                },
            }

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")
        if self.multi_horizon:
            for h in HORIZONS:
                wandb.define_metric(f"f1_score_h{h}", summary="max")

    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses (and per-horizon σ_h in multi-horizon mode)."""
        if not self.is_wandb:
            return
        log_dict = {
            "losses": {"train": train_loss, "validation": val_loss},
            "epoch": self.global_step,
        }
        if self.multi_horizon:
            # Per-horizon training CE and learned uncertainty σ_h
            for i, h in enumerate(HORIZONS):
                if self.train_losses_per_h[i]:
                    mean_val = float(torch.stack(self.train_losses_per_h[i]).mean())
                    log_dict[f"losses/train_h{h}"] = mean_val
                sigma_h = float(torch.exp(0.5 * self.log_vars[i]).detach().cpu())
                log_dict[f"uncertainty/sigma_h{h}"] = sigma_h
        wandb.log(log_dict)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def model_checkpointing(self, loss):
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = "val_loss=" + str(round(loss, 3)) + "_epoch=" + str(self.current_epoch) + ".pt"
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)

        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt)

            # ONNX export — single-head only (multi-horizon output is a list, not yet ONNX-friendly)
            # Skip for DPVN family (DPVN/DAVN) — value-regression models that don't need ONNX.
            if not self.multi_horizon and self.model_type not in ("DPVN", "DAVN"):
                onnx_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "onnx")
                os.makedirs(onnx_dir, exist_ok=True)
                onnx_filename = "val_loss=" + str(round(loss, 3)) + "_epoch=" + str(self.current_epoch) + ".onnx"
                onnx_path = os.path.join(onnx_dir, onnx_filename)
                dummy_input = torch.randn(1, self.seq_size, self.num_features, device=self.device)
                export_model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                previous_fast_attention_state = None
                try:
                    if hasattr(export_model, "set_fast_attention"):
                        previous_fast_attention_state = export_model.use_fast_attention
                        export_model.set_fast_attention(False)
                    onnx_logger = logging.getLogger("torch.onnx")
                    onnx_schemas_logger = logging.getLogger("torch.onnx._internal.exporter._schemas")
                    previous_onnx_logger_level = onnx_logger.level
                    previous_onnx_schemas_logger_level = onnx_schemas_logger.level
                    onnx_logger.setLevel(logging.ERROR)
                    onnx_schemas_logger.setLevel(logging.ERROR)
                    try:
                        with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                            warnings.simplefilter("ignore")
                            torch.onnx.export(
                                export_model,
                                dummy_input,
                                onnx_path,
                                dynamo=True,
                                export_params=True,
                                opset_version=25,
                                do_constant_folding=True,
                                input_names=["input"],
                                output_names=["output"],
                                dynamic_shapes={"input": {0: torch.export.Dim("batch_size")}},
                            )
                    finally:
                        onnx_logger.setLevel(previous_onnx_logger_level)
                        onnx_schemas_logger.setLevel(previous_onnx_schemas_logger_level)
                except Exception as e:
                    self._console(f"Failed to export ONNX model: {e}")
                finally:
                    if hasattr(export_model, "set_fast_attention") and previous_fast_attention_state is not None:
                        export_model.set_fast_attention(previous_fast_attention_state)

        self.last_path_ckpt = path_ckpt

    # ------------------------------------------------------------------
    # PR Curve
    # ------------------------------------------------------------------
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label="Precision-Recall", color="black")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        if is_wandb:
            tag = "multi_horizon" if self.multi_horizon else self.dataset_type
            wandb.log({f"precision_recall_curve_{tag}": wandb.Image(plt)})
        plt.savefig(
            cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" + f"precision_recall_curve_{self.dataset_type}.svg"
        )
        plt.close()


_DPVN_ACTION_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float64)


def _dpvn_spread_argmax_predictions(
    v_values: np.ndarray,
    z_half_spread: np.ndarray,
    segment_boundaries: np.ndarray | None = None,
    z_transaction_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Sequential spread-aware argmax decision rule for DPVN.

    Mirrors evaluate_trading._spread_argmax_predictions so the training-time
    trading log reflects the architecture's intended inference path.
    Decision rule: a* = argmax_a [V(t,a) - |a - pos_prev| * (z_hs[t] + z_tc[t])].
    """
    n = v_values.shape[0]
    z_hs = np.abs(np.asarray(z_half_spread, dtype=np.float64))
    if z_transaction_cost is None:
        z_tc = np.zeros(n, dtype=np.float64)
    else:
        z_tc = np.abs(np.asarray(z_transaction_cost, dtype=np.float64))
        if len(z_tc) != n:
            raise ValueError(
                f"z_transaction_cost length {len(z_tc)} must match v_values length {n}"
            )
    per_unit = z_hs + z_tc
    seg_starts = set(int(b) for b in segment_boundaries) if segment_boundaries is not None else set()
    positions = np.zeros(n, dtype=np.float64)
    pos_prev = 0.0
    for t in range(n):
        if t in seg_starts:
            pos_prev = 0.0
        best_idx = 0
        best_score = -np.inf
        for a_idx in range(3):
            a = _DPVN_ACTION_VALUES[a_idx]
            score = v_values[t, a_idx] - abs(a - pos_prev) * per_unit[t]
            if score > best_score:
                best_score = score
                best_idx = a_idx
        positions[t] = _DPVN_ACTION_VALUES[best_idx]
        pos_prev = positions[t]
    predictions = np.ones(n, dtype=np.int64)
    predictions[positions > 0.5] = 0
    predictions[positions < -0.5] = 2
    return predictions


def compute_most_attended(att_feature):
    """att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features)"""
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)
    """ att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) """
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                most_frequent_indices[layer, head, seq] = most_frequent_index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values
