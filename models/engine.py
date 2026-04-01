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
    compute_baselines,
    compute_metrics,
    format_confidence_stats,
    format_horizon_table,
    format_prediction_distribution,
    plot_confusion_matrices,
)
import constants as cst
from scipy.stats import mode
from models.losses import FocalLoss

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

        if self.loss_type == "cross_entropy":
            if self.multi_horizon:
                # Per-horizon weights applied manually in _multi_horizon_loss
                self.loss_function = nn.CrossEntropyLoss()
            else:
                self.loss_function = nn.CrossEntropyLoss(
                    weight=self.class_weights if self.class_weights is not None else None
                )
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
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.min_loss = np.inf
        if self.loss_type == "cross_entropy":
            self.save_hyperparameters(ignore=["focal_gamma", "ordinal_smoothing"])
        else:
            self.save_hyperparameters()
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
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
    def forward(self, x, batch_idx=None):
        return self.model(x)

    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)

    def _multi_horizon_loss(self, y_hat_list, y_multi):
        """Homoscedastic uncertainty-weighted multi-task loss (Kendall et al. 2018).

        L = Σ_h [ CE(ŷ_h, y_h) / (2σ_h²) + log σ_h ]
        where log σ_h² = self.log_vars[h] (log-variance parameterisation).
        """
        horizon_losses = []
        for horizon_index, y_hat_h in enumerate(y_hat_list):
            if self.loss_type == "cross_entropy":
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
            else:
                ce_h = self.horizon_losses[horizon_index](y_hat_h, y_multi[:, horizon_index])
            horizon_losses.append(ce_h)

        ce_per_h = torch.stack(horizon_losses)  # (H,)
        # Uncertainty weighting (vectorised)
        sigma2 = torch.exp(self.log_vars)  # (H,)
        total_loss = (ce_per_h / (2.0 * sigma2) + 0.5 * self.log_vars).sum()
        return total_loss, ce_per_h.detach()  # scalar, (H,) tensor

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.multi_horizon:
            y_hat_list = self.forward(x)
            batch_loss, per_h_ce = self._multi_horizon_loss(y_hat_list, y)
            # per_h_ce is an (H,) tensor — store each horizon's CE as a scalar tensor
            for i, ce_val in enumerate(per_h_ce.unbind()):
                self.train_losses_per_h[i].append(ce_val)
        else:
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)

        batch_loss_mean = batch_loss if self.multi_horizon else torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.detach())
        self.epoch_iteration_count += 1
        self.epoch_sample_count += int(y.shape[0])
        self.ema.update()
        return batch_loss_mean

    def on_train_epoch_start(self) -> None:
        self.train_epoch_start_time = time.perf_counter()
        self.epoch_sample_count = 0
        self.epoch_iteration_count = 0

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
            f"Epoch {self.current_epoch} throughput - samples/s: {samples_per_sec:.2f}, it/s: {it_per_sec:.2f}"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with self.ema.average_parameters():
            if self.multi_horizon:
                y_hat_list = self.forward(x)
                batch_loss, _ = self._multi_horizon_loss(y_hat_list, y)
                for i, y_hat in enumerate(y_hat_list):
                    self.val_targets_per_h[i].append(y[:, i])
                    self.val_predictions_per_h[i].append(y_hat.argmax(dim=1))
                # Keep primary horizon tracking for early-stopping compatibility.
                self.val_targets.append(y[:, 0])
                self.val_predictions.append(y_hat_list[0].argmax(dim=1))
                batch_loss_mean = batch_loss
            else:
                y_hat = self.forward(x)
                batch_loss = self.loss(y_hat, y)
                self.val_targets.append(y)
                self.val_predictions.append(y_hat.argmax(dim=1))
                batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.detach())
        return batch_loss_mean

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        x, y = batch
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)

        if self.multi_horizon:
            if self.experiment_type == "TRAINING":
                with self.ema.average_parameters():
                    y_hat_list = self.forward(x, batch_idx)
                    batch_loss, per_h_ce = self._multi_horizon_loss(y_hat_list, y)
            else:
                y_hat_list = self.forward(x, batch_idx)
                batch_loss, per_h_ce = self._multi_horizon_loss(y_hat_list, y)

            for i, y_hat in enumerate(y_hat_list):
                self.test_targets_per_h[i].append(y[:, i])
                self.test_predictions_per_h[i].append(y_hat.argmax(dim=1))
                self.test_proba_per_h[i].append(torch.softmax(y_hat, dim=1))
                self.test_losses_per_h[i].append(per_h_ce[i])

            # Keep h10 in primary accumulators for compatibility with downstream logic.
            self.test_targets.append(y[:, 0])
            self.test_predictions.append(y_hat_list[0].argmax(dim=1))
            batch_loss_mean = batch_loss
        else:
            if self.experiment_type == "TRAINING":
                with self.ema.average_parameters():
                    y_hat = self.forward(x, batch_idx)
                    batch_loss = self.loss(y_hat, y)
                    self.test_targets.append(y)
                    self.test_predictions.append(y_hat.argmax(dim=1))
                    self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1])
                    batch_loss_mean = torch.mean(batch_loss)
            else:
                y_hat = self.forward(x, batch_idx)
                batch_loss = self.loss(y_hat, y)
                self.test_targets.append(y)
                self.test_predictions.append(y_hat.argmax(dim=1))
                self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1])
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

        # Classification report on primary horizon (always h10 for multi, full for single)
        targets = torch.cat(self.val_targets).cpu().numpy()
        predictions = torch.cat(self.val_predictions).cpu().numpy()
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        if self.multi_horizon:
            self._console("Primary horizon (h10) classification report")
        self._console(classification_report(targets, predictions, digits=4))
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

            low_mcc_horizons = [f"h{HORIZONS[i]}" for i, metrics in enumerate(val_metrics) if metrics["mcc"] < 0.15]
            if low_mcc_horizons:
                self._console(
                    "Warning: near-random validation behaviour (MCC < 0.15) at " + ", ".join(low_mcc_horizons)
                )

        self.val_targets = []
        self.val_predictions = []
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
        predictions_path = os.path.join(save_dir, "predictions")
        np.save(predictions_path, predictions)
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

            self._console("")
            self._console("Detailed classification reports")
            for i, h in enumerate(HORIZONS):
                self._console(f"--- Horizon h={h} ---")
                self._console(classification_report(targets_per_h[i], predictions_per_h[i], digits=4))

            self.log("test_loss", sum(self.test_losses) / len(self.test_losses))

            # Reset per-horizon accumulators
            self.test_targets_per_h = [[] for _ in HORIZONS]
            self.test_predictions_per_h = [[] for _ in HORIZONS]
            self.test_proba_per_h = [[] for _ in HORIZONS]
            self.test_losses_per_h = [[] for _ in HORIZONS]
        else:
            class_report = classification_report(targets, predictions, digits=4, output_dict=True)
            self._console(classification_report(targets, predictions, digits=4))
            self.log("test_loss", sum(self.test_losses) / len(self.test_losses))
            self.log("f1_score", class_report["macro avg"]["f1-score"])
            self.log("accuracy", class_report["accuracy"])
            self.log("precision", class_report["macro avg"]["precision"])
            self.log("recall", class_report["macro avg"]["recall"])

        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []
        self.first_test = False

        if not self.multi_horizon and self.test_proba:
            test_proba = torch.cat(self.test_proba).cpu().numpy()
            precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
            self.plot_pr_curves(recall, precision, self.is_wandb)

        self.test_proba = []

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
            if not self.multi_horizon:
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
