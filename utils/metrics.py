import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


CLASS_NAMES = {
    0: "up",
    1: "stat",
    2: "down",
}


def _to_numpy(array_like: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _resolve_labels(targets: np.ndarray, predictions: np.ndarray | None = None) -> np.ndarray:
    if predictions is None:
        merged = targets
    else:
        merged = np.concatenate([targets, predictions])

    if merged.size == 0:
        return np.array([0, 1, 2], dtype=np.int64)

    merged = merged.astype(np.int64, copy=False)
    if merged.min() >= 0 and merged.max() <= 2:
        return np.array([0, 1, 2], dtype=np.int64)
    return np.unique(merged)


def _format_lift(score: float, baseline: float) -> str:
    if baseline <= 0:
        return "   n/a   "
    lift_pct = ((score / baseline) - 1.0) * 100.0
    return f"{lift_pct:+7.1f}%"


def _format_int_space(value: int) -> str:
    return f"{int(value):,}"


def compute_metrics(targets, predictions) -> dict[str, Any]:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)
    labels = _resolve_labels(targets, predictions)

    conf = confusion_matrix(targets, predictions, labels=labels)
    per_class_precision = precision_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_recall = recall_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        targets,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    try:
        mcc = float(matthews_corrcoef(targets, predictions))
    except ValueError:
        mcc = 0.0

    try:
        kappa = float(cohen_kappa_score(targets, predictions, labels=labels))
    except ValueError:
        kappa = 0.0

    return {
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(targets, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        "mcc": mcc,
        "kappa": kappa,
        "accuracy": float(accuracy_score(targets, predictions)),
        "per_class_f1": {int(lbl): float(val) for lbl, val in zip(labels, per_class_f1)},
        "per_class_precision": {int(lbl): float(val) for lbl, val in zip(labels, per_class_precision)},
        "per_class_recall": {int(lbl): float(val) for lbl, val in zip(labels, per_class_recall)},
        "confusion_matrix": conf,
        "support": conf.sum(axis=1),
        "labels": labels,
    }


def compute_baselines(targets) -> dict[str, Any]:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    labels = _resolve_labels(targets)
    counts = np.array([(targets == lbl).sum() for lbl in labels], dtype=np.int64)
    total = max(int(counts.sum()), 1)
    distribution = counts / total

    majority_idx = int(np.argmax(counts))
    majority_class = int(labels[majority_idx])
    majority_predictions = np.full_like(targets, fill_value=majority_class)

    majority_f1 = float(
        f1_score(
            targets,
            majority_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )
    random_f1 = 1.0 / float(len(labels)) if len(labels) > 0 else 0.0

    return {
        "majority_class": majority_class,
        "majority_f1": majority_f1,
        "random_f1": random_f1,
        "class_distribution": {int(lbl): float(pct) for lbl, pct in zip(labels, distribution)},
    }


def format_horizon_table(metrics_list, horizons, baselines) -> str:
    header = (
        "Horizon | F1(mac) | F1(wtd) |   MCC   |  Kappa  |   Acc   |     N     | vs Random | vs Majority"
    )
    separator = (
        "--------|---------|---------|---------|---------|---------|-----------|-----------|------------"
    )
    lines = [header, separator]

    for idx, horizon in enumerate(horizons):
        metrics = metrics_list[idx]
        baseline = baselines[idx]
        n_samples = int(np.sum(metrics["support"]))
        vs_random = _format_lift(metrics["macro_f1"], baseline["random_f1"])
        vs_majority = _format_lift(metrics["macro_f1"], baseline["majority_f1"])

        lines.append(
            f"h{horizon:<6}|"
            f" {metrics['macro_f1']:>7.4f} |"
            f" {metrics['weighted_f1']:>7.4f} |"
            f" {metrics['mcc']:>7.4f} |"
            f" {metrics['kappa']:>7.4f} |"
            f" {metrics['accuracy']:>7.4f} |"
            f" {_format_int_space(n_samples):>9} |"
            f" {vs_random:>9} |"
            f" {vs_majority:>10}"
        )

    return "\n".join(lines)


def format_prediction_distribution(targets, predictions) -> str:
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)
    labels = _resolve_labels(targets, predictions)

    target_total = max(int(targets.shape[0]), 1)
    pred_total = max(int(predictions.shape[0]), 1)

    lines = [
        "Class  | Actual % | Pred %  | Actual n | Pred n",
        "-------|----------|---------|----------|-------",
    ]

    for label in labels:
        name = CLASS_NAMES.get(int(label), f"class{int(label)}")
        actual_n = int((targets == label).sum())
        pred_n = int((predictions == label).sum())
        lines.append(
            f"{name:<6} |"
            f" {100.0 * actual_n / target_total:>7.2f}% |"
            f" {100.0 * pred_n / pred_total:>6.2f}% |"
            f" {_format_int_space(actual_n):>8} |"
            f" {_format_int_space(pred_n):>6}"
        )

    return "\n".join(lines)


def format_confidence_stats(probas, targets, predictions) -> str:
    probas = _to_numpy(probas)
    targets = _to_numpy(targets).astype(np.int64, copy=False)
    predictions = _to_numpy(predictions).astype(np.int64, copy=False)

    if probas.ndim == 2:
        confidence = probas.max(axis=1)
    elif probas.ndim == 1:
        confidence = probas
    else:
        raise ValueError(f"Expected 1D or 2D probabilities, got shape={probas.shape}")

    n = min(int(confidence.shape[0]), int(targets.shape[0]), int(predictions.shape[0]))
    if n == 0:
        return "No confidence stats available (empty inputs)."

    confidence = confidence[:n]
    targets = targets[:n]
    predictions = predictions[:n]

    correct_mask = predictions == targets
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]

    mean_correct = float(correct_conf.mean()) if correct_conf.size > 0 else float("nan")
    mean_incorrect = float(incorrect_conf.mean()) if incorrect_conf.size > 0 else float("nan")
    overconfident_errors = (
        float((incorrect_conf >= 0.80).mean()) if incorrect_conf.size > 0 else float("nan")
    )

    def _fmt(value: float) -> str:
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"

    lines = [
        f"Samples: {_format_int_space(n)}",
        f"Mean confidence (correct):   {_fmt(mean_correct)}",
        f"Mean confidence (incorrect): {_fmt(mean_incorrect)}",
        f"Overconfident errors (>=0.80): {_fmt(overconfident_errors)}",
    ]

    return "\n".join(lines)


def plot_confusion_matrices(metrics_list, horizons, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    n = len(metrics_list)
    n_cols = 2 if n > 1 else 1
    n_rows = int(math.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.5 * n_rows), dpi=140)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, horizon in enumerate(horizons):
        metrics = metrics_list[idx]
        cm = np.asarray(metrics["confusion_matrix"])
        labels = [int(v) for v in np.asarray(metrics["labels"]).tolist()]

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"h{horizon} confusion matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = int(cm[row, col])
                color = "white" if value > threshold else "black"
                ax.text(col, row, _format_int_space(value), ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        single_path = os.path.join(save_dir, f"confusion_matrix_h{horizon}.png")
        single_fig, single_ax = plt.subplots(figsize=(6, 5), dpi=140)
        single_im = single_ax.imshow(cm, cmap="Blues")
        single_ax.set_title(f"h{horizon} confusion matrix")
        single_ax.set_xlabel("Predicted")
        single_ax.set_ylabel("Actual")
        single_ax.set_xticks(range(len(labels)))
        single_ax.set_yticks(range(len(labels)))
        single_ax.set_xticklabels(labels)
        single_ax.set_yticklabels(labels)

        single_threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                value = int(cm[row, col])
                color = "white" if value > single_threshold else "black"
                single_ax.text(col, row, _format_int_space(value), ha="center", va="center", color=color, fontsize=10)
        single_fig.colorbar(single_im, ax=single_ax, fraction=0.046, pad=0.04)
        single_fig.tight_layout()
        single_fig.savefig(single_path, bbox_inches="tight")
        plt.close(single_fig)
        saved_paths.append(single_path)

    for idx in range(len(horizons), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    combined_path = os.path.join(save_dir, "confusion_matrices.png")
    fig.savefig(combined_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(combined_path)

    return saved_paths
