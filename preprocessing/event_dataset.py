"""
Dataset classes for event-based LOB models.

EventSnapshotDataset loads both LOB snapshots (.npy) and event data (events.npz)
for a single product, returning aligned (snapshot_window, event_window, event_mask,
labels) tuples.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils import data



class EventSnapshotDataset(data.Dataset):
    """Dataset returning aligned snapshot windows + event windows for one product.

    Each sample i returns:
        snapshot_window : (seq_size, num_snapshot_features) float32
        event_window    : (seq_size, max_events, N_EVENT_FEATURES) float32
        event_mask      : (seq_size, max_events) bool
        labels          : (num_horizons,) int64

    The snapshot and event data must be pre-aligned: row i in the .npy and
    row i in events.npz correspond to the same time window.
    """

    def __init__(
        self,
        snapshot_input: torch.Tensor | np.ndarray,
        event_features: np.ndarray,
        event_mask: np.ndarray,
        labels: torch.Tensor | np.ndarray,
        seq_size: int,
    ):
        self.seq_size = seq_size

        # Snapshots
        if isinstance(snapshot_input, np.ndarray):
            self.snapshots = torch.from_numpy(snapshot_input).float()
        else:
            self.snapshots = snapshot_input.float()

        # Events: (N, max_events, 7) and (N, max_events) bool
        if isinstance(event_features, np.ndarray):
            self.event_features = torch.from_numpy(event_features).float()
        else:
            self.event_features = event_features.float()

        if isinstance(event_mask, np.ndarray):
            self.event_mask = torch.from_numpy(event_mask).bool()
        else:
            self.event_mask = event_mask.bool()

        # Labels
        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

        # Usable length: need seq_size consecutive rows
        self.length = min(
            self.labels.shape[0],
            self.snapshots.shape[0] - seq_size + 1,
        )

        # For compatibility with DataModule.pin_memory check
        self.data = self.snapshots

    @property
    def x(self):
        """Alias for snapshot data, compatible with Dataset.x"""
        return self.snapshots

    @property
    def y(self):
        """Alias for labels, compatible with Dataset.y (returns h10 for multi-horizon)."""
        if self.labels.ndim == 2:
            return self.labels[:, 0]
        return self.labels

    @property
    def y_multi(self):
        """Alias for multi-horizon labels, compatible with MultiHorizonDataset.y_multi."""
        if self.labels.ndim == 2:
            return self.labels
        return None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        snap_window = self.snapshots[i : i + self.seq_size]  # (seq_size, F)
        event_window = self.event_features[i : i + self.seq_size]  # (seq_size, E, 7)
        mask_window = self.event_mask[i : i + self.seq_size]  # (seq_size, E)
        label = self.labels[i]  # (num_horizons,) or scalar
        return snap_window, event_window, mask_window, label


class EventOnlyDataset(data.Dataset):
    """Dataset returning only event windows (no snapshots) for one product.

    For PerceiverLOB and event-only ablation.
    """

    def __init__(
        self,
        event_features: np.ndarray,
        event_mask: np.ndarray,
        labels: torch.Tensor | np.ndarray,
        seq_size: int,
    ):
        self.seq_size = seq_size

        if isinstance(event_features, np.ndarray):
            self.event_features = torch.from_numpy(event_features).float()
        else:
            self.event_features = event_features.float()

        if isinstance(event_mask, np.ndarray):
            self.event_mask = torch.from_numpy(event_mask).bool()
        else:
            self.event_mask = event_mask.bool()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

        self.length = min(
            self.labels.shape[0],
            self.event_features.shape[0] - seq_size + 1,
        )

        # For DataModule compatibility
        self.data = self.event_features

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        event_window = self.event_features[i : i + self.seq_size]
        mask_window = self.event_mask[i : i + self.seq_size]
        label = self.labels[i]
        return event_window, mask_window, label


def load_events_for_product(product_dir: str) -> dict[str, np.ndarray] | None:
    """Load events.npz from a product directory.

    Returns dict with keys 'event_features', 'event_mask', 'n_events',
    or None if events.npz does not exist.
    """
    import os

    events_path = os.path.join(product_dir, "events.npz")
    if not os.path.exists(events_path):
        return None

    data = np.load(events_path)
    return {
        "event_features": data["event_features"],
        "event_mask": data["event_mask"],
        "n_events": data["n_events"],
    }
