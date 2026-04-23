import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import constants as cst
from torch.utils import data

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch.

    Args:
        x: Input tensor/array (N, num_features).
        y: Label tensor/array (N,) — single horizon.
        seq_size: Sliding window length.
        q_targets: Optional (N, 3) array of DP-distilled Q targets for DPVN.
                   When present, __getitem__ also returns the target aligned
                   with the last-step index of the window.
        dfl_data:  Optional (half_spreads (N,), transaction_costs (N,)) pair of
                   raw price-unit arrays. When present, __getitem__ yields them
                   alongside each sample so eval-time trading metrics can see
                   the per-step cost without needing the full DFL batch tuple.
    """

    def __init__(self, x, y, seq_size, q_targets=None, dfl_data=None):
        self.seq_size = seq_size
        self.x = x
        self.y = y
        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(x).float()
        if type(self.y) == np.ndarray:
            self.y = torch.from_numpy(y).long()
        self.length = min(y.shape[0], self.x.shape[0] - seq_size + 1)
        self.data = self.x
        self.has_q_targets = q_targets is not None
        if self.has_q_targets:
            qt = q_targets if isinstance(q_targets, torch.Tensor) else torch.from_numpy(q_targets)
            self.q_targets = qt.float()
        self.has_dfl = dfl_data is not None
        if self.has_dfl:
            if len(dfl_data) != 2:
                raise ValueError(
                    f"Dataset.dfl_data must be (half_spreads, transaction_costs); got length {len(dfl_data)}"
                )
            self.half_spreads = dfl_data[0] if isinstance(dfl_data[0], torch.Tensor) else torch.from_numpy(dfl_data[0]).float()
            self.transaction_costs = dfl_data[1] if isinstance(dfl_data[1], torch.Tensor) else torch.from_numpy(dfl_data[1]).float()

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]
        if self.has_q_targets and self.has_dfl:
            return input, self.y[i], self.q_targets[i], self.half_spreads[i], self.transaction_costs[i]
        if self.has_q_targets:
            return input, self.y[i], self.q_targets[i]
        if self.has_dfl:
            return input, self.y[i], self.half_spreads[i], self.transaction_costs[i]
        return input, self.y[i]


class MultiHorizonDataset(data.Dataset):
    """Dataset that returns labels for all horizons simultaneously.

    Args:
        x:       Input tensor of shape (N, num_features).
        y_multi: Label tensor of shape (N, num_horizons), ordered h={10,20,50,100}.
        seq_size: Sequence window length.
        dfl_data: Optional (delta_mids, half_spreads, transaction_costs) for DFL trading loss.
                  delta_mids:        (N, num_horizons) raw mid-price changes.
                  half_spreads:      (N,) raw half bid-ask spread per timestep.
                  transaction_costs: (N,) exchange fee per unit |Δpos| in raw units.
                                     Required (pass zeros to disable); preprocessors always
                                     emit it when the .npy includes the DFL columns.
    """
    def __init__(self, x, y_multi, seq_size, dfl_data=None):
        self.seq_size = seq_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y_multi = y_multi if isinstance(y_multi, torch.Tensor) else torch.from_numpy(y_multi).long()
        self.length = min(self.y_multi.shape[0], self.x.shape[0] - seq_size + 1)
        self.data = self.x
        self.has_dfl = dfl_data is not None
        if self.has_dfl:
            if len(dfl_data) != 3:
                raise ValueError(
                    f"dfl_data must be a 3-tuple (delta_mids, half_spreads, transaction_costs); got {len(dfl_data)}"
                )
            self.delta_mids = dfl_data[0] if isinstance(dfl_data[0], torch.Tensor) else torch.from_numpy(dfl_data[0]).float()
            self.half_spreads = dfl_data[1] if isinstance(dfl_data[1], torch.Tensor) else torch.from_numpy(dfl_data[1]).float()
            self.transaction_costs = dfl_data[2] if isinstance(dfl_data[2], torch.Tensor) else torch.from_numpy(dfl_data[2]).float()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x_window = self.x[i:i + self.seq_size, :]   # (seq_size, num_features)
        y_all = self.y_multi[i]                       # (num_horizons,)
        if self.has_dfl:
            return x_window, y_all, self.delta_mids[i], self.half_spreads[i], self.transaction_costs[i]
        return x_window, y_all


class DataModule(pl.LightningDataModule):
    def   __init__(self, train_set, val_set, batch_size, test_batch_size,  is_shuffle_train=True, test_set=None, num_workers=16):
        super().__init__()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.is_shuffle_train = is_shuffle_train
        if hasattr(train_set, 'data'):
            self.pin_memory = train_set.data.device.type != cst.DEVICE
        else:
            # ConcatDataset: data lives on CPU, pin when using GPU
            self.pin_memory = cst.DEVICE != "cpu"
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.is_shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.test_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context='spawn' if self.num_workers > 0 else None,
        )