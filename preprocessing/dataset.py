import torch
from torch.utils import data
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
import constants as cst
from torch.utils import data

class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, x, y, seq_size):
        """Initialization""" 
        self.seq_size = seq_size
        self.x = x
        self.y = y
        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(x).float()
        if type(self.y) == np.ndarray:
            self.y = torch.from_numpy(y).long()
        self.length = min(y.shape[0], self.x.shape[0] - seq_size + 1)
        self.data = self.x

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, i):
        input = self.x[i:i+self.seq_size, :]
        return input, self.y[i]


class MultiHorizonDataset(data.Dataset):
    """Dataset that returns labels for all horizons simultaneously.

    Args:
        x:       Input tensor of shape (N, num_features).
        y_multi: Label tensor of shape (N, num_horizons), ordered h={10,20,50,100}.
        seq_size: Sequence window length.
        dfl_data: Optional (delta_mids, half_spreads) for DFL trading loss.
                  delta_mids: (N, num_horizons) raw mid-price changes.
                  half_spreads: (N,) half bid-ask spread at each timestep.
        dp_data:  Optional (dp_trade, dp_prev_pos) for CPT trade filter supervision.
                  dp_trade: (N,) binary {0=hold, 1=trade} from DP optimal.
                  dp_prev_pos: (N,) DP previous position {-1, 0, +1}.
    """
    def __init__(self, x, y_multi, seq_size, dfl_data=None, dp_data=None):
        self.seq_size = seq_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y_multi = y_multi if isinstance(y_multi, torch.Tensor) else torch.from_numpy(y_multi).long()
        self.length = min(self.y_multi.shape[0], self.x.shape[0] - seq_size + 1)
        self.data = self.x
        self.has_dfl = dfl_data is not None
        if self.has_dfl:
            self.delta_mids = dfl_data[0] if isinstance(dfl_data[0], torch.Tensor) else torch.from_numpy(dfl_data[0]).float()
            self.half_spreads = dfl_data[1] if isinstance(dfl_data[1], torch.Tensor) else torch.from_numpy(dfl_data[1]).float()
        self.has_dp = dp_data is not None
        if self.has_dp:
            self.dp_trade = dp_data[0] if isinstance(dp_data[0], torch.Tensor) else torch.from_numpy(dp_data[0]).float()
            self.dp_prev_pos = dp_data[1] if isinstance(dp_data[1], torch.Tensor) else torch.from_numpy(dp_data[1]).float()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x_window = self.x[i:i + self.seq_size, :]   # (seq_size, num_features)
        y_all = self.y_multi[i]                       # (num_horizons,)
        if self.has_dfl and self.has_dp:
            return x_window, y_all, self.delta_mids[i], self.half_spreads[i], self.dp_trade[i], self.dp_prev_pos[i]
        if self.has_dfl:
            return x_window, y_all, self.delta_mids[i], self.half_spreads[i]
        return x_window, y_all


class SequentialChunkDataset(data.Dataset):
    """Dataset yielding consecutive T-step chunks for sequential training.

    Unlike standard Dataset which yields random individual samples, this
    yields (chunk_size, seq_size, features) blocks of consecutive LOB snapshots
    so that position state can carry forward within each chunk.

    Respects segment boundaries: chunks never cross product/segment borders.

    Args:
        x: Input tensor of shape (N, num_features).
        y_multi: Labels of shape (N, num_horizons).
        seq_size: Window length for each LOB snapshot (standard TLOB input).
        chunk_size: Number of consecutive timesteps per chunk (T).
        dfl_data: Optional (delta_mids, half_spreads) for PnL computation.
        segment_boundaries: Indices where segments change (product switches).
    """

    def __init__(
        self,
        x,
        y_multi,
        seq_size: int,
        chunk_size: int = 64,
        dfl_data=None,
        segment_boundaries=None,
    ):
        self.seq_size = seq_size
        self.chunk_size = chunk_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y_multi = y_multi if isinstance(y_multi, torch.Tensor) else torch.from_numpy(y_multi).long()

        self.has_dfl = dfl_data is not None
        if self.has_dfl:
            self.delta_mids = dfl_data[0] if isinstance(dfl_data[0], torch.Tensor) else torch.from_numpy(dfl_data[0]).float()
            self.half_spreads = dfl_data[1] if isinstance(dfl_data[1], torch.Tensor) else torch.from_numpy(dfl_data[1]).float()

        # Total valid samples (accounting for seq_size lookback)
        n_valid = min(self.y_multi.shape[0], self.x.shape[0] - seq_size + 1)

        # Build segment ranges
        if segment_boundaries is not None:
            bounds = np.asarray(segment_boundaries, dtype=np.int64)
            seg_starts = np.concatenate([[0], bounds])
            seg_ends = np.concatenate([bounds, [n_valid]])
        else:
            seg_starts = np.array([0])
            seg_ends = np.array([n_valid])

        # Generate non-overlapping chunk indices within each segment
        self.chunk_starts = []
        for s_start, s_end in zip(seg_starts, seg_ends):
            seg_len = s_end - s_start
            if seg_len < chunk_size:
                continue
            for i in range(s_start, s_end - chunk_size + 1, chunk_size):
                self.chunk_starts.append(i)

    def __len__(self):
        return len(self.chunk_starts)

    def __getitem__(self, idx):
        start = self.chunk_starts[idx]
        T = self.chunk_size

        # Build chunk: T consecutive (seq_size, features) windows
        x_chunk = torch.stack([
            self.x[start + t: start + t + self.seq_size, :]
            for t in range(T)
        ])  # (T, seq_size, features)

        y_chunk = self.y_multi[start: start + T]  # (T, num_horizons)

        if self.has_dfl:
            dm_chunk = self.delta_mids[start: start + T]  # (T, num_horizons)
            hs_chunk = self.half_spreads[start: start + T]  # (T,)
            return x_chunk, y_chunk, dm_chunk, hs_chunk

        return x_chunk, y_chunk


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