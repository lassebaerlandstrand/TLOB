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
        self.length = y.shape[0]
        self.x = x
        self.y = y
        if type(self.x) == np.ndarray:
            self.x = torch.from_numpy(x).float()
        if type(self.y) == np.ndarray:
            self.y = torch.from_numpy(y).long()
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
    """
    def __init__(self, x, y_multi, seq_size):
        self.seq_size = seq_size
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y_multi = y_multi if isinstance(y_multi, torch.Tensor) else torch.from_numpy(y_multi).long()
        self.length = self.y_multi.shape[0]
        self.data = self.x

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x_window = self.x[i:i + self.seq_size, :]   # (seq_size, num_features)
        y_all = self.y_multi[i]                       # (num_horizons,)
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
        if train_set.data.device.type != cst.DEVICE:       #this is true only when we are using a GPU but the data is still on the CPU
            self.pin_memory = True
        else:
            self.pin_memory = False
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