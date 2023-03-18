import pandas as pd
import numpy as np
import datetime
from typing import Optional
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .configuration_dataset import TSAnomalyConfig, SPCAnomalyConfig

class TSAnomalyDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config: TSAnomalyConfig):
        super().__init__()
        self.window_size = config.window_size
        self.time_idx = config.time_idx
        if self.time_idx is not None:
            self.time_stamp = data[self.time_idx]

        if self.time_idx is not None:
            self.X = torch.as_tensor(np.asarray(
                data.drop(columns=[config.time_idx])), dtype=torch.float)
        self.X = torch.as_tensor(np.asarray(
            data.drop(columns=[config.target_col])), dtype=torch.float)
        self.Y = torch.as_tensor(np.expand_dims(
            np.asarray(data[config.target_col]), axis=-1), dtype=torch.float).squeeze()
        if len(self.X) < self.window_size:
            raise ValueError('Datalength must bigger than lag.')

    def __getitem__(self, idx) -> tuple:
        x = self.X[idx: idx + self.window_size]
        y = self.Y[idx + self.window_size]

        if self.time_idx is not None:
            t = self.time_stamp[idx: idx + self.window_size]
            return x, t, y
        else:
            return x, y

    def __len__(self):
        return len(self.X) - self.window_size

    def __gettime(self, data):
        time_lst = []
        for t in data[self.time_idx]:
            tmp = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            time_lst.append([tmp.hour, tmp.minute, tmp.second])

        return torch.as_tensor(time_lst, dtype=torch.float)

class SPCAnomalyDataset(TSAnomalyDataset):
    def __init__(self, data: pd.DataFrame, config: SPCAnomalyConfig):
        super().__init__(data, config)
        self.spc = torch.as_tensor(np.expand_dims(
            np.asarray(data[config.spc_col]), axis=-1), dtype=torch.float).squeeze()

    def __getitem__(self, idx) -> tuple:
        spc = self.spc[idx + idx + self.window_size]
        if self.time_idx is not None:
            x, t, y = super().__getitem__(idx)
            return x, t, spc, y
        else:
            x, y = super().__getitem__(idx)
            return x, spc, y

