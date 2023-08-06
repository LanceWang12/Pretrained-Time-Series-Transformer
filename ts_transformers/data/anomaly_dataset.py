import pandas as pd
import numpy as np
import datetime
from typing import Optional, Tuple, Any
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .configuration_dataset import TSAnomalyConfig, SPCAnomalyConfig
# from .utils import load_data, fix_seed


class TSAnomalyDataset(Dataset):

    def __init__(self, data: pd.DataFrame, config: TSAnomalyConfig):
        super().__init__()
        self.output_every_anomaly_label = config.output_every_anomaly_label
        self.window_size = config.window_size
        self.time_idx = config.time_idx
        if self.time_idx is not None:
            self.time_stamp = data[self.time_idx]
        if self.time_idx is not None:
            self.X = torch.as_tensor(np.asarray(
                data.drop(columns=[config.time_idx, config.target_col])),
                dtype=torch.float)
        self.X = torch.as_tensor(np.asarray(data.drop(columns=[config.target_col])),
                                 dtype=torch.float)
        self.Y = torch.as_tensor(np.expand_dims(np.asarray(data[config.target_col]), axis=-1),
                                 dtype=torch.float).squeeze()
        if len(self.X) < self.window_size:
            raise ValueError('Datalength must bigger than lag.')

    def __getitem__(self, idx) -> tuple:
        x = self.X[idx:idx + self.window_size]

        if self.output_every_anomaly_label:
            y = self.Y[idx:idx + self.window_size]
        else:
            y = self.Y[idx + self.window_size]

        if self.time_idx is not None:
            t = self.time_stamp[idx:idx + self.window_size]
            return x, t, y
        else:
            return x, y

    def __len__(self) -> int:
        return len(self.X) - self.window_size

    def __gettime(self, data):
        time_lst = []
        for t in data[self.time_idx]:
            tmp = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            time_lst.append([tmp.hour, tmp.minute, tmp.second])

        return torch.as_tensor(time_lst, dtype=torch.float)


class SPCAnomalyDataset(Dataset):

    def __init__(self, data: pd.DataFrame, config: SPCAnomalyConfig):
        super().__init__()
        # Real window size = seq_len - spc_label_num
        # Need to pad the input series to the shape from
        # (batch_size, real_window_size, features)
        # to (batch_size, seq_len, features) ---> important!!!
        self.output_every_anomaly_label = config.output_every_anomaly_label
        self.spc_label_num = len(config.spc_col)
        self.window_size = config.window_size
        self.time_idx = config.time_idx
        self.padding = config.padding
        if self.time_idx:
            self.time_stamp = data[self.time_idx]
            self.X = torch.as_tensor(np.asarray(
                data.drop(columns=[config.time_idx, config.target_col])),
                dtype=torch.float)
        else:
            self.X = torch.as_tensor(
                np.asarray(data.drop(columns=[config.target_col]).drop(
                    columns=config.spc_col)),
                dtype=torch.float
            )
        self.Y = torch.as_tensor(np.expand_dims(np.asarray(data[config.target_col]), axis=-1),
                                 dtype=torch.float).squeeze()
        if len(self.X) < self.window_size:
            raise ValueError('Datalength must bigger than lag.')
        self.spc = torch.as_tensor(np.asarray(
            data[config.spc_col]), dtype=torch.float).squeeze()

    def __getitem__(self, idx) -> tuple:
        x = self.X[idx:idx + self.window_size]
        # padding input series: first token is spc token
        if self.padding:
            x_pad = F.pad(x, (0, 0, 1, 0), value=0)
        else:  # no padding
            x_pad = x

        spc = self.spc[idx + self.window_size]

        if self.output_every_anomaly_label:
            y = self.Y[idx:idx + self.window_size]
        else:
            y = self.Y[idx + self.window_size]

        # if self.time_idx is not None:
        #     t = self.time_stamp[idx:idx + self.window_size]
        #     return x, x_pad, spc, t, y
        # else:
        return x, x_pad, spc, y

    def __len__(self) -> int:
        return len(self.X) - self.window_size


def get_loader(
    config: SPCAnomalyConfig,
    flag: str = "DMDS",
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    if flag == "DMDS":
        csv_dir = "data/DMDS/csv_fe/"
        train_df = pd.read_csv(csv_dir + "30102001_spc_label.csv")

        # 固定 [57030: ] 為 validation set
        # 因為需要剩一些 anomaly time points 來設定 threshold
        if config.val_idx is not None:
            val_idx = config.val_idx
        else:
            val_idx = int((1 - config.val_size) * train_df.shape[0])

        val_df = train_df[val_idx:]
        train_df = train_df[:val_idx]
        test_df = pd.read_csv(csv_dir + "17112001_spc_label.csv")

        trainset = SPCAnomalyDataset(train_df, config)
        valset = SPCAnomalyDataset(val_df, config)

        config.output_every_anomaly_label = True
        set_for_threshold = SPCAnomalyDataset(val_df, config)
        testset = SPCAnomalyDataset(test_df, config)  # set_for_threshold

        if config.echo:
            print()
            print("-" * 25)
            print(f"Trainset size: {len(trainset)}")
            print(f"Valset size: {len(valset)}")
            print(f"Thresholdset size: {len(set_for_threshold)}")
            print(f"Testset size: {len(testset)}")
            print("-" * 25)
            print()

        trainloader = load_data(trainset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers)
        valloader = load_data(valset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)
        thresloader = load_data(set_for_threshold,
                                batch_size=config.batch_size,
                                shuffle=True,
                                num_workers=config.num_workers)
        testloader = load_data(testset,
                               batch_size=config.batch_size,
                               shuffle=True,
                               num_workers=config.num_workers)

    return trainloader, valloader, thresloader, testloader


def get_pattern_matching_loader(
    config: SPCAnomalyConfig,
    flag: str = "DMDS",
) -> Tuple[DataLoader, DataLoader]:
    csv_dir = "data/DMDS/csv_fe"
    df = pd.read_csv(
        f"{csv_dir}/09112001_spc_label_user_added.csv")
    val_df = df[0: 7500]
    test_df = df[12000:]
    # config.output_every_anomaly_label = True
    valset = SPCAnomalyDataset(val_df, config)
    testset = SPCAnomalyDataset(test_df, config)

    if config.echo:
        print()
        print("-" * 25)
        print(f"Valset size: {len(valset)}")
        print(f"Testset size: {len(testset)}")
        print("-" * 25)
        print()

    valloader = load_data(valset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=config.num_workers)
    testloader = load_data(testset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)

    return valloader, testloader


def load_data(dataset: Any,
              batch_size: int,
              shuffle: bool,
              sampler=None,
              num_workers: int = -1) -> DataLoader:
    if num_workers == -1:
        num_workers = len(os.sched_getaffinity(0))

    if sampler:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=shuffle,
                                 drop_last=True,
                                 sampler=sampler)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=shuffle,
                                 drop_last=True)

    return data_loader
