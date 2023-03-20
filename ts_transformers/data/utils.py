import os
from typing import Any
import random
import numpy as np
import torch
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def load_data(dataset: Any, batch_size: int, shuffle: bool = True,
              sampler: Sampler = None, num_workers: int = -1) -> DataLoader:
    if num_workers == -1:
        num_workers = len(os.sched_getaffinity(0))

    if sampler:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True
        )

    return data_loader
