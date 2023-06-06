import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re
from typing import Tuple


# -------------- for plot -----------------
def my_plot(x, Ys, label_lst, title="", figsize=(16, 9)):
    plt.figure(figsize=figsize, dpi=100)
    plt.title(title)
    for y, label in zip(Ys, label_lst):
        plt.plot(x, y, label=label)
    plt.legend()
    plt.show()


def analysis(file: str, start: int, end: int, columns: list):
    df = pd.read_csv(file)
    Ys = df[columns][start: end].to_numpy().transpose()
    x = np.arange(start, end)
    my_plot(x, Ys, columns, file)


# --------------- for spc label generator ---------------
# spc labels generator
def generate_mean_std(
    df: pd.DataFrame,
    target_features: list,
    label_features: list,
    wnd_size: int = 5000
) -> pd.DataFrame:
    df_fe = df[target_features].copy()
    for target in target_features:
        op = "mean"
        df_fe[f"{target}_{op}_{wnd_size}"] = df_fe[target].rolling(
            wnd_size, min_periods=200).mean()

    for target in target_features:
        op = "std"
        df_fe[f"{target}_{op}_{wnd_size}"] = df_fe[target].rolling(
            wnd_size, min_periods=200).std()
    df_fe[label_features] = df[label_features]
    df_fe.dropna(axis=0, inplace=True)

    return df_fe

class SPCRules(object):
    def __init__(self, data, mean, std):
        super().__init__()
        self.data = data
        self.n = len(data)
        self.index = np.array(range(self.n))
        self.mean = mean
        self.std = std
        self.upper_c = mean + std
        self.upper_b = mean + 2 * std
        self.upper_a = mean + 3 * std
        self.lower_c = mean - std
        self.lower_b = mean - 2 * std
        self.lower_a = mean - 3 * std

    def detect(self, idx: int):
        if idx == 1:
            # One point beyond the 3 σ control limit
            out = (self.data < self.lower_a) | (self.data > self.upper_a)
        elif idx == 2:
            # Nine or more points on one side of the centerline without crossing
            counter = 9
            upside = (self.data > self.mean).rolling(counter).sum()
            downside = (self.data < self.mean).rolling(counter).sum()
            out = (upside >= counter) | (downside >= counter)
        elif idx == 3:
            # Two out of three points in zone A or beyond
            counter = 3
            upside = (self.data > self.upper_b).rolling(counter).sum()
            downside = (self.data < self.lower_b).rolling(counter).sum()
            counter -= 1
            out = (upside >= counter) | (downside >= counter)
        elif idx == 4:
            # Four out of five points in zone B or beyond
            counter = 5
            upside = (self.data > self.upper_c).rolling(counter).sum()
            downside = (self.data < self.lower_c).rolling(counter).sum()
            counter -= 1
            out = (upside >= counter) | (downside >= counter)
        elif idx == 5:
            # Fifteen points are all in zone c
            counter = 15
            out = (self.data < self.upper_c) & (self.data > self.lower_c)
            out = out.rolling(counter).sum()
            out = (out == counter)
        elif idx == 6:
            # Eight continual points with none in zone c
            counter = 8
            out = (self.data > self.upper_c) | (self.data < self.lower_c)
            out = out.rolling(counter).sum()
            out = (out == counter)
        else:
            raise ValueError("Only implement rule 1~6")
        return out

    # Six or more points are continually increasing or decreasing
    def rule7(self):
        pass

    # Fourteen or more points alternate in direction
    def rule8(self):
        ofc8_ind = []
        for i in range(self.n - 13):
            d = self.data[i:i + 14]
            idx = self.index[i:i + 14]
            diff = list(v - u for u, v in zip(d, d[1:]))
            # if all(u * v < 0):
            #     pass

# rearange spc_labels to fit SPCPatchBert input
def rearange_spc_labels(
    df: pd.DataFrame,
    features: list,
    spc_labels: list,
    gt: list,
) -> Tuple[pd.DataFrame, list]:
    spc_lst = []
    spc_head_lst = []
    for fea in features:
        counter = 0
        for spc in spc_labels:
            result = re.match(fea, spc)
            if result is not None:
                counter += 1
                spc_lst.append(spc)
        spc_head_lst.append(counter)

    rearange_lst = features + spc_lst + gt
    df_rearange = df[rearange_lst]

    return df_rearange, spc_head_lst


# ------------- Generate some patterns which users add ---------------
