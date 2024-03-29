from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


# @inproceedings{kim2021reversible,
#   title     = {Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift, ICLR2022},
#   author    = {Kim, Taesung and
#                Kim, Jinhee and
#                Tae, Yunwon and
#                Park, Cheonbok and
#                Choi, Jang-Ho and
#                Choo, Jaegul},
#   booktitle = {International Conference on Learning Representations},
#   year      = {2021},
#   url       = {https://openreview.net/forum?id=cGDAkQo1C0p}
# }


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=True,
        dropout_rate=0.1,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.dropout = nn.Dropout(dropout_rate)
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str = "norm"):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
            x = self.dropout(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # print(x.shape)
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):

    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2

        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float = 0.0,
        norm: bool = False,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super(DataEmbedding, self).__init__()

        # Deep Adaptive Input Normalization
        if norm:
            self.norm = RevIN(
                num_features=c_in,
                eps=eps, affine=affine,
                dropout_rate=dropout
            )
        else:
            self.norm = None

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        norm_out = x
        if self.norm:
            # print(x.shape)
            # x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            # print(x.shape)
            norm_out = self.norm(norm_out)
        out = self.value_embedding(norm_out) + \
            self.position_embedding(norm_out)
        out = self.dropout(out)
        if self.norm:
            return norm_out, out
        else:
            return out

    def denormalize(self, x):
        if self.norm:
            # return self.norm.denormalize(x.transpose(1, 2)).transpose(1, 2)
            return self.norm(x, mode="denorm")
        return x


class Patchify1D(nn.Module):
    def __init__(self, config):
        super(Patchify1D, self).__init__()
        # patch
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.padding_patch = config.padding_patch
        self.patch_num = int(
            (config.window_size - config.patch_len) / config.stride + 1
        )
        if config.padding_patch:  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
            self.patch_num += 1

    def forward(self, x):
        # x: (batch, window_size, features_num)
        # features_num: the number of features
        # window_size: the size of a window

        # padding (because of the stride)
        # Padding and patchifing is channel-first,
        # so transform "x" from (batch, seq, dim) to (batch, dim, seq)
        x = x.permute(0, 2, 1)
        if self.padding_patch:
            x = self.padding_patch_layer(x)

        # patching
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x


class UnPatchify1D(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.unpatchify_lst = nn.ModuleList([
                UnPatchify1DPerFeature(nf, target_window, head_dropout)
                for _ in range(n_vars)
            ])
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    # x: [bs x nvars x d_model x patch_num]
    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.unpatchify_lst[i](x[:, i, :, :])
                x_out.append(z)
            # x: [bs x nvars x target_window]
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x


class UnPatchify1DPerFeature(nn.Module):
    def __init__(
        self,
        nf: int,
        window_size: int = 256,
        dropout: float = 0.,
    ):
        super().__init__()
        # self.unpatch = nn.Sequential(OrderedDict([
        #     ("flatten", nn.Flatten(start_dim=-2)),
        #     ("linear", nn.Linear(hidden_size, window_size)),
        #     ("unpatchify_dropout", nn.Dropout(dropout))
        # ]))
        self.flatten = nn.Flatten(start_dim=-2)
        self.project = nn.Linear(nf, window_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # print(f"In: {x.shape}")
        out = self.flatten(x)
        out = self.project(out)
        out = self.drop(out)
        # print(f"Out: {out.shape}")
        return out


if __name__ == "__main__":

    def test_RevIN(batch_size, seq_len, dim):
        print("Start to test RevIN Layer...")
        modes = ["avg", "adaptive_avg", "adaptive_scale", "full"]
        x = torch.arange(batch_size * seq_len * dim,
                         dtype=torch.float).reshape(batch_size, seq_len, dim)

        with torch.no_grad():
            norm = RevIN(num_features=dim)
            norm.eval()
            with torch.no_grad():
                y = norm(x)
                y_denormalize = norm._denormalize(y)
            err = torch.mean((x - y_denormalize)**2)
            assert (err < 1e-3)
            # print(x)
            # print(y)
            # print(y_denormalize)

            print("CPU Pass")

            device = "cuda:0"
            x = x.to(device)
            norm = RevIN(num_features=dim).to(device)
            norm.eval()
            with torch.no_grad():
                y = norm(x)
                y_denormalize = norm._denormalize(y)
            err = torch.mean((x - y_denormalize)**2)
            assert (err < 1e-3)
            # print(x)
            # print(y)
            # print(y_denormalize)

            print("GPU Pass")

    batch_size, seq_len, dim = 64, 128, 38
    test_RevIN(batch_size, seq_len, dim)
