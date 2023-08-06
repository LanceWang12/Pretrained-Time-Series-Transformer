if __name__ == "__main__":
    import argparse
    from configuration_spcbert import SPCBertConfig
    from embed import DataEmbedding, Patchify1D, RevIN, UnPatchify1D, UnPatchify1DPerFeature
else:
    from .configuration_spcbert import SPCBertConfig
    from .embed import DataEmbedding, Patchify1D, RevIN, UnPatchify1D, UnPatchify1DPerFeature

from transformers.models.bert.modeling_bert import BertEncoder
import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np
from math import sqrt
from typing import Tuple


class TriangularCausalMask():

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(
                mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class SPCBERTOutput():
    def __init__(
        self,
        attn,
        norm_out,
        spc_out,
        out,
        denorm_out
    ):
        super().__init__()
        self.record = dict()
        self.record["attn"] = attn
        self.record["norm_out"] = norm_out
        self.record["spc_out"] = spc_out
        self.record["out"] = out
        self.record["denorm_out"] = denorm_out

def fuzzy_output(obs: float, d1: float = 3, d2: float = 5):
    normal, anomaly = 0, 0
    if obs <= d1:
        normal = 1
        anomaly = 0
    elif obs > d1 and obs <= d2:
        normal = (obs - d2) / (d1 - d2)
        anomaly = (obs - d1) / (d2 - d1)
    else:
        normal = 0
        anomaly = 1
    return normal, anomaly

class SPCBert(nn.Module):
    def __init__(self, config: SPCBertConfig) -> None:
        super(SPCBert, self).__init__()
        self.norm = config.norm
        self.embed = DataEmbedding(
            config.input_dim,
            config.hidden_size,
            config.hidden_dropout_prob,
            norm=config.norm,
        )
        if config.backbone == "bert":
            self.encoder = BertEncoder(config)
        else:
            raise NotImplementedError("Other backbones are not implemented.")

        self.output_attention = config.output_attention
        self.spc_rule_num = config.spc_rule_num
        if self.spc_rule_num:
            self.spc_heads = nn.Linear(config.hidden_size, self.spc_rule_num)
        self.output_heads = nn.Linear(config.hidden_size, config.output_dim)
        self.sigmoid = nn.Sigmoid()

    def _forward(self, x: torch.Tensor, mode=0, denorm=True, attn=True):
        if self.norm:
            norm_out, x = self.embed(x)
            norm_out = norm_out[:, 1:, :]
        else:
            x = self.embed(x)
        encoder_out = self.encoder(x, output_attentions=True)
        x = encoder_out['last_hidden_state']

        if self.spc_rule_num:
            spc_out = x[:, 0, :]
            spc_out = self.spc_heads(spc_out)
            spc_out = self.sigmoid(spc_out)
        else:
            spc_out = None
            spc_out = None
        series_out = x[:, 1:, :]
        series_out = self.output_heads(series_out)

        attn = encoder_out['attentions']

        if mode == 0:  # in training stage and anomaly detection stage
            if denorm:
                series_out = self.embed.denormalize(series_out)

            if self.output_attention:
                return attn, norm_out, spc_out, series_out

            return norm_out, spc_out, series_out
        else:  # get representation
            return spc_out, attn

    def forward(self, x: torch.Tensor, denorm=True):
        return self._forward(x, mode=0, denorm=denorm, attn=self.output_attention)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        spc_representation, attn = self._forward(x, mode=1)
        attn = torch.cat(attn, axis=0)
        attn_map = torch.mean(attn, axis=(0, 1))[:, 0]
        spc_representation = torch.mean(spc_representation, axis=0)
        return spc_representation, attn_map


class SPCPatchBert(nn.Module):
    def __init__(self, config: SPCBertConfig) -> None:
        super(SPCPatchBert, self).__init__()
        self.verbose = config.verbose
        self.window_size = config.window_size
        self.hidden_size = config.hidden_size
        self.input_dim = config.input_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.spc_head_lst = config.spc_head_lst
        # Adaptive Normalization
        if config.norm:
            self.norm = RevIN(
                num_features=config.input_dim,
                eps=config.eps, affine=config.affine,
                dropout_rate=config.hidden_dropout_prob
            )
            # Preload the weight of norm layer
            if config.load_norm:
                self.norm.load_state_dict(torch.load(config.load_norm))
                print(f"Load norm layer from {config.load_norm}\n")
        else:
            self.norm = None

        # Patchify1D
        self.patchify = Patchify1D(config)
        # because of spc token at the first place
        config.max_position_embeddings = self.patchify.patch_num + 1

        # Position encoding & Token embedding (don't use norm)
        self.embed = DataEmbedding(
            config.patch_len,
            config.hidden_size,
            config.hidden_dropout_prob,
            norm=False,
        )
        # Preload the weight of embedding layer
        if config.load_embed:
            self.embed.load_state_dict(torch.load(config.load_embed))
            print(f"Load embedding layer from {config.load_embed}\n")

        # Ensemble model
        self.ensemble = config.ensemble
        if self.ensemble:
            if config.backbone == "bert":
                self.encoders = nn.ModuleList(
                    [BertEncoder(config) for _ in range(config.input_dim)]
                )
            else:
                raise NotImplementedError(
                    "Other backbones are not implemented.")
            # Preload the weight of Encoder backbone
            if config.load_encoder:
                for i in range(config.input_dim):
                    self.encoders[i].load_state_dict(
                        torch.load(config.load_encoder))
                print(
                    f"Load {config.input_dim} encoders from {config.load_norm}\n")
        else:
            if config.backbone == "bert":
                self.encoder = BertEncoder(config)
            else:
                raise NotImplementedError(
                    "Other backbones are not implemented.")

        self.output_attention = config.output_attention
        self.spc_rule_num = config.spc_rule_num
        if self.spc_rule_num:
            # each features may have different spc rules
            self.spc_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, i) for i in config.spc_head_lst]
            )

        if self.ensemble:
            self.output_heads = nn.ModuleList([
                UnPatchify1DPerFeature(
                    self.hidden_size * self.patchify.patch_num,
                    config.window_size, config.hidden_dropout_prob)
                for _ in range(self.input_dim)
            ])
        else:
            self.output_heads = UnPatchify1D(
                config.individual, config.input_dim,
                self.hidden_size * self.patchify.patch_num,
                config.window_size, head_dropout=config.hidden_dropout_prob
            )
        self.sigmoid = nn.Sigmoid()

    def _forward(
        self,
        x: torch.Tensor,
        mode: int = 0,
        idx_lst: list = [],
    ):
        # -----------------------------------
        # | mode: 0: training mode, validation mode, 2: get representation
        # -----------------------------------

        # --------------------------------------------------------------------------
        # |  Dataflow
        # |  x -> RevIN -> Patchify -> position encoding -> token embedding -> Encoder -> Denorm -> End
        # |                                                                            -> SPC out
        # --------------------------------------------------------------------------

        # << RevIN >>
        # out: (batch, seq_len, features_num)
        device = x.device
        norm_out = x
        if self.norm:
            norm_out = self.norm(x)
        # << Patchify>>
        # out: (batch, features_num, patch_num, patch_len)
        out = self.patchify(norm_out)
        # Add a spc token at the first place
        out = F.pad(out, (0, 0, 1, 0), value=0)
        batch_size, features_num, patch_num, patch_len = out.shape
        # Because embed & encoder can't deal with 4D shape, reshape the tensor
        out = torch.reshape(
            out, (batch_size * features_num, patch_num, patch_len))
        # << Position encoding & Token embedding >>
        # out: (batch_size * features_num, patch_num, hidden_size)
        out = self.embed(out)

        # << Encoder >>
        if self.ensemble:
            # Get each features to input in each responsible model
            input_tokens = out.reshape(batch_size, features_num,
                                       patch_num, self.hidden_size)
            if mode == 0:  # training, inference, validation, test
                # Store all model's output
                out = torch.zeros(
                    batch_size, features_num, patch_num, self.hidden_size).to(device)
                # Store all model's attention output
                attn = torch.zeros(
                    batch_size, features_num, patch_num, patch_num).to(device)
                # Store each model's output in "encoder_out", and store attention in "attn"
                for i in range(self.input_dim):
                    # input_tokens[:, i]: (batch_size, patch_num, hidden_size)
                    encoder_out = self.encoders[i](
                        input_tokens[:, i], output_attentions=True)
                    out[:, i] = encoder_out['last_hidden_state']
                    # attention map shape:
                    # (layers, batch, heads, patch_num, patch_num) ->
                    # (batch, heads, patch_num, patch_num))
                    attn_tmp = torch.cat(encoder_out['attentions'], dim=0).reshape(
                        self.num_hidden_layers, batch_size,
                        self.num_attention_heads, patch_num, patch_num
                    ).mean(axis=(0, 2))
                    attn[:, i] = attn_tmp
            else:  # get representation
                # Store the model's attention output in idx_lst
                idx_lst_len = len(idx_lst)
                attn = torch.zeros(
                    batch_size, idx_lst_len, patch_num, patch_num).to(device)
                for i, idx in enumerate(idx_lst):
                    encoder_out = self.encoders[idx](
                        input_tokens[:, idx], output_attentions=True)
                    attn_tmp = torch.cat(encoder_out['attentions'], dim=0).reshape(
                        self.num_hidden_layers, batch_size,
                        self.num_attention_heads, patch_num, patch_num
                    ).mean(axis=(0, 2))
                    attn[:, i] = attn_tmp
        else:
            encoder_out = self.encoder(out, output_attentions=True)
            out = encoder_out['last_hidden_state'].reshape(
                (batch_size, features_num, patch_num, self.hidden_size)
            )
            attn = torch.cat(encoder_out['attentions'], dim=0).reshape(
                self.num_hidden_layers, batch_size, features_num,
                self.num_attention_heads, patch_num, patch_num
            ).mean(axis=(0, 3))

        if self.verbose:
            if mode == 1:
                print(f"attn: {attn.shape}")
            else:
                print(f"encoder_out: {out.shape}, attn: {attn.shape}")

        # << SPC Heads >>
        if self.ensemble:
            if mode == 0:  # inference, validation & test
                # out: (batch_size, input_dim, patch_num, hidden_size)
                spc_out = torch.zeros(batch_size, self.spc_rule_num).to(device)
                start = 0
                for i in range(self.input_dim):
                    spc_out[:, start: start + self.spc_head_lst[i]
                            ] = self.spc_heads[i](out[:, i, 0])
                    start += self.spc_head_lst[i]
                spc_out = self.sigmoid(spc_out)
            else:  # get representation
                spc_out = None
        else:
            # x: x: (batch_size, features_num, patch_num, patch_len)
            tmp_spc_out = out[:, :, 0, :]
            spc_out = [
                self.spc_heads[i](tmp_spc_out[:, i]) for i in range(len(self.spc_heads))
            ]
            spc_out = torch.cat(spc_out, dim=1)
            spc_out = self.sigmoid(spc_out)
        if (spc_out is not None) and self.verbose:
            print(f"spc_out: {spc_out.shape}")

        # << Series Output Heads >>
        if self.ensemble:
            if mode == 0:  # inference, validation & test
                # out: (batch_size, input_dim, patch_num, hidden_size)
                series_out = torch.zeros(
                    batch_size, self.window_size, self.input_dim).to(device)
                for i in range(self.input_dim):
                    series_out[:, :, i] = self.output_heads[i](
                        out[:, i, 1:, :]
                    )
            else:  # get representation
                series_out = None
        else:
            series_out = self.output_heads(out[:, :, 1:])
        if (series_out is not None) and self.verbose:
            print(f"unpatchified_series_out: {series_out.shape}")

        if (series_out is not None) and self.norm:
            series_out = self.norm._denormalize(series_out)
        if (series_out is not None) and self.verbose:
            print(f"denormed_series_out: {series_out.shape}")

        # << Final Output Stage >>
        if mode == 0:  # training & inference
            if self.output_attention:
                return attn, norm_out, spc_out, series_out
            else:
                return norm_out, spc_out, series_out
        else:  # get representation
            return attn  # spc_out, attn

    def forward(self, x: torch.Tensor):
        return self._forward(x, mode=0)

    # -> Tuple[torch.Tensor, torch.Tensor]:
    def get_representation(self, x: torch.Tensor, idx_lst: list) -> torch.Tensor:
        attn = self._forward(x, mode=1, idx_lst=idx_lst)

        return attn
        # return spc_representation, attn_map


if __name__ == "__main__":
    origin, green = "\033[0m", "\033[32m"
    red_background = "\033[41m"

    def parse() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--target', type=str,
            default="SPCBert",
            choices=["SPCBert", "SPCPatchBert"]
        )
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--input_dim', type=int, default=10)
        parser.add_argument('--seq_len', type=int, default=101)
        parser.add_argument('--window_size', type=int, default=256)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--num_hidden_layers', type=int, default=8)
        parser.add_argument('--num_attention_heads', type=int, default=8)
        parser.add_argument('--attention_probs_dropout',
                            type=int, default=0.1)
        parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
        parser.add_argument('--output_attention', type=int, default=1)
        parser.add_argument('--norm', type=int, default=1)
        parser.add_argument('--patch_len', type=int, default=16)
        parser.add_argument('--stride', type=int, default=4)
        parser.add_argument('--patch_padding', type=int, default=1)
        parser.add_argument('--ensemble', type=int, default=1)
        parser.add_argument('--load_norm', type=str, default="")
        parser.add_argument('--load_embed', type=str, default="")
        parser.add_argument('--load_encoder', type=str, default="")
        parser.add_argument('--verbose', type=int, default=0)
        args = parser.parse_args()
        args.spc_head_lst = [2, 3, 3, 4, 2, 2, 1, 2, 3, 2]
        args.spc_rule_num = sum(args.spc_head_lst)

        print('=' * 70)
        for key, value in vars(args).items():
            print(f'{key}: {value}')
        print('=' * 70)

        return args

    def test_SPCBert(args):
        config = SPCBertConfig(
            input_dim=args.input_dim,
            output_dim=args.input_dim,
            spc_rule_num=args.spc_rule_num,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            hidden_size=args.hidden_size,
            hidden_act="gelu",
            attention_probs_dropout_prob=args.attention_probs_dropout,
            hidden_dropout_prob=args.hidden_dropout_prob,
            max_position_embeddings=args.seq_len,
            output_attention=args.output_attention,
            norm=args.norm,
        )
        model = SPCBert(config)

        x = torch.rand(args.batch_size, args.seq_len, args.input_dim)
        attn, norm_out, spc_label, y = model(x)
        spc_representation, attn_representation = model.get_representation(x)

        if args.verbose:
            print(f"SPC label: {spc_label.shape}")
            print(f"Series out: {y.shape}")
            print(f"Attention scores: {attn[-1].shape}")
            print(f"SPC Representation: {spc_representation.shape}")
            print(f"Attn Representation: {attn_representation.shape}")
        print(f"{green}{args.target} pass the test on cpu{origin}")

        x = x.cuda()
        model = model.cuda()
        attn, norm_out, spc_label, y = model(x)
        if args.verbose:
            print(f"SPC label: {spc_label.shape}")
            print(f"Series out: {y.shape}")
            print(f"Attention scores: {attn[-1].shape}")
        print(f"{green}{args.target} pass the test on gpu{origin}")

        config.spc_rule_num = 0
        config.output_attention = False
        model = SPCBert(config).cuda()
        norm_out, spc_label, y = model(x)
        if args.verbose:
            print(f"SPC label: {spc_label}")
            print(f"Series out: {y.shape}")
        print(f"{green}{args.target} pass the test about zero spc label")

    def test_SPCPatchBert(args):
        config = SPCBertConfig(
            input_dim=args.input_dim,
            output_dim=args.input_dim,
            window_size=args.window_size,
            spc_rule_num=args.spc_rule_num,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            hidden_act="gelu",
            attention_probs_dropout_prob=args.attention_probs_dropout,
            hidden_dropout_prob=args.hidden_dropout_prob,
            output_attention=args.output_attention,
            norm=args.norm,
            patch_len=args.patch_len,
            stride=args.stride,
            spc_head_lst=args.spc_head_lst,
            ensemble=args.ensemble,
            load_norm=args.load_norm,
            load_embed=args.load_embed,
            load_encoder=args.load_encoder,
            verbose=args.verbose,
        )
        model = SPCPatchBert(config)

        # test on cpu
        x = torch.rand(args.batch_size, args.window_size, args.input_dim)
        if args.verbose:
            print(f"Input shape: {x.shape}")
        if args.verbose:
            print(f"{red_background}Test inference flow...{origin}")
        attn, norm_out, spc_out, series_out = model(x)
        idx_lst = [1, 3, 5]
        if args.verbose:
            print(f"{red_background}Test representation flow...{origin}")
            print(
                f"Getting representations from features: {idx_lst}")
        attn = model.get_representation(x, idx_lst)
        print(f"{green}{args.target} pass the test on cpu{origin}")

        # test on gpu
        x = x.cuda()
        model = model.cuda()
        if args.verbose:
            print(f"{red_background}Test inference flow...{origin}")
        attn, norm_out, spc_out, series_out = model(x)
        if args.verbose:
            print(f"{red_background}Test representation flow...{origin}")
            print(
                f"Getting representations from features: {idx_lst}")
        idx_lst = [1, 3, 5]
        attn = model.get_representation(x, idx_lst)
        print(f"{green}{args.target} pass the test on gpu{origin}")

    def main():
        args = parse()
        if args.target == "SPCBert":
            test_SPCBert(args)
        else:
            test_SPCPatchBert(args)

    main()
