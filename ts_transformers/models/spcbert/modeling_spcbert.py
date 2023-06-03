if __name__ == "__main__":
    import argparse
    from configuration_spcbert import SPCBertConfig
    from embed import DataEmbedding, Patchify1D, RevIN, UnPatchify1D
else:
    from .configuration_spcbert import SPCBertConfig
    from .embed import DataEmbedding, Patchify1D, RevIN, UnPatchify1D

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
        spc_pred,
        out,
        denorm_out
    ):
        super().__init__()
        self.record = dict()
        self.record["attn"] = attn
        self.record["norm_out"] = norm_out
        self.record["spc_pred"] = spc_pred
        self.record["out"] = out
        self.record["denorm_out"] = denorm_out


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
            spc_pred = self.spc_heads(spc_out)
            spc_pred = self.sigmoid(spc_pred)
        else:
            spc_out = None
            spc_pred = None
        series_out = x[:, 1:, :]
        series_out = self.output_heads(series_out)

        attn = encoder_out['attentions']

        if mode == 0:  # in training stage and anomaly detection stage
            if denorm:
                series_out = self.embed.denormalize(series_out)

            if self.output_attention:
                return attn, norm_out, spc_pred, series_out

            return norm_out, spc_pred, series_out
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
        self.hidden_size = config.hidden_size
        # Adaptive Normalization
        if config.norm:
            self.norm = RevIN(
                num_features=config.input_dim,
                eps=config.eps, affine=config.affine,
                dropout_rate=config.hidden_dropout_prob
            )
        else:
            self.norm = None

        # Patchify1D
        self.patchify = Patchify1D(config)
        config.max_position_embeddings = self.patchify.patch_num

        # Position encoding & Token embedding (don't use norm)
        self.embed = DataEmbedding(
            config.patch_len,
            config.hidden_size,
            config.hidden_dropout_prob,
            norm=False,
        )

        if config.backbone == "bert":
            self.encoder = BertEncoder(config)
        else:
            raise NotImplementedError("Other backbones are not implemented.")

        self.output_attention = config.output_attention
        self.spc_rule_num = config.spc_rule_num
        if self.spc_rule_num:
            # each features may have different spc rules
            self.spc_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, i) for i in config.spc_head_lst]
            )
        self.output_heads = UnPatchify1D(
            config.individual, config.input_dim,
            self.hidden_size * self.patchify.patch_num,
            config.window_size, head_dropout=config.hidden_dropout_prob
        )
        self.sigmoid = nn.Sigmoid()

    def _forward(
        self, x: torch.Tensor, mode=0, denorm=True, attn=True
    ):
        # Dataflow
        # x -> RevIN -> Patchify -> position encoding -> token embedding -> Encoder

        # RevIN, out: (batch, seq_len, features_num)
        norm_out = x
        if self.norm:
            norm_out = self.norm(x)
        # Patchify, out: (batch, features_num, patch_num, patch_len)
        out = self.patchify(norm_out)
        # Add a spc token at the first place
        out = F.pad(out, (0, 0, 1, 0), value=0)
        batch_size, features_num, patch_num, patch_len = out.shape
        # Because embed & encoder can't deal with 4D shape, reshape the tensor
        out = torch.reshape(
            out, (batch_size * features_num, patch_num, patch_len))
        # Position encoding & Token embedding,
        # out: (batch_size * features_num, patch_num, patch_len)
        out = self.embed(out)
        # Encoder, out: (batch_size, features_num, patch_num, patch_len)
        encoder_out = self.encoder(out, output_attentions=True)
        out = encoder_out['last_hidden_state'].reshape(
            (batch_size, features_num, patch_num, self.hidden_size)
        )
        attn = encoder_out['attentions']

        if self.verbose:
            print(f"encoder_out: {out.shape}, attn: {attn[0].shape}")

        if self.spc_rule_num:
            # x: x: (batch_size, features_num, patch_num, patch_len)
            spc_out = out[:, :, 0, :]
            spc_pred = [
                self.spc_heads[i](spc_out[:, i]) for i in range(len(self.spc_heads))
            ]
            spc_pred = torch.cat(spc_pred, dim=1)
            spc_pred = self.sigmoid(spc_pred)
            if self.verbose:
                print(f"spc_pred: {spc_pred.shape}")
        else:
            spc_out = None
            spc_pred = None
        series_out = out[:, :, 1:, :]
        series_out = self.output_heads(series_out)
        if self.verbose:
            print(f"series_out: {series_out.shape}")

        if mode == 0:  # in training stage and anomaly detection stage
            if denorm:
                series_out = self.embed.denormalize(series_out)

            if self.output_attention:
                return attn, norm_out, spc_pred, series_out

            return norm_out, spc_pred, series_out
        else:  # get representation
            return spc_out, attn

    def forward(self, x: torch.Tensor, denorm=True):
        return self._forward(x, mode=0, denorm=denorm, attn=self.output_attention)

    def get_representation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Function: get_representation is not implemented!"
        )
        spc_representation, attn = self._forward(x, mode=1)
        attn = torch.cat(attn, axis=0)
        attn_map = torch.mean(attn, axis=(0, 1))[:, 0]
        spc_representation = torch.mean(spc_representation, axis=0)
        return spc_representation, attn_map


if __name__ == "__main__":
    origin, green = "\033[0m", "\033[32m"

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
        parser.add_argument('--attention_probs_dropout', type=int, default=0.1)
        parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
        parser.add_argument('--output_attention', type=int, default=1)
        parser.add_argument('--norm', type=int, default=1)
        parser.add_argument('--spc_rule_num', type=int, default=1)
        parser.add_argument('--patch_len', type=int, default=16)
        parser.add_argument('--stride', type=int, default=4)
        parser.add_argument('--patch_padding', type=int, default=1)
        parser.add_argument('--verbose', type=int, default=0)
        args = parser.parse_args()
        args.spc_head_lst = [2, 3, 3, 4, 2, 2, 1, 2, 3, 2]

        print('=' * 70)
        for key, value in vars(args).items():
            print(f'{key}: {value}')
        print(f"spc_rule_num: {sum(args.spc_head_lst)}")
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
            verbose=args.verbose,
        )
        model = SPCPatchBert(config)

        # test on cpu
        x = torch.rand(args.batch_size, args.window_size, args.input_dim)
        attn, norm_out, spc_pred, series_out = model(x)
        print(f"{green}{args.target} pass the test on cpu{origin}")

        # test on gpu
        x = x.cuda()
        model = model.cuda()
        attn, norm_out, spc_pred, series_out = model(x)
        print(f"{green}{args.target} pass the test on gpu{origin}")

    def main():
        args = parse()
        if args.target == "SPCBert":
            test_SPCBert(args)
        else:
            test_SPCPatchBert(args)

    main()
