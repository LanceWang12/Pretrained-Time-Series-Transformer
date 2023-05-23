if __name__ == "__main__":
    from configuration_spcbert import SPCBertConfig
    from embed import DataEmbedding
else:
    from .configuration_spcbert import SPCBertConfig
    from .embed import DataEmbedding
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
        # print(config.max_position_embeddings)
        # exit(1)
        self.norm = config.norm
        self.embed = DataEmbedding(config.input_dim,
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

    def forward(self, x, denorm=True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.norm:
            norm_out, x = self.embed(x)
            norm_out = norm_out[:, 1:, :]
        else:
            x = self.embed(x)
        encoder_out = self.encoder(x, output_attentions=True)
        x = encoder_out['last_hidden_state']

        if self.spc_rule_num:
            spc_pred = x[:, 1, :]
            spc_pred = self.spc_heads(spc_pred)
            spc_pred = self.sigmoid(spc_pred)
        else:
            spc_pred = None
        series_out = x[:, 1:, :]
        series_out = self.output_heads(series_out)

        if denorm:
            series_out = self.embed.denormalize(series_out)

        if self.output_attention:
            attn = encoder_out['attentions']
            return attn, norm_out, spc_pred, series_out

        return norm_out, spc_pred, series_out


if __name__ == "__main__":
    batch_size, cin, seq_len, hidden_size = 64, 38, 100, 128
    n_head = 8
    seq_len += 1
    config = SPCBertConfig(
        input_dim=cin,
        output_dim=cin,
        spc_rule_num=25,
        num_hidden_layers=3,
        num_attention_heads=n_head,
        hidden_size=hidden_size,
        hidden_act="gelu",
        max_position_embeddings=seq_len,
        norm=True,
        output_attention=True,
    )
    model = SPCBert(config)

    x = torch.rand(batch_size, seq_len, cin)
    # print(x.shape)
    print("Test on cpu:")
    attn, spc_label, y = model(x)

    print(f"SPC label: {spc_label.shape}")
    print(f"Series out: {y.shape}")
    print(f"Attention scores: {attn[-1].shape}")
    print("pass\n")

    print("Test on gpu:")
    x = x.cuda()
    model = model.cuda()
    attn, spc_label, y = model(x)
    print(f"SPC label: {spc_label.shape}")
    print(f"Series out: {y.shape}")
    print(f"Attention scores: {attn[-1].shape}")
    print("pass\n")

    config = SPCBertConfig(
        input_dim=cin,
        output_dim=cin,
        spc_rule_num=0,
        num_hidden_layers=3,
        num_attention_heads=n_head,
        hidden_size=hidden_size,
        hidden_act="gelu",
        max_position_embeddings=seq_len,
        norm=True,
        output_attention=False,
    )
    model = SPCBert(config).cuda()
    print("Test 0 spc label on gpu:")
    spc_label, y = model(x)

    print(f"SPC label: {spc_label}")
    print(f"Series out: {y.shape}")
    print("pass\n")
