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


class SPCBert(nn.Module):
    def __init__(self, config: SPCBertConfig) -> None:
        super(SPCBert, self).__init__()
        self.embed = DataEmbedding(
            config.input_dim, config.hidden_size, config.hidden_dropout_prob,
            norm=config.norm, mode=config.mode, mean_lr=config.mean_lr,
            gate_lr=config.gate_lr, scale_lr=config.scale_lr
        )
        if config.backbone == "bert":
            self.encoder = BertEncoder(config)
        else:
            raise NotImplementedError("Other backbones are not implemented.")

        self.spc_rule_num = config.spc_rule_num
        if self.spc_rule_num:
            self.spc_heads = nn.Linear(config.hidden_size, 1)
        self.output_heads = nn.Linear(config.hidden_size, config.output_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        x = self.encoder(x)['last_hidden_state']
        if self.spc_rule_num:
            spc_pred = x[:, : self.spc_rule_num, :]
            spc_pred = self.spc_heads(spc_pred)
        else:
            spc_pred = None
        series_out = x[:, self.spc_rule_num:, :]
        series_out = self.output_heads(series_out)

        return spc_pred, series_out


if __name__ == "__main__":
    batch_size, cin, seq_len, hidden_size = 64, 38, 100, 128
    n_head = 8
    config = SPCBertConfig(
        input_dim=cin,
        output_dim=cin,
        spc_rule_num=4,
        num_hidden_layers=3,
        num_attention_heads=n_head,
        hidden_size=hidden_size,
        hidden_act="gelu",
        max_position_embeddings=seq_len,
        norm=True,
    )
    model = SPCBert(config)

    x = torch.rand(batch_size, seq_len, cin)
    # print(x.shape)
    print("Test on cpu:")
    spc_label, y = model(x)

    print(f"SPC label: {spc_label.shape}")
    print(f"Series out: {y.shape}")
    print("pass\n")

    print("Test on gpu:")
    x = x.cuda()
    model = model.cuda()
    spc_label, y = model(x)
    print(f"SPC label: {spc_label.shape}")
    print(f"Series out: {y.shape}")
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
    )
    model = SPCBert(config).cuda()
    print("Test 0 spc label on gpu:")
    spc_label, y = model(x)

    print(f"SPC label: {spc_label}")
    print(f"Series out: {y.shape}")
    print("pass\n")
