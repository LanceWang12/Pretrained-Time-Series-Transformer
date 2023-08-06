# -------- general --------
import os
import time
import argparse
import pandas as pd

# -------- Torch --------
import torch
import torch.nn as nn
import torch.optim as optim

# -------- My Lib ---------
from ts_transformers.models import TSRunner
from ts_transformers.data import SPCAnomalyConfig
from ts_transformers.models.ckpt import EarlyStopping
from ts_transformers.data import fix_seed, get_loader
# from ts_transformers.models.bert import AnomalyBert, AnomalyBertConfig
from ts_transformers.models.spcbert import SPCPatchBert, SPCBertConfig


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--trainable', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--delta', type=float, default=1e-2)
    parser.add_argument('--val_size', type=float, default=None)
    parser.add_argument('--val_idx', type=float, default=None)
    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--params', type=int, default=1)

    anomaly_tasks = ["DMDS", "SWaT"]
    parser.add_argument('--task', type=str, required=True,
                        choices=anomaly_tasks)

    # -------- hyperparameter for SPCBERT --------
    parser.add_argument('--model_name', type=str,
                        default="Norm & Embedding Layer")
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--attention_probs_dropout', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--output_attention', type=int, default=1)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--patch_padding', type=int, default=1)
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--sensitive_level', type=float, default=20)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    if args.task == "DMDS":
        args.model_name = f"{args.model_name}_{args.task}"
        args.input_dim = 15
        # In DMDS dataset
        # Force to set val_idx = 57030
        #          set testset as 09112001.csv
        args.spc_head_lst = [5, 5, 6, 5, 2, 5, 4, 6, 3, 4, 4, 6, 6, 6, 6]
        args.spc_rule_num = sum(args.spc_head_lst)

        print('=' * 70)
        for key, value in vars(args).items():
            print(f'{key}: {value}')
        print('=' * 70)

    elif args.task == "SWaT":
        args.input_dim = 30
        # In SWaT dataset
        # Train[380000: 440000], Val[450000: 470000], Test[478254: 498254]
        args.spc_head_lst = [5, 3, 4, 5, 3, 5, 5, 5, 5, 4, 4, 3,
                             4, 4, 4, 3, 5, 3, 5, 5, 3, 3, 5, 6, 5, 5, 5, 4, 4, 4]
        args.spc_rule_num = sum(args.spc_head_lst)

    return args


def main() -> None:
    args = parse()
    fix_seed()

    # seq_len = original_seq_len + a spc token
    model_config = SPCBertConfig(
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
        sensitive_level=args.sensitive_level,
        ensemble=args.ensemble,
        verbose=args.verbose,
    )
    net = SPCPatchBert(model_config)

    net.load_state_dict(torch.load(args.load))

    save_dir = "./ckpt/Norm"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"{save_dir} is created.")
    save_file = f"{save_dir}/NormLayer_{args.task}.pt"
    print(f"Saving the norm layer from {args.load} to {save_file}")
    torch.save(net.norm.state_dict(), save_file)

    save_dir = "./ckpt/Embed"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"{save_dir} is created.")
    save_file = f"{save_dir}/EmbeddingLayer_{args.task}.pt"
    print(f"Saving the embedding layer from {args.load} to {save_file}")
    torch.save(net.embed.state_dict(), save_file)

    save_dir = "./ckpt/Encoder"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"{save_dir} is created.")
    save_file = f"{save_dir}/EncoderBackbone_{args.task}.pt"
    print(f"Saving the Encoder backbone from {args.load} to {save_file}")
    torch.save(net.encoder.state_dict(), save_file)


if __name__ == "__main__":
    main()
