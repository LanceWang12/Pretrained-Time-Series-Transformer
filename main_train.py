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
from ts_transformers.models.bert import AnomalyBert, AnomalyBertConfig
from ts_transformers.models.ckpt import EarlyStopping
from ts_transformers.models import TSRunner
from ts_transformers.data import SPCAnomalyConfig
from ts_transformers.data import load_data, fix_seed, get_loader


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--trainable', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)

    anomaly_tasks = ["DMDS"]
    parser.add_argument('--task', type=str, default="DMDS",
                        choices=anomaly_tasks)

    # -------- hyperparameter for SPCBERT --------
    parser.add_argument('--model_name', type=str, default="SPCBERT")
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=38)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--attention_probs_dropout', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--output_attention', type=int, default=1)
    parser.add_argument('--norm', type=int, default=1)
    args = parser.parse_args()
    print('=' * 70)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 70)

    if args.task == "DMDS":
        args.input_dim = 33

    return args


def main() -> None:
    args = parse()
    fix_seed()
    # -------- Prepare dataloader --------
    data_config = SPCAnomalyConfig(
        spc_col="fault_label",
        target_col="anomaly_label",
        window_size=args.seq_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    train_loader, val_loader, test_loader = get_loader(data_config, "DMDS")

    # -------- Prepare model --------
    model_config = AnomalyBertConfig(
        input_dim=args.input_dim,
        output_dim=args.input_dim,
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
    net = AnomalyBert(model_config)

    if args.load:
        net.load_state_dict(torch.load(args.load))

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-7)

    ckpt_path = f'ckpt/{args.model_name}/'
    os.makedirs(ckpt_path, exist_ok=True)
    model_ckpt = [
        EarlyStopping(
            patience=args.patience,
            verbose=True,
            delta=args.delta,
            path=ckpt_path + f"{args.model_name}_ckpt.pt"
        )
    ]

    trainer = TSRunner(
        net, model_config,
        optimizer, criterion,
        model_ckpt, args.device
    )

    if args.trainable:
        print("Start to train...\n")
        start = time.time()
        trainer.train(
            args.epochs, train_loader,
            valid_loader=val_loader, scheduler=scheduler
        )
        end = time.time()
        print(f'End in {end - start:.4f}s...\n')

        for key, value in trainer.weights.items():
            torch.save(value.state_dict(), os.path.join(
                ckpt_path, "last_epoch.pt"))

    trainer.test(test_loader)


if __name__ == "__main__":
    main()
