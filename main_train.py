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
from ts_transformers.data import load_data, fix_seed, get_loader
# from ts_transformers.models.bert import AnomalyBert, AnomalyBertConfig
from ts_transformers.models.spcbert import SPCBert, SPCBertConfig


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default=None)
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

    anomaly_tasks = ["DMDS"]
    parser.add_argument('--task', type=str, default="DMDS",
                        choices=anomaly_tasks)

    # -------- hyperparameter for SPCBERT --------
    parser.add_argument('--model_name', type=str, default="SPCBERT")
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--attention_probs_dropout', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--output_attention', type=int, default=1)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--spc_rule_num', type=int, default=1)
    parser.add_argument('--sensitive_level', type=float, default=4)
    parser.add_argument('--alpha', type=float, default=0.5)
    # parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()
    print('=' * 70)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 70)

    if args.task == "DMDS":
        args.input_dim = 15
        # In DMDS dataset
        # Force to set val_idx = 57030
        #          set testset as 09112001.csv
        args.val_idx = 57030
        args.spc_col = [
            'LC51_03CV_rule1', 'LC51_03X_rule1', 'LC51_03PV_rule1', 'P51_06_rule1',
            'T51_01_rule1', 'F51_01_rule1', 'P57_03_rule1', 'P57_04_rule1',
            'FC57_03PV_rule1', 'FC57_03CV_rule1', 'FC57_03X_rule1', 'F74_00_rule1',
            'LC74_20CV_rule1', 'LC74_20X_rule1', 'LC74_20PV_rule1',
            'LC51_03PV_rule2', 'P57_04_rule2', 'F74_00_rule2', 'LC74_20CV_rule2',
            'LC74_20X_rule2', 'LC74_20PV_rule2', 'LC51_03CV_rule3',
            'LC51_03X_rule3', 'LC51_03PV_rule3', 'P51_06_rule3', 'T51_01_rule3',
            'F51_01_rule3', 'P57_03_rule3', 'P57_04_rule3', 'FC57_03PV_rule3',
            'FC57_03CV_rule3', 'FC57_03X_rule3', 'F74_00_rule3', 'LC74_20CV_rule3',
            'LC74_20X_rule3', 'LC74_20PV_rule3', 'LC51_03CV_rule4',
            'LC51_03X_rule4', 'LC51_03PV_rule4', 'P51_06_rule4', 'F51_01_rule4',
            'P57_04_rule4', 'F74_00_rule4', 'LC74_20CV_rule4', 'LC74_20X_rule4',
            'LC74_20PV_rule4', 'LC51_03CV_rule5', 'LC51_03X_rule5',
            'LC51_03PV_rule5', 'P51_06_rule5', 'F51_01_rule5', 'P57_03_rule5',
            'P57_04_rule5', 'FC57_03PV_rule5', 'FC57_03CV_rule5', 'FC57_03X_rule5',
            'F74_00_rule5', 'LC74_20CV_rule5', 'LC74_20X_rule5', 'LC74_20PV_rule5',
            'LC51_03CV_rule6', 'LC51_03X_rule6', 'LC51_03PV_rule6', 'P51_06_rule6',
            'F51_01_rule6', 'P57_03_rule6', 'P57_04_rule6', 'FC57_03CV_rule6',
            'FC57_03X_rule6', 'F74_00_rule6', 'LC74_20CV_rule6', 'LC74_20X_rule6',
            'LC74_20PV_rule6'
        ]
        args.seq_len = args.seq_len + len(args.spc_col)
        args.spc_rule_num = len(args.spc_col)

    return args


def main() -> None:
    args = parse()
    fix_seed()
    # -------- Prepare dataloader --------
    data_config = SPCAnomalyConfig(
        spc_col=args.spc_col,
        target_col="anomaly_label",
        window_size=args.seq_len,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        val_idx=args.val_idx,
        echo=True,
    )
    train_loader, val_loader, thres_loader, test_loader = get_loader(
        data_config, "DMDS")

    # -------- Prepare model --------
    # model_config = AnomalyBertConfig(
    #     input_dim=args.input_dim,
    #     output_dim=args.input_dim,
    #     num_hidden_layers=args.num_hidden_layers,
    #     num_attention_heads=args.num_attention_heads,
    #     hidden_size=args.hidden_size,
    #     hidden_act="gelu",
    #     attention_probs_dropout_prob=args.attention_probs_dropout,
    #     hidden_dropout_prob=args.hidden_dropout_prob,
    #     max_position_embeddings=args.seq_len,
    #     output_attention=args.output_attention,
    #     norm=args.norm,
    #     k=args.k,
    # )
    # net = AnomalyBert(model_config)
    model_config = SPCBertConfig(
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
        sensitive_level=args.sensitive_level,
    )
    net = SPCBert(model_config)
    if args.load:
        net.load_state_dict(torch.load(args.load))

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-7)

    ckpt_path = f'ckpt/{args.model_name}/'
    os.makedirs(ckpt_path, exist_ok=True)
    model_ckpt = [
        EarlyStopping(target='loss',
                      path=ckpt_path +
                      f"{args.model_name}_{args.task}_ckpt.pt",
                      patience=args.patience,
                      verbose=True,
                      delta=args.delta)
    ]

    trainer = TSRunner(net, model_config, optimizer, model_ckpt, args.device)

    if args.trainable:
        print("Start to train...\n")
        start = time.time()
        trainer.train(args.epochs, train_loader,
                      valid_loader=val_loader, scheduler=scheduler)
        end = time.time()
        print(f'End in {end - start:.4f}s...\n')
        # for i in range(args.spc_rule_num):
        #     recall, precision, f1 = trainer.history._history[f"recall_{i}"][-1], trainer.history._history[
        #         f"precision_{i}"][-1], trainer.history._history[f"f1-score_{i}"][-1]
        #     print(
        #         f"{args.spc_col[i]:18s}: precision({precision:.3f}), recall({recall:.3f}), f1-score({f1:.3f})")
        for key, value in trainer.weights.items():
            torch.save(value.state_dict(), os.path.join(
                ckpt_path, "last_epoch.pt"))

    trainer.set_threshold(thres_loader)
    anomaly_report, spc_report = trainer.test(test_loader)


if __name__ == "__main__":
    main()
