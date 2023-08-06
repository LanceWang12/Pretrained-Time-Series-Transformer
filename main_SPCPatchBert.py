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
    parser.add_argument('--params', type=int, default=1)

    anomaly_tasks = ["DMDS", "SWaT", "SMD"]
    parser.add_argument('--task', type=str, required=True,
                        choices=anomaly_tasks)

    # -------- hyperparameter for SPCBERT --------
    parser.add_argument('--model_name', type=str, default="MultiSPCPatchBert")
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=32)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_attention_heads', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--intermediate_size', type=int, default=256)
    parser.add_argument('--attention_probs_dropout', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--output_attention', type=int, default=1)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--patch_padding', type=int, default=1)
    parser.add_argument('--ensemble', type=int, default=1)
    # DMDS: 20, SWaT:
    parser.add_argument('--sensitive_level', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--load_norm', type=str, default="")
    parser.add_argument('--load_embed', type=str, default="")
    parser.add_argument('--load_encoder', type=str, default="")
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    if args.task == "DMDS":
        args.input_dim = 15
        # In DMDS dataset
        # Force to set val_idx = 57030
        #          set testset as 09112001.csv
        args.val_idx = 57030
        args.spc_col = [
            'LC51_03CV_rule1', 'LC51_03CV_rule3', 'LC51_03CV_rule4',
            'LC51_03CV_rule5', 'LC51_03CV_rule6', 'LC51_03X_rule1',
            'LC51_03X_rule3', 'LC51_03X_rule4', 'LC51_03X_rule5', 'LC51_03X_rule6',
            'LC51_03PV_rule1', 'LC51_03PV_rule2', 'LC51_03PV_rule3',
            'LC51_03PV_rule4', 'LC51_03PV_rule5', 'LC51_03PV_rule6', 'P51_06_rule1',
            'P51_06_rule3', 'P51_06_rule4', 'P51_06_rule5', 'P51_06_rule6',
            'T51_01_rule1', 'T51_01_rule3', 'F51_01_rule1', 'F51_01_rule3',
            'F51_01_rule4', 'F51_01_rule5', 'F51_01_rule6', 'P57_03_rule1',
            'P57_03_rule3', 'P57_03_rule5', 'P57_03_rule6', 'P57_04_rule1',
            'P57_04_rule2', 'P57_04_rule3', 'P57_04_rule4', 'P57_04_rule5',
            'P57_04_rule6', 'FC57_03PV_rule1', 'FC57_03PV_rule3', 'FC57_03PV_rule5',
            'FC57_03CV_rule1', 'FC57_03CV_rule3', 'FC57_03CV_rule5',
            'FC57_03CV_rule6', 'FC57_03X_rule1', 'FC57_03X_rule3', 'FC57_03X_rule5',
            'FC57_03X_rule6', 'F74_00_rule1', 'F74_00_rule2', 'F74_00_rule3',
            'F74_00_rule4', 'F74_00_rule5', 'F74_00_rule6', 'LC74_20CV_rule1',
            'LC74_20CV_rule2', 'LC74_20CV_rule3', 'LC74_20CV_rule4',
            'LC74_20CV_rule5', 'LC74_20CV_rule6', 'LC74_20X_rule1',
            'LC74_20X_rule2', 'LC74_20X_rule3', 'LC74_20X_rule4', 'LC74_20X_rule5',
            'LC74_20X_rule6', 'LC74_20PV_rule1', 'LC74_20PV_rule2',
            'LC74_20PV_rule3', 'LC74_20PV_rule4', 'LC74_20PV_rule5',
            'LC74_20PV_rule6'
        ]
        args.spc_head_lst = [5, 5, 6, 5, 2, 5, 4, 6, 3, 4, 4, 6, 6, 6, 6]
        args.spc_rule_num = len(args.spc_col)
        # args.sensitive_level = 20
    elif args.task == "SWaT":
        args.input_dim = 30
        # In SWaT dataset
        # Train[380000: 440000], Val[450000: 470000], Test[478254: 498254]
        args.spc_col = [
            'FIT101_rule1', 'LIT101_rule1', 'MV101_rule1', 'P101_rule1', 'AIT203_rule1', 'FIT201_rule1', 'MV201_rule1', 'P203_rule1', 'P205_rule1', 'DPIT301_rule1', 'FIT301_rule1', 'LIT301_rule1', 'MV302_rule1', 'MV304_rule1', 'P302_rule1', 'AIT402_rule1', 'FIT401_rule1', 'LIT401_rule1', 'P402_rule1', 'UV401_rule1', 'AIT501_rule1', 'AIT502_rule1', 'FIT501_rule1', 'FIT502_rule1', 'FIT503_rule1', 'FIT504_rule1', 'P501_rule1', 'PIT501_rule1', 'PIT502_rule1', 'PIT503_rule1', 'P402_rule2', 'UV401_rule2', 'FIT502_rule2', 'P501_rule2', 'FIT101_rule3', 'LIT101_rule3', 'MV101_rule3', 'P101_rule3', 'AIT203_rule3', 'FIT201_rule3', 'MV201_rule3', 'P203_rule3', 'P205_rule3', 'DPIT301_rule3', 'FIT301_rule3', 'LIT301_rule3', 'MV302_rule3', 'MV304_rule3', 'P302_rule3', 'AIT402_rule3', 'FIT401_rule3', 'LIT401_rule3', 'P402_rule3', 'UV401_rule3', 'AIT501_rule3', 'AIT502_rule3', 'FIT501_rule3', 'FIT502_rule3', 'FIT503_rule3', 'FIT504_rule3', 'P501_rule3', 'PIT501_rule3', 'PIT502_rule3',
            'PIT503_rule3', 'FIT101_rule4', 'MV101_rule4', 'P101_rule4', 'FIT201_rule4', 'MV201_rule4', 'P203_rule4', 'P205_rule4', 'DPIT301_rule4', 'FIT301_rule4', 'MV302_rule4', 'MV304_rule4', 'P302_rule4', 'FIT401_rule4', 'P402_rule4', 'UV401_rule4', 'FIT501_rule4', 'FIT502_rule4', 'FIT503_rule4', 'FIT504_rule4', 'P501_rule4', 'PIT502_rule4', 'FIT101_rule5', 'LIT101_rule5', 'P101_rule5', 'AIT203_rule5', 'FIT201_rule5', 'MV201_rule5', 'P203_rule5', 'P205_rule5', 'LIT301_rule5', 'AIT402_rule5', 'FIT401_rule5', 'LIT401_rule5', 'AIT501_rule5', 'AIT502_rule5', 'FIT501_rule5', 'FIT502_rule5', 'FIT503_rule5', 'FIT504_rule5', 'PIT501_rule5', 'PIT503_rule5', 'FIT101_rule6', 'MV101_rule6', 'P101_rule6', 'FIT201_rule6', 'MV201_rule6', 'P203_rule6', 'P205_rule6', 'DPIT301_rule6', 'FIT301_rule6', 'MV302_rule6', 'MV304_rule6', 'P302_rule6', 'FIT401_rule6', 'P402_rule6', 'UV401_rule6', 'FIT501_rule6', 'FIT502_rule6', 'FIT503_rule6', 'FIT504_rule6', 'P501_rule6', 'PIT501_rule6', 'PIT502_rule6', 'PIT503_rule6'
        ]
        args.spc_head_lst = [5, 3, 4, 5, 3, 5, 5, 5, 5, 4, 4, 3,
                             4, 4, 4, 3, 5, 3, 5, 5, 3, 3, 5, 6, 5, 5, 5, 4, 4, 4]
        args.spc_rule_num = len(args.spc_col)

    print('=' * 70)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 70)

    return args


def main() -> None:
    args = parse()
    fix_seed()
    # -------- Prepare dataloader --------
    data_config = SPCAnomalyConfig(
        spc_col=args.spc_col,
        target_col="anomaly_label",
        time_idx=None,
        window_size=args.window_size,
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        val_idx=args.val_idx,
        padding=False,
        echo=True,
    )
    train_loader, val_loader, thres_loader, test_loader = get_loader(
        data_config, args.task)

    # seq_len = original_seq_len + a spc token
    model_config = SPCBertConfig(
        input_dim=args.input_dim,
        output_dim=args.input_dim,
        window_size=args.window_size,
        spc_rule_num=args.spc_rule_num,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
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
        load_norm=args.load_norm,
        load_embed=args.load_embed,
        load_encoder=args.load_encoder,
        verbose=args.verbose,
    )
    net = SPCPatchBert(model_config)

    if args.params:
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameter: %.2fM\n" % (total / 1e6))

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
