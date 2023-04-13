import torch

from ts_transformers.models.bert import AnomalyBert
from ts_transformers.models.bert import AnomalyBertConfig
from ts_transformers.data import SPCAnomalyConfig
from ts_transformers.data import load_data, fix_seed, get_loader

# -------- Hyperparameter Setting --------
input_dim = 38
batch_size = 32
seq_len = 256
output_attention = True
test_size = 0.2
val_size = 0.2


def test_model() -> None:
    print("Start model test...")
    # -------- Model Preparation --------
    config = AnomalyBertConfig(
        input_dim=input_dim,
        output_dim=input_dim,
        num_hidden_layers=3,
        num_attention_heads=8,
        hidden_size=512,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        max_position_embeddings=seq_len,
        output_attention=output_attention,
        norm=True,
    )

    model = AnomalyBert(config)

    # -------- Data Preparation --------
    X = torch.rand(batch_size, seq_len, input_dim)

    # -------- Test --------
    # test on cpu
    print("Test on cpu: ", end="")
    Y = model(X)[0]
    assert (X.shape == Y.shape)
    print("pass\n")

    # test on gpu
    print("Test on gpu ", end="")
    X = X.cuda()
    model = model.cuda()
    Y = model(X)[0]
    assert (X.shape == Y.shape)
    print("pass\n")

    print(f"Input.shape = {X.shape}")
    print(f"Output.shape = {Y.shape}")
    print("End model test...\n")


def test_data() -> None:
    print("Start data test...")
    fix_seed()
    # -------- Prepare dataloader --------
    data_config = SPCAnomalyConfig(
        spc_col=["fault_label", "T74_30"], # just mimic some spc label
        target_col="anomaly_label",
        window_size=seq_len,
        batch_size=batch_size,
        test_size=test_size,
        val_size=val_size,
    )
    train_loader, val_loader, test_loader = get_loader(data_config, "DMDS")

    print("Test DataLoader: ", end="")
    try:
        counter = 0
        for x, spc, y in train_loader:
            if not counter:
                print(x.shape, spc.shape, y.shape)
                print(x[0])
                counter += 1
            pass
    except IndexError as idx_err:
        print("DataLoader error..., please check!\n")
        print(idx_err)
        exit(1)
    else:
        print("pass\n")
    print("End data test\n")


if __name__ == "__main__":
    # test_model()
    test_data()
