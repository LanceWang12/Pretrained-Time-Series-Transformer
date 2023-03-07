import torch

from ts_transformers.models.bert import AnomalyBert
from ts_transformers.models.bert import AnomalyBertConfig

# -------- Hyperparameter Setting --------
input_dim = 38
batch_size = 32
seq_len = 256
output_attention = True

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
print("Test on cpu...")
Y = model(X)[0]
assert (X.shape == Y.shape)
print("pass\n")

# test on gpu
print("Test on gpu...")
X = X.cuda()
model = model.cuda()
Y = model(X)[0]
assert (X.shape == Y.shape)
print("pass\n")

print(f"Input.shape = {X.shape}")
print(f"Output.shape = {Y.shape}")
