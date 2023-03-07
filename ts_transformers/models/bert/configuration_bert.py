from transformers import BertConfig


class AnomalyBertConfig(BertConfig):
    def __init__(
        self,
        input_dim,
        output_dim,
        output_attention=True,
        norm=True,
        mode="adaptive_avg",
        mean_lr=0.00001,
        gate_lr=0.001,
        scale_lr=0.00001,
        ** kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_attention = output_attention
        self.norm = norm
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr
