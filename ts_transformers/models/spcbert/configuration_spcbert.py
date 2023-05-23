from transformers import BertConfig


class SPCBertConfig(BertConfig):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spc_rule_num: int,
        backbone: str = "bert",
        output_attention: bool = True,
        norm: bool = True,
        mode: str = "full",
        sensitive_level: int = 0,
        alpha: float = 0.5,
        ** kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spc_rule_num = spc_rule_num
        self.backbone = backbone
        self.output_attention = output_attention
        self.norm = norm
        self.mode = mode
        self.sensitive_level = sensitive_level

        # total_loss = alpha * reconstruction + (1 - alpha) * spc_label
        self.alpha = alpha
