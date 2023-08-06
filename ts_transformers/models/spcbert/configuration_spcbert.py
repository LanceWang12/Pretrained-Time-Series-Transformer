from transformers import BertConfig


class SPCBertConfig(BertConfig):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        spc_rule_num: int,
        spc_head_lst: list = [],
        backbone: str = "bert",
        output_attention: bool = True,
        verbose: bool = False,

        # for RevIN
        norm: bool = True,
        eps: float = 1e-5,
        affine: bool = True,

        mode: str = "full",
        sensitive_level: int = 0,
        alpha: float = 0.5,
        load_norm: str = "",
        load_embed: str = "",
        load_encoder: str = "",

        # for patch version
        window_size: int = 256,
        patch_len: int = 16,
        stride: int = 1,
        padding_patch: bool = True,
        individual: bool = False,

        # ensemble
        ensemble: bool = True,

        # pattern matching
        target_features_idx: list = [],
        top_k_report: int = 600,
        ** kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.spc_rule_num = spc_rule_num
        self.spc_head_lst = spc_head_lst
        self.backbone = backbone
        self.output_attention = output_attention
        self.verbose = verbose

        # RevIN
        self.norm = norm
        self.eps = eps
        self.affine = affine
        self.load_norm = load_norm
        self.load_embed = load_embed
        self.load_encoder = load_encoder

        self.mode = mode
        self.sensitive_level = sensitive_level

        # total_loss = alpha * reconstruction + (1 - alpha) * spc_label
        self.alpha = alpha

        # Patchify1D
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.individual = individual

        # Each feature has its own BERT
        self.ensemble = ensemble

        # pattern matching
        self.target_features_idx = target_features_idx
        self.top_k_report = top_k_report
