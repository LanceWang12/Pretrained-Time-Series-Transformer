from typing import Optional


class TSAnomalyConfig(object):
    def __init__(
        self,
        target_col: str,
        time_idx: Optional[str] = None,
        window_size: int = 256,
        batch_size: int = 32,
        val_size: float = 0.2,
        val_idx: int = None,
        test_size: float = 0.2,
        test_idx: int = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.time_idx = time_idx
        self.target_col = target_col
        self.window_size = window_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers


class SPCAnomalyConfig(TSAnomalyConfig):
    def __init__(
        self,
        spc_col: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spc_col = spc_col
