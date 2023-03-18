from typing import Optional

class TSAnomalyConfig(object):
    def __init__(
        self,
        target_col: str,
        time_idx: Optional[str] = None,
        window_size: int = 256,
    ) -> None:
        super().__init__()
        self.time_idx = time_idx
        self.target_col = target_col
        self.window_size = window_size

class SPCAnomalyConfig(TSAnomalyConfig):
    def __init__(
        self,
        target_col: str,
        spc_col: str,
        time_idx: Optional[str] = None,
        window_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(target_col, time_idx, window_size)
        self.spc_col = spc_col
