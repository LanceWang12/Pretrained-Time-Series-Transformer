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
        output_every_anomaly_label: bool = False,
        num_workers: int = 4,
        echo: bool = True,
    ) -> None:
        super().__init__()
        self.time_idx = time_idx
        self.target_col = target_col
        self.window_size = window_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size
        self.test_idx = test_idx
        self.val_idx = val_idx

        # True: Anomaly_Label[idx: idx + wnd_size]
        # False: Anomaly_Label[idx + wnd_size]
        self.output_every_anomaly_label = output_every_anomaly_label
        self.num_workers = num_workers
        self.echo = echo


class SPCAnomalyConfig(TSAnomalyConfig):

    def __init__(
        self,
        spc_col: list,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spc_col = spc_col
