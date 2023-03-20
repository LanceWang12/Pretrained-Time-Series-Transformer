# -------- general --------
from typing import Any, Callable, Optional, Tuple

# -------- torch --------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------- my lib --------
from .bert import AnomalyBertConfig
from .utils import ProgressBar, _History, _BaseRunner, my_kl_loss


class TSRunner(_BaseRunner):
    def __init__(
        self,
        net,
        config: AnomalyBertConfig,
        optimizer: optim.Optimizer,
        criterion: Callable,
        model_ckpt: Optional[Callable] = None,
        device: str = 'cuda'
    ) -> None:
        super().__init__(device=device)

        self.history = _History(
            metrics=['loss', 'accuracy'], additional_keys=['loss2']
        )
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_ckpt = model_ckpt
        self.config = config

    @torch.no_grad()
    def _step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        x = x.to(self.device)
        y = y.to(self.device)

        if self.config.output_attention:
            output, series, prior, sigmas = self.net(x)
        else:
            output = self.net(x)

        y_hat = torch.argmax(output, dim=-1)
        running_loss = self.criterion(output, y.squeeze())

        nn.utils.clip_grad_value_(self.net.parameters(), clip_value=1.0)

        self.history.log('count', y.shape[0])
        self.history.log('loss', running_loss)
        self.history.log('correct', int(
            torch.sum(y_hat == y.squeeze()).item()))

        return running_loss

    def _train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        y = y.to(self.device)

        if self.config.output_attention:
            output, series, prior, _ = self.net(x)
        else:
            output = self.net(x)

        # calculate Association discrepancy
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                .repeat(1, 1, 1, self.config.max_position_embeddings)).detach()))
                + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1)
                    .repeat(1, 1, 1, self.config.max_position_embeddings)).detach(), series[u])))
            prior_loss += (torch.mean(my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1),
                 dim=-1).repeat(1, 1, 1, self.config.max_position_embeddings)),
                series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(
                    torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.config.max_position_embeddings)))))

        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        rec_loss = self.criterion(output, x)

        loss1 = rec_loss - self.config.k * series_loss
        loss2 = rec_loss + self.config.k * prior_loss

        self.history.log('count', y.shape[0])
        self.history.log('loss', loss1)
        self.history.log('loss2', loss2)
        self.history.log('correct', y.shape[0])

        return loss1, loss2

    def train(
            self, epochs: int,
            train_loader: DataLoader,
            valid_loader: Optional[DataLoader] = None,
            scheduler: Any = None
    ) -> None:
        epoch_length = len(str(epochs))
        for epoch in range(epochs):
            self.net.train()
            for i, (x, spc, y) in enumerate(train_loader):
                loss1, loss2 = self._train_step(x, y)

                self.optimizer.zero_grad()
                loss1.backward(retain_graph=True)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = str(self.history)
                ProgressBar.show(prefix, postfix, i, len(train_loader))

            self.history.summary()

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, len(train_loader),
                             len(train_loader), newline=True)

            self.history.reset()

            if valid_loader:
                flag = self.val(valid_loader)
                if self.model_ckpt.early_stop:
                    break

            if scheduler:
                scheduler.step()

    @torch.no_grad()
    def val(self, test_loader: DataLoader) -> bool:
        self.net.eval()
        flag = True
        for i, (x, spc, y) in enumerate(test_loader):
            loss1, loss2 = self._train_step(x, y)
            prefix = 'Val'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        self.history.summary()

        prefix = 'Val'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        if self.model_ckpt is not None:
            # val_loss, val_loss2, model
            self.model_ckpt(self.history[-1]['loss'],
                            self.history[-1]['loss2'], self.net)

        self.history.reset()
        return flag

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()

        for i, (x, spc, y) in enumerate(test_loader):
            running_loss = self._step(x, y)
            prefix = 'Test'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        self.history.summary()

        prefix = 'Test'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        self.history.reset()

    @property
    @torch.no_grad()
    def weights(self):
        return {'net': self.net}
