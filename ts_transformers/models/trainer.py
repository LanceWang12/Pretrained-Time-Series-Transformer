from typing import Any, Callable, Optional, Sequence, Dict, Tuple, Union, List
import random
import numpy as np
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from sklearn.metrics import r2_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from .bert import AnomalyBertConfig


class ProgressBar:
    last_length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int,
             total: int, newline: bool = False) -> None:
        progress = (current + 1) / total
        if current == total:
            progress = 1

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = ''

        if len(prefix) > 0:
            message += f'{prefix}, [{progress_bar:<20}]'

            if not newline:
                message += f' {current_progress:6.2f}%'

        if len(postfix) > 0:
            message += f', {postfix}'

        print(f'\r{" " * ProgressBar.last_length}', end='')
        print(f'\r{message}', end='')

        if newline:
            print()
            ProgressBar.last_length = 0
        else:
            ProgressBar.last_length = len(message) + 1


class _History:
    def __init__(self, metrics: Sequence[str] = ['loss', 'accuracy'],
                 additional_keys: Sequence[str] = []) -> None:
        self.metrics = metrics
        self.additional_keys = additional_keys

        self._history = {
            'count': [],
            'loss': [],
            'correct': [],
            'accuracy': []
        }

        for key in self.additional_keys:
            self._history[key] = []

    def __str__(self) -> str:
        results = []

        for metric in self.metrics:
            results.append(f'{metric}: {self._history[metric][-1]:.6f}')

        return ', '.join(results)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        results = {}

        for metric in self.metrics:
            results[metric] = self._history[metric][idx]

        return results

    def reset(self) -> None:
        for key in self._history.keys():
            self._history[key].clear()

    def log(self, key: str, value: Any) -> None:
        self._history[key].append(value)

        if len(self._history['count']) == len(self._history['correct']) and \
                len(self._history['count']) > len(self._history['accuracy']):
            self._history['accuracy'].append(
                self._history['correct'][-1] / self._history['count'][-1])

    def summary(self) -> None:
        _count = sum(self._history['count'])
        if _count == 0:
            _count = 1

        _loss = sum(self._history['loss']) / len(self._history['loss'])
        _correct = sum(self._history['correct'])
        _accuracy = _correct / _count

        self._history['count'].append(_count)
        self._history['loss'].append(_loss)
        self._history['correct'].append(_correct)
        self._history['accuracy'].append(_accuracy)

        for key in self.additional_keys:
            _value = sum(self._history[key]) / len(self._history[key])
            self._history[key].append(_value)


class _BaseRunner:
    def __init__(self, device='cuda') -> None:
        self.device = device if cuda.is_available() else 'cpu'

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


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

        self.history = _History(metrics=['loss', 'accuracy'])
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_ckpt = model_ckpt
        self.config = config

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

    def train(self, epochs: int, train_loader: DataLoader,
              valid_loader: Optional[DataLoader] = None, scheduler: Any = None) -> None:
        epoch_length = len(str(epochs))
        for epoch in range(epochs):
            self.net.train()
            for i, (x, spc, y) in enumerate(train_loader):
                running_loss = self._step(x, y)

                self.optimizer.zero_grad()
                running_loss.backward()  # retain_graph = True)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

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
                if not flag:
                    break

            if scheduler:
                scheduler.step()

    @torch.no_grad()
    def val(self, test_loader: DataLoader) -> None:
        self.net.eval()
        flag = True
        for i, (x, spc, y) in enumerate(test_loader):
            running_loss = self._step(x, y)
            prefix = 'Val'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        self.history.summary()

        prefix = 'Val'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        if self.model_ckpt is not None:
            flag = self.model_ckpt(self.history[-1]['loss'], self.net)

        self.history.reset()
        # return flag

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
