from typing import Sequence, Dict, Union, Any
import torch
import torch.cuda as cuda


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
    def __init__(
            self, metrics: Sequence[str] = ['loss', 'accuracy'],
            additional_keys: Sequence[str] = []
    ) -> None:
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
            # print(key)
            _value = sum(self._history[key]) / len(self._history[key])
            self._history[key].append(_value)


class _BaseRunner:
    def __init__(self, device='cuda') -> None:
        self.device = device if cuda.is_available() else 'cpu'

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
