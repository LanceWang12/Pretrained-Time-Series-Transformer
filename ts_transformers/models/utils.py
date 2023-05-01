from typing import Sequence, Dict, Union, Any
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score


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
            self,
            label_num: int,
            metrics: Sequence[str] = [
                'loss', 'recall', 'precision', 'f1-score'],
    ) -> None:
        self.label_num = label_num
        tmp = [f"tp_{i}" for i in range(label_num)]
        tmp += [f"fp_{i}" for i in range(label_num)]
        tmp += [f"fn_{i}" for i in range(label_num)]
        tmp += [f"tn_{i}" for i in range(label_num)]
        tmp += [f"recall_{i}" for i in range(label_num)]
        tmp += [f"precision_{i}" for i in range(label_num)]
        tmp += [f"f1-score_{i}" for i in range(label_num)]
        self.metrics = metrics

        self._history = {
            'count': [],
            'loss': [],
            'recall': [],
            'precision': [],
            'f1-score': []
        }

        for key in tmp:
            self._history[key] = []

    def __str__(self) -> str:
        results = []
        for metric in self.metrics:
            results.append(f'{metric}: {self._history[metric][-1]:.4f}')

        return ', '.join(results)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, float]]:
        results = {}

        for metric in self.metrics:
            results[metric] = self._history[metric][idx]

        return results

    def reset(self) -> None:
        for key in self._history.keys():
            self._history[key].clear()

    def log(self, key: str, value: Any, idx: int = None) -> None:
        self._history[key].append(value)
        if idx is not None and\
                len(
                    set([
                        len(self._history[f"tp_{idx}"]),
                        len(self._history[f"fp_{idx}"]),
                        len(self._history[f"tn_{idx}"]),
                        len(self._history[f"fn_{idx}"])])
                ) == 1 and\
                len(self._history['count']) == len(self._history[f'tp_{idx}']) and\
                len(self._history['count']) > len(self._history[f'f1-score_{idx}']):
            recall = self._history[f"tp_{idx}"][-1] / (
                self._history[f"tp_{idx}"][-1] + self._history[f"fn_{idx}"][-1])
            precision = self._history[f"tp_{idx}"][-1] / (
                self._history[f"tp_{idx}"][-1] + self._history[f"fp_{idx}"][-1])
            f1 = 2 / (1 / precision + 1 / recall)
            self._history[f"recall_{idx}"].append(recall)
            self._history[f"precision_{idx}"].append(precision)
            self._history[f"f1-score_{idx}"].append(f1)

            if idx == (self.label_num - 1):
                recall_avg = sum([self._history[f"recall_{i}"][-1] for i in range(
                    self.label_num)]) / self.label_num
                precision_avg = sum([self._history[f"precision_{i}"][-1] for i in range(
                    self.label_num)]) / self.label_num
                f1_avg = sum([self._history[f"f1-score_{i}"][-1] for i in range(
                    self.label_num)]) / self.label_num
                self._history['recall'].append(recall_avg)
                self._history['precision'].append(precision_avg)
                self._history['f1-score'].append(f1_avg)

    def summary(self) -> None:
        _count = sum(self._history['count'])
        if _count == 0:
            _count = 1

        _loss = sum(self._history['loss']) / len(self._history['loss'])
        self._history["loss"].append(_loss)

        target_lst = ['tp', 'fp', 'tn', 'fn']
        for i in range(self.label_num):
            # summarize confusion matrix
            for target in target_lst:
                tmp = sum(self._history[f"{target}_{i}"])
                self._history[f"{target}_{i}"].append(tmp)

            # summarize recall, precision, f1-score
            recall = self._history[f"tp_{i}"][-1] / (
                self._history[f"tp_{i}"][-1] + self._history[f"fn_{i}"][-1])
            precision = self._history[f"tp_{i}"][-1] / (
                self._history[f"tp_{i}"][-1] + self._history[f"fp_{i}"][-1])
            f1 = 2 / (1 / precision + 1 / recall)
            self._history[f"recall_{i}"].append(recall)
            self._history[f"precision_{i}"].append(precision)
            self._history[f"f1-score_{i}"].append(f1)

        recall_avg = sum([self._history[f"recall_{i}"][-1] for i in range(
            self.label_num)]) / self.label_num
        precision_avg = sum([self._history[f"precision_{i}"][-1] for i in range(
            self.label_num)]) / self.label_num
        f1_avg = sum([self._history[f"f1-score_{i}"][-1] for i in range(
            self.label_num)]) / self.label_num

        self._history['recall'].append(recall_avg)
        self._history['precision'].append(precision_avg)
        self._history['f1-score'].append(f1_avg)


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
