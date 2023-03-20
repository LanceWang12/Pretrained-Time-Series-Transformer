import os
import numpy as np
import torch

'''
# old version for one loss
class EarlyStopping:
    """
        Early stops the training if validation loss doesn't
        improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0,
                 path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool):
                If True, prints a message for each validation loss improvement.
                Default: False
            delta (float):
                Minimum change in the monitored quantity to qualify as an
                improvement.
                Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_min = 0.0
        self.delta = delta

        self.path = path
        directory = '/'.join(path.split('/')[: -1])
        os.makedirs(directory, exist_ok=True)

        self.trace_func = trace_func

    def __call__(self, val_accuracy, model):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.trace_func('Early stop!!!!!\n')
                return False
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

        return True

    def save_checkpoint(self, val_accuracy, model):
        """Saves model when validation accuracy decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation accuracy increased ({self.val_accuracy_min:.6f} \
                    --> {val_accuracy:.6f}).  Saving model ...\n'
            )
        torch.save(model.state_dict(), self.path)
        self.val_accuracy_min = val_accuracy

'''

# -------- New version for two loss ---------


class EarlyStopping:
    def __init__(self, path, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, val_loss2, model):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, self.path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, self.path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  \
                    Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
