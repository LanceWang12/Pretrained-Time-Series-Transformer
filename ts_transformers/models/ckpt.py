import os
import torch

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
        '''Saves model when validation accuracy decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation accuracy increased ({self.val_accuracy_min:.6f} \
                    --> {val_accuracy:.6f}).  Saving model ...\n'
            )
        torch.save(model.state_dict(), self.path)
        self.val_accuracy_min = val_accuracy
