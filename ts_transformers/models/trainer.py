# -------- general --------
from typing import Any, Callable, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import time
import numpy as np

# -------- torch --------
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------- my lib --------
# from .bert import AnomalyBertConfig
from .spcbert import SPCBertConfig
from .utils import ProgressBar, _History, my_kl_loss


class _BaseRunner:
    def __init__(self, device='cuda') -> None:
        self.device = device if cuda.is_available() else 'cpu'

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


class TSRunner(_BaseRunner):

    def __init__(self,
                 net,
                 config: SPCBertConfig,
                 optimizer: optim.Optimizer,
                 model_ckpt: Optional[Callable] = None,
                 device: str = 'cuda') -> None:
        super().__init__(device=device)

        self.history = _History(
            metrics=['loss', 'recall', 'precision', 'f1-score'],
            label_num=config.spc_rule_num,
        )
        self.net = net.to(self.device)
        self.optimizer = optimizer
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        # self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.model_ckpt = model_ckpt
        self.config = config

    def _train_step(self, x: torch.Tensor, x_pad: torch.Tensor, spc_target: torch.Tensor,
                    y: torch.Tensor, val: bool = False) -> torch.Tensor:
        x = x.to(self.device)
        x_pad = x_pad.to(self.device)
        y = y.to(self.device)
        spc_target = spc_target.to(self.device)

        if self.config.output_attention:
            attn, norm_out, spc_out, out = self.net(x_pad)
        else:
            norm_out, spc_out, out = self.net(x_pad)

        if val:
            # compute the relative distance on validation set
            mse_loss = self.mse(out, norm_out)
        else:
            # compute the absolute distance on training set
            mse_loss = self.mse(out, x)

        spc_target = spc_target.unsqueeze(dim=-1)
        assert torch.isfinite(mse_loss), f"mse_loss is infinite: {mse_loss}"

        spc_target = spc_target.squeeze(-1)
        bce_loss = self.bce(spc_out, spc_target)
        loss = self.config.alpha * mse_loss + \
            (1 - self.config.alpha) * bce_loss
        self.history.log('count', y.shape[0])
        self.history.log('loss', loss)

        # compute tp, fp, fn, tn
        spc_out = spc_out.squeeze(-1).detach().cpu().numpy()
        spc_out = spc_out > 0.5
        spc_target = spc_target.squeeze(-1).detach().cpu().numpy()
        for i in range(self.config.spc_rule_num):
            y_true, y_pred = spc_target[:, i], spc_out[:, i]
            cm = confusion_matrix(y_true, y_pred)
            tp = np.diagonal(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            tn = cm.sum() - (tp + fp + fn)
            tp, fp, tn, fn = tp[0], fp[0], tn[0], fn[0]
            self.history.log(f"tp_{i}", tp, idx=i)
            self.history.log(f"fp_{i}", fp, idx=i)
            self.history.log(f"fn_{i}", fn, idx=i)
            self.history.log(f"tn_{i}", tn, idx=i)

        return loss

    def train(self,
              epochs: int,
              train_loader: DataLoader,
              valid_loader: Optional[DataLoader] = None,
              scheduler: Any = None) -> None:
        epoch_length = len(str(epochs))
        for epoch in range(epochs):
            self.net.train()
            for i, (x, x_pad, spc, y) in enumerate(train_loader):
                loss = self._train_step(x, x_pad, spc, y, val=False)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()
                # torch.cuda.empty_cache()
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
                self.val(valid_loader)
                if self.model_ckpt[0].early_stop:
                    break

            if scheduler:
                scheduler.step()

    @torch.no_grad()
    def val(self, test_loader: DataLoader) -> None:
        self.net.eval()
        flag = True
        for i, (x, x_pad, spc, y) in enumerate(test_loader):
            loss = self._train_step(x, x_pad, spc, y, val=True)
            prefix = 'Val'
            postfix = str(self.history)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        self.history.summary()

        prefix = 'Val'
        postfix = str(self.history)
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        if self.model_ckpt is not None:
            self.model_ckpt[0](self.history[-1]['loss'], self.net)

        # self.history.reset()

    @torch.no_grad()
    def set_threshold(self, thres_loader: DataLoader) -> None:
        self.net.eval()
        err_record = []
        print()
        print('-' * 55)
        start = time.time()
        prefix = "Set threshold on valset"
        postfix = ""
        for i, (x, x_pad, _, y) in enumerate(thres_loader):
            # x: [batch, seq, dim], y: [batch, seq]
            x, x_pad, y = x.to(self.device), x_pad.to(
                self.device), y.to(self.device)
            _, norm_out, spc_pred, series_out = self.net(x_pad, denorm=True)
            errors = (series_out - x)**2
            # errors = (series_out - norm_out)**2
            for j in range(len(spc_pred)):
                time_points = y[j]  # [seq,]
                normal_errors = (errors[j])[time_points == 0]  # [<seq, dim]
                if normal_errors.shape[0]:
                    err_record.append(normal_errors)
            ProgressBar.show(prefix, postfix, i, len(thres_loader))
        ProgressBar.show(prefix, postfix, len(thres_loader),
                         len(thres_loader), newline=True)
        err_record = torch.cat(err_record, dim=0)  # [anomalies_num, dim]

        # 這裡以後要實現加權平均
        err_record = torch.mean(err_record, dim=1)  # [anomalies_num,]
        self.threshold_mean = torch.mean(err_record)
        self.threshold_std = torch.std(err_record)
        end = time.time()
        duration = end - start
        print(f"Set the anomaly threshold by valset in {duration:.2f}sec.")
        print(f"Mean: {self.threshold_mean:.6f}")
        print(f"Stdard Error: {self.threshold_std:.6f}")
        print('-' * 55)
        print()

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()
        real_threshold = self.threshold_mean + \
            self.config.sensitive_level * self.threshold_std
        preds = []
        ground_truth = []
        spc_preds = []
        spc_labels = []
        print('*' * 60)
        prefix = "Test, produce classification report"
        postfix = ""
        for i, (x, x_pad, spc_label, y) in enumerate(test_loader):
            x, x_pad, spc_label, y = x.to(self.device), x_pad.to(
                self.device), spc_label.to(self.device), y.to(self.device)

            #  <model prediction>
            # series_out.shape = (batch, seq_len, input_dim)
            _, norm_out, spc_pred, series_out = self.net(x_pad, denorm=True)
            # errs.shape = (batch, seq_len)
            errs = torch.mean((series_out - x)**2, dim=2)
            # errs = torch.mean((series_out - norm_out)**2, dim=2)
            # (batch, seq_len) -> (batch,)
            pred = torch.sum((errs > real_threshold), dim=1)
            # pred.shape = (batch,)
            pred = pred > 0
            preds.append(pred)
            spc_preds.append(spc_pred)

            # <ground truth>
            # (batch, seq_len) -> (batch,)
            gt = torch.sum(y, dim=1) > 0
            ground_truth.append(gt)
            spc_labels.append(spc_label)
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        # Anomaly detection report
        preds = torch.cat(preds).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truth).detach().cpu().numpy()
        anomaly_report = classification_report(ground_truth, preds)

        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        self.history.reset()
        print()
        print(anomaly_report)
        print('*' * 60)

        # SPC label classification report
        spc_preds = torch.cat(spc_preds).squeeze(-1).detach().cpu().numpy()
        spc_labels = torch.cat(spc_labels).detach().cpu().numpy()
        spc_report = []
        # exit(1)
        for i in range(self.config.spc_rule_num):
            spc_pred = spc_preds[:, i]
            spc_pred = spc_pred > 0.5
            spc_label = spc_labels[:, i]
            precision, recall, f1, _ = precision_recall_fscore_support(
                spc_label, spc_pred, average="binary", zero_division=0)
            # print()
            # print(
            #     f"SPC label {i:02}: precision, recall, f1-score = {precision:.3f}, {recall:.3f}, {f1:.3f}")
            spc_report.append((precision, recall, f1))

        return anomaly_report, spc_report

    @property
    @torch.no_grad()
    def weights(self):
        return {'net': self.net}


# class TSRunner(_BaseRunner):
#     def __init__(
#         self,
#         net,
#         config: AnomalyBertConfig,
#         optimizer: optim.Optimizer,
#         model_ckpt: Optional[Callable] = None,
#         device: str = 'cuda'
#     ) -> None:
#         super().__init__(device=device)

#         self.history = _History(
#             metrics=['loss', 'accuracy'], additional_keys=['loss2']
#         )
#         self.net = net.to(self.device)
#         self.optimizer = optimizer
#         self.mse = nn.MSELoss()
#         self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)
#         self.model_ckpt = model_ckpt
#         self.config = config

#     @torch.no_grad()
#     def _step(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor
#     ) -> torch.Tensor:
#         x = x.to(self.device)
#         y = y.to(self.device)

#         if self.config.output_attention:
#             output, series, prior, sigmas = self.net(x)
#         else:
#             output = self.net(x)

#         y_hat = torch.argmax(output, dim=-1)
#         running_loss = self.criterion(output, y.squeeze())

#         nn.utils.clip_grad_value_(self.net.parameters(), clip_value=1.0)

#         self.history.log('count', y.shape[0])
#         self.history.log('loss', running_loss)
#         self.history.log('correct', int(
#             torch.sum(y_hat == y.squeeze()).item()))

#         return running_loss

#     def _train_step(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = x.to(self.device)
#         y = y.to(self.device)

#         if self.config.output_attention:
#             output, series, prior, _ = self.net(x)
#         else:
#             output = self.net(x)

#         # calculate Association discrepancy
#         series_loss = 0.0
#         prior_loss = 0.0
#         for u in range(len(prior)):
#             kl_a = series[u]
#             kl_b = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
#                 1, 1, 1, self.config.max_position_embeddings)
#             series_loss += torch.mean(self.kl(kl_a, kl_b.detach()) +
#                                       self.kl(kl_b.detach(), kl_a))
#             prior_loss += torch.mean(self.kl(kl_b, kl_a.detach()) +
#                                      self.kl(kl_a.detach(), kl_b))

#         series_loss = series_loss / len(prior)
#         prior_loss = prior_loss / len(prior)

#         rec_loss = self.mse(output, x)

#         loss1 = rec_loss - self.config.k * series_loss
#         loss2 = rec_loss + self.config.k * prior_loss

#         self.history.log('count', y.shape[0])
#         self.history.log('loss', loss1)
#         self.history.log('loss2', loss2)
#         self.history.log('correct', y.shape[0])

#         return loss1, loss2

#     def train(
#             self, epochs: int,
#             train_loader: DataLoader,
#             valid_loader: Optional[DataLoader] = None,
#             scheduler: Any = None
#     ) -> None:
#         epoch_length = len(str(epochs))
#         for epoch in range(epochs):
#             self.net.train()
#             for i, (x, spc, y) in enumerate(train_loader):
#                 loss1, loss2 = self._train_step(x, y)

#                 self.optimizer.zero_grad()
#                 loss1.backward(retain_graph=True)
#                 loss2.backward()
#                 torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
#                 self.optimizer.step()
#                 torch.cuda.empty_cache()
#                 prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
#                 postfix = str(self.history)
#                 ProgressBar.show(prefix, postfix, i, len(train_loader))

#             self.history.summary()

#             prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
#             postfix = str(self.history)
#             ProgressBar.show(prefix, postfix, len(train_loader),
#                              len(train_loader), newline=True)

#             self.history.reset()

#             if valid_loader:
#                 flag = self.val(valid_loader)
#                 if self.model_ckpt.early_stop:
#                     break

#             if scheduler:
#                 scheduler.step()

#     @torch.no_grad()
#     def val(self, test_loader: DataLoader) -> bool:
#         self.net.eval()
#         flag = True
#         for i, (x, spc, y) in enumerate(test_loader):
#             loss1, loss2 = self._train_step(x, y)
#             prefix = 'Val'
#             postfix = str(self.history)
#             ProgressBar.show(prefix, postfix, i, len(test_loader))

#         self.history.summary()

#         prefix = 'Val'
#         postfix = str(self.history)
#         ProgressBar.show(prefix, postfix, len(test_loader),
#                          len(test_loader), newline=True)

#         if self.model_ckpt is not None:
#             # val_loss, val_loss2, model
#             self.model_ckpt(self.history[-1]['loss'],
#                             self.history[-1]['loss2'], self.net)

#         self.history.reset()
#         return flag

#     @torch.no_grad()
#     def test(self, test_loader: DataLoader) -> None:
#         self.net.eval()

#         for i, (x, spc, y) in enumerate(test_loader):
#             running_loss = self._step(x, y)
#             prefix = 'Test'
#             postfix = str(self.history)
#             ProgressBar.show(prefix, postfix, i, len(test_loader))

#         self.history.summary()

#         prefix = 'Test'
#         postfix = str(self.history)
#         ProgressBar.show(prefix, postfix, len(test_loader),
#                          len(test_loader), newline=True)

#         self.history.reset()

#     @property
#     @torch.no_grad()
#     def weights(self):
#         return {'net': self.net}
