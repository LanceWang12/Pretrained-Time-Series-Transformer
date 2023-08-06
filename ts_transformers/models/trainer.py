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
                 model_ckpt=None,
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

        mse_loss = self.mse(out, x)
        # if val:
        #     # compute the relative distance on validation set
        #     mse_loss = self.mse(out, norm_out)
        # else:
        #     # compute the absolute distance on training set
        #     mse_loss = self.mse(out, x)

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
            if self.config.load_norm:  # fixed the weight in norm layer
                for _, param in self.net.norm.named_parameters():
                    param.requires_grad = False
            if self.config.load_embed:  # fixed the weight in embedding layer
                for _, param in self.net.embed.named_parameters():
                    param.requires_grad = False
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
            _, norm_out, spc_pred, series_out = self.net(x_pad)
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

        err_record = torch.mean(err_record, dim=1)  # [anomalies_num,]
        self.threshold_mean = torch.mean(err_record)
        self.threshold_std = torch.std(err_record)
        end = time.time()
        duration = end - start
        print(f"Set the anomaly threshold by valset in {duration:.2f}sec.")
        print(f"Mean: {self.threshold_mean:.6f}")
        print(f"Stdandard Error: {self.threshold_std:.6f}")
        print('-' * 55)
        print()

    @torch.no_grad()
    def test(self, test_loader: DataLoader):
        self.net.eval()
        d1 = self.threshold_mean + self.config.sensitive_level * 0.5 * self.threshold_std
        d2 = self.threshold_mean + self.config.sensitive_level * self.threshold_std
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
            _, norm_out, spc_pred, series_out = self.net(x_pad)
            # errs.shape = (batch, seq_len)
            errs = torch.mean((series_out - x)**2, dim=2)
            # (batch, seq_len, features) -> (batch, seq_len)
            # pred = torch.sum((errs > real_threshold), dim=1)

            # Fuzzify binary output
            pred = self.fuzzy_output(errs, d1, d2)
            # pred.shape = (batch,)
            pred = torch.sum(pred > 0.5, dim=1)  # 機率大於 0.5 算 anomaly
            pred = pred > 1
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
        anomaly_report = classification_report(
            ground_truth, preds, zero_division=0)

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

    def fuzzy_output(self, obs: torch.tensor, d1: torch.tensor, d2: torch.tensor):
        out = torch.zeros_like(obs).to(obs.device)
        # out[obs <= d1] = 0
        target = torch.logical_and(obs > d1, obs <= d2)
        out[target] = (obs[target] - d1) / (d2 - d1)
        out[obs > d2] = 1
        # if obs <= d1:
        #     normal = torch.ones_like(obs)
        #     anomaly = torch.zeros_like(obs)
        # elif obs > d1 and obs <= d2:
        #     normal = (obs - d2) / (d1 - d2)
        #     anomaly = (obs - d1) / (d2 - d1)
        # else:
        #     normal = torch.zeros_like(obs)
        #     anomaly = torch.ones_like(obs)
        return out

    @torch.no_grad()
    def get_pattern_attn(self, test_loader: DataLoader):
        self.net.eval()
        attn_record = []
        print()
        print('-' * 55)
        start = time.time()
        prefix = "Get attention map from val loader"
        postfix = ""
        for i, (x, x_pad, _, y) in enumerate(test_loader):
            # x: [batch, seq, dim], y: [batch, seq]
            x, x_pad, y = x.to(self.device), x_pad.to(
                self.device), y.to(self.device)
            attn = self.net.get_representation(
                x_pad, idx_lst=self.config.target_features_idx)
            attn = attn[y == 1]
            attn_record.append(attn)

            ProgressBar.show(prefix, postfix, i, len(test_loader))
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)
        # [anomalies_num, patch_num, patch_num]
        attn_record = torch.cat(attn_record, dim=0)

        # 這裡以後要實現加權平均
        self.attn_mean = attn_record.mean(dim=0).unsqueeze(0)
        self.attn_std = attn_record.std(dim=0).unsqueeze(0)
        end = time.time()
        duration = end - start
        print(f"Get attention map in {duration:.2f}sec.")
        print(self.attn_mean.shape)
        print('-' * 55)
        print()

    @torch.no_grad()
    # -> Tuple[np.ndarray, float]:
    def test_pattern_matching(self, test_loader: DataLoader):
        self.net.eval()
        preds_record = []
        gt_record = []
        print('*' * 60)
        prefix = f"Test Pattern Matching, produce top-{self.config.top_k_report} report"
        postfix = ""
        length = len(self.config.target_features_idx)
        seq_len_square = (self.net.patchify.patch_num + 1)**2
        for i, (x, x_pad, spc_label, y) in enumerate(test_loader):
            x, x_pad, spc_label, y = x.to(self.device), x_pad.to(
                self.device), spc_label.to(self.device), y.to(self.device)

            #  <model prediction>
            # series_out.shape = (batch, seq_len, input_dim)
            attn = self.net.get_representation(
                x_pad, idx_lst=self.config.target_features_idx)
            # errs.shape = (batch,)
            # print(attn.shape, self.attn_mean.shape)
            errs = torch.sqrt((attn - self.attn_mean)**2)
            # print("haha", errs.shape)
            errs = errs > (self.config.sensitive_level * self.attn_std)
            errs = torch.sum(errs, dim=(1, 2, 3))
            errs = errs > length * seq_len_square
            preds_record.append(errs)
            gt_record.append(y)
            ProgressBar.show(prefix, postfix, i, len(test_loader))
        ProgressBar.show(prefix, postfix, len(test_loader),
                         len(test_loader), newline=True)

        preds = torch.cat(preds_record).detach().cpu().numpy()
        ground_truth = torch.cat(gt_record).detach().cpu().numpy()
        report = classification_report(ground_truth, preds, zero_division=0)
        # Anomaly detection report
        # preds = torch.cat(errs_record).detach().cpu().numpy()
        # ground_truth = torch.cat(gt_record).detach().cpu().numpy()
        # # print(preds.shape)
        # # exit(1)
        # idx = np.argsort(preds)
        # # ground_truth[ground_truth == 1]
        # ans = ground_truth[idx[: self.config.top_k_report]]
        # print(ans)
        # top_k_acc = sum(ans) / len(ans)

        self.history.reset()
        print()
        # print(f"Top k accuracy: {top_k_acc * 100:.3f}")
        print(report)
        print('*' * 60)

        return report
        # return ans, top_k_acc

    @property
    @torch.no_grad()
    def weights(self):
        return {'net': self.net}
