import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import average_precision_score

from networks.ClfCelebA import ClfImg, ClfText

class ClfCelebA(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.clfs = nn.ModuleList(
            [ClfImg(cfg).to(cfg.model.device), ClfText(cfg).to(cfg.model.device)]
        )
        self.cfg = cfg
        self.loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []
        self.training_step_outputs = []

        self.save_hyperparameters()

    def training_step(self, batch):
        out = self.forward(self.cfg, batch)
        _, losses = out
        loss = losses.mean(dim=1).mean(dim=0)
        self.log("train/loss/loss", loss)
        return loss

    def validation_step(self, batch):
        out = self.forward(self.cfg, batch)
        _, labels = batch
        preds, losses = out
        loss = losses.mean(dim=1).mean(dim=0)

        n_labels = labels.shape[1]
        accs = []
        for m in range(self.cfg.dataset.num_views):
            loss_m = losses[:, m, :].mean(dim=0)
            pred_m = preds[:, m, :]
            accs_m = torch.zeros(n_labels)
            for k in range(0, n_labels):
                accs_m[k] = average_precision_score(
                    labels[:, k].cpu(), pred_m[:, k].detach().cpu().numpy()
                )
            accs.append(accs_m.mean())
            self.log("val/loss/v" + str(m), loss_m)
            self.log("val/accuracy/v" + str(m), accs_m.mean())
        mean_ap = torch.tensor(accs).mean()
        self.log("val/loss/loss", loss)
        self.log("val/loss/mean_ap", mean_ap)
        self.validation_step_outputs.append(mean_ap)
        return loss

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()  # free memory

    def forward(self, cfg, batch):
        data, labels = batch
        n_labels = labels.shape[1]
        preds = torch.zeros(
            (cfg.model.batch_size_eval, cfg.dataset.num_views, n_labels),
            device=cfg.model.device,
        )
        losses = torch.zeros(
            (cfg.model.batch_size_eval, cfg.dataset.num_views, 1),
            device=cfg.model.device,
        )
        for m, m_key in enumerate(data.keys()):
            m_val = data[m_key]
            pred_m = self.clfs[m](m_val)
            preds[:, m, :] = pred_m
            loss_m = self.loss(pred_m, labels)
            losses[:, m, :] = loss_m
        return preds, losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.model.lr,
        )
        return {
            "optimizer": optimizer,
        }
