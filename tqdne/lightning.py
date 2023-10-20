import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from typing import List


class LightningClassifier(pl.LightningModule):
    """A PyTorch Lightning classifier module.

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    lr_rate : float
        The learning rate for the optimizer.

    """

    def __init__(self, net: torch.nn.Module, lr_rate: float = 1e-3):
        super(LightningClassifier, self).__init__()

        self.net = net
        self.lr_rate = lr_rate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return F.nll_loss(logits, labels)

    def global_step(self, batch: List, batch_idx: int, train: bool = False):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if train:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def training_step(self, train_batch: List, batch_idx: int):
        return self.global_step(train_batch, batch_idx, train=True)

    def validation_step(self, val_batch: List, batch_idx: int):
        return self.global_step(val_batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            "name": "expo_lr",
        }
        return [optimizer], [lr_scheduler]


