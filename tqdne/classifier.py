import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Metric, MetricCollection

from .blocks import Encoder


class LithningClassifier(pl.LightningModule):
    """A PyTorch Lightning module for training a classifier.

    It encodes the input signal with an encoder, average-pools the latent before passing it through the output layer.

    Parameters
    ----------
    encoder : Encoder
        The encoder module.
    output_layer : nn.Module
        The output layer projecting the latent to the number of classes.
    loss : nn.Module
        The loss function.
    metrics : list
        A list of torchmetrics.Metric instances.
    """

    def __init__(
        self,
        encoder: Encoder,
        output_layer: nn.Module,
        loss: nn.Module,
        metrics: list[Metric],
        lr=1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.output_layer = output_layer
        self.loss = loss
        self.metrics = MetricCollection(metrics)
        self.lr = lr
        self.save_hyperparameters()

    def embed(self, x):
        h = self.encoder(x)  # batch_size x channels x ...
        return torch.mean(h, dim=list(range(2, len(h.shape))))

    def forward(self, x):
        h = self.embed(x)
        return self.output_layer(h)

    def training_step(self, batch, batch_idx):
        x = batch["signal"]
        y = batch["label"]
        loss = self.loss(self(x), y)
        self.log("training/loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["signal"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("validation/loss", loss.item())
        self.metrics(y_hat, y)
        self.log_dict(self.metrics, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
