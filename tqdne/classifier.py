import pytorch_lightning as pl
import torch as th
from torch import nn
from torchmetrics import Metric, MetricCollection

from .blocks import Encoder


class LithningClassifier(pl.LightningModule):
    """A PyTorch Lightning module for training a classifier.

    It encodes the input signal with an encoder, average-pools the latent before passing it through the output layer.

    Parameters
    ----------
    encoder_config : dict
        The configuration for the encoder.
    num_classes : int
        The number of classes to predict.
    loss : nn.Module
        The loss function.
    metrics : list
        A list of torchmetrics.Metric instances.
    optimizer_params : dict
        The parameters for the optimizer.
    """

    def __init__(
        self,
        encoder_config: dict,
        num_classes: int,
        loss: nn.Module,
        metrics: list[Metric],
        optimizer_params: dict,
    ):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        out_channels = encoder_config["out_channels"]
        self.output_MLP = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.output_layer = nn.Linear(out_channels, num_classes)
        self.loss = loss
        self.metrics = MetricCollection(metrics)
        self.optimizer_params = optimizer_params
        self.save_hyperparameters()

    def embed(self, x):
        h = self.encoder(x)  # batch_size x channels x ...
        h = th.mean(h, dim=list(range(2, len(h.shape))))
        h = self.output_MLP(h)
        return h

    def forward(self, x):
        h = self.embed(x)
        return self.output_layer(h)

    def training_step(self, batch, batch_idx):
        x = batch["signal"]
        y = batch["label"]
        loss = self.loss(self(x), y)
        self.log("training/loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["signal"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("validation/loss", loss.item(), sync_dist=True)
        self.metrics(y_hat, y)
        self.log_dict(self.metrics, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.optimizer_params["learning_rate"])
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_params["max_steps"],
            eta_min=self.optimizer_params["eta_min"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }
