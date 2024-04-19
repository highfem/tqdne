import ml_collections
import pytorch_lightning as pl
import torch
from diffusers.optimization import get_scheduler

class LightningClassification(pl.LightningModule):
    """A PyTorch Lightning module for training a classification model

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    example_input_array : torch.Tensor, optional
        An example input array for the network.
    ml_config : ml_collections.ConfigDict, optional
        A configuration object for the model.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_params: dict,
        loss: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        metrics: list = [],
        example_input_array: torch.Tensor = None,
        ml_config: ml_collections.ConfigDict = None,
    ):
        super().__init__()

        self.net = net
        self.loss = loss
        self.metrics = metrics
        self.optimizer_params = optimizer_params
        self.example_input_array = example_input_array
        self.ml_config = ml_config
        self.save_hyperparameters(ignore=["example_input_array"])

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward(self, x):
        return self.net(x)
    
    def evaluate(self, batch):
        x = batch["repr"]
        return self(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["repr"], batch["classes"] 
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_value(loss, "loss")
        for metric in self.metrics:
            metric = metric.to(self.device)
            self.log_value(metric(y_hat, y), metric.__class__.__name__)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["repr"], batch["classes"] 
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_value(loss, "loss", train=False)
        for metric in self.metrics:
            metric = metric.to(self.device)
            self.log_value(metric(y_hat, y), metric.__class__.__name__, train=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        return optimizer
        