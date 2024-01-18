import time

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
from torch import Tensor

import wandb


class LogCallback(Callback):
    """
    Callback for logging metrics and visualizations during training and validation.

    Args:
        wandb_logger (wandb.Logger): Wandb logger for logging metrics and visualizations.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        metrics (list): List of metrics to compute and log.
        limit_batches (int): Limit the number of validation batches to process (-1 for all).
        every (int): Log metrics and visualizations every `every` validation epochs.
    """

    def __init__(self, wandb_logger, val_loader, metrics, limit_batches=1, every=1):
        super().__init__()
        self.wandb_logger = wandb_logger
        self.metrics = metrics
        self.val_loader = val_loader
        self.limit_batches = limit_batches
        self.total_time = 0
        self.every = every

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called when the validation epoch ends.

        Args:
            trainer (pytorch_lightning.Trainer): PyTorch Lightning trainer object.
            pl_module (pytorch_lightning.LightningModule): PyTorch Lightning module.
        """
        if pl_module.current_epoch % self.every != 0:
            return

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        # Make predictions and update metrics
        for i, batch in enumerate(self.val_loader):
            if self.limit_batches != -1 and i >= self.limit_batches:
                break
            batch = {
                k: v.to(pl_module.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            pred = pl_module.evaluate(batch)
            for metric in self.metrics:
                metric.update(pred, batch)

        # Log metrics
        for metric in self.metrics:
            result = metric.compute()
            if isinstance(result, dict):
                pl_module.log_dict(result)
            elif result is not None:
                pl_module.log(metric.__class__.__name__, result)

        # Log metric plots
        for metric in self.metrics:
            try:
                plot = metric.plot()
                name = metric.__class__.__name__
                try:
                    trainer.logger.experiment.log({f"{name} (Plot)": plot})
                except:
                    pass
                trainer.logger.experiment.log({f"{name} (Image)": wandb.Image(plot)})
            except NotImplementedError:
                pass

    def on_train_batch_start(self, *args, **kwargs):
        """
        Called when the train batch starts.
        """
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        """
        Called when the train batch ends.

        Args:
            trainer (pytorch_lightning.Trainer): PyTorch Lightning trainer object.
            pl_module (pytorch_lightning.LightningModule): PyTorch Lightning module.
        """
        batch_time = time.time() - self.start_time
        self.total_time += batch_time
        pl_module.log("traintime", self.total_time, on_step=True, on_epoch=False)
