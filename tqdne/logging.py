import time
import warnings

import torch
from pytorch_lightning.callbacks import Callback
from torch import Tensor

import wandb


class LogCallback(Callback):
    def __init__(self, task, val_loader, metrics, plots, limit_batches=1, every=1):
        super().__init__()
        self.task = task
        self.val_loader = val_loader
        self.metrics = metrics
        self.plots = plots
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

        # Make predictions
        batches = []
        preds = []
        for i, batch in enumerate(self.val_loader):
            if self.limit_batches != -1 and i >= self.limit_batches:
                break
            batch = {
                k: v.to(pl_module.device) if isinstance(v, Tensor) else v for k, v in batch.items()
            }
            batches.append(batch)
            pred = pl_module.evaluate(batch)
            preds.append(pred)

        batch = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0].keys()}
        pred = torch.cat(preds, dim=0)
        if self.task == "classification":
            target = batch["classes"]
        elif self.task == "generation":
            target = batch["repr"] 
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # TODO: cache the inverted representation to speed up the evaluation

        # Log metrics
        if isinstance(self.metrics, list):
            self.metrics = {(metric.name if hasattr(metric, "name") else metric.__class__.__name__): metric for metric in self.metrics}

        for metric_name, metric in self.metrics.items():
            result = metric(preds=pred, target=target)
            pl_module.log(metric_name, result)

        # Log plots
        for plot in self.plots:
            fig = plot(
                preds=pred,
                target=target,
                cond_signal=batch["cond_signal"] if "cond_signal" in batch else None,
                cond=batch["cond"] if "cond" in batch else None,
            )
            try:
                trainer.logger.experiment.log({f"{plot.name} (Image)": wandb.Image(fig)})
                trainer.logger.experiment.log({f"{plot.name} (Plot)": fig})
            except Exception as e:
                warnings.warn(f"Failed to log plot: {e}")

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
