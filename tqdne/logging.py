import time
import warnings

import numpy as np
import torch
import torch as th
import wandb
from pytorch_lightning.callbacks import Callback
from torch import Tensor


class LogCallback(Callback):
    """
    Callback for logging metrics and visualizations during training and validation.

    Parameters
    ----------
    val_loader : torch.utils.data.DataLoader
        Validation data loader.
    representation : tqdne.representation.Representation
        Representation object.
    metrics : list
        List of metrics to compute and log.
    plots : list
        List of plots to compute and log.
    limit_batches : int, optional
        Limit the number of validation batches to process (-1 for all).
    every : int, optional
        Log metrics and visualizations every `every` validation epochs.
    """

    def __init__(self, val_loader, representation, metrics, plots, limit_batches=1, every=1):
        super().__init__()
        self.val_loader = val_loader
        self.representation = representation
        self.metrics = metrics
        self.plots = plots
        self.limit_batches = limit_batches
        self.total_time = 0
        self.every = every

    def on_validation_epoch_end(self, trainer, pl_module):
        if pl_module.current_epoch % self.every != 0:
            return

        # Make predictions
        batches = []
        preds = []
        for i, batch in enumerate(self.val_loader):
            if self.limit_batches != -1 and i >= self.limit_batches:
                break
            batches.append(batch)
            batch = {
                k: v.to(pl_module.device) if isinstance(v, Tensor) else v for k, v in batch.items()
            }
            pred = pl_module.evaluate(batch)
            if th.any(th.isnan(pred)):
                warnings.warn("found nan in prediction, setting to zero")
                pred = th.nan_to_num(pred)
            pred = self.representation.invert_representation(pred)
            preds.append(pred)

        pred = np.concatenate(preds, axis=0)
        batch = {
            k: torch.cat([b[k] for b in batches], dim=0).numpy(force=True)
            for k in batches[0].keys()
        }

        # Log metrics
        for metric in self.metrics:
            result = metric(pred=pred, target=batch["waveform"])
            pl_module.log(metric.name, result, sync_dist=True)

        # Log plots
        for plot in self.plots:
            fig = plot(
                pred=pred,
                target=batch["waveform"],
                cond_signal=batch["cond_waveform"] if "cond_waveform" in batch else None,
                cond=batch["cond"] if "cond" in batch else None,
            )
            try:
                trainer.logger.experiment.log(
                    {f"{plot.name} (Image)": wandb.Image(fig)}, step=pl_module.global_step
                )
                # trainer.logger.experiment.log(
                #     {f"{plot.name} (Plot)": fig}, step=pl_module.global_step
                # )
            except Exception as e:
                warnings.warn(f"Failed to log plot: {e}")

    def on_train_batch_start(self, *args, **kwargs):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        batch_time = time.time() - self.start_time
        self.total_time += batch_time
        pl_module.log("traintime", self.total_time, on_step=True, on_epoch=False)
