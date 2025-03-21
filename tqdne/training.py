from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tqdne.ema import EMA
from tqdne.logging import LogCallback


def get_pl_trainer(
    name,
    val_loader,
    config,
    metrics=None,
    plots=None,
    ema_decay=0.0,
    eval_every=1,
    limit_eval_batches=1,
    log_to_wandb=True,
    **trainer_params,
):
    # wandb logger
    if log_to_wandb:
        wandb_logger = WandbLogger(
            project=config.project_name, 
            name=name,
            resume="allow",
        )
    else:
        wandb_logger = None

    # learning rate logger
    callbacks = [LearningRateMonitor()]
    if ema_decay > 0:
        callbacks.append(EMA(decay=ema_decay))

    # log callback
    if metrics or plots:
        callbacks.append(
            LogCallback(
                val_loader,
                config.representation,
                metrics,
                plots,
                limit_batches=limit_eval_batches,
                every=eval_every,
            )
        )

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    if "enable_checkpointing" not in trainer_params or trainer_params["enable_checkpointing"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.outputdir / Path(name),
                filename="{name}_{epoch}-val_loss={validation/loss:.2e}",
                monitor="validation/loss",
                auto_insert_metric_name=False,
                mode="min",
                save_top_k=3,
                save_last=True,
            )
        )

    output_dir = config.outputdir / Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
    )

    return trainer
