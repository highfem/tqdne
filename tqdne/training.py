from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

from tqdne.conf import Config
from tqdne.logging import LogCallback


def get_pl_trainer(
    name,
    val_loader,
    metrics,
    plots,
    eval_every=1,
    limit_eval_batches=1,
    log_to_wandb=True,
    config=Config(),
    **trainer_params,
):
    # wandb logger
    if log_to_wandb:
        wandb_logger = WandbLogger(project=config.project_name, name=name)
    else:
        wandb_logger = None

    # learning rate logger
    callbacks = [LearningRateMonitor()]

    # log callback
    callbacks.append(
        LogCallback(val_loader, metrics, plots, limit_batches=limit_eval_batches, every=eval_every)
    )

    # set early stopping
    # early_stopping = EarlyStopping('val_loss', mode='min', patience=5)

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    if "enable_checkpointing" not in trainer_params or trainer_params["enable_checkpointing"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.outputdir / Path(name),
                filename="{name}_{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=5,
            )
        )
        
    # model summary
    callbacks.append(ModelSummary(max_depth=-1))    

    output_dir = config.outputdir / Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        #enable_progress_bar=False
    )

    return trainer
