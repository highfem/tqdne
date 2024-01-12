from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tqdne.conf import Config
from tqdne.logging import LogCallback


def get_pl_trainer(name, val_loader, metrics, eval_every, config=Config(), **trainer_params):
    # wandb logger
    wandb_logger = WandbLogger(
        project=config.project_name
    )  # add project='projectname' to log to a specific project

    # learning rate logger
    lr_logger = LearningRateMonitor()

    # set early stopping
    # early_stopping = EarlyStopping('val_loss', mode='min', patience=5)

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.outputdir / Path(name),
        filename="{name}_{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
    )

    # log callback
    log_callback = LogCallback(wandb_logger, val_loader, metrics, every=eval_every)

    output_dir = config.outputdir / Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=[lr_logger, log_callback, checkpoint_callback],
        default_root_dir=output_dir
    )

    return trainer
