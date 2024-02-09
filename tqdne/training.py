from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from tqdne.conf import Config
from tqdne.logging import LogCallback


def get_pl_trainer(
    name,
    val_loader,
    metrics,
    eval_every,
    log_to_wandb=True,
    config=Config(),
    **trainer_params
):
    # wandb logger
    if log_to_wandb:
        wandb_logger = WandbLogger(project=config.project_name, name=name)
    else:
        wandb_logger = None

<<<<<<< HEAD
def get_pl_trainer(name, project=PROJECT_NAME, specific_callbacks = [], **trainer_params):
=======
    # learning rate logger
    callbacks = [LearningRateMonitor()]
>>>>>>> main

    # log callback
    callbacks.append(LogCallback(val_loader, metrics, every=eval_every))

    # set early stopping
    # early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
<<<<<<< HEAD
    # 4. saves checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(dirpath=OUTPUTDIR / Path(name), filename='{name}_{epoch}-{val_loss:.2f}',
                                        monitor='val_loss', mode='min', save_top_k=5)
    # 5. My custom callback
    lst_cbk = [lr_logger, checkpoint_callback, *specific_callbacks]
    print(lst_cbk)
    output_dir = (OUTPUTDIR/Path(name))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(**trainer_params, logger=wandb_logger, callbacks=lst_cbk, 
                        default_root_dir=output_dir ) 
    
=======

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    if (
        "enable_checkpointing" not in trainer_params
        or trainer_params["enable_checkpointing"]
    ):
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.outputdir / Path(name),
                filename="{name}_{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=5,
            )
        )

    output_dir = config.outputdir / Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=output_dir
    )

>>>>>>> main
    return trainer
