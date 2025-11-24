from pathlib import Path

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

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
    checkpoint_every_n_steps=None,
    max_checkpoints_to_keep=None,
    early_stopping_patience=None,
    early_stopping_min_delta=None,
    sampling_every_n_steps=None,
    sampling_batches=None,
    enable_profiler=False,
    **trainer_params,
):
    # wandb logger
    if log_to_wandb:
        wandb_logger = WandbLogger(
            project=config.project_name,
            name=name,
            resume="allow",
            settings=wandb.Settings(init_timeout=300),
        )
    else:
        wandb_logger = None

    # learning rate logger
    callbacks = [LearningRateMonitor()]
    if ema_decay > 0:
        callbacks.append(EMA(decay=ema_decay))

    # early stopping
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="validation/loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta if early_stopping_min_delta else 0.0,
                mode="min",
                verbose=True,
            )
        )

    # log callback for evaluation
    if metrics or plots:
        callbacks.append(
            LogCallback(
                val_loader,
                config.representation,
                metrics,
                plots,
                limit_batches=limit_eval_batches,
                every=eval_every,
                use_steps=True,
            )
        )

    # sampling callback (separate from evaluation)
    if sampling_every_n_steps is not None and plots:
        callbacks.append(
            LogCallback(
                val_loader,
                config.representation,
                metrics=None,  # No metrics for sampling callback
                plots=plots,
                limit_batches=sampling_batches if sampling_batches else limit_eval_batches,
                every=sampling_every_n_steps,
                use_steps=True,
            )
        )

    # save checkpoints to 'model_path' whenever 'val_loss' has a new min
    if "enable_checkpointing" not in trainer_params or trainer_params["enable_checkpointing"]:
        # Validation loss based checkpoint
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
        # Step-based checkpoint (if configured)
        if checkpoint_every_n_steps is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=config.outputdir / Path(name),
                    filename="{name}_step={step:07d}",
                    every_n_train_steps=checkpoint_every_n_steps,
                    save_top_k=-1,  # Save all step-based checkpoints (no metric to monitor)
                    save_last=False,
                )
            )

    output_dir = config.outputdir / Path(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup profiler if enabled
    profiler = None
    if enable_profiler:
        profiler = PyTorchProfiler(
            dirpath=output_dir / "profiler",
            filename="profile",
            row_limit=20,
            export_to_chrome=True,
            profile_memory=True,
            with_flops=True,  # Enable FLOPs counting
            with_stack=False,
        )

    # Define Trainer
    trainer = pl.Trainer(
        **trainer_params,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        profiler=profiler,
    )

    return trainer
