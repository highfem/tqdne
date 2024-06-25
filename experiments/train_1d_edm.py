import logging

import torch
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.config import MovingAverageEnvelopeConfig
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    logging.info("Set parameters...")

    name = "EDM-MovingAvg"
    config = MovingAverageEnvelopeConfig()
    max_epochs = 300
    batch_size = 320
    lr = 1e-4
    ema_decay = 0.999
    resume = True

    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="train"
    )
    test_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="test"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # metrics
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ]

    # plots
    plots = [plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    # Unet parameters
    unet_config = {
        "in_channels": config.channels,
        "out_channels": config.channels,
        "cond_features": len(config.features_keys),
        "dims": 1,
        "conv_kernel_size": 5,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attention_resolutions": (8,),
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.1,
        "flash_attention": False,  # flash attention not tested (potentially faster)
    }

    max_steps = max_epochs * len(train_loader)
    trainer_params = {
        "precision": 32,
        "max_steps": max_steps,
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": 5,
    }

    optimizer_params = {"learning_rate": lr, "max_steps": max_steps}

    logging.info("Build lightning module...")
    model = LightningEDM(unet_config, optimizer_params)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        config.representation,
        metrics,
        plots,
        ema_decay=ema_decay,
        eval_every=10,
        limit_eval_batches=2,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir) if resume else None
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")
