import logging

import torch
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.autoencoder import LithningAutoencoder
from tqdne.blocks import Decoder, Encoder
from tqdne.config import LatentSpectrogramConfig
from tqdne.dataset import Dataset
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    logging.info("Set parameters...")

    name = "Autoencoder-32x32x4-LogSpectrogram-150"
    config = LatentSpectrogramConfig()
    batch_size = 64
    lr = 1e-4
    max_epochs = 150
    resume = True

    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=False, split="train"
    )
    test_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=False, split="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True)
        for c in range(config.channels)
    ] + [metric.MeanSquaredError(channel=c) for c in range(config.channels)]

    # plots
    plots = [
        plot.SamplePlot(plot_target=True, fs=config.fs, channel=c) for c in range(config.channels)
    ] + [plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(config.channels)]

    # Parameters
    base_params = {
        "model_channels": 64,
        "channel_mult": (1, 2, 4),
        "num_res_blocks": 2,
        "dims": 2,
        "conv_kernel_size": 3,
        "num_heads": 4,
        "flash_attention": False,
    }
    encoder_params = base_params | {
        "in_channels": config.channels,
        "out_channels": config.latent_channels * 2,
    }
    decoder_params = base_params | {
        "in_channels": config.latent_channels,
        "out_channels": config.channels,
    }
    max_steps = max_epochs * len(train_loader)
    optimizer_params = {"learning_rate": lr, "max_steps": max_steps}
    autoencoder_params = {"kl_weight": config.kl_weight, "optimizer_params": optimizer_params}

    trainer_params = {
        "precision": 32,
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": max_steps,
    }

    logging.info("Build lightning module...")
    autoencoder = LithningAutoencoder(
        encoder=Encoder(**encoder_params), decoder=Decoder(**decoder_params), **autoencoder_params
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        config.representation,
        metrics=metrics,
        plots=plots,
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")
