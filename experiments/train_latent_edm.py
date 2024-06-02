import logging

import torch
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.autoencoder import LithningAutoencoder
from tqdne.conf import LatentSpectrogramConfig
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.unet import UNetModel
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    logging.info("Set parameters...")

    name = "Latent-EDM-LogSpectrogram"
    config = LatentSpectrogramConfig()
    max_epochs = 300
    batch_size = 2048
    lr = 1e-4
    resume = True

    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="train"
    )
    test_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="test"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True)
        for c in range(config.channels)
    ]

    # plots
    plots = [
        plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(config.channels)
    ] + [plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(config.channels)]

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "in_channels": config.latent_channels,
        "out_channels": config.latent_channels,
        "cond_features": len(config.features_keys),
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),  # might want to change to (1, 2, 4, 8)
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.2,
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
    }

    optimizer_params = {"learning_rate": lr, "max_steps": max_steps}

    logging.info("Build network...")
    net = UNetModel(**unet_params)

    logging.info("Loading autoencoder...")
    checkpoint = config.outputdir / "Autoencoder-32x32x4-LogSpectrogram-New" / "last.ckpt"
    autoencoder = LithningAutoencoder.load_from_checkpoint(checkpoint)

    logging.info("Build lightning module...")
    model = LightningEDM(net, optimizer_params=optimizer_params, autoencoder=autoencoder)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        config.representation,
        metrics,
        plots,
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
