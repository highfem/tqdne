import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.autoencoder import LithningAutoencoder
from tqdne.conf import Config
from tqdne.dataset import Dataset
from tqdne.representation import LogSpectrogram
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    resume = False
    logging.info("Loading data...")
    t = 4096
    batch_size = 24

    name = "Autoencoder-32x32x3-LogSpectrogram"
    config = Config()

    representation = LogSpectrogram(output_shape=(256, 256))

    path_train = config.datasetdir / Path(config.data_upsample_train)
    path_test = config.datasetdir / Path(config.data_upsample_test)
    train_dataset = Dataset(path_train, representation, cut=t, cond=True)
    test_dataset = Dataset(path_test, representation, cut=t, cond=True)

    channels = train_dataset[0]["signal"].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    metrics = [metric.PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)] + [
        metric.MeanSquaredError(channel=c) for c in range(channels)
    ]

    # plots
    plots = [
        plot.SamplePlot(plot_target=True, fs=config.fs, channel=c) for c in range(channels)
    ] + [plot.PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]

    logging.info("Set parameters...")

    # Unet parameters
    autoencoder_params = {
        "kl_weight": 0.000001,
        "in_channels": channels,
        "latent_channels": channels,
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": 32,
        "channel_mult": (1, 2, 4, 4),  # might want to change to (1, 2, 4, 8)
        "num_res_blocks": 2,
        "num_heads": 4,
        "flash_attention": False,  # flash attention not tested (potentially faster)
        "lr": 4.5e-6,  # taken from the original implementation
    }

    trainer_params = {
        "precision": 32,
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
    }

    logging.info("Build lightning module...")
    autoencoder = LithningAutoencoder(**autoencoder_params)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        representation,
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
