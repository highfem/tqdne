import logging
from pathlib import Path
import torch

from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.conf import Config
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.representation import LogSpectrogram
from tqdne.training import get_pl_trainer
from tqdne.unet import UNetModel
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    resume = True
    logging.info("Loading data...")
    batch_size = 128
    stft_channels = 128
    hop_size = stft_channels // 4
    t = 4096 - hop_size  # subtract hop_size to make sure spectrogram has even number of frames

    name = "EDM-LogSpectrogram"
    config = Config()

    representation = LogSpectrogram()

    path_train = config.datasetdir / Path(config.data_upsample_train)
    path_test = config.datasetdir / Path(config.data_upsample_test)
    train_dataset = Dataset(path_train, representation, cut=t, cond=True)
    test_dataset = Dataset(path_test, representation, cut=t, cond=True)

    channels = train_dataset[0]["signal"].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    metrics = [metric.PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]

    # plots
    plots = [
        plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(channels)
    ] + [plot.PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "in_channels": channels,
        "out_channels": channels,
        "cond_features": 5,  # set to 5 if cond=True
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": 64,
        "channel_mult": (1, 2, 1, 2),  # might want to change to (1, 2, 4, 8)
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "flash_attention": False,  # flash attention not tested (potentially faster)
    }

    max_epochs = 100
    trainer_params = {
        "precision": 32,
        "max_steps": max_epochs * len(train_loader),
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
    }

    optimizer_params = {
        "lr": 1e-4,
    }

    logging.info("Build network...")
    net = UNetModel(**unet_params)

    logging.info("Build lightning module...")
    model = LightningEDM(net, optimizer_params=optimizer_params)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        representation,
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
