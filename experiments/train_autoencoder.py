import logging
import os
import sys

import torch
from config import LatentSpectrogramConfig
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataset import Dataset
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint


def run(args):
    logging.info("Set parameters...")
    name = "Autoencoder-32x32x4-LogSpectrogram"
    config = LatentSpectrogramConfig(args.workdir, args.infile)
    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    batch_size = 64
    lr = 1e-4
    max_epochs = 200
    resume = True


    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=False, split="train"
    )
    test_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=False, split="test"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=os.cpu_count())

    # metrics
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ] + [metric.MeanSquaredError(channel=c) for c in range(3)]

    # plots
    plots = [plot.SamplePlot(plot_target=True, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    # Parameters
    base_config = {
        "model_channels": 64,
        "channel_mult": (1, 2, 4),
        "attention_resolutions": (),
        "num_res_blocks": 2,
        "dims": 2,
        "conv_kernel_size": 3,
        "dropout": 0.1,
    }
    encoder_config = base_config | {
        "in_channels": config.channels,
        "out_channels": config.latent_channels * 2,
    }
    decoder_config = base_config | {
        "in_channels": config.latent_channels,
        "out_channels": config.channels,
    }
    max_steps = max_epochs * len(train_loader)
    optimizer_params = {"learning_rate": lr, "max_steps": max_steps}

    trainer_params = {
        "precision": 32,
        "accelerator": "auto",
        "devices": 1,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": max_steps,
        "max_steps": max_steps,
    }

    logging.info("Build lightning module...")
    autoencoder = LightningAutoencoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        optimizer_params=optimizer_params,
        kl_weight=config.kl_weight,
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=test_loader,
        config=config,
        metrics=metrics,
        plots=plots,
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=False,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir) if resume else None
    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train a variational autoencoder")
    parser.add_argument("--workdir", type=str, help="the working directory in which checkpoints and all output are saved to")
    parser.add_argument("--infile", type=str, default=None, help="location of the training file; if not given assumes training data is located as `workdir/datasets/preprocessed_waveforms.h5`")
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)

