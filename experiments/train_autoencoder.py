import logging
import sys

import torch
from config import LatentSpectrogramConfig

from tqdne import metric, plot
from tqdne.architectures import get_2d_autoencoder_configs
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataloader import get_train_and_val_loader
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint


def fake_represent(representation, leng_signal):
    signal = torch.ones((1, leng_signal))
    spectr = representation.get_representation(signal)
    return spectr


def run(args):
    config = LatentSpectrogramConfig(args.workdir)
    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    spectr = fake_represent(config.representation)
    name = f"Autoencoder-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x4-LogSpectrogram"

    train_loader, val_loader = get_train_and_val_loader(config, args.num_workers, args.batchsize)
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ] + [metric.MeanSquaredError(channel=c) for c in range(3)]
    plots = [plot.SamplePlot(plot_target=True, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    optimizer_params = {
        "learning_rate": 0.0001,
        "max_steps": 300 * len(train_loader),
        "eta_min": 0.0,
    }
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": 300 * len(train_loader),
    }

    logging.info("Build lightning module...")
    encoder_config, decoder_config = get_2d_autoencoder_configs(config)
    autoencoder = LightningAutoencoder(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        optimizer_params=optimizer_params,
        kl_weight=config.kl_weight,
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        plots=plots,
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir)
    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )
    logging.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train a 2D variational autoencoder")
    parser.add_argument(
        "--workdir",
        type=str,
        help="the working directory in which checkpoints and all output are saved to",
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, help="size of a batch of each gradient step", default=128
    )
    parser.add_argument(
        "-w", "--num-workers", type=int, help="number of separate processes for file/io", default=32
    )
    parser.add_argument(
        "-d", "--num-devices", type=int, help="number of CPUs/GPUs to train on", default=4
    )
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
