import logging
import sys

import torch
from config import LatentSpectrogramConfig

from tqdne import metric, plot
from tqdne.architectures import get_2d_unet_config
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataloader import get_train_and_val_loader
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint


def fake_represent(representation, leng_signal):
    signal = torch.ones((1, leng_signal))
    spectr = representation.get_representation(signal)
    return spectr


def run(args):
    config = LatentSpectrogramConfig(args.workdir)
    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    spectr = fake_represent(config.representation, config.t)
    name = f"Latent-EDM-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x4-LogSpectrogram"    

    train_loader, val_loader = get_train_and_val_loader(
        config, args.num_workers, args.batchsize, cond=True
    )
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ]
    plots = [plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    optimizer_params = {
        "learning_rate": 0.0001,
        "max_steps": 200 * len(train_loader),
        "eta_min": 0.0,
    }
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": 200 * len(train_loader),
    }

    
    checkpoint = config.outputdir / f"Autoencoder-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x4-LogSpectrogram" / "last.ckpt"
    logging.info(f"Loading autoencoder: {checkpoint}")
    autoencoder = LightningAutoencoder.load_from_checkpoint(checkpoint)
    
    model = LightningEDM(
        get_2d_unet_config(config, config.latent_channels, config.latent_channels),
        optimizer_params,
        autoencoder=autoencoder,
        frequency_weights=weight,
        mask=mask
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        plots=plots,
        ema_decay=0.999,
        eval_every=10,
        limit_eval_batches=2,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir)
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    import argparse

    parser = argparse.ArgumentParser("Train a 2D latent diffusion model")
    parser.add_argument(
        "--workdir",
        type=str,
        help="the working directory in which checkpoints and all output are saved to",
    )       
    parser.add_argument(
        "-b", "--batchsize", type=int, help="size of a batch of each gradient step", default=256
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
