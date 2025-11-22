import logging
import sys

import torch
from config import LatentSpectrogramConfig

from tqdne import metric, plot
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataloader import get_train_and_val_loader
from tqdne.flow_matching import LightningFlowMatching
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint


def fake_represent(representation, leng_signal):
    signal = torch.ones((1, leng_signal))
    spectr = representation.get_representation(signal)
    return spectr


def get_dit_config(config, latent_channels, model_channels=384):
    """Get DiT-S/2 configuration for flow matching.

    Note: Flow matching reuses the same DiT architecture as EDM+DiT.
    """
    dit_config = {
        "input_size": 32,  # latent spatial size
        "patch_size": 2,
        "in_channels": latent_channels,
        "hidden_size": model_channels,  # DiT-S default: 384
        "depth": 12,  # DiT-S
        "num_heads": 6,  # DiT-S
        "mlp_ratio": 4.0,
        "cond_features": len(config.features_keys),
    }
    return dit_config


def run(args):
    config = LatentSpectrogramConfig(args.workdir)

    # Override config with command-line arguments
    if args.maxlen is not None:
        config.t = args.maxlen
    if args.nlatent is not None:
        config.latent_channels = args.nlatent

    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    spectr = fake_represent(config.representation, config.t)

    # Use provided name or generate default
    if args.name is not None:
        name = args.name
    else:
        name = f"Latent-FlowMatching-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x{config.latent_channels}-LogSpectrogram"

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

    # Load autoencoder checkpoint
    if args.autoencodername is not None:
        autoencoder_checkpoint = config.outputdir / args.autoencodername / "last.ckpt"
    else:
        autoencoder_checkpoint = (
            config.outputdir
            / f"Autoencoder-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x4-LogSpectrogram"
            / "last.ckpt"
        )
    logging.info(f"Loading autoencoder: {autoencoder_checkpoint}")
    autoencoder = LightningAutoencoder.load_from_checkpoint(autoencoder_checkpoint)

    model = LightningFlowMatching(
        get_dit_config(config, config.latent_channels, args.modelchannels),
        optimizer_params,
        num_sampling_steps=args.sampling_steps,
        autoencoder=autoencoder,
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    import argparse

    parser = argparse.ArgumentParser("Train a 2D latent Flow Matching model with DiT")
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
    parser.add_argument(
        "--maxlen", type=int, help="maximum signal length (overrides config.t)", default=None
    )
    parser.add_argument(
        "--nlatent", type=int, help="number of latent channels (overrides config.latent_channels)", default=None
    )
    parser.add_argument(
        "--modelchannels", type=int, help="number of model hidden channels for DiT", default=384
    )
    parser.add_argument(
        "--sampling-steps", type=int, help="number of ODE integration steps for sampling", default=50
    )
    parser.add_argument(
        "--autoencodername", type=str, help="name of autoencoder checkpoint directory", default=None
    )
    parser.add_argument(
        "--name", type=str, help="name for the experiment (overrides default naming)", default=None
    )
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
