"""Transfer learning script for autoencoder on new seismic data from different location.

This script loads a pre-trained autoencoder and fine-tunes it on seismic data from a new location.
"""

import logging
import sys
from pathlib import Path

import torch
from config import TransferConfig

from tqdne import metric, plot
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataloader import get_train_and_val_loader
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint


def fake_represent(representation, leng_signal):
    signal = torch.ones((1, leng_signal))
    spectr = representation.get_representation(signal)
    return spectr


def freeze_parameters(model, parameter_names):
    """Freeze specific parameters in the model."""
    frozen_count = 0
    for name, param in model.named_parameters():
        for param_pattern in parameter_names:
            if param_pattern in name:
                param.requires_grad = False
                frozen_count += 1
                logging.info(f"  Frozen: {name}")
                break
    return frozen_count


def run(args):
    config = TransferConfig(
        workdir=args.workdir,
        target_workdir=args.target_workdir,
        source_autoencoder_checkpoint=args.source_checkpoint,
        transfer_strategy=args.strategy,
    )

    # Override config with command-line arguments if provided
    if args.freeze_encoder is not None:
        config.freeze_encoder = args.freeze_encoder
    if args.freeze_decoder is not None:
        config.freeze_decoder = args.freeze_decoder
    if args.learning_rate is not None:
        config.autoencoder_learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.autoencoder_max_steps = args.max_steps

    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    spectr = fake_represent(config.representation, config.t)
    name = f"Transfer-Autoencoder-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x4-LogSpectrogram"

    if args.name is not None:
        name = args.name

    # Load target domain data
    logging.info(f"Loading target domain data from: {config.datapath}")
    train_loader, val_loader = get_train_and_val_loader(config, args.num_workers, args.batchsize)

    # Metrics and plots
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ] + [metric.MeanSquaredError(channel=c) for c in range(3)]
    plots = [plot.SamplePlot(plot_target=True, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    # Optimizer parameters for transfer learning (lower LR, fewer steps)
    optimizer_params = {
        "learning_rate": config.autoencoder_learning_rate,
        "max_steps": config.autoencoder_max_steps,
        "eta_min": 0.0,
    }
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": config.autoencoder_max_steps,
    }

    # Load pre-trained autoencoder from source domain
    source_checkpoint_path = Path(config.source_autoencoder_checkpoint)
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Source autoencoder checkpoint not found: {source_checkpoint_path}"
        )

    logging.info(f"Loading pre-trained autoencoder from: {source_checkpoint_path}")
    autoencoder = LightningAutoencoder.load_from_checkpoint(
        source_checkpoint_path,
        optimizer_params=optimizer_params,  # Update optimizer params for transfer learning
        kl_weight=config.kl_weight,
        strict=False,  # Allow loading even if optimizer state differs
    )

    # Apply freezing strategy
    logging.info(f"Transfer strategy: {config.transfer_strategy}")
    logging.info(f"Freeze encoder: {config.freeze_encoder}")
    logging.info(f"Freeze decoder: {config.freeze_decoder}")

    if config.freeze_encoder:
        logging.info("Freezing encoder parameters...")
        frozen = freeze_parameters(autoencoder, ["encoder"])
        logging.info(f"  Total encoder parameters frozen: {frozen}")

    if config.freeze_decoder:
        logging.info("Freezing decoder parameters...")
        frozen = freeze_parameters(autoencoder, ["decoder"])
        logging.info(f"  Total decoder parameters frozen: {frozen}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    logging.info("Building Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        plots=plots,
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=True,
        checkpoint_every_n_steps=5_000,
        max_checkpoints_to_keep=5,
        **trainer_params,
    )

    logging.info("Starting transfer learning (fine-tuning)...")
    torch.set_float32_matmul_precision("high")

    # Check if we should resume from existing transfer checkpoint
    checkpoint = None
    if args.resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)

    trainer.fit(
        autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Transfer learning complete!")
    logging.info(f"Checkpoints saved to: {trainer.default_root_dir}")


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    import argparse

    parser = argparse.ArgumentParser(
        description="Transfer learning for autoencoder on new seismic data"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        help="Source domain working directory (for reference)",
    )
    parser.add_argument(
        "--target-workdir",
        type=str,
        required=True,
        help="Target domain working directory (where target data and outputs will be saved)",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=str,
        required=True,
        help="Path to source autoencoder checkpoint (e.g., workdir/outputs/Autoencoder-.../last.ckpt)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="conservative",
        choices=["conservative", "aggressive"],
        help="Transfer strategy: 'conservative' (freeze encoder) or 'aggressive' (fine-tune all)",
    )
    parser.add_argument(
        "--freeze-encoder",
        type=bool,
        default=None,
        help="Override: freeze encoder (True/False)",
    )
    parser.add_argument(
        "--freeze-decoder",
        type=bool,
        default=None,
        help="Override: freeze decoder (True/False)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override: learning rate for fine-tuning",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override: maximum training steps",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=128,
        help="Batch size for each gradient step",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=32,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "-d",
        "--num-devices",
        type=int,
        default=4,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for the experiment (overrides default naming)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint in target directory",
    )

    args = parser.parse_args()
    run(args)
