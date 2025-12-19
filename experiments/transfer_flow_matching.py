"""Transfer learning script for flow matching model on new seismic data from different location.

This script loads a pre-trained Flow Matching model and fine-tunes it on seismic data
from a new location. It uses the transferred autoencoder from Phase 1.

It also supports cross-transfer: loading DiT weights from a diffusion checkpoint
and using them for flow matching training.
"""

import logging
import sys
from pathlib import Path

import torch
from config import TransferConfig

from tqdne import metric, plot
from tqdne.autoencoder import LightningAutoencoder
from tqdne.dataloader import get_train_and_val_loader
from tqdne.flow_matching import LightningFlowMatching
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint


def fake_represent(representation, leng_signal):
    """Generate fake representation for shape inference."""
    signal = torch.ones((1, leng_signal))
    spectr = representation.get_representation(signal)
    return spectr


def get_dit_config(config, latent_channels, model_channels=768):
    """Get DiT-B/2 configuration for flow matching."""
    dit_config = {
        "input_size": 32,  # latent spatial size
        "patch_size": 2,
        "in_channels": latent_channels,
        "hidden_size": model_channels,  # DiT-B default: 768
        "depth": 12,  # DiT-B
        "num_heads": 12,  # DiT-B
        "mlp_ratio": 4.0,
        "cond_features": len(config.features_keys),
    }
    return dit_config


def cross_transfer_dit_to_flow_matching(dit_checkpoint_path, flow_matching_model):
    """Transfer DiT weights from diffusion model to flow matching model.

    Parameters
    ----------
    dit_checkpoint_path : Path
        Path to the DiT diffusion checkpoint
    flow_matching_model : LightningFlowMatching
        Target flow matching model (already initialized with correct architecture)

    Returns
    -------
    LightningFlowMatching
        Flow matching model with transferred DiT weights
    """
    logging.info("Cross-transfer: Loading DiT weights from diffusion checkpoint...")

    # Load the diffusion checkpoint
    dit_checkpoint = torch.load(dit_checkpoint_path, map_location="cpu")

    # Extract only the DiT model weights (not optimizer state)
    dit_state_dict = {}
    for key, value in dit_checkpoint["state_dict"].items():
        if key.startswith("dit."):
            # Keep the same key structure
            dit_state_dict[key] = value

    # Load DiT weights into flow matching model
    # Use strict=False to allow optimizer state mismatch
    missing_keys, unexpected_keys = flow_matching_model.load_state_dict(
        dit_state_dict, strict=False
    )

    # Log transfer details
    logging.info(f"  Transferred {len(dit_state_dict)} DiT parameters")
    if missing_keys:
        logging.info(f"  Missing keys (expected, will be randomly initialized): {len(missing_keys)}")
        # These should only be optimizer state and other non-DiT parameters
    if unexpected_keys:
        logging.warning(f"  Unexpected keys: {unexpected_keys}")

    return flow_matching_model


def run(args):
    config = TransferConfig(
        workdir=args.workdir,
        target_workdir=args.target_workdir,
        source_autoencoder_checkpoint=args.source_autoencoder_checkpoint,
        source_flow_matching_checkpoint=args.source_flow_matching_checkpoint,
        source_diffusion_checkpoint=args.source_diffusion_checkpoint,
        transfer_strategy=args.strategy,
    )

    # Override config with command-line arguments
    if args.maxlen is not None:
        config.t = args.maxlen
    if args.nlatent is not None:
        config.latent_channels = args.nlatent
    if args.learning_rate is not None:
        config.flow_matching_learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.flow_matching_max_steps = args.max_steps

    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    spectr = fake_represent(config.representation, config.t)

    # Determine transfer mode
    is_flow_matching_source = args.source_flow_matching_checkpoint is not None
    is_cross_transfer = args.source_diffusion_checkpoint is not None

    if is_flow_matching_source and is_cross_transfer:
        raise ValueError(
            "Cannot specify both --source-flow-matching-checkpoint and --source-diffusion-checkpoint. "
            "Use one or the other."
        )

    if not is_flow_matching_source and not is_cross_transfer:
        raise ValueError(
            "Must specify either --source-flow-matching-checkpoint or --source-diffusion-checkpoint"
        )

    # Generate name
    if args.name is not None:
        name = args.name
    else:
        transfer_type = "FlowMatching" if is_flow_matching_source else "DiT-to-FlowMatching"
        name = f"Transfer-{transfer_type}-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x{config.latent_channels}-LogSpectrogram"

    # Load target domain data
    logging.info(f"Loading target domain data from: {config.datapath}")
    train_loader, val_loader = get_train_and_val_loader(
        config, args.num_workers, args.batchsize, cond=True
    )

    # Metrics and plots
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ]
    plots = [plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    # Adjust hyperparameters for cross-transfer if needed
    warmup_steps = config.flow_matching_warmup_steps
    max_steps = config.flow_matching_max_steps

    if is_cross_transfer:
        # Cross-transfer benefits from longer warmup and more training steps
        warmup_steps = 1000  # 2x longer than standard (500)
        max_steps = 150_000  # 50% more than standard (100k)
        logging.info("Cross-transfer detected: using extended training schedule")
        logging.info(f"  Warmup steps: {warmup_steps}, Max steps: {max_steps}")

    # Optimizer parameters for transfer learning
    optimizer_params = {
        "learning_rate": config.flow_matching_learning_rate,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
        "decay_steps": max_steps,
        "end_learning_rate": config.flow_matching_end_learning_rate,
        "gradient_clipping": config.gradient_clipping,
        "b1": 0.9,
        "b2": 0.999,
        "weight_decay": config.weight_decay,
    }
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": max_steps,
    }

    # Load autoencoder checkpoint (transferred from Phase 1)
    if args.transferred_autoencoder_checkpoint is not None:
        autoencoder_checkpoint = Path(args.transferred_autoencoder_checkpoint)
    else:
        # Try to find the transferred autoencoder automatically
        autoencoder_checkpoint = (
            config.outputdir
            / f"Transfer-Autoencoder-{spectr.shape[1] // 4}x{spectr.shape[2] // 4}x{config.latent_channels}-LogSpectrogram"
            / "last.ckpt"
        )

    if not autoencoder_checkpoint.exists():
        raise FileNotFoundError(
            f"Transferred autoencoder checkpoint not found: {autoencoder_checkpoint}\n"
            f"Please run transfer_autoencoder.py first or specify --transferred-autoencoder-checkpoint"
        )

    logging.info(f"Loading transferred autoencoder from: {autoencoder_checkpoint}")
    autoencoder = LightningAutoencoder.load_from_checkpoint(autoencoder_checkpoint)
    autoencoder.eval()

    # Option to freeze autoencoder completely
    if args.freeze_autoencoder:
        logging.info("Freezing autoencoder parameters...")
        for param in autoencoder.parameters():
            param.requires_grad = False
    else:
        logging.info("Autoencoder will be fine-tuned jointly with flow matching model")

    # Load pre-trained model
    if is_flow_matching_source:
        # Standard transfer: Load from flow matching checkpoint
        source_checkpoint_path = Path(config.source_flow_matching_checkpoint)

        if not source_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Source flow matching checkpoint not found: {source_checkpoint_path}"
            )

        logging.info(f"Loading pre-trained flow matching model from: {source_checkpoint_path}")
        logging.info("Transfer mode: Flow Matching -> Flow Matching")

        model = LightningFlowMatching.load_from_checkpoint(
            source_checkpoint_path,
            autoencoder=autoencoder,
            optimizer_params=optimizer_params,  # Update optimizer for transfer learning
            num_sampling_steps=25,
            strict=False,  # Allow loading even if optimizer state differs
        )

    else:
        # Cross-transfer: Load DiT weights from diffusion checkpoint
        source_checkpoint_path = Path(config.source_diffusion_checkpoint)

        if not source_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Source diffusion checkpoint not found: {source_checkpoint_path}"
            )

        logging.info(f"Loading pre-trained diffusion model from: {source_checkpoint_path}")
        logging.info("Transfer mode: DiT (Diffusion) -> Flow Matching (CROSS-TRANSFER)")
        logging.info("  This will extract DiT weights and initialize a Flow Matching model")

        # First, create a new flow matching model with the correct architecture
        model = LightningFlowMatching(
            dit_config=get_dit_config(config, config.latent_channels, args.model_channels),
            optimizer_params=optimizer_params,
            num_sampling_steps=25,
            autoencoder=autoencoder,
        )

        # Then, transfer DiT weights from the diffusion checkpoint
        model = cross_transfer_dit_to_flow_matching(source_checkpoint_path, model)

    # Apply user's autoencoder freezing preference
    # (override the default freezing from model constructor)
    if not args.freeze_autoencoder:
        logging.info("Unfreezing autoencoder for joint fine-tuning...")
        for param in model.autoencoder.parameters():
            param.requires_grad = True

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    logging.info("Building Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=val_loader,
        config=config,
        metrics=metrics,
        plots=plots,
        ema_decay=config.ema_decay,
        eval_every=5000,
        limit_eval_batches=25,
        checkpoint_every_n_steps=10_000,
        max_checkpoints_to_keep=10,
        early_stopping_patience=50,
        early_stopping_min_delta=0.001,
        sampling_every_n_steps=10_000,
        sampling_batches=5,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Starting transfer learning (fine-tuning)...")
    torch.set_float32_matmul_precision("high")

    # Check if we should resume from existing transfer checkpoint
    checkpoint = None
    if args.resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)

    trainer.fit(
        model,
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
        description="Transfer learning for flow matching model on new seismic data"
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
        "--source-autoencoder-checkpoint",
        type=str,
        required=True,
        help="Path to SOURCE autoencoder checkpoint (for reference, not used directly)",
    )
    parser.add_argument(
        "--source-flow-matching-checkpoint",
        type=str,
        default=None,
        help="Path to source flow matching checkpoint (e.g., workdir/outputs/Latent-FlowMatching-.../last.ckpt)",
    )
    parser.add_argument(
        "--source-diffusion-checkpoint",
        type=str,
        default=None,
        help="Path to source DiT diffusion checkpoint for cross-transfer (e.g., workdir/outputs/Latent-DiT-.../last.ckpt)",
    )
    parser.add_argument(
        "--transferred-autoencoder-checkpoint",
        type=str,
        default=None,
        help="Path to transferred autoencoder checkpoint from Phase 1. If not specified, will auto-detect.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="conservative",
        choices=["conservative", "aggressive"],
        help="Transfer strategy: affects learning rates and training duration",
    )
    parser.add_argument(
        "--freeze-autoencoder",
        action="store_true",
        help="Freeze autoencoder parameters (recommended if autoencoder transfer was successful)",
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
        "--maxlen",
        type=int,
        default=None,
        help="Maximum signal length (overrides config.t)",
    )
    parser.add_argument(
        "--nlatent",
        type=int,
        default=None,
        help="Number of latent channels (overrides config.latent_channels)",
    )
    parser.add_argument(
        "--model-channels",
        type=int,
        default=768,
        help="DiT hidden size (must be divisible by 12, e.g., 768, 1152, 1536). Only used for cross-transfer.",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=256,
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
