import logging

import torch
from config import LatentSpectrogramConfig
from torch.utils.data import DataLoader

from tqdne import metric, plot
from tqdne.autoencoder import LithningAutoencoder
from tqdne.dataset import Dataset
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

def run(args):
    logging.info("Set parameters...")
    name = "Latent-EDM-LogSpectrogram"
    config = LatentSpectrogramConfig(args.workdir, args.infile)
    config.representation.disable_multiprocessing()  # needed for Pytorch Lightning
    max_epochs = 300
    batch_size = 2048 * 4
    lr = 1e-4
    ema_decay = 0.999
    resume = True

    train_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="train"
    )
    val_dataset = Dataset(
        config.datapath, config.representation, cut=config.t, cond=True, split="validation"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        num_workers=32, 
        shuffle=True,         
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=32, 
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # metrics
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ]

    # plots
    plots = [plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    # Unet parameters
    unet_config = {
        "in_channels": config.latent_channels,
        "out_channels": config.latent_channels,
        "cond_features": len(config.features_keys),
        "dims": 2,
        "conv_kernel_size": 3,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "attention_resolutions": (8,),
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.1,
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
        "check_val_every_n_epoch": 5,
    }

    optimizer_params = {"learning_rate": lr, "max_steps": max_steps}

    logging.info("Loading autoencoder...")
    checkpoint = (
        config.outputdir / "Autoencoder-32x32x4-LogSpectrogram" / "best.ckpt"
    )
    autoencoder = LithningAutoencoder.load_from_checkpoint(checkpoint)

    logging.info("Build lightning module...")
    model = LightningEDM(unet_config, optimizer_params, autoencoder=autoencoder)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        val_loader,
        config.representation,
        metrics,
        plots,
        ema_decay=ema_decay,
        eval_every=20,
        limit_eval_batches=1,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir) if resume else None
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train a variational autoencoder")
    parser.add_argument("--workdir", type=str, help="the working directory in which checkpoints and all output are saved to")
    parser.add_argument("--infile", type=str, default=None, help="location of the training file; if not given assumes training data is located as `workdir/data/preprocessed_waveforms.h5`")
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)

