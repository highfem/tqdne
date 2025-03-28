import logging

import torch

from config import MovingAverageEnvelopeConfig
from tqdne import metric, plot
from tqdne.architectures import get_1d_unet_config
from tqdne.dataloader import get_train_and_val_loader
from tqdne.edm import LightningEDM
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint, get_device


def run(args):
    name = "EDM-MovingAvg"
    config = MovingAverageEnvelopeConfig(args.workdir, args.infile)

    train_loader, val_loader = get_train_and_val_loader(config, args.num_workers, args.batchsize)
    metrics = [
        metric.AmplitudeSpectralDensity(fs=config.fs, channel=c, isotropic=True) for c in range(3)
    ]
    plots = [plot.SamplePlot(plot_target=False, fs=config.fs, channel=c) for c in range(3)] + [
        plot.AmplitudeSpectralDensity(fs=config.fs, channel=c) for c in range(3)
    ]

    optimizer_params = {"learning_rate": 0.0001, "max_steps": 200 * len(train_loader), "eta_min": 0.0}
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,        
        "max_steps": optimizer_params["max_steps"],
    }

    logging.info("Build lightning module...")
    model = LightningEDM(get_1d_unet_config(config), optimizer_params)

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
    import argparse
    parser = argparse.ArgumentParser(
        "Train a 1D diffusion model"
    )
    parser.add_argument("--workdir", type=str, help="the working directory in which checkpoints and all output are saved to")
    parser.add_argument("--infile", type=str, default=None, help="location of the training file; if not given assumes training data is located as `workdir/data/preprocessed_waveforms.h5`")
    parser.add_argument('-b', '--batchsize', type=int, help='size of a batch of each gradient step', default=256)
    parser.add_argument('-w', '--num-workers', type=int, help='number of separate processes for file/io', default=32)
    parser.add_argument('-d', '--num-devices', type=int, help='number of CPUs/GPUs to train on', default=4)
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
