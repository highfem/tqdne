import logging

import torch
from config import SpectrogramClassificationConfig
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from tqdne.classifier import LithningClassifier
from tqdne.dataset import ClassificationDataset
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint, get_device


def run(args):
    name = "Classifier-LogSpectrogram"
    config = SpectrogramClassificationConfig(args.workdir, None)

    train_dataset = ClassificationDataset(
        config.datapath,
        config.representation,
        mag_bins=config.mag_bins,
        dist_bins=config.dist_bins,
        cut=config.t,
        split="train",
    )

    val_dataset = ClassificationDataset(
        config.datapath,
        config.representation,
        mag_bins=config.mag_bins,
        dist_bins=config.dist_bins,
        cut=config.t,
        split="validation",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=True, drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batchsize, num_workers=args.num_workers,
        prefetch_factor=2,
        drop_last=True,
        persistent_workers=True,
    )

    # loss and metrics
    class_weights = train_dataset.get_class_weights()
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    metrics = [
        MulticlassAccuracy(len(class_weights)),
        MulticlassRecall(len(class_weights)),
        MulticlassPrecision(len(class_weights)),
        MulticlassF1Score(len(class_weights)),
    ]

    # Parameters
    encoder_config = {
        "in_channels": config.channels,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "out_channels": 256,
        "num_res_blocks": 2,
        "attention_resolutions": (8,),
        "dims": 2,
        "conv_kernel_size": 3,
        "num_heads": 4,
        "dropout": 0.1,
        "flash_attention": False,
    }

    optimizer_params = {"learning_rate": 0.001, "max_steps": 100 * len(train_loader), "eta_min": 0.0}
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": optimizer_params["max_steps"],
    }

    logging.info("Build lightning module...")
    classifier = LithningClassifier(
        encoder_config=encoder_config,
        num_classes=len(class_weights),
        loss=loss,
        metrics=metrics,
        optimizer_params=optimizer_params,
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        val_loader,
        config.representation,
        metrics=[],
        plots=[],
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir)
    trainer.fit(
        classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(
        "Train a classifier"
    )
    parser.add_argument("--workdir", type=str, help="the working directory in which checkpoints and all output are saved to")
    parser.add_argument('-b', '--batchsize', type=int, help='size of a batch of each gradient step', default=128)
    parser.add_argument('-w', '--num-workers', type=int, help='number of separate processes for file/io', default=32)
    parser.add_argument('-d', '--num-devices', type=int, help='number of CPUs/GPUs to train on', default=4)
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
