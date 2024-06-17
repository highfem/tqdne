import logging

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from tqdne.classifier import LithningClassifier
from tqdne.config import SpectrogramClassificationConfig
from tqdne.dataset import ClassificationDataset
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    logging.info("Set parameters...")

    name = "Classifier-LogSpectrogram"
    config = SpectrogramClassificationConfig()
    max_epochs = 100
    batch_size = 128
    lr = 1e-4
    resume = True

    train_dataset = ClassificationDataset(
        config.datapath,
        config.representation,
        mag_bins=config.mag_bins,
        dist_bins=config.dist_bins,
        cut=config.t,
        split="train",
    )

    test_dataset = ClassificationDataset(
        config.datapath,
        config.representation,
        mag_bins=config.mag_bins,
        dist_bins=config.dist_bins,
        cut=config.t,
        split="test",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=5, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # loss and metrics
    class_weights = train_dataset.get_class_weights()
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    metrics = [
        MulticlassAccuracy(len(class_weights)),
        MulticlassRecall(len(class_weights)),
        MulticlassPrecision(len(class_weights)),
        MulticlassF1Score(len(class_weights)),
    ]

    logging.info("Set parameters...")

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

    max_steps = max_epochs * len(train_loader)
    trainer_params = {
        "precision": 32,
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": max_steps,
    }

    logging.info("Build lightning module...")
    output_layer = torch.nn.Linear(128, len(class_weights))
    classifier = LithningClassifier(
        encoder_config=encoder_config,
        num_classes=len(class_weights),
        loss=loss,
        metrics=metrics,
        optimizer_params={"learning_rate": lr, "max_steps": max_steps},
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
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
    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    trainer.fit(
        classifier,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")
