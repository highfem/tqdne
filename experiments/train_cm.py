import os

# select GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from tqdne.conf import Config
from tqdne.consistency_model import LightningConsistencyModel
from tqdne.dataset import UpsamplingDataset
from tqdne.metric import PowerSpectralDensity, SamplePlot
from tqdne.training import get_pl_trainer
from tqdne.unet import UNetModel
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    resume = False
    logging.info("Loading data...")
    t = 4096
    batch_size = 64

    name = "CM-Unet1D-Upsample"
    config = Config()

    path_train = config.datasetdir / Path(config.data_upsample_train)
    path_test = config.datasetdir / Path(config.data_upsample_test)
    train_dataset = UpsamplingDataset(path_train, cut=t, cond=False)
    test_dataset = UpsamplingDataset(path_test, cut=t, cond=False)

    channels = train_dataset[0]["high_res"].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels)]
    psd = [PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]
    metrics = plots + psd

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "in_channels": channels * 2,  # high_res and low_res
        "out_channels": channels,
        "cond_features": None,  # set to 5 if cond=True
        "dims": 1,
        "conv_kernel_size": 3,  # might want to change to 5
        "model_channels": 32,
        "num_res_blocks": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "flash_attention": False,  # flash attention not tested (potentially faster)
    }

    max_epochs = 100
    trainer_params = {
        "precision": 32,
        "max_steps": max_epochs * len(train_loader),
        "accelerator": "auto",
        "devices": "1",
        "num_nodes": 1,
    }

    logging.info("Build network...")
    net = UNetModel(**unet_params)

    logging.info("Build lightning module...")
    model = LightningConsistencyModel(net, lr=1e-4)

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name,
        test_loader,
        metrics,
        eval_every=5,
        limit_eval_batches=-1,
        log_to_wandb=True,
        **trainer_params
    )

    logging.info("Start training...")

    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")
