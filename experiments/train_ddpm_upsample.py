from diffusers import DiffusionPipeline
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdne.conf import DATASETDIR
from pathlib import Path
from tqdne.dataset import H5Dataset
from torch.utils.data import DataLoader
from diffusers import UNet1DModel
from diffusers import DDPMScheduler
from tqdne.diffusers import DDPMPipeline1DCond
from tqdne.lightning import LightningDDMP, LogCallback

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger

from tqdne.conf import OUTPUTDIR, PROJECT_NAME

import pytorch_lightning as pl
import logging

if __name__ == '__main__':

    logging.info("Loading data...")
    t = (5501 // 32) * 32
    batch_size = 64
    max_epochs = 100
    name = '1D-UNET-UPSAMPLE-DDPM'


    path_train = DATASETDIR / Path("data_train.h5")
    path_test = DATASETDIR / Path("data_test.h5")
    train_dataset = H5Dataset(path_train, cut=t)
    test_dataset = H5Dataset(path_test, cut=t)

    channels = train_dataset[0][0].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "sample_size":t,
        "in_channels":2*channels, 
        "out_channels":channels,
        "block_out_channels":  (32, 64, 128, 256),
        "down_block_types": ('DownBlock1D', 'DownBlock1D', 'DownBlock1D', 'AttnDownBlock1D'),
        "up_block_types": ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1D', 'UpBlock1D'),
        "mid_block_type": 'UNetMidBlock1D',
        "extra_in_channels" : 0 
    }

    scheduler_params = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_train_timesteps": 1000,
    }

    optimizer_params = {
        "learning_rate": 1e-4,
        "lr_warmup_steps": 500,
        "n_train": len(train_dataset) // batch_size,
        "seed": 0,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
    }

    trainer_params = {
        # trainer parameters
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1,
        "precision": "32-true",  
        # Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
        # 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        # Can be used on CPU, GPU, TPUs, HPUs or IPUs.
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "num_nodes": 1}
    
    logging.info("Build network...")
    net = UNet1DModel(**unet_params)
    logging.info(net.config)

    logging.info("Build scheduler...")
    scheduler = DDPMScheduler(**scheduler_params)
    logging.info(scheduler.config)

    logging.info("Build pipeline...")
    pipeline = DDPMPipeline1DCond(net, scheduler)
    logging.info(pipeline.config)

    logging.info("Build lightning module...")
    model = LightningDDMP(net, scheduler, optimizer_params)


    logging.info("Build Pytorch Lightning Trainer...")

    # 1. Wandb Logger
    wandb_logger = WandbLogger(project=PROJECT_NAME) # add project='projectname' to log to a specific project

    # 2. Learning Rate Logger
    lr_logger = LearningRateMonitor()
    # 3. Set Early Stopping
    # early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
    # 4. saves checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(dirpath=OUTPUTDIR / Path(name), filename='{name}_{epoch}-{val_loss:.2f}',
                                        monitor='val_loss', mode='min', save_top_k=5)
    # 5. My custom callback
    log_callback = LogCallback(wandb_logger, test_dataset)

    (OUTPUTDIR/Path(name)).mkdir(parents=True, exist_ok=True)
    # Define Trainer
    trainer = pl.Trainer(**trainer_params, logger=wandb_logger, callbacks=[lr_logger, log_callback, checkpoint_callback], 
                        default_root_dir=OUTPUTDIR/Path(name)) 
    
    logging.info("Start training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    logging.info("Done!")
