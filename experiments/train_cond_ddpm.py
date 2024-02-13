import os

from tqdne.representations import SignalWithEnvelope

# select GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from diffusers import DDPMScheduler
from tqdne.conf import Config
from tqdne.dataset import EnvelopeDataset
from tqdne.diffusion import LightningDDMP
from tqdne.metric import (
    BinMetric,
    MeanSquaredError,
    PowerSpectralDensity,
    RepresentationInversion,
    SamplePlot,
)
from tqdne.training import get_pl_trainer
from tqdne.unet_1d import UNet1DModel
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    resume = False
    logging.info("Loading data...")
    t = (5501 // 32) * 32 # TO ASK: why 5501? why 32?
    batch_size = 64
    max_epochs = 150
    prediction_type = "sample"  # `epsilon` (predicts the noise of the diffusion process) or `sample` (directly predicts the noisy sample)

    name = "COND-1D-UNET-DDPM-envelope"
    config = Config()

    path_train = config.datasetdir / config.data_train
    path_test = config.datasetdir / config.data_test
    
    # DEBUG
    #path_train = Path("/users/abosisio/scratch/tqdne/datasets/small_data_upsample_train.h5")
    #path_test = Path("/users/abosisio/scratch/tqdne/datasets/small_data_upsample_test.h5")

    train_dataset = EnvelopeDataset(path_train, SignalWithEnvelope(config), cut=t) # to check
    test_dataset = EnvelopeDataset(path_test, SignalWithEnvelope(config), cut=t) # to check

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    channels = train_dataset[0]["representation"].shape[0] # aclready acounts for both envelope and signal (i.e. 6 channels in total)

    # train_dataset[0]['representation'].shape --> torch.Size([6, 5472])
    # rain_dataset[0]['cond'].shape --> torch.Size([5])
    
    plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels//2)]
    plots = [RepresentationInversion(metric, SignalWithEnvelope(config)) for metric in plots]
    plots_debug = [SamplePlot(fs=config.fs, channel=c) for c in range(channels)]
    psd = [PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels//2)]
    psd = [RepresentationInversion(metric, SignalWithEnvelope(config)) for metric in psd]
    bin_metrics = [BinMetric(metric) for metric in psd]
    metrics = plots + plots_debug + psd + bin_metrics

    logging.info("Set parameters...")

    # Unet parameters # BLOCKS ALREADY FIXED
    unet_params = {
        "sample_size": t,
        "in_channels": channels, 
        "out_channels": channels,
        "block_out_channels": (32, 64, 128, 256),
        "down_block_types": (
            "DownBlock1D",
            "DownBlock1D",
            "DownBlock1D",
            "AttnDownBlock1D",
        ),
        "up_block_types": (
            "AttnUpBlock1D", 
            "UpBlock1D", 
            "UpBlock1D", 
            "UpBlock1D",
        ),
        "mid_block_type": "UNetMidBlock1D",
        "out_block_type": "OutConv1DBlock",
        "extra_in_channels": 0,
        "act_fn": "relu",
        "cond_dim": len(config.features_keys),
        "cond_concat": True,
    }

    scheduler_params = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_train_timesteps": 1000,
        "prediction_type": prediction_type,
        "clip_sample": False,
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
        "num_nodes": 1,
    }

    logging.info("Build network...")
    net = UNet1DModel(**unet_params)
    logging.info(net.config)

    logging.info("Build scheduler...")
    scheduler = DDPMScheduler(**scheduler_params)
    logging.info(scheduler.config)

    logging.info("Build lightning module...")
    model = LightningDDMP(
        net,
        scheduler,
        prediction_type=prediction_type,
        optimizer_params=optimizer_params,
        low_res_input=False,
        cond_input=True,
    )

    logging.info("Build Pytorch Lightning Trainer...")

    trainer = get_pl_trainer(name, test_loader, metrics, eval_every=5, **trainer_params)

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
