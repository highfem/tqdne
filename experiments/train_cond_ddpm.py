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

    path_train = config.datapath / config.data_train
    path_test = config.datapath / config.data_test
    train_dataset = EnvelopeDataset(path_train, cut=t) # to check
    test_dataset = EnvelopeDataset(path_test, cut=t) # to check

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    channels = train_dataset[0]["signals"].shape[0]


    # metrics #TODO: use also the invertrepresentation metric thing. Plots: signal_scaled and envelope, signal_scaled*envelope, PSD of the signal_scaled*envelope, MSE for all, bin of cond for MSE  
    #plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels)] 
    #mse = [MeanSquaredError(channel=c) for c in range(channels)] # doesn't make sense here
    #psd = [PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]
    #bin_metrics = [BinMetric(metric) for metric in mse + psd] # to ask: why bin psd? it's a scalar (Wassertein-2) between the test-set mean PSD and generated mean PSD (batch)
    #metrics = plots + mse + psd + bin_metrics

    plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels)]
    plots = [RepresentationInversion(metric, SignalWithEnvelope) for metric in plots]
    psd = [PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]
    bin_metrics = [BinMetric(metric) for metric in psd]
    metrics = plots + psd + bin_metrics

    logging.info("Set parameters...")

    # Unet parameters # BLOCKS ALREADY FIXED
    unet_params = {
        "sample_size": t,
        "in_channels": 2 * channels, 
        "out_channels": channels,
        "block_out_channels": (32, 64, 128, 256),
        "down_block_types": (
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "AttnDownBlock1D",
        ),
        "up_block_types": ("AttnUpBlock1D", "UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
        "mid_block_type": "MidResTemporalBlock1D",
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
        low_res_input=True,
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
