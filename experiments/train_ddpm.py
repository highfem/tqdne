import os

# select GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from diffusers import DDPMScheduler, UNet1DModel
from tqdne.conf import Config
from tqdne.dataset import UpsamplingDataset
from tqdne.diffusion import LightningDDMP
from tqdne.metric import PowerSpectralDensity, SamplePlot, RepresentationInversion
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":
    resume = False
    logging.info("Loading data...")
    t = (5501 // 32) * 32
    batch_size = 64
    max_epochs = 100
    prediction_type = "sample"  # `epsilon` (predicts the noise of the diffusion process) or `sample` (directly predicts the noisy sample`

    name = "1D-UNET"
    config = Config()

    path_train = config.datasetdir / Path(config.data_upsample_train)
    path_test = config.datasetdir / Path(config.data_upsample_test)
    train_dataset = UpsamplingDataset(path_train, cut=t)
    test_dataset = UpsamplingDataset(path_test, cut=t)

    channels = train_dataset[0]["high_res"].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=5)

    # metrics
    plots = [SamplePlot(fs=config.fs, channel=c) for c in range(channels)]
    plots = [RepresentationInversion(metric, envelope_repesntation) for metric in plots]
    psd = [PowerSpectralDensity(fs=config.fs, channel=c) for c in range(channels)]
    metrics = plots + psd

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "sample_size": t,
        "in_channels": channels,
        "out_channels": channels,
        "block_out_channels": (32, 64, 128, 256),
        "down_block_types": (
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "DownResnetBlock1D",
            "AttnDownBlock1D",
        ),
        "up_block_types": ("AttnUpBlock1D", "UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
        "mid_block_type": "UNetMidBlock1D",
        "out_block_type": "OutConv1DBlock",
        "extra_in_channels": 0,
        "act_fn": "relu",
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
        cond_input=False,
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
