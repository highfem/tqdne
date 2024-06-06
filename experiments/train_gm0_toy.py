import os

# select GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader

from tqdne.conf import Config
from tqdne.dataset import LowFreqDataset
from tqdne.diffusion import LightningDiffusion
from tqdne.unet import UNetModel
from tqdne.metric import PowerSpectralDensity
from tqdne.representations import Signal
from tqdne.plot import SamplePlot, PowerSpectralDensityPlot, LogEnvelopePlot
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint

if __name__ == "__main__":

    logging.info("Loading data...")
    resume = False
    t = (2500 // 32) * 32
    batch_size = 128
    max_epochs = 500
    prediction_type = "sample"  # `epsilon` (predicts the noise of the diffusion process) or `sample` (directly predicts the noisy sample)

    name = "1D-UNET-TOY-lowfreq-DDIM"
    config = Config()

    train_dataset = LowFreqDataset(t=t)
    test_dataset = LowFreqDataset(512, t=t)

    channels = train_dataset[0]["repr"].shape[0]
    num_cond_features = train_dataset[0]["cond"].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # metrics
    metrics = [PowerSpectralDensity(fs=config.fs, channel=c, data_representation=Signal(), invert_representation=False) for c in range(channels)]

    # plots
    plots = [SamplePlot(fs=config.fs, channel=c, data_representation=Signal(), invert_representation=False) for c in range(channels)]
    plots += [PowerSpectralDensityPlot(fs=config.fs, channel=c, data_representation=Signal(), invert_representation=False) for c in range(channels)]
    plots += [LogEnvelopePlot(fs=config.fs, channel=c, data_representation=Signal(), invert_representation=False) for c in range(channels)]

    logging.info("Build network...")
    net = UNetModel(
        dims=1,
        in_channels=channels, 
        out_channels=channels, 
        model_channels=32, 
        conv_kernel_size=5, 
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.2,
        num_heads=4,
        flash_attention=False,
        cond_emb_scale=None,
        cond_features=num_cond_features
    )

    logging.info(net.config)

    logging.info("Build scheduler...")
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        num_train_timesteps=1000,
        prediction_type="sample",
        clip_sample=False,
    )
    scheduler.set_timesteps(70)
    logging.info(scheduler.config)

    logging.info("Build lightning module...")
    optimizer_params = {
        "learning_rate":3e-4,
        "scheduler_name":"cosine",
        "lr_warmup_steps":500,
        "batch_size":128,
        "seed":0,
        "n_train": len(train_dataset) // batch_size,
        "max_epochs": max_epochs,
    }
    model = LightningDiffusion(
        net=net,
        noise_scheduler=scheduler,
        prediction_type=prediction_type,
        optimizer_params=optimizer_params,
        cond_input=True
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name, 
        task="generation",
        val_loader=test_loader, 
        metrics=metrics, 
        plots=plots,
        eval_every=5, 
        log_to_wandb=True, 
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        gradient_clip_val=1,
        precision="32-true",
        accelerator="auto",
        devices="auto",
        flags={}
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
