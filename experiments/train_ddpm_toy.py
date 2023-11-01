import os
# select GPU 1
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from tqdne.dataset import RandomDataset
from torch.utils.data import DataLoader
from diffusers import UNet1DModel
from diffusers import DDPMScheduler
from tqdne.diffusers import DDPMPipeline1DCond
from tqdne.lightning import LightningDDMP
from tqdne.training import get_pl_trainer
from tqdne.utils import get_last_checkpoint
import logging


if __name__ == '__main__':

    logging.info("Loading data...")
    resume = True
    t = (5501 // 32) * 32
    batch_size = 64
    max_epochs = 1000
    prediction_type = "sample" # `epsilon` (predicts the noise of the diffusion process) or `sample` (directly predicts the noisy sample`

    name = '1D-UNET-TOY-DDPM'


    train_dataset = RandomDataset(1024*8, t=t)
    test_dataset = RandomDataset(512, t=t)


    channels = train_dataset[0][0].shape[0]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "sample_size":t,
        "in_channels":channels*2, 
        "out_channels":channels,
        "block_out_channels":  (32, 64, 128),
        "down_block_types": ('DownBlock1D', 'DownBlock1D', 'AttnDownBlock1D'),
        "up_block_types": ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1D'),
        "mid_block_type": 'UNetMidBlock1D',
        "out_block_type": "OutConv1DBlock",
        "extra_in_channels" : 0,
    }

    scheduler_params = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_train_timesteps": 1000,
        "clip_sample": False,
        "prediction_type": prediction_type, 
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
    model = LightningDDMP(net, scheduler, prediction_type=prediction_type, optimizer_params=optimizer_params)

    logging.info("Build Pytorch Lightning Trainer...")

    trainer = get_pl_trainer(name, test_loader, **trainer_params)
    
    logging.info("Start training...")

    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=checkpoint)

    logging.info("Done!")
