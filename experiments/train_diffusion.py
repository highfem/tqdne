from diffusers import UNet1DModel
from diffusers import DDPMScheduler
from tqdne.diffusion_lightning import DiffusionL
from tqdne.training import get_pl_trainer
from pathlib import Path
from tqdne.utils.model_utils import get_last_checkpoint
from tqdne.simple_dataset import StationarySignalDM
from tqdne.callbacks.sample_callback import SimplePlotCallback
import logging


if __name__ == '__main__':

    resume = False
    logging.info("Loading data...")
    batch_size = 64
    max_epochs = 100
    frac_train = 0.8
    wfs_expected_size = 1024
    name = '1D-UNET-UPSAMPLE-DDPM-Moreira'
    dataset_size = 10000
    dm = StationarySignalDM(dataset_size, wfs_expected_size, batch_size, frac_train)    

    logging.info("Set parameters...")

    # Unet parameters
    unet_params = {
        "sample_size":wfs_expected_size,
        "in_channels":1, 
        "out_channels":1,
        "block_out_channels":  (32, 64, 128, 256),
        "down_block_types": ('DownBlock1D', 'DownBlock1D', 'DownBlock1D', 'AttnDownBlock1D'),
        "up_block_types": ('AttnUpBlock1D', 'UpBlock1D', 'UpBlock1D', 'UpBlock1D'),
        "mid_block_type": 'UNetMidBlock1D',
        "out_block_type": "OutConv1DBlock",
        "act_fn": "relu",
        "extra_in_channels" : 0 
    }

    scheduler_params = {
        "beta_schedule": "linear",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "num_train_timesteps": 1000, 
        "clip_sample": False,
    }

    optimizer_params = {
        "learning_rate": 1e-4,
        "lr_warmup_steps": 500,
        "n_train":  dataset_size // batch_size,
        "seed": 0,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
    }

    trainer_params = {
        # trainer parameters
        "accumulate_grad_batches": 1,
        "gradient_clip_val": 1,
        # "precision": "32-true",  
        # Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
        # 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
        # Can be used on CPU, GPU, TPUs, HPUs or IPUs.
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "num_nodes": 1
    }
    plot_callback_parameters = {
        "dataset": dm,
        "every": 5,
        "n_waveforms": 3,
    }
    specific_callbacks = [
        SimplePlotCallback(**plot_callback_parameters)
    ]
    
    logging.info("Build network...")
    net = UNet1DModel(**unet_params)
    logging.info(net.config)

    logging.info("Build scheduler...")
    scheduler = DDPMScheduler(**scheduler_params)
    logging.info(scheduler.config)

    logging.info("Build lightning module...")
    model = DiffusionL(net, scheduler, optimizer_params=optimizer_params)

    logging.info("Build Pytorch Lightning Trainer...")

    trainer = get_pl_trainer(name, project="diffusion-dummy", specific_callbacks=specific_callbacks, **trainer_params)
    
    logging.info("Start training...")

    if resume:
        checkpoint = get_last_checkpoint(trainer.default_root_dir)
    else:
        checkpoint = None
    trainer.fit(model, dm, ckpt_path=checkpoint)

    logging.info("Done!")
