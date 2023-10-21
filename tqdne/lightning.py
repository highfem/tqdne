import torch
from torch.nn import functional as F
from typing import List
from diffusers.optimization import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from tqdne.diffusers import DDPMPipeline1DCond
from pytorch_lightning.callbacks import Callback
import time
from tqdne.diffusers import to_inputs
from tqdne.utils import fig2PIL


class LogPredictionsCallback(Callback):
    def __init__(self, wandb_logger, dataset, dataset_train=None) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.scores = []
        self.scores_val = []
        self.dataset = dataset
        self.dataset_train = dataset_train
        self.total_time = 0


    def log_images(self, low_res, high_res, reconstructed):
        pass
    #  self.wandb_logger.log_image(key=f'{"train " if train else "valid "}samples - channel {i}', images=images, caption=captions)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation batch ends."""
        if pl_module.current_epoch==0 or pl_module.current_epoch % 5 != 0:
            # Computation takes time, let us computed it every 10 epochs
            return
        low_res, high_res = next(iter(self.dataset))
        reconstructed = trainer.evaluate(low_res)
        self.log_images(low_res, high_res, reconstructed)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self.start_time = time.time()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        time_for_epoch = time.time()-self.start_time
        self.total_time = self.total_time + time_for_epoch
        self.log("traintime",  self.total_time, on_step=True, on_epoch=False)


class LightningClassifier(pl.LightningModule):
    """A PyTorch Lightning classifier module.

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    lr_rate : float
        The learning rate for the optimizer.

    """

    def __init__(self, net: torch.nn.Module, lr_rate: float = 1e-3):
        super(LightningClassifier, self).__init__()

        self.net = net
        self.lr_rate = lr_rate
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return F.nll_loss(logits, labels)

    def global_step(self, batch: List, batch_idx: int, train: bool = False):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        if train:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def training_step(self, train_batch: List, batch_idx: int):
        return self.global_step(train_batch, batch_idx, train=True)

    def validation_step(self, val_batch: List, batch_idx: int):
        return self.global_step(val_batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            "name": "expo_lr",
        }
        return [optimizer], [lr_scheduler]
    




class LightningDDMP(pl.LightningModule):
    """A PyTorch Lightning module for training a diffusion model.

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    noise_scheduler : DDPMScheduler
        A scheduler for adding noise to the clean images.
    config : TrainingConfig
        A dataclass containing the training configuration.

    """

    def __init__(self, net: torch.nn.Module, noise_scheduler: DDPMScheduler, optimizer_params:dict):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        self.pipeline = DDPMPipeline1DCond(self.net, self.noise_scheduler)
        self.save_hyperparameters()


    # def forward(self, x: torch.Tensor):
    #     return self.net(x)
    def evaluate(self, low_res):
        # Sample some signaol from random noise (this is the backward diffusion process).
        sig = self.pipeline(
            low_res = low_res,
            generator=torch.manual_seed(self.optimizer_params["seed"]),
        ).audios

        return sig

    def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        return F.nll_loss(logits, labels)

    def global_step(self, batch: List, batch_idx: int, train: bool = False):
        low_res, high_res = batch

        # Sample noise to add to the high_res
        noise = torch.randn(high_res.shape).to(high_res.device)
        batch_size = high_res.shape[0]

        # Sample a random timestep for each signal
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=high_res.device
        ).long()

        # Add noise to the clean high_res according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_hig_res = self.noise_scheduler.add_noise(high_res, noise, timesteps)

        # Predict the noise residual
        inputs = to_inputs(low_res, noisy_hig_res)
        noise_pred = self.net(inputs, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        if train:
            self.log("train_loss", loss, prog_bar=True)
        else:
            self.log("val_loss", loss, prog_bar=True)
        return loss

    def training_step(self, train_batch: List, batch_idx: int):
        return self.global_step(train_batch, batch_idx, train=True)

    def validation_step(self, val_batch: List, batch_idx: int):
        return self.global_step(val_batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.optimizer_params["learning_rate"])
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_params["lr_warmup_steps"],
            num_training_steps=(self.optimizer_params["n_train"] * self.optimizer_params["max_epochs"]),
        )
        return [optimizer], [lr_scheduler]
    





