import torch
from torch.nn import functional as F
from typing import List
from diffusers.optimization import get_cosine_schedule_with_warmup
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from diffusers import DDPMPipeline
torch.set_default_dtype(torch.float64)

class DiffusionL(pl.LightningModule):
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

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        optimizer_params: dict,
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        self.pipeline = DDPMPipeline(self.net, self.noise_scheduler)
        self.save_hyperparameters()

    # def forward(self, x: torch.Tensor):
    #     return self.net(x)
        
    def sample(self, num_waveforms):
        noise = torch.randn(num_waveforms, 1, 1024, device=self.device)
        timesteps = torch.full((num_waveforms,), self.noise_scheduler.config.num_train_timesteps - 1, dtype=torch.long, device=self.device)
        out = self.net(noise, timesteps, return_dict=False)[0]
        return out.detach().cpu().numpy()
    
    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward_step(self, high_res):
        # Sample noise to add to the high_res
        noise = torch.randn(high_res.shape).to(high_res.device)
        batch_size = high_res.shape[0]

        # Sample a random timestep for each signal
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=high_res.device,
        ).long()

        # Add noise to the clean high_res according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_hig_res = self.noise_scheduler.add_noise(high_res, noise, timesteps)

        # Predict the noise residual
        inputs = noisy_hig_res
        output = self.net(inputs, timesteps, return_dict=False)[0]

        target = high_res
        
        return output, target, timesteps

    def global_step(self, batch: List, batch_idx: int, train: bool = False):
        high_res = batch

        output, target, _ = self.forward_step(high_res)

        loss = F.mse_loss(output, target)

        self.log_value(loss, "loss", train=train, prog_bar=True)

        return loss

    def training_step(self, train_batch: List, batch_idx: int):
        return self.global_step(train_batch, batch_idx, train=True)

    def validation_step(self, val_batch: List, batch_idx: int):
        return self.global_step(val_batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.optimizer_params["learning_rate"]
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.optimizer_params["lr_warmup_steps"],
            num_training_steps=(
                self.optimizer_params["n_train"] * self.optimizer_params["max_epochs"]
            ),
        )
        return [optimizer], [lr_scheduler]
