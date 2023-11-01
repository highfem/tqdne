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
import matplotlib.pyplot as plt
import numpy as np
import wandb 

class LogCallback(Callback):
    def __init__(self, wandb_logger, dataset, dataset_train=None, every=5) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.scores = []
        self.scores_val = []
        self.dataset = dataset
        self.dataset_train = dataset_train
        self.total_time = 0
        self.every = every

    def log_images(self, low_res, high_res, reconstructed):
        b, c, t = reconstructed.shape
        fs = 100
        time = np.arange(0, t) / fs
        for i in range(max(4, b)):
            for j in range(c):
                fig = plt.figure(figsize=(6, 3))
                plt.plot(time, low_res[i, j].cpu().numpy(), "b", label="Input")
                plt.plot(time, high_res[i, j].cpu().numpy(), "r", label="Target")
                plt.plot(
                    time,
                    reconstructed[i, j].cpu().numpy(),
                    "g",
                    alpha=0.5,
                    label="Reconstructed",
                )
                plt.legend()
                plt.tight_layout()
                # image = fig2PIL(fig)
                # self.wandb_logger.log_image(
                #     key=f"samples {i} - channel {j}", images=[image]
                # )
                wandb.log({f"Generation - channel {j}": fig})

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation batch ends."""
        low_res, high_res = next(iter(self.dataset))
        device = pl_module.device
        n = 4
        with torch.no_grad():
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            output, target, timesteps = pl_module.forward_step(low_res[:n], high_res[:n])
            for i in range(n):
                self.log_plot(output[i], target[i], f"Timestep {timesteps[i]}")

            if pl_module.current_epoch == 0 or pl_module.current_epoch % self.every != 0:
                # Computation takes time, let us computed it every 10 epochs
                return

            reconstructed = pl_module.evaluate(low_res[:n])
            self.log_images(low_res[:n], high_res[:n], reconstructed)


    def log_plot(self, pred, target, name):
        fs = 100
        assert len(pred.shape) == 2
        c, n = pred.shape
        t = np.arange(0, n) / fs
        mask = np.logical_and(t < 5, t > 1)

        for i in range(c):
            wandb.log({f"{name}_pred_vs_target_{i}" : wandb.plot.line_series(
            xs=t[mask],
            ys=[pred[i, mask], target[i, mask]],
            keys=["Prediction", "Target"],
            title= name + f" - channel {i}",
            xname="Time (s)")})
     
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        time_for_epoch = time.time() - self.start_time
        self.total_time = self.total_time + time_for_epoch
        self.log("traintime", self.total_time, on_step=True, on_epoch=False)


# class LightningClassifier(pl.LightningModule):
#     """A PyTorch Lightning classifier module.

#     Parameters
#     ----------
#     net : torch.nn.Module
#         A PyTorch neural network.
#     lr_rate : float
#         The learning rate for the optimizer.

#     """

#     def __init__(self, net: torch.nn.Module, lr_rate: float = 1e-3):
#         super(LightningClassifier, self).__init__()

#         self.net = net
#         self.lr_rate = lr_rate
#         self.save_hyperparameters()

#     def forward(self, x: torch.Tensor):
#         return self.net(x)

#     def cross_entropy_loss(self, logits: torch.Tensor, labels: torch.Tensor):
#         raise ValueError("You probably want to replace this loss with the CrossEntropyLoss")
#         return F.nll_loss(logits, labels)

#     def global_step(self, batch: List, batch_idx: int, train: bool = False):
#         x, y = batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         if train:
#             self.log("train_loss", loss, prog_bar=True)
#         else:
#             self.log("val_loss", loss, prog_bar=True)
#         return loss

#     def training_step(self, train_batch: List, batch_idx: int):
#         return self.global_step(train_batch, batch_idx, train=True)

#     def validation_step(self, val_batch: List, batch_idx: int):
#         return self.global_step(val_batch, batch_idx)

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
#         lr_scheduler = {
#             "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
#             "name": "expo_lr",
#         }
#         return [optimizer], [lr_scheduler]


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

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        optimizer_params: dict,
        prediction_type: str = "epsilon"
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        if prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        self.prediction_type = prediction_type
        self.pipeline = DDPMPipeline1DCond(self.net, self.noise_scheduler)
        self.save_hyperparameters()

    # def forward(self, x: torch.Tensor):
    #     return self.net(x)
    def evaluate(self, low_res):
        # Sample some signaol from random noise (this is the backward diffusion process).
        sig = self.pipeline(
            low_res=low_res,
            generator=torch.manual_seed(self.optimizer_params["seed"]),
        ).audios

        return sig

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward_step(self, low_res, high_res):

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
        inputs = to_inputs(low_res, noisy_hig_res)
        output = self.net(inputs, timesteps, return_dict=False)[0]

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = high_res
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        
        return output, target, timesteps

    def global_step(self, batch: List, batch_idx: int, train: bool = False):
        low_res, high_res = batch

        output, target, _ = self.forward_step(low_res, high_res)

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
