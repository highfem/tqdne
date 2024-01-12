import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm

from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup



class LightningDDMP(pl.LightningModule):
    """A PyTorch Lightning module for training a diffusion model

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    noise_scheduler : DDPMScheduler
        A scheduler for adding noise to the clean images.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    prediction_type : str, optional
        The type of prediction to make. One of "epsilon" or "sample".
    low_res_input : bool, optional
        Whether low resolution input is provided.
    cond_input : bool, optional
        Whether conditional input is provided.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: DDPMScheduler,
        optimizer_params: dict,
        prediction_type: str = "epsilon",
        low_res_input: bool = False,
        cond_input: bool = False,
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        if prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        self.prediction_type = prediction_type
        self.low_res_input = low_res_input
        self.cond_input = cond_input
        self.save_hyperparameters()

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward(self, input, t, low_res=None, cond=None):
        """Make a forward pass through the network."""

        # input
        if self.low_res_input:
            assert low_res is not None
            input = torch.cat((low_res, input), dim=1)

        # predict
        if self.cond_input:
            assert cond is not None
            return self.net(input, t, cond)[0]
        else:
            return self.net(input, t)[0]

    def sample(self, shape, low_res=None, cond=None):
        """Sample from the diffusion model."""
        # initialize noise
        sample = torch.randn(shape, device=self.device)

        # sample iteratively
        for t in tqdm(self.noise_scheduler.timesteps):
            pred = self.forward(sample, t, low_res, cond)
            sample = self.noise_scheduler.step(pred, t, sample).prev_sample

        return sample

    def evaluate(self, batch):
        """Evaluate diffusion model."""
        shape = batch["high_res"].shape
        low_res = batch["low_res"] if self.low_res_input else None
        cond = batch["cond"] if self.cond_input else None
        sample = self.sample(shape, low_res, cond)
        return {"high_res": sample}

    def step(self, batch, train):
        high_res = batch["high_res"]

        # add noise
        noise = torch.randn(high_res.shape, device=high_res.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (high_res.shape[0],),
            device=high_res.device,
        ).long()
        noisy_hig_res = self.noise_scheduler.add_noise(high_res, noise, timesteps)

        # loss
        pred = self.forward(noisy_hig_res, timesteps, batch["low_res"], batch["cond"])
        target = noise if self.prediction_type == "epsilon" else high_res
        loss = F.mse_loss(pred, target)
        self.log_value(loss, "loss", train=train, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, train=True)

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, train=False)

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

