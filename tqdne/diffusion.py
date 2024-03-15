import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from tqdm import tqdm

from diffusers import ConfigMixin, SchedulerMixin
from diffusers.optimization import get_cosine_schedule_with_warmup


class LightningDiffusion(pl.LightningModule):
    """A PyTorch Lightning module for training a diffusion model

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    noise_scheduler : SchedulerMixin
        A scheduler for adding noise to the clean images.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    prediction_type : str, optional
        The type of prediction to make. One of "epsilon" or "sample".
    cond_signal_input : bool, optional
        Whether low resolution input is provided.
    cond_input : bool, optional
        Whether conditional input is provided.
    example_input_array : torch.Tensor, optional
        An example input array for the network.    
    """

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: SchedulerMixin,
        optimizer_params: dict,
        prediction_type: str = "epsilon",
        cond_signal_input: bool = False,
        cond_input: bool = False,
        example_input_array: torch.Tensor = None,
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.noise_scheduler = noise_scheduler
        if prediction_type not in ["epsilon", "sample"]:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        self.prediction_type = prediction_type
        self.cond_signal_input = cond_signal_input
        self.cond_input = cond_input
        self.save_hyperparameters()
        self.example_input_array = example_input_array

    def log_value(self, value, name, train=True, prog_bar=True):
        if train:
            self.log(f"train_{name}", value, prog_bar=prog_bar)
        else:
            self.log(f"val_{name}", value, prog_bar=prog_bar)

    def forward(self, input, t, cond_signal=None, cond=None):
        """Make a forward pass through the network."""
        
        # TODO: maybe remove assertion to speed up training

        # input
        if self.cond_signal_input:
            assert cond_signal is not None
            assert torch.isfinite(cond_signal).all().item()
            input = torch.cat((cond_signal, input), dim=1)

        # predict
        if self.cond_input:
            assert torch.isfinite(cond).all().item()
            cond = cond 
        else:
            cond = None

        assert torch.isfinite(input).all().item()
        out = self.net(input, t, cond=cond)
        #assert torch.isfinite(out).all().item() 
        #print("Model outputs all 0s: ", torch.all(out == 0).item())
        return out

    def sample(self, shape, cond_signal=None, cond=None):
        """Sample from the diffusion model."""
        # initialize noise
        sample = torch.randn(shape, device=self.device)

        # sample iteratively
        for t in tqdm(self.noise_scheduler.timesteps):
            pred = self.forward(
                sample, t * torch.ones(shape[0], device=self.device), cond_signal, cond
            )
            sample = self.noise_scheduler.step(pred, t, sample).prev_sample

        return sample

    def evaluate(self, batch):
        """Evaluate diffusion model."""
        shape = batch["representation"].shape
        cond_signal = batch["cond_signal"] if self.cond_signal_input else None
        cond = batch["cond"] if self.cond_input else None
        return self.sample(shape, cond_signal, cond)

    def step(self, batch, train):
        signal_batch = batch["representation"]
        cond_signal_batch = batch["cond_signal"] if self.cond_signal_input else None
        cond_batch = batch["cond"] if self.cond_input else None

        # add noise
        noise = torch.randn(signal_batch.shape, device=signal_batch.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (signal_batch.shape[0],),
            device=signal_batch.device,
        ).long()
        noisy_signal = self.noise_scheduler.add_noise(signal_batch, noise, timesteps)

        # loss
        pred = self.forward(noisy_signal, timesteps, cond_signal_batch, cond_batch)
        target = noise if self.prediction_type == "epsilon" else signal_batch
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
