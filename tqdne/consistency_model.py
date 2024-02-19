import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.seed import isolate_rng

from tqdne.nn import append_dims


class LithningConsistencyModel(pl.LightningModule):
    """A PyTorch Lightning module for training a consistency model.

    Implements Improved Techniques for Training Consistency Models (https://arxiv.org/abs/2310.14189).

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_max : float, default=80.0
        Maximum standard deviation of the noise.
    rho : float, default=7.0
        Schedule hyper-parameter.
    sigma_data : float, default=0.5
        Standard deviation of the data.
    initial_timesteps : int, default=10
        Schedule timesteps at the start of training.
    final_timesteps : int, default=1280
        Schedule timesteps at the end of training.
    lognormal_mean : float, default=-1.1
        Mean of the lognormal timestep distribution.
    lognormal_std : float, default=2.0
        Standard deviation of the lognormal timestep distribution.
    lr : float, default=1e-4
        Learning rate.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        sigma_data: float = 0.5,
        initial_timesteps: int = 10,
        final_timesteps: int = 1280,
        lognormal_mean: float = -1.1,
        lognormal_std: float = 2.0,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.initial_timesteps = initial_timesteps
        self.final_timesteps = final_timesteps
        self.lognormal_mean = lognormal_mean
        self.lognormal_std = lognormal_std
        self.lr = lr

    def forward(self, sample, sigma, low_res=None, cond=None):
        """Make a forward pass through the network with skip connection."""
        # concatenate optional low resolution input
        input = sample if low_res is None else torch.cat((sample, low_res), dim=1)

        # skip coefficients
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_skip = append_dims(c_skip, sample.dim())
        c_out = (self.sigma_data * (sigma - self.sigma_min)) / (
            self.sigma_data**2 + sigma**2
        ) ** 0.5
        c_out = append_dims(c_out, sample.dim())

        # forward pass
        out = self.net(input, sigma, cond)
        out = c_out * out + c_skip * sample
        return out

    @torch.no_grad()
    def sample(self, shape, sigmas=[1.0], low_res=None, cond=None):
        """Sample from the consistency model.

        Parameters
        ----------
        shape : tuple
            Shape of the sample.
        sigmas : list, optional
            List of standard deviations for optional refinements.
        low_res : torch.Tensor, optional
            Low resolution input.
        cond : torch.Tensor, optional
            Conditional input.

        """
        epsilon = torch.randn(shape, device=self.device)
        ones = torch.ones(shape[0], device=self.device)
        sample = self(epsilon, ones * self.sigma_max, low_res, cond)

        # optional refinements
        for sigma in sigmas:
            sample += torch.rand_like(sample) * sigma
            sample = self(sample, ones * sigma, low_res, cond)

        return sample

    def evaluate(self, batch, sigmas=[1]):
        """Evaluate the model on a batch of data."""
        sample = batch["high_res"]
        low_res = batch["low_res"] if "low_res" in batch else None
        cond = batch["cond"] if "cond" in batch else None
        # return self.sample(sample.shape, sigmas, low_res, cond) TODO: change interface
        return {"high_res": self.sample(sample.shape, sigmas, low_res, cond)}

    def step(self, batch):
        """A single step of training or validation."""
        sample = batch["high_res"]
        low_res = batch["low_res"] if "low_res" in batch else None
        cond = batch["cond"] if "cond" in batch else None

        # calculate timesteps
        total_training_steps_prime = np.floor(
            self.trainer.max_steps
            / (np.log2(np.floor(self.final_timesteps / self.initial_timesteps)) + 1)
        )
        num_timesteps = self.initial_timesteps * 2 ** np.floor(
            self.global_step / total_training_steps_prime
        )
        num_timesteps = min(num_timesteps, self.final_timesteps) + 1

        # calculate sigmas
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_timesteps, device=self.device) / (num_timesteps - 1)
        sigmas = self.sigma_min**rho_inv + steps * (
            self.sigma_max**rho_inv - self.sigma_min**rho_inv
        )
        sigmas = sigmas**self.rho

        # sample timesteps from lognormal distribution
        pdf = torch.erf(
            (torch.log(sigmas[1:]) - self.lognormal_mean)
            / (self.lognormal_std * np.sqrt(2))
        ) - torch.erf(
            (torch.log(sigmas[:-1]) - self.lognormal_mean)
            / (self.lognormal_std * np.sqrt(2))
        )
        pdf = pdf / pdf.sum()
        timesteps = torch.multinomial(pdf, sample.shape[0], replacement=True)

        # sample noise
        epsilon = torch.randn_like(sample)

        # obtain target with teacher
        teacher_sigma = sigmas[timesteps]
        teacher_sample = sample + epsilon * append_dims(teacher_sigma, sample.dim())
        with torch.no_grad():
            with isolate_rng():
                # teacher and student using the same random seed (for dropout)
                target = self(teacher_sample, teacher_sigma, low_res, cond)

        # make prediction with student
        student_sigma = sigmas[timesteps + 1]
        student_sample = sample + epsilon * append_dims(student_sigma, sample.dim())
        prediction = self(student_sample, student_sigma, low_res, cond)

        # compute pseudo huber loss
        sample_dim = np.prod(sample.shape[2:])  # assume (N, C, ...)
        c = 0.00054 * np.sqrt(sample_dim)  # heuristic proposed in the paper
        loss = torch.sqrt((prediction - target) ** 2 + c**2) - c

        # weight loss
        weights = (1 / (sigmas[1:] - sigmas[:-1]))[timesteps]
        loss = loss * append_dims(weights, loss.dim())

        return loss.mean()

    def training_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.step(batch)
        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.net.parameters(), lr=self.lr)
        return opt
