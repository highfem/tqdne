"""Flow Matching implementation for seismic waveform generation.

This module implements Conditional Flow Matching (CFM) adapted for latent diffusion models,
using the DiT architecture as the velocity field network.

References
----------
[1] Flow Matching for Generative Modeling (Lipman et al., 2023)
[2] Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
"""

import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from tqdne.autoencoder import LightningAutoencoder
from tqdne.dit import DiT


class FlowMatching:
    """Flow Matching configuration and utilities.

    This class implements Conditional Flow Matching (CFM) which learns
    continuous normalizing flows via optimal transport paths.

    The flow is defined as: x_t = t * x_1 + (1 - t) * x_0
    where x_0 ~ N(0, I) (noise) and x_1 is the data.

    The model learns the velocity field v_θ(x_t, t) to predict (x_1 - x_0).
    """

    sigma_min: float = 1e-4  # Minimum noise level for numerical stability

    def time_embedding(self, t):
        """Convert time t ∈ [0, 1] to model input.

        Parameters
        ----------
        t : torch.Tensor
            Time values in [0, 1]

        Returns
        -------
        torch.Tensor
            Time embeddings for the model
        """
        # Scale to similar range as EDM noise conditioning
        # EDM uses 0.25 * log(sigma), so we map [0,1] to a similar range
        # t=0 (noise) -> large negative, t=1 (data) -> ~0
        return th.log(t + self.sigma_min)

    def sample_time(self, batch_size, device):
        """Sample random time steps for training.

        Parameters
        ----------
        batch_size : int
            Number of time steps to sample
        device : torch.device
            Device to create tensor on

        Returns
        -------
        torch.Tensor
            Time steps uniformly sampled from [0, 1]
        """
        return th.rand(batch_size, device=device)

    def interpolate(self, x0, x1, t):
        """Linear interpolation between noise and data.

        Parameters
        ----------
        x0 : torch.Tensor
            Noise samples (source)
        x1 : torch.Tensor
            Data samples (target)
        t : torch.Tensor
            Time values in [0, 1] of shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Interpolated samples x_t = t * x_1 + (1 - t) * x_0
        """
        t = t.view(-1, *([1] * (x1.dim() - 1)))  # Reshape for broadcasting
        return t * x1 + (1 - t) * x0

    def target_velocity(self, x0, x1):
        """Compute the target velocity field (x_1 - x_0).

        For linear interpolation, the velocity is constant and equals x_1 - x_0.

        Parameters
        ----------
        x0 : torch.Tensor
            Noise samples
        x1 : torch.Tensor
            Data samples

        Returns
        -------
        torch.Tensor
            Target velocity field
        """
        return x1 - x0

    def sampling_schedule(self, num_steps, device=None):
        """Generate time schedule for ODE sampling.

        Parameters
        ----------
        num_steps : int
            Number of integration steps
        device : torch.device, optional
            Device to create tensor on

        Returns
        -------
        torch.Tensor
            Time values from 0 to 1
        """
        return th.linspace(0, 1, num_steps + 1, device=device)


class LightningFlowMatching(pl.LightningModule):
    """A PyTorch Lightning module for Flow Matching with DiT.

    This module implements Conditional Flow Matching (CFM) training and sampling
    using the DiT architecture to learn the velocity field of continuous normalizing flows.

    Parameters
    ----------
    dit_config : dict
        Configuration for the DiT model
    optimizer_params : dict
        Parameters for the optimizer (learning_rate, max_steps, eta_min)
    num_sampling_steps : int, optional
        Number of ODE integration steps during sampling (default: 50)
    flow_matching : FlowMatching, optional
        Flow matching configuration (default: FlowMatching())
    autoencoder : LightningAutoencoder, optional
        Autoencoder for latent diffusion (default: None)
    """

    def __init__(
        self,
        dit_config: dict,
        optimizer_params: dict,
        num_sampling_steps: int = 50,
        flow_matching: FlowMatching = FlowMatching(),
        autoencoder: None | LightningAutoencoder = None,
    ):
        super().__init__()

        self.dit = DiT(**dit_config)
        self.optimizer_params = optimizer_params
        self.num_sampling_steps = num_sampling_steps
        self.flow_matching = flow_matching
        self.autoencoder = autoencoder.eval() if autoencoder else None
        self.config = dit_config

        if self.autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.save_hyperparameters(ignore=("autoencoder",))

    def forward(self, x_t, t, cond=None):
        """Forward pass through the velocity field network.

        Parameters
        ----------
        x_t : torch.Tensor
            Interpolated sample at time t
        t : torch.Tensor
            Time values in [0, 1] of shape (batch_size,)
        cond : torch.Tensor, optional
            Conditional features

        Returns
        -------
        torch.Tensor
            Predicted velocity field
        """
        # Convert time to model conditioning
        t_emb = self.flow_matching.time_embedding(t)

        # DiT expects time conditioning similar to noise level in EDM
        # We reuse the same interface
        return self.dit(x_t, t_emb, cond=cond)

    def step(self, batch, batch_idx):
        """A single training/validation step.

        Implements the CFM training objective:
        L = E_{t, x_0, x_1} [||v_θ(x_t, t) - (x_1 - x_0)||^2]
        """
        x1 = batch["signal"]  # Data (target)
        cond = batch["cond"] if "cond" in batch else None

        # Encode to latent space if using autoencoder
        if self.autoencoder:
            x1 = self.autoencoder.encode(x1)

        # Sample noise (source)
        x0 = th.randn_like(x1)

        # Sample random time steps
        t = self.flow_matching.sample_time(x1.shape[0], device=self.device)

        # Interpolate between noise and data
        x_t = self.flow_matching.interpolate(x0, x1, t)

        # Compute target velocity
        target = self.flow_matching.target_velocity(x0, x1)

        # Predict velocity
        pred = self(x_t, t, cond)

        # Flow matching loss (simple MSE)
        loss = th.mean((pred - target) ** 2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("training/loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("validation/loss", loss.item(), sync_dist=True)
        return loss

    @th.no_grad()
    def sample(self, shape, cond_sample=None, cond=None):
        """Sample by integrating the ODE dx/dt = v_θ(x, t) from t=0 to t=1.

        Uses Heun's method (2nd order Runge-Kutta) for ODE integration.

        Parameters
        ----------
        shape : tuple
            Shape of samples to generate
        cond_sample : torch.Tensor, optional
            Conditional sample (not used in current implementation)
        cond : torch.Tensor, optional
            Conditional features

        Returns
        -------
        torch.Tensor
            Generated samples
        """
        dtype = th.float32 if self.device.type == "mps" else th.float64

        if self.autoencoder:
            # Infer latent shape
            dummy = th.zeros(shape, device=self.device)
            latent = self.autoencoder.encode(dummy)
            shape = latent.shape

        # Start from noise at t=0
        x = th.randn(shape, device=self.device, dtype=dtype)

        # Get time schedule
        times = self.flow_matching.sampling_schedule(self.num_sampling_steps, device=self.device)

        # Integrate ODE using Heun's method
        x = self.sample_heun(x, times, cond)

        # Decode from latent space if using autoencoder
        x = x.to(th.float32)
        if self.autoencoder:
            return self.autoencoder.decode(x)
        return x

    def sample_heun(self, x, times, cond=None):
        """Heun's method (2nd order Runge-Kutta) for ODE integration.

        This is a 2nd order ODE solver that provides good accuracy with
        reasonable computational cost.

        Parameters
        ----------
        x : torch.Tensor
            Initial state (noise)
        times : torch.Tensor
            Time schedule from 0 to 1
        cond : torch.Tensor, optional
            Conditional features

        Returns
        -------
        torch.Tensor
            Final state (data)
        """
        dtype = x.dtype

        for i, (t_curr, t_next) in enumerate(zip(times[:-1], times[1:])):
            dt = t_next - t_curr

            # First evaluation (Euler step)
            t_curr_batch = t_curr.repeat(len(x))
            v_curr = self(
                x.to(self.dtype),
                t_curr_batch.to(self.dtype),
                cond
            ).to(dtype)

            # Euler predictor
            x_next = x + dt * v_curr

            # Second evaluation (correction)
            if i < len(times) - 2:  # Skip correction on last step
                t_next_batch = t_next.repeat(len(x))
                v_next = self(
                    x_next.to(self.dtype),
                    t_next_batch.to(self.dtype),
                    cond
                ).to(dtype)

                # Heun's method: average of two slopes
                x = x + dt * (0.5 * v_curr + 0.5 * v_next)
            else:
                x = x_next

        return x

    @th.no_grad()
    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["signal"]
        cond = batch["cond"] if "cond" in batch else None
        return self.sample(sample.shape, cond=cond)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.optimizer_params["learning_rate"])
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_params["max_steps"],
            eta_min=self.optimizer_params["eta_min"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
