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

    time_eps: float = 1e-3  # Minimum time value for numerical stability
    time_max: float = 1.0   # Maximum time value

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
        # Linear scaling: maps time [0, 1] to [0, 999] for DiT timestep embedding
        return t * 999.0

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
            Time steps uniformly sampled from [time_eps, time_max]
        """
        t = th.rand(batch_size, device=device)
        return t * (self.time_max - self.time_eps) + self.time_eps

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
            Time values from time_eps to time_max
        """
        # Generate uniform time steps, then scale to [time_eps, time_max]
        t = th.linspace(0, 1, num_steps + 1, device=device)
        return t * (self.time_max - self.time_eps) + self.time_eps


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
        num_sampling_steps: int = 25,
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

        # Integrate ODE using Euler method
        x = self.sample_euler(x, times, cond)

        # Decode from latent space if using autoencoder
        x = x.to(th.float32)
        if self.autoencoder:
            return self.autoencoder.decode(x)
        return x

    def sample_euler(self, x, times, cond=None):
        """Euler method (1st order) for ODE integration.

        Implements simple Euler integration for the flow ODE.

        Parameters
        ----------
        x : torch.Tensor
            Initial state (noise)
        times : torch.Tensor
            Time schedule from time_eps to time_max
        cond : torch.Tensor, optional
            Conditional features

        Returns
        -------
        torch.Tensor
            Final state (data)
        """
        dtype = x.dtype
        dt = 1.0 / self.num_sampling_steps

        for i in range(self.num_sampling_steps):
            # Get time for this step
            t = times[i].repeat(len(x))

            # Predict velocity
            v = self(
                x.to(self.dtype),
                t.to(self.dtype),
                cond
            ).to(dtype)

            # Euler step: x = x + v * dt
            x = x + v * dt

        return x

    @th.no_grad()
    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["signal"]
        cond = batch["cond"] if "cond" in batch else None
        return self.sample(sample.shape, cond=cond)

    def configure_optimizers(self):
        optimizer = th.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_params["learning_rate"],
            betas=(self.optimizer_params.get("b1", 0.9), self.optimizer_params.get("b2", 0.999)),
            weight_decay=self.optimizer_params.get("weight_decay", 0.0),
        )

        # Custom LR scheduler with warmup and linear decay
        def lr_lambda(step):
            warmup_steps = self.optimizer_params.get("warmup_steps", 0)
            decay_steps = self.optimizer_params.get("decay_steps", self.optimizer_params["max_steps"])
            end_lr = self.optimizer_params.get("end_learning_rate", 0.0)
            start_lr = self.optimizer_params["learning_rate"]

            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Linear decay
                progress = (step - warmup_steps) / (decay_steps - warmup_steps)
                progress = min(progress, 1.0)
                return (1.0 - progress) + progress * (end_lr / start_lr)

        lr_scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

        # Add gradient clipping if specified
        if "gradient_clipping" in self.optimizer_params:
            config["gradient_clip_val"] = self.optimizer_params["gradient_clipping"]

        return config
