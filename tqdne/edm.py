import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from tqdne.autoencoder import LithningAutoencoder
from tqdne.nn import append_dims


class EDM:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 0.5
    P_mean: float = -1.2
    P_std: float = 1.2
    S_churn: float = 40
    S_min: float = 0.05
    S_max: float = 50
    S_noise: float = 1.003

    def sigma(self, eps):
        return (eps * self.P_std + self.P_mean).exp()

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def skip_scaling(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def out_scaling(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5

    def in_scaling(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def noise_conditioning(self, sigma):
        return 0.25 * sigma.log()

    def sampling_sigmas(self, num_steps, device=None):
        rho_inv = 1 / self.rho
        step_idxs = torch.arange(num_steps, dtype=torch.float64, device=device)
        sigmas = (
            self.sigma_max**rho_inv
            + step_idxs / (num_steps - 1) * (self.sigma_min**rho_inv - self.sigma_max**rho_inv)
        ) ** self.rho
        return torch.cat([sigmas, torch.zeros_like(sigmas[:1])])  # add sigma=0

    def sigma_hat(self, sigma, num_steps):
        gamma = (
            min(self.S_churn / num_steps, 2**0.5 - 1) if self.S_min <= sigma <= self.S_max else 0
        )
        return sigma + gamma * sigma


class LightningEDM(pl.LightningModule):
    """A PyTorch Lightning module of the EDM model [1].

    Parameters
    ----------
    net : torch.nn.Module
        A PyTorch neural network.
    optimizer_params : dict
        A dictionary of parameters for the optimizer.
    cond_signal_input : bool, optional
        Whether low resolution input is provided.
    cond_input : bool, optional
        Whether conditional input is provided.
    autoencoder : None | LithningAutoencoder, optional
        If provided, the autoencoder used to obtain the latent representations.
        The diffusion model will then generate these latent representations instead of the original signal [2].

    References
    ----------
    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    [2] High-Resolution Image Synthesis with Latent Diffusion Models
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer_params: dict,
        prediction_type: str = "epsilon",
        cond_signal_input: bool = False,
        cond_input: bool = False,
        num_sampling_steps: int = 256,
        edm: EDM = EDM(),
        autoencoder: None | LithningAutoencoder = None,
    ):
        super().__init__()

        self.net = net
        self.optimizer_params = optimizer_params
        self.prediction_type = prediction_type
        self.cond_signal_input = cond_signal_input
        self.cond_input = cond_input
        self.num_sampling_steps = num_sampling_steps
        self.edm = edm
        self._autoencoder = autoencoder  # use underscore to not save in checkpoints
        if self._autoencoder:
            for param in self._autoencoder.parameters():
                param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, sample, sigma, cond_sample=None, cond=None):
        """Make a forward pass through the network with skip connection."""
        dim = sample.dim()
        sample_in = sample * append_dims(self.edm.in_scaling(sigma), dim)
        input = sample_in if cond_sample is None else torch.cat((sample_in, cond_sample), dim=1)
        noise_cond = self.edm.noise_conditioning(sigma)
        out = self.net(input, noise_cond, cond=cond)
        skip = append_dims(self.edm.skip_scaling(sigma), dim) * sample
        return out * append_dims(self.edm.out_scaling(sigma), dim) + skip

    def step(self, batch, batch_idx):
        """A single step in the training loop."""
        sample = batch["signal"]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None

        if self._autoencoder:
            sample = self._autoencoder.encode(sample)
            if cond_sample:
                cond_sample = self._autoencoder.encode(cond_sample)

        eps = torch.randn(sample.shape[0], device=self.device)
        sigma = self.edm.sigma(eps)
        noise = torch.randn_like(sample) * append_dims(sigma, sample.dim())
        pred = self(sample + noise, sigma, cond_sample, cond)

        loss = (pred - sample) ** 2
        loss_weight = append_dims(self.edm.loss_weight(sigma), loss.dim())

        return (loss * loss_weight).mean()

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("training/loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("validation/loss", loss.item())
        return loss

    def sample(self, shape, cond_sample=None, cond=None):
        """Sample using Heun's second order method."""
        if self._autoencoder:
            if cond_sample:
                cond_sample = self._autoencoder.encode(cond_sample)

            # infer latent shape
            dummy = torch.zeros(shape, device=self.device)
            latent = self._autoencoder.encode(dummy)
            shape = latent.shape

        sigmas = self.edm.sampling_sigmas(self.num_sampling_steps, device=self.device)

        sample_next = torch.randn(shape, device=self.device, dtype=torch.float64) * sigmas[0]
        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next

            # increase noise temporarily
            sigma_hat = self.edm.sigma_hat(sigma, self.num_sampling_steps)
            noise = torch.randn(shape, device=self.device, dtype=torch.float64) * self.edm.S_noise
            sample_hat = sample_curr + noise * (sigma_hat**2 - sigma**2) ** 0.5

            # euler step
            pred_hat = self(
                sample_hat.to(self.dtype),
                sigma.to(self.dtype).repeat(len(sample_hat)),
                cond_sample,
                cond,
            ).to(torch.float64)
            d_cur = (sample_curr - pred_hat) / sigma_hat
            x_next = sample_curr + d_cur * (sigma_next - sigma_hat)

            # second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self(
                    x_next.to(self.dtype),
                    sigma_next.to(self.dtype).repeat(len(sample_hat)),
                    cond_sample,
                    cond,
                ).to(torch.float64)
                d_next = (x_next - pred_next) / sigma_next
                sample_next = sample_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_next)

        sample = sample_next.to(self.dtype)
        if self._autoencoder:
            return self._autoencoder.decode(sample)
        return sample

    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["signal"]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None
        return self.sample(sample.shape, cond_sample, cond)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_params)
        return optimizer
