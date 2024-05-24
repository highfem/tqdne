import pytorch_lightning as pl
import torch

from .blocks import Decoder, Encoder


class LithningAutoencoder(pl.LightningModule):
    """A PyTorch Lightning module for training an autoencoder.

    Parameters
    ----------
    encoder : Encoder
        The encoder module.
        It should output a tensor with twice the number of latent channels.
    decoder : Decoder
        The decoder module.
    kl_weight : float, optional
        The weight of the KL divergence term in the loss function.
    lr : float, optional
        The learning rate for the optimizer.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, kl_weight: float = 1e-6, lr=1e-5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        self.lr = lr
        self.save_hyperparameters()

    def _encode(self, x):
        mean, log_std = torch.chunk(self.encoder(x), 2, dim=1)
        latent = mean + torch.randn_like(mean) * torch.exp(log_std)
        return latent, mean, log_std

    def encode(self, x):
        return self._encode(x)[0]

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self._encode(x)[0])

    def evaluate(self, batch):
        return self(batch["signal"])

    def kl_divergence(self, mean, log_std):
        """Computes the KL divergence between the latent distribution and an isotropic Gaussian distribution."""
        log_var = 2 * log_std
        return 0.5 * torch.sum(mean**2 + torch.exp(log_var) - log_var - 1, dim=1)

    def step(self, batch, stage="training"):
        x = batch["signal"]
        latent, mean, log_std = self._encode(x)
        x_recon = self.decode(latent)
        recon_loss = torch.mean((x - x_recon) ** 2)
        kl_div = self.kl_divergence(mean, log_std).mean()
        loss = recon_loss + self.kl_weight * kl_div
        self.log(f"{stage}/reconstruction_loss", recon_loss.item())
        self.log(f"{stage}/kl_divergence", kl_div.item())
        self.log(f"{stage}/loss", loss.item())

        if "cond_signal" not in batch:
            return loss

        # Conditional signal
        cond_x = batch["cond_signal"]
        cond_latent, cond_mean, cond_log_std = self._encode(cond_x)
        cond_x_recon = self.decode(cond_latent)
        cond_recon_loss = torch.mean((cond_x - cond_x_recon) ** 2)
        cond_kl_div = self.kl_divergence(cond_mean, cond_log_std).mean()
        cond_loss = cond_recon_loss + self.kl_weight * cond_kl_div
        self.log(f"{stage}/cond_reconstruction_loss", cond_recon_loss.item())
        self.log(f"{stage}/cond_kl_divergence", cond_kl_div.item())
        self.log(f"{stage}/cond_loss", cond_loss.item())
        return loss + cond_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="training")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="validation")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
