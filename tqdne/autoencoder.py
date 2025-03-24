import pytorch_lightning as pl
import torch as th

from .blocks import Decoder, Encoder


class LightningAutoencoder(pl.LightningModule):
    """A PyTorch Lightning module for training an autoencoder.

    Parameters
    ----------
    encoder_config : dict
        The configuration for the encoder.
    decoder_config : dict
        The configuration for the decoder.
    optimizer_params : dict
        The parameters for the optimizer.
    kl_weight : float, optional
        The weight of the KL divergence term in the loss function.
    """

    def __init__(
        self,
        encoder_config: dict,
        decoder_config: dict,
        optimizer_params: dict,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.optimizer_params = optimizer_params
        self.kl_weight = kl_weight
        self.save_hyperparameters()

    def _encode(self, x):
        mean, log_std = th.chunk(self.encoder(x), 2, dim=1)
        latent = mean + th.randn_like(mean) * th.exp(log_std)
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
        return 0.5 * th.sum(mean**2 + th.exp(log_var) - log_var - 1, dim=1)

    def step(self, batch, stage="training"):
        x = batch["signal"]
        latent, mean, log_std = self._encode(x)
        x_recon = self.decode(latent)
        recon_loss = th.mean((x - x_recon) ** 2)
        kl_div = self.kl_divergence(mean, log_std).mean()
        loss = recon_loss + self.kl_weight * kl_div
        self.log(f"{stage}/reconstruction_loss", recon_loss.item(), sync_dist=True)
        self.log(f"{stage}/kl_divergence", kl_div.item(), sync_dist=True)
        self.log(f"{stage}/loss", loss.item(), sync_dist=True)

        if "cond_signal" not in batch:
            return loss

        # Conditional signal
        cond_x = batch["cond_signal"]
        cond_latent, cond_mean, cond_log_std = self._encode(cond_x)
        cond_x_recon = self.decode(cond_latent)
        cond_recon_loss = th.mean((cond_x - cond_x_recon) ** 2)
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
        optimizer = th.optim.Adam(self.parameters(), lr=self.optimizer_params["learning_rate"])
        lr_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.optimizer_params["max_steps"], eta_min=self.optimizer_params["eta_min"]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
