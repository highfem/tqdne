import pytorch_lightning as pl
import torch
from torch import nn

from .nn import conv_nd, normalization, zero_module
from .unet import AttentionBlock, Downsample, Upsample


class ResBlock(nn.Module):
    """
    A residual block similar to the one used in the UNet but without conditional embeddings and dropout.
    """

    def __init__(self, channels, out_channels=None, kernel_size=3, dims=2):
        super().__init__()
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, kernel_size, padding="same"),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, out_channels, out_channels, kernel_size, padding="same")),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions=[],
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        num_heads=1,
        flash_attention=True,
    ):
        super().__init__()
        ch = int(channel_mult[0] * model_channels)
        self.input_layer = conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")

        ds = 1
        down_blocks = []
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                down_blocks.append(
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                    )
                )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    down_blocks.append(
                        AttentionBlock(
                            ch, num_heads=num_heads, dims=dims, flash_attention=flash_attention
                        )
                    )
            if level != len(channel_mult) - 1:
                out_ch = ch
                down_blocks.append(Downsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                ch = out_ch
                ds *= 2
        self.down_blocks = nn.Sequential(*down_blocks)

        self.middle_blocks = nn.Sequential(
            ResBlock(ch, kernel_size=conv_kernel_size, dims=dims),
            AttentionBlock(ch, num_heads=num_heads, dims=dims, flash_attention=flash_attention),
            ResBlock(ch, kernel_size=conv_kernel_size, dims=dims),
        )

        self.output_layer = conv_nd(dims, ch, out_channels, conv_kernel_size, padding="same")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.down_blocks(x)
        x = self.middle_blocks(x)
        x = self.output_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions=[],
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        num_heads=1,
        flash_attention=True,
    ):
        super().__init__()
        ch = int(channel_mult[-1] * model_channels)
        self.input_layer = conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")

        self.middle_blocks = nn.Sequential(
            ResBlock(ch, kernel_size=conv_kernel_size, dims=dims),
            AttentionBlock(ch, num_heads=num_heads, dims=dims, flash_attention=flash_attention),
            ResBlock(ch, kernel_size=conv_kernel_size, dims=dims),
        )

        ds = 2 ** (len(channel_mult) - 1)
        up_blocks = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            if level != len(channel_mult) - 1:
                out_ch = ch
                up_blocks.append(Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                ch = out_ch
                ds //= 2
            for _ in range(num_res_blocks):
                up_blocks.append(
                    ResBlock(
                        ch,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                    )
                )
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    up_blocks.append(
                        AttentionBlock(
                            ch, num_heads=num_heads, dims=dims, flash_attention=flash_attention
                        )
                    )
        self.up_blocks = nn.Sequential(*up_blocks)

        self.output_layer = conv_nd(dims, ch, out_channels, conv_kernel_size, padding="same")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.middle_blocks(x)
        x = self.up_blocks(x)
        x = self.output_layer(x)
        return x


class LithningAutoencoder(pl.LightningModule):
    def __init__(
        self,
        kl_weight,
        in_channels,
        latent_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions=[],
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        num_heads=1,
        flash_attention=True,
        lr=1e-5,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.lr = lr
        self.save_hyperparameters()

        self.encoder = Encoder(
            in_channels,
            model_channels,
            latent_channels * 2,
            num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            conv_kernel_size=conv_kernel_size,
            conv_resample=conv_resample,
            dims=dims,
            num_heads=num_heads,
            flash_attention=flash_attention,
        )
        self.decoder = Decoder(
            latent_channels,
            model_channels,
            in_channels,
            num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            conv_kernel_size=conv_kernel_size,
            conv_resample=conv_resample,
            dims=dims,
            num_heads=num_heads,
            flash_attention=flash_attention,
        )

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
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="training")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="validation")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
