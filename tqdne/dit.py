"""Diffusion Transformer (DiT) implementation for seismic waveform generation.

This module implements the DiT architecture adapted for latent diffusion models,
using adaptive layer normalization (adaLN) for conditioning on both noise levels
and earthquake parameters.

References
----------
[1] Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
[2] Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., 2022)
"""

import math

import pytorch_lightning as pl
import torch as th
import torch.nn as nn

from tqdne.autoencoder import LightningAutoencoder
from tqdne.edm import EDM
from tqdne.nn import append_dims


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(half, dtype=th.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds conditional features (earthquake parameters) into vector representations."""

    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        """
        Parameters
        ----------
        cond : torch.Tensor
            Conditional features of shape (batch_size, num_features)
        """
        return self.mlp(cond)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding for latent representations."""

    def __init__(self, patch_size=2, in_channels=8, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input latent of shape (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Patches of shape (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape (B, N, hidden_size)
        c : torch.Tensor
            Conditioning embedding of shape (B, hidden_size)

        Returns
        -------
        torch.Tensor
            Output tokens of shape (B, N, hidden_size)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FinalLayer(nn.Module):
    """
    The final layer of DiT with adaLN conditioning.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def modulate(self, x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) for latent diffusion.

    Parameters
    ----------
    input_size : int
        Spatial size of input latent (assumes square, e.g., 32 for 32x32)
    patch_size : int
        Size of each patch (default: 2)
    in_channels : int
        Number of input channels (default: 8 for latent space)
    hidden_size : int
        Hidden size of transformer (default: 384 for DiT-S)
    depth : int
        Number of transformer blocks (default: 12 for DiT-S)
    num_heads : int
        Number of attention heads (default: 6 for DiT-S)
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim (default: 4.0)
    cond_features : int
        Number of conditional features (default: 5 for earthquake parameters)
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=8,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        cond_features=5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(cond_features, hidden_size)

        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(th.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize transformer weights following standard practices."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embedding with 2D sinusoidal embedding
        pos_embed = self.get_2d_sincos_pos_embed(self.hidden_size, int(self.pos_embed.shape[1] ** 0.5))
        self.pos_embed.data.copy_(th.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize label embedding MLP
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size):
        """Generate 2D sinusoidal positional embeddings."""
        import numpy as np

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = DiT.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        """Generate 2D sinusoidal positional embeddings from grid."""
        import numpy as np

        assert embed_dim % 2 == 0
        emb_h = DiT.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = DiT.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = np.concatenate([emb_h, emb_w], axis=1)
        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """Generate 1D sinusoidal positional embeddings."""
        import numpy as np

        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega

        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)

        emb_sin = np.sin(out)
        emb_cos = np.cos(out)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)
        return emb

    def unpatchify(self, x, h, w):
        """
        Reconstruct latent from patches.

        Parameters
        ----------
        x : torch.Tensor
            Patches of shape (B, num_patches, patch_size^2 * C)
        h : int
            Height in patches
        w : int
            Width in patches

        Returns
        -------
        torch.Tensor
            Latent of shape (B, C, H, W)
        """
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = th.einsum("nhwpqc->nchpwq", x)
        latent = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        return latent

    def forward(self, x, t, cond=None):
        """
        Forward pass through DiT.

        Parameters
        ----------
        x : torch.Tensor
            Noisy latent of shape (B, C, H, W)
        t : torch.Tensor
            Noise conditioning (preprocessed timesteps) of shape (B,)
        cond : torch.Tensor, optional
            Conditional features of shape (B, num_features)

        Returns
        -------
        torch.Tensor
            Predicted noise or denoised latent of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size

        # Embed patches
        x = self.x_embedder(x) + self.pos_embed  # (B, num_patches, hidden_size)

        # Embed timestep
        t_emb = self.t_embedder(t)  # (B, hidden_size)

        # Embed conditional features and combine with timestep
        if cond is not None:
            y_emb = self.y_embedder(cond)  # (B, hidden_size)
            c = t_emb + y_emb  # Combine timestep and condition embeddings
        else:
            c = t_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)  # (B, num_patches, patch_size^2 * out_channels)

        # Unpatchify
        x = self.unpatchify(x, h, w)  # (B, out_channels, H, W)

        return x


class LightningDiT(pl.LightningModule):
    """
    A PyTorch Lightning module for the DiT model with EDM noise scheduling.

    This module adapts the Diffusion Transformer (DiT) architecture to work with
    EDM-style noise scheduling and provides the same interface as LightningEDM
    for compatibility with the existing training infrastructure.

    Parameters
    ----------
    dit_config : dict
        Configuration for the DiT model
    optimizer_params : dict
        Parameters for the optimizer (learning_rate, max_steps, eta_min)
    num_sampling_steps : int, optional
        Number of sampling steps during inference (default: 25)
    deterministic_sampling : bool, optional
        If True, use deterministic sampling (default: True)
    edm : EDM, optional
        EDM model parameters for noise scheduling (default: EDM())
    autoencoder : LightningAutoencoder, optional
        Autoencoder for latent diffusion (default: None)
    """

    def __init__(
        self,
        dit_config: dict,
        optimizer_params: dict,
        num_sampling_steps: int = 25,
        deterministic_sampling: bool = True,
        edm: EDM = EDM(),
        autoencoder: None | LightningAutoencoder = None,
    ):
        super().__init__()

        self.dit = DiT(**dit_config)
        self.optimizer_params = optimizer_params
        self.num_sampling_steps = num_sampling_steps
        self.deterministic_sampling = deterministic_sampling
        self.edm = edm
        self.autoencoder = autoencoder.eval() if autoencoder else None
        self.config = dit_config

        if self.autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.save_hyperparameters(ignore=("autoencoder",))

    def forward(self, sample, sigma, cond_sample=None, cond=None):
        """
        Forward pass with EDM-style skip connection.

        Parameters
        ----------
        sample : torch.Tensor
            Clean or noisy latent
        sigma : torch.Tensor
            Noise level
        cond_sample : torch.Tensor, optional
            Conditional sample (not used in current implementation)
        cond : torch.Tensor, optional
            Conditional features

        Returns
        -------
        torch.Tensor
            Denoised prediction
        """
        dim = sample.dim()
        sample_in = sample * append_dims(self.edm.in_scaling(sigma), dim)

        # DiT doesn't support cond_sample concatenation in the same way as UNet
        # If needed, this can be added as an additional conditioning mechanism
        if cond_sample is not None:
            raise NotImplementedError("cond_sample concatenation not implemented for DiT")

        noise_cond = self.edm.noise_conditioning(sigma)
        out = self.dit(sample_in, noise_cond, cond=cond)
        skip = append_dims(self.edm.skip_scaling(sigma), dim) * sample
        return out * append_dims(self.edm.out_scaling(sigma), dim) + skip

    def step(self, batch, batch_idx):
        """A single training/validation step."""
        sample = batch["signal"]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None

        if self.autoencoder:
            sample = self.autoencoder.encode(sample)
            if cond_sample is not None:
                cond_sample = self.autoencoder.encode(cond_sample)

        eps = th.randn(sample.shape[0], device=self.device)
        sigma = self.edm.sigma(eps)
        noise = th.randn_like(sample) * append_dims(sigma, sample.dim())
        pred = self(sample + noise, sigma, cond_sample, cond)

        loss = (pred - sample) ** 2
        loss_weight = append_dims(self.edm.loss_weight(sigma), loss.dim())

        return th.mean(loss * loss_weight)

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
        """Sample using Heun's second order method (same as EDM)."""
        dtype = th.float32 if self.device.type == "mps" else th.float64

        if self.autoencoder:
            if cond_sample is not None:
                cond_sample = self.autoencoder.encode(cond_sample)

            # Infer latent shape
            dummy = th.zeros(shape, device=self.device)
            latent = self.autoencoder.encode(dummy)
            shape = latent.shape

        sigmas = self.edm.sampling_sigmas(self.num_sampling_steps, device=self.device)
        eps = th.randn(shape, device=self.device, dtype=dtype) * sigmas[0]

        if self.deterministic_sampling:
            sample = self.sample_deterministically(eps, sigmas, cond_sample, cond)
        else:
            sample = self.sample_stochastically(eps, sigmas, cond_sample, cond)

        sample = sample.to(th.float32)
        if self.autoencoder:
            return self.autoencoder.decode(sample)
        return sample

    def sample_deterministically(self, eps, sigmas, cond_sample=None, cond=None):
        """Deterministic sampling using Heun's method."""
        dtype = th.float32 if self.device.type == "mps" else th.float64
        sample_next = eps

        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next
            pred_curr = self(
                sample_curr.to(self.dtype),
                sigma.to(self.dtype).repeat(len(sample_curr)),
                cond_sample,
                cond,
            ).to(dtype)
            d_cur = (sample_curr - pred_curr) / sigma
            sample_next = sample_curr + d_cur * (sigma_next - sigma)

            # Second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self(
                    sample_next.to(self.dtype),
                    sigma_next.to(self.dtype).repeat(len(sample_curr)),
                    cond_sample,
                    cond,
                ).to(dtype)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_curr + (sigma_next - sigma) * (0.5 * d_cur + 0.5 * d_prime)

        return sample_next

    def sample_stochastically(self, eps, sigmas, cond_sample=None, cond=None):
        """Stochastic sampling using Heun's method with noise injection."""
        dtype = th.float32 if self.device.type == "mps" else th.float64
        sample_next = eps

        for i, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            sample_curr = sample_next

            # Increase noise temporarily
            sigma_hat = self.edm.sigma_hat(sigma, self.num_sampling_steps)
            noise = th.randn_like(sample_curr) * self.edm.S_noise
            sample_hat = sample_curr + noise * (sigma_hat**2 - sigma**2) ** 0.5

            # Euler step
            pred_hat = self(
                sample_hat.to(self.dtype),
                sigma_hat.to(self.dtype).repeat(len(sample_hat)),
                cond_sample,
                cond,
            ).to(dtype)
            d_cur = (sample_hat - pred_hat) / sigma_hat
            sample_next = sample_hat + d_cur * (sigma_next - sigma_hat)

            # Second order correction
            if i < self.num_sampling_steps - 1:
                pred_next = self(
                    sample_next.to(self.dtype),
                    sigma_next.to(self.dtype).repeat(len(sample_hat)),
                    cond_sample,
                    cond,
                ).to(dtype)
                d_prime = (sample_next - pred_next) / sigma_next
                sample_next = sample_hat + (sigma_next - sigma_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return sample_next

    @th.no_grad()
    def evaluate(self, batch):
        """Evaluate the model on a batch of data."""
        sample = batch["signal"]
        cond_sample = batch["cond_signal"] if "cond_signal" in batch else None
        cond = batch["cond"] if "cond" in batch else None
        return self.sample(sample.shape, cond_sample, cond)

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
