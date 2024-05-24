"""
Various neural network blocks.
"""

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import avg_pool_nd, checkpoint, conv_nd, normalization, zero_module


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier projection layer."""

    def __init__(self, channels: int, scale: float = 0.02) -> None:
        super().__init__()
        self.W = nn.Parameter(th.randn(channels // 2) * scale, requires_grad=False)

    def forward(self, x):
        h = x[:, None] * self.W[None, :] * 2 * th.pi

        h = th.cat([th.sin(h), th.cos(h)], dim=-1)
        return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    :param out_channels: if specified, the number of out channels.
    :param kernel_size: kernel size for the spatial convolutions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, kernel_size, padding="same")

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    :param out_channels: if specified, the number of out channels.
    :param kernel_size: kernel size for the spatial convolutions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        use_checkpoint=False,
        flash_attention=True,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        if flash_attention:
            self.attention = QKVFlashAttention(channels, self.num_heads)
        else:
            self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        h = self.attention(qkv)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)


class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        from einops import rearrange
        from flash_attn.flash_attention import FlashAttention

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads)
        qkv, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


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
