################################################################################
# This file is a adapted from https://github.com/openai/consistency_models
################################################################################

import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    GaussianFourierProjection,
    append_dims,
    avg_pool_nd,
    global_avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    zero_module,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.optimization import get_scheduler


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    :param output_size_list: list of output sizes for the upsampling operation at each upsampling level.             
    :param out_channels: if specified, the number of out channels.
    :param kernel_size: kernel size for the spatial convolutions.
    """

    def __init__(self, channels, use_conv, output_size_list, dims=2, out_channels=None, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, kernel_size, padding="same"
            )
        # Contains the output size of the upsampling operation, upddated at each downsampling level    
        self.output_size_list = output_size_list    
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        output_size = self.output_size_list.pop()
        if self.dims == 3:
            x = F.interpolate(
                x, (output_size[0], output_size[1] * 2, output_size[2] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, size=output_size, mode="nearest")
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


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param kernel_size: the size of the spatial convolution kernel.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        kernel_size=3,
        use_conv=False,
        use_scale_shift_norm=False, #TODO: what's that?
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding="same"),
        )
        if emb_channels > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding="same",
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding="same"
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x) 
        if emb is not None:
            emb_out = self.emb_layers(emb).type(h.dtype) 
            emb_out = append_dims(emb_out, h.dim()) 
        else:
            emb_out = 0    
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out 
            h = self.out_layers(h) 
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Args:
        channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        use_checkpoint (bool, optional): Whether to use checkpointing for memory optimization. Defaults to False.
        flash_attention (bool, optional): Whether to use QKVFlashAttention. Defaults to True.
        dims (int, optional): Number of dimensions for the input. Defaults to 2.
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
        #from flash_attn.flash_attention import FlashAttention # TODO: requires CUDA 11.6

        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        #self.inner_attn = FlashAttention(
        #    attention_dropout=attention_dropout, **factory_kwargs
        #)
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False.
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


class UNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_kernel_size: kernel size for the spatial convolutions.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_cond_features: if specified (as an int), then this model will use
        this number of features for conditioning.
        They will be embedded and added to timestep embeddings.
    :param cond_emb_scale: if specified (as a float), conditional inputs will be
        embedded with fourier embeddings of this scale.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param flash_attention: use the flash attention implementation.
    """
    @register_to_config
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions=(8, 16, 32),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        cond_features=None,
        cond_emb_scale=None,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
        flash_attention=True,
    ):
        super().__init__()

        self.dims = dims

        embed_dim = model_channels * 4
        self.time_embed = GaussianFourierProjection(model_channels)
        self.time_mlp = nn.Sequential(
            linear(model_channels, embed_dim), nn.SiLU(), linear(embed_dim, embed_dim)
        )

        self.cond_features = cond_features
        if cond_features is not None:
            if cond_emb_scale is not None:
                self.cond_embed = GaussianFourierProjection(
                    model_channels, cond_emb_scale
                )
                cond_features *= model_channels
            else:
                self.cond_embed = None

            self.cond_mlp = nn.Sequential(
                linear(cond_features, embed_dim),
                nn.SiLU(),
                linear(embed_dim, embed_dim),
            )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        self.upsample_output_sizes = []
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            flash_attention=flash_attention,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                embed_dim,
                dropout,
                kernel_size=conv_kernel_size,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                dims=dims,
                use_checkpoint=use_checkpoint,
                flash_attention=flash_attention,
            ),
            ResBlock(
                ch,
                embed_dim,
                dropout,
                kernel_size=conv_kernel_size,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            flash_attention=flash_attention,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(
                            ch,
                            conv_resample,
                            output_size_list=self.upsample_output_sizes,
                            dims=dims,
                            out_channels=out_ch,
                            kernel_size=conv_kernel_size,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(
                conv_nd(dims, input_ch, out_channels, conv_kernel_size, padding="same")
            ),
        )

    def forward(self, x, timesteps, cond=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond: an [N x cond_features] Tensor of conditioning features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (cond is not None) == (
            self.cond_features is not None
        ), "must specify cond if and only if the model is conditioned"

        hs = []
        emb = self.time_mlp(self.time_embed(timesteps))

        if self.cond_features is not None:
            if self.cond_embed is not None:
                cond = self.cond_embed(cond).view(cond.shape[0], -1)  
            emb += self.cond_mlp(cond) 

        h = x
        for module in self.input_blocks:
            if isinstance(module[-1], Downsample):
                self.upsample_output_sizes.append(h.shape[-self.dims:])
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h) 
    
    def get_signal_length(self, original_signal_length: int) -> int:
        """
        Calculates the adjusted signal length based on the maximum channel multiplier.

        Parameters:
            original_signal_length (int): The original length of the signal.

        Returns:
            int: The adjusted signal length.
        """
        max_ch_mult = max(self.channel_mult)
        return round(original_signal_length / max_ch_mult) * max_ch_mult



class HalfUNetClassifierModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels,
        model_channels,
        num_classes,
        num_res_blocks,
        attention_resolutions=(8, 16, 32),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_kernel_size=3,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
        flash_attention=True,
    ):  
        super().__init__()

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(dims, in_channels, ch, conv_kernel_size, padding="same")
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        emb_channels=0,
                        dropout=dropout,
                        out_channels=int(mult * model_channels),
                        kernel_size=conv_kernel_size,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            flash_attention=flash_attention,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.mlp = nn.Sequential(
                    normalization(ch),
                    nn.SiLU(),
                    global_avg_pool_nd(dims),
                    nn.Flatten(1),
                    linear(ch, num_classes)
                )


    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = x
        for module in self.input_blocks:
            h = module(h)
        return self.mlp(h)    
    
    def get_embeddings(self, x):
        """
        Get the embeddings from the model.

        :param x: an [N x C x ...] Tensor of inputs.
        :return: an [N x C x ...] Tensor of embeddings.
        """
        with th.no_grad():
            h = x
            for module in self.input_blocks:
                h = module(h)
            for block in self.mlp[:-1]:
                h = block(h)
            return h

    def get_predictions(self, x, from_embeddings):
        """
        Get the predictions from the model.

        :param x: an [N x C x ...] Tensor of inputs.
        :param from_embeddings: a bool determining if the predictions should be
            made from the embeddings.
        :return: an [N x C x ...] Tensor of predictions.
        """
        if from_embeddings:
            return self.mlp[-1](x)
        return self(x)      