"""
UNet model with attention and timestep embeddings.
Implementation adapted from https://github.com/openai/consistency_models.
"""

from abc import abstractmethod

import torch as th
from torch import nn

from .blocks import AttentionBlock, Downsample, GaussianFourierProjection, Upsample
from .nn import append_dims, checkpoint, conv_nd, normalization, zero_module


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


class ResBlock(TimestepBlock):
    """A residual block that can optionally change the number of channels.

    Parameters
    ----------
    channels : int
        The number of input channels.
    emb_channels : int
        The number of timestep embedding channels.
    dropout : float
        The rate of dropout.
    out_channels : int, optional
        If specified, the number of output channels. Default is None.
    kernel_size : int, optional
        The size of the spatial convolution kernel. Default is 3.
    use_conv : bool, optional
        If True and out_channels is specified, use a spatial convolution instead
        of a smaller 1x1 convolution to change the channels in the skip connection.
        Default is False.
    dims : int, optional
        Determines if the signal is 1D, 2D, or 3D. Default is 2.
    use_checkpoint : bool, optional
        If True, use gradient checkpointing on this module. Default is False.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        kernel_size=3,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, kernel_size, padding="same"),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, out_channels, out_channels, kernel_size, padding="same")),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, out_channels, kernel_size, padding="same"
            )
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        Parameters
        ----------
        x : ndarray
            An [N x C x ...] array of features.
        emb : ndarray
            An [N x emb_channels] array of timestep embeddings.

        Returns
        -------
        ndarray
            An [N x C x ...] array of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = append_dims(emb_out, h.dim())
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    Parameters
    ----------
    in_channels : int
        Channels in the input Tensor.
    model_channels : int
        Base channel count for the model.
    out_channels : int
        Channels in the output Tensor.
    num_res_blocks : int
        Number of residual blocks per downsample.
    attention_resolutions : collection
        A collection of downsample rates at which attention will take place.
        May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention will be used.
    dropout : float, optional
        The dropout probability.
    channel_mult : tuple, optional
        Channel multiplier for each level of the UNet.
    conv_kernel_size : int, optional
        Kernel size for the spatial convolutions.
    conv_resample : bool, optional
        If True, use learned convolutions for upsampling and downsampling.
    dims : int, optional
        Determines if the signal is 1D, 2D, or 3D.
    cond_features : int, optional
        If specified, this model will use this number of features for conditioning.
        They will be embedded and added to timestep embeddings.
    cond_emb_scale : float, optional
        If specified, conditional inputs will be embedded with Fourier embeddings of this scale.
    use_checkpoint : bool, optional
        Use gradient checkpointing to reduce memory usage.
    num_heads : int, optional
        The number of attention heads in each attention layer.
    use_scale_shift_norm : bool, optional
        Use a FiLM-like conditioning mechanism.
    flash_attention : bool, optional
        Use the flash attention implementation.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions=(8, 16, 32),
        dropout=0,
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
        use_causal_mask=False,
    ):
        super().__init__()
        embed_dim = model_channels * 4
        self.time_embed = GaussianFourierProjection(model_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        self.cond_features = cond_features
        if cond_features is not None:
            if cond_emb_scale is not None:
                self.cond_embed = GaussianFourierProjection(model_channels, cond_emb_scale)
                cond_features *= model_channels
            else:
                self.cond_embed = None

            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_features, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
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
                            use_causal_mask=use_causal_mask,
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
                use_causal_mask=use_causal_mask,
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
                            use_causal_mask=use_causal_mask,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(
                            ch,
                            conv_resample,
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
            zero_module(conv_nd(dims, input_ch, out_channels, conv_kernel_size, padding="same")),
        )

    def forward(self, x, timesteps, cond=None):
        """
        Apply the model to an input batch.

        Parameters
        ----------
        x : torch.Tensor
            An [N x C x ...] Tensor of inputs.
        timesteps : torch.Tensor
            A 1-D batch of timesteps.
        cond : torch.Tensor, optional
            An [N x cond_features] Tensor of conditioning features.

        Returns
        -------
        torch.Tensor
            An [N x C x ...] Tensor of outputs.
        """
        assert (cond is not None) == (self.cond_features is not None), (
            "must specify cond if and only if the model is conditioned"
        )

        hs = []
        emb = self.time_mlp(self.time_embed(timesteps))

        if self.cond_features is not None:
            if self.cond_embed is not None:
                cond = self.cond_embed(cond).view(cond.shape[0], -1)
            emb += self.cond_mlp(cond)

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
