from functools import partial
from turtle import up
from typing import Callable, Union, Sequence

import torch
import torch.nn as nn
from pynop.core.ops import ConvLayer, MaxPoolConv, InterpolateConvUpSampleLayer
from pynop.core.utils import *
from pynop.core.norm import LayerNorm2d, RMSNorm2d


class unet(nn.Module):
    """UNET architecture"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: Union[Callable, nn.Module, Sequence],
        filters: Sequence = (32, 64, 128, 256),
        repeats: Sequence = (2, 2, 2, 2),
        downblock: Union[Callable, nn.Module] = partial(MaxPoolConv, factor=2, kernel_size=2),
        upblock: Union[Callable, nn.Module] = partial(InterpolateConvUpSampleLayer, factor=2, kernel_size=1),
        fusion="concat",
        stem_activation: Union[Callable, nn.Module] = nn.GELU,
        stem_norm: Union[Callable, nn.Module] = LayerNorm2d,
        stem_kernel_size=3,
        stem_stride=1,
    ):
        """
        UNET architecture.
        The UNET architecture is a convolutional neural network that consists of an encoder and a decoder.
        The encoder is a series of convolutional layers that downsample the input image,
        while the decoder is a series of convolutional layers that upsample the feature maps.
        The encoder and decoder are connected by skip connections, which concatenate the feature maps
        from the encoder to the decoder.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            block (Union[Callable, nn.Module, Sequence): Block to use for the encoder and decoder.
            If a list of blocks is provided, its dimensions must be equal to 2 * filters - 1.
            filters (Sequence): List of filter sizes for each stage.
            repeats (Sequence): List of number of repeats for each stage.
            downblock (Union[Callable, nn.Module]): Downsampling block.
            upblock (Union[Callable, nn.Module]): Upsampling block.
            fusion (str): Fusion method. Can be 'concat' or 'add'.
            stem_activation (Union[Callable, nn.Module]): Activation function for the stem layer.
            stem_norm (Union[Callable, nn.Module]): Normalization layer for the stem layer.
            stem_kernel_size (int): Kernel size for the stem layer.
            stem_stride (int): Stride for the stem layer.
        Returns:
            When calling the forward method, the output will be a tensor of shape (batch_size, out_channels, height, width).
            with height and width being the same as the input image.
        """
        super().__init__()

        assert len(repeats) == len(filters), "unet: filters must have the same length as repeats"
        self.filters = list(filters)
        self.repeats = list(repeats)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.downblocks = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.shortcut_filters = self.filters[:-1][::-1]
        self.shortcut_filters.append(in_channels)
        self.fusion = fusion

        # stem
        self.stem = ConvLayer(
            in_channels,
            filters[0],
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=stem_norm,
            activation=stem_activation,
        )
        # Encoder
        for i, n in enumerate(repeats):
            for j in range(n):
                in_ch = filters[max(i - 1, 0)] if j == 0 else filters[i]
                self.encoder.append(
                    block(
                        in_channels=in_ch,
                        out_channels=filters[i],
                        kernel_size=3,
                        stride=1,
                    )
                )
                # Downsample the last block of each stage, ecept for the last stage, which is the bottleneck
                if i < len(repeats) - 1 and j == n - 1:
                    self.downblocks.append(downblock(in_channels=filters[i], out_channels=filters[i]))
        # Decoder
        upconvs = self.repeats[:-1][::-1]
        upfilters = filters[:-1][::-1]

        for i, n in enumerate(upconvs):
            in_ch = filters[-1] if i == 0 else upfilters[i - 1]
            self.upblocks.append(upblock(in_ch, upfilters[i]))
            if fusion == "concat":
                in_ch = upfilters[i] + self.shortcut_filters[i]
            elif fusion == "add":
                in_ch = upfilters[i]
            else:
                raise ValueError(f"unet: fusion must be 'concat' or 'add', got {fusion}")
            self.proj.append(
                ConvLayer(
                    in_channels=in_ch,
                    out_channels=upfilters[i],
                    kernel_size=1,
                    stride=1,
                    use_bias=True,
                )
            )
            for j in range(n):

                self.decoder.append(
                    block(
                        in_channels=upfilters[i],
                        out_channels=upfilters[i],
                    )
                )
        # Project to output channels. If stem stride is 2 then we have to upscale the last block
        # to the original size of the input image
        self.final = nn.ModuleList()
        if stem_stride == 2:
            self.final.append(
                upblock(
                    in_channels=upfilters[i],
                    out_channels=upfilters[i],
                )
            )
        self.final.append(ConvLayer(upfilters[i], out_channels, kernel_size=1, stride=1))

    def forward(self, x):

        shortcuts = []
        counter = -1
        # print(self.shortcut_filters)
        x = self.stem(x)

        for i, n in enumerate(self.repeats):

            for j in range(n):
                counter += 1
                x = self.encoder[counter](x)

            if i < len(self.repeats) - 1 and j == n - 1:
                # shortcut before downsampling
                shortcuts.append(x)
                x = self.downblocks[i](x)

        upconvs = self.repeats[:-1][::-1]
        counter = -1
        shortcuts = shortcuts[::-1]

        for i, n in enumerate(upconvs):
            x = self.upblocks[i](x)
            if self.fusion == "concat":
                x = torch.cat([shortcuts[i], x], 1)
            elif self.fusion == "add":
                x = x + shortcuts[i]
            x = self.proj[i](x)
            for j in range(n):
                counter += 1
                x = self.decoder[counter](x)
        for op in self.final:
            x = op(x)

        return x
