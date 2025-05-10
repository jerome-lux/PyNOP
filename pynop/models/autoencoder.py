import torch.nn as nn
import typing


from pynop.core.ops import ConvLayer
from pynop.core.utils import *


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels,
        latent_dim,
        block: nn.Module,
        downblock=None,
        stem_kernel_size=3,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_activation="silu",
        stem_norm="bn",
    ):
        """in_channels: number of channels of the input image
        latent_dim: number of channel of the latent representation
        block: basic block used to construct the network
        downblock: if None, the downsampling is done in the first block of each stage. If a nn.Module is given, then it is responsible for the downsampling


        """
        super().__init__()

        self.op_list = nn.ModuleList()
        self.stem_conv = ConvLayer(
            in_channels, dims[0], kernel_size=stem_kernel_size, norm=stem_norm, activation=stem_activation
        )

        for stage, repeats in enumerate(depths):

            if downblock is not None:
                input_channels = dims[stage] if stage == 0 else dims[stage - 1]
                self.op_list.append(downblock(in_channels=input_channels, out_channels=dims[stage]))
                input_channels = dims[stage]
            else:
                input_channels = dims[stage] if stage == 0 else dims[stage - 1]

            for i in range(repeats):
                stride = 2 if downblock is None and i == 0 else 1
                self.op_list.append(block(in_channels=input_channels, out_channels=dims[stage], stride=stride))

        # Project to latent space
        self.bottleneck = nn.Conv2d(dims[-1], latent_dim, 1, bias=True)

    def forward(self, x):

        x = self.stem_conv(x)
        for op in self.op_list:
            x = op(x)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels,
        latent_dim,
        block: nn.Module,
        upblock: nn.Module,
        depths=[3, 6, 3, 3],
        dims=[1024, 512, 256, 1024],
    ):
        """upblock: it must be a nn.Module responsible for the upsampling AND the projection to the stage's number of channels"""

        super().__init__()

        self.op_list = nn.ModuleList()
        for stage, repeats in enumerate(depths):

            input_channels = latent_dim if stage == 0 else dims[stage - 1]
            self.op_list.append(upblock(in_channels=input_channels, out_channels=dims[stage]))

            for i in range(repeats):
                self.op_list.append(block(in_channels=dims[stage], out_channels=dims[stage]))

        # Project back to in_channels
        self.projet = ConvLayer(dims[-1], out_channels, 1, use_bias=True, activation=None, norm=None)

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        x = self.projet(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
