from typing import Sequence, Union
import numpy as np
import collections.abc as abc
import torch
import torch.nn as nn
from pynop.core.blocks import FNOBlock, FNOBlockv2, UFNOBlock, ConvFNOBlock
from pynop.core.ops import CartesianEmbedding
from pynop.core.ops import ConvLayer
from pynop.core.norm import LayerNorm2d
from pynop.core.utils import make_tuple

REGISTERED_FNO = {"FNO": FNOBlock, "FNOv2": FNOBlockv2, "UFNO": UFNOBlock, "ConvFNO": ConvFNOBlock}


class FNO(nn.Module):
    """Implementation of Fourier Neural Operator
    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    modes: Union[int, Sequence[int]]
        Number of Fourier modes to use in each dimension. If a single integer is given, it is used for all dimensions.
    hidden_channels: Sequence[int]
        Number of hidden channels in each block. If a single integer is given, it is used for all blocks.
    blocks: Union[str, Sequence[str]]
        Type of blocks to use in the network. If a single string is given, it is used for all blocks. Supported values are 'FNO', 'FNOv2', 'UFNO'.
    spectral_compression_factor: Sequence
        Factor to compress the spectral layer. If a single integer is given, it is used for all blocks.
    activation: nn.module
        Activation function to use in the network. Default is nn.GELU.
    norm: nn.module
        Normalization layer to use in the network. Default is LayerNorm2d.
    fixed_pos_encoding: bool
        Whether to use fixed positional encoding based on the grid coordinates. Default is True.
    trainable_pos_encoding: bool
        Whether to use trainable positional encoding. Default is False.
    trainable_pos_encoding_modes: tuple
        Modes to use for the trainable positional encoding. Only useful if `trainable_pos_encoding` is True. Default is (16, 16).
    trainable_pos_encoding_dims: int
        Number of dimensions for the trainable positional encoding. Only useful if `trainable_pos_encoding` is True. Default is 8.
    block_kwargs: dict
        Additional keyword arguments to pass to the block constructor. Default is an empty dictionary.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_steps: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Sequence[int] = (64, 64, 64, 64),
        blocks: Union[str, Sequence[str]] = "FNO",
        spectral_compression_factor: Sequence = (1, 1, 1),
        activation=nn.GELU,
        norm=LayerNorm2d,
        fixed_pos_encoding: bool = True,
        trainable_pos_encoding: bool = False,
        trainable_pos_encoding_modes=(16, 16),  # Only useful if pos_encoding == 'trainable'
        trainable_pos_encoding_dims=8,
        block_kwargs: dict = {},   
    ):
        super().__init__()
        self.fixed_pos_encoding = fixed_pos_encoding
        self.trainable_pos_encoding = trainable_pos_encoding
        self.time_steps = 1

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        if isinstance(blocks, str):
            blocks = [blocks] * len(hidden_channels)

        assert len(blocks) == len(
            hidden_channels
        ), "Number of elements in hidden_channels must match the number of element in block list"

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        if trainable_pos_encoding:
            self.pos_embedding_weights = nn.Parameter(
                torch.randn(
                    1,
                    trainable_pos_encoding_dims,
                    *trainable_pos_encoding_modes,
                    dtype=torch.cfloat,
                )
            )

            in_channels += trainable_pos_encoding_dims

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)

        self.ops = nn.ModuleList()

        for i, channels in enumerate(hidden_channels):
            in_channels = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            block = REGISTERED_FNO.get(blocks[i], FNOBlock)
            ranks = [in_channels, channels, np.prod(modes)]
            ranks = np.ceil(np.divide(ranks, spectral_compression_factor)).astype(int)
            self.ops.append(
                block(
                    in_channels=in_channels,
                    out_channels=channels,
                    modes=modes,
                    activation=activation,
                    normalization=norm,
                    spectral_layer_type="tucker",
                    ranks=ranks,
                    **block_kwargs,  # Additional keyword arguments for the block
                )
            )

        # self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)
        self.projection = nn.Conv2d(hidden_channels[-1], out_channels * time_steps, 1, bias=True)

    def forward(self, x, return_coords=False):

        if return_coords and not self.fixed_pos_encoding:
            raise ValueError(
                "return_coords is only available when fixed_pos_encoding or trainable_pos_encoding is True"
            )

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)
            if return_coords:
                coords = x[:, -2:, :, :]

        if self.trainable_pos_encoding:
            pos_embeddings = torch.fft.irfftn(self.pos_embedding_weights, s=x.shape[-2:])
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat([x, pos_embeddings.repeat(*repeat_shape)], dim=1)

        x = self.lifting(x)

        for op in self.ops:
            x = op(x)

        x = self.projection(x) 

        B, C_times_T, H, W = x.shape
        if C_times_T % self.time_steps != 0:
            raise ValueError(f"Cannot reshape output with shape {x.shape} into [B, T={self.time_steps}, C, H, W]")

        C = C_times_T // self.time_steps
        x = x.view(B, self.time_steps, C, H, W)

        if self.fixed_pos_encoding and return_coords:
            return x, coords
        else:
            return x

