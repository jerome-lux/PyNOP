from typing import Sequence, Union
import numpy as np
from collections.abc import Iterable
import torch
import torch.nn as nn
from pynop.core.blocks import FNOBlock, FNOBlockv2, UFNOBlock
from pynop.core.ops import CartesianEmbedding
from pynop.core.ops import ConvLayer
from pynop.core.norm import LayerNorm2d
from pynop.core.utils import make_tuple

REGISTERED_FNO = {"FNO": FNOBlock, "FNOv2": FNOBlockv2, "UFNO": UFNOBlock}


class FNO(nn.Module):
    """Implementation of Fourier Neural Operator"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
    ):
        super().__init__()
        self.fixed_pos_encoding = fixed_pos_encoding
        self.trainable_pos_encoding = trainable_pos_encoding

        if isinstance(blocks, str):
            blocks = [blocks] * len(hidden_channels)
        elif isinstance(hidden_channels, Iterable):
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
                )
            )

        self.projection = nn.Conv2d(channels, out_channels, 1, bias=True)

    def forward(self, x):

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        if self.trainable_pos_encoding:
            pos_embeddings = torch.fft.irfftn(self.pos_embedding_weights, s=x.shape[-2:])
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]  # repeat along the batch size to match input
            x = torch.cat([x, pos_embeddings.repeat(*repeat_shape)], dim=1)  # cat along the channel axis

        x = self.lifting(x)

        for op in self.ops:
            x = op(x)

        x = self.projection(x)

        return x
