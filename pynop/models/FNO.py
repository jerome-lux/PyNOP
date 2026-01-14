from typing import Sequence, Union
import numpy as np
import collections.abc as abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from pynop.core.blocks import FNOBlock, FNOBlockv2, UFNOBlock, ConvFNOBlock, SpectralConv2d_fast
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



class FNO2d_PDEBench(nn.Module):
    """
    PDEBench-style FNO2d

    Expects:
      inp : [B, H, W, initial_step * C]
      grid: [B, H, W, 2]  (x,y coords)

    Returns:
      [B, H, W, 1, C]  (one-step prediction with time axis length 1)
    """

    def __init__(self, num_channels: int, modes1: int = 12, modes2: int = 12, width: int = 20, initial_step: int = 10):
        super().__init__()
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.initial_step = initial_step

        self.padding = 2  

        # "first 10 timesteps + 2 coords" -> lift to width
        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)

        self.conv0 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d_fast(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d_fast(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # projection back to channels
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, inp: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # inp:  [B,H,W, initial_step*C]
        # grid: [B,H,W,2]
        x = torch.cat((inp, grid), dim=-1)     # [B,H,W, initial_step*C + 2]
        x = self.fc0(x)                        # [B,H,W,width]
        x = x.permute(0, 3, 1, 2)              # [B,width,H,W]

        # pad (right/bottom) like PDEBench
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        # unpad
        x = x[..., : -self.padding, : -self.padding]   # [B,width,H,W]
        x = x.permute(0, 2, 3, 1)                      # [B,H,W,width]

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)                                # [B,H,W,C]

        return x.unsqueeze(-2)                          # [B,H,W,1,C]
