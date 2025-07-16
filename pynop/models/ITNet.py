import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.norm import LayerNorm2d
from pynop.core.blocks import LITBlock


class CITNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Sequence[int],
        block: Callable = LITBlock,
        mlp_layers: int = 2,
        mpl_dim: int = 64,
        activation: Callable = nn.GELU,
        norm: Callable = LayerNorm2d,
    ):
        super().__init__()

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)

        self.ops = nn.ModuleList()

        for i, channels in enumerate(hidden_channels):
            in_channels = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            self.ops.append(
                block(
                    in_channels=in_channels,
                    out_channels=channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    mlp_hidden_dim=mpl_dim,
                    mlp_num_layers=mlp_layers,
                    activation=activation,
                    norm=norm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x):

        x = self.lifting(x)

        for op in self.ops:
            x = op(x)

        x = self.projection(x)

        return x
