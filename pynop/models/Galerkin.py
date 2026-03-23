import math
from functools import partial
from sympy import chebyshevt_poly
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import (
    MLPBlock,
    GalerkinTransformerBlock,
)
from pynop.core.norm import AdaptiveLayerNorm, AdaRMSNorm
from pynop.core.activations import gumbel_softmax, Sine
from pynop.core.utils import ChebyshevBasis, print_stats


class GalerkinTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dt: float = 1.0,
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
        dim: int = 2,
        verbose: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dt = dt

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.pelayer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=dim,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.norm = AdaRMSNorm(hidden_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                GalerkinTransformerBlock(hidden_channels, num_heads, mlp_dim=hidden_channels * mlp_factor, delta=1e-2)
                for _ in range(num_blocks)
            ]
        )

        self.time_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
        )

        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, time, residual=True):
        # x: [batch, channels, height, width]
        B, C, H, W = x.shape

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        x = x.view(B, C, -1).transpose(1, 2)  # [batch, n, c]

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        coords = coords.view(B, -1, 2)

        pe = self.pelayer(coords)  # Add 2D Positional Encoding
        x = self.input_proj(x)

        encoded_time = self.time_embedding(time)

        x = self.norm(x + encoded_time[:, None, :], encoded_time)

        for layer in self.layers:
            x = x + pe
            x = layer(x)

        x = self.output_proj(x)
        x = x.transpose(1, 2).view(B, -1, H, W)

        if residual:
            return shortcut + x * self.dt
        else:
            return x