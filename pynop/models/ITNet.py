import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
from pynop.core.blocks import LITBlock, MLPBlock, GalerkinTransolverBlock
from pynop.core.encoding import CartesianEmbedding
from pynop.core.utils import print_stats


class LITNet(nn.Module):
    """Learned Integral Transform Network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: int,
        n_layers: int,
        block: Callable = LITBlock,
        mlp_layers: int = 1,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        nonlinear_kernel=True,
        dt=1,
        cond_dim=None,
        separable_conv=True,
        verbose=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dt = dt
        self.verbose = verbose

        in_channels += 2
        self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)
        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=activation,
            )

        self.timstep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
        )

        self.ops = nn.ModuleList()

        for i in range(n_layers):
            self.ops.append(
                block(
                    in_channels=hidden_channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    mlp_hidden_dim=mlp_dim,
                    mlp_num_layers=mlp_layers,
                    activation=activation,
                    mlp_act=activation,
                    nonlinear=nonlinear_kernel,
                    separable=separable_conv,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(
        self,
        x,
        time=None,
        cond=None,
        return_derivative: bool = True,
    ):

        if not return_derivative:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                return_derivative = True

        B, C, H, W = x.shape

        x = self.grid_encoding(x)
        x = self.lifting(x.permute(0, 2, 3, 1))

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        if time is not None:
            time = self.timstep_embedding(time).unsqueeze(1).unsqueeze(1)
            x = x + time

        for i, op in enumerate(self.ops):
            x = op(x, time)
            if self.verbose:
                print_stats(x, -1, f"after block {i+1}")

        x = self.projection(x).permute(0, 3, 1, 2)

        if return_derivative:
            return x
        else:
            return shortcut + x * self.dt


class GalerkinTransolver(nn.Module):
    """Learned Integral Transform Network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int,
        n_layers: int,
        num_heads: int = 4,
        mlp_layers: int = 1,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        dt=1,
        cond_dim=None,
        kv_normalization=False,
        dropout=0.1,
        verbose=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dt = dt
        self.verbose = verbose

        in_channels += 2
        self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=activation,
            )

        self.timestep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
        )

        self.ops = nn.ModuleList()

        for i in range(n_layers):
            self.ops.append(
                GalerkinTransolverBlock(
                    dim=hidden_channels,
                    num_heads=num_heads,
                    modes=modes * modes,
                    mlp_factor=2,
                    activation=activation,
                    kv_normalization=kv_normalization,
                    dropout=dropout,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(
        self,
        x,
        time=None,
        cond=None,
        return_derivative: bool = True,
    ):

        if not return_derivative:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                return_derivative = True

        B, C, H, W = x.shape

        x = self.grid_encoding(x)

        x = self.lifting(x.permute(0, 2, 3, 1))  # B H W D

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        if time is not None:
            time = self.timestep_embedding(time).unsqueeze(1).unsqueeze(1)
            x = x + time
        xhat = x.view(B, H * W, -1)
        for i, op in enumerate(self.ops):
            xhat = op(xhat)
            if self.verbose:
                print_stats(xhat, -1, f"after block {i+1}")

        x = self.projection(xhat).view(B, H, W, -1)  # B Cout H W
        x = x.permute(0, 3, 1, 2)

        if return_derivative:
            return x
        else:
            return shortcut + x * self.dt
