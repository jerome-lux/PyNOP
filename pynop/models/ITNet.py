import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.norm import AdaptiveLayerNorm, LayerNorm2d
from pynop.core.blocks import LITBlock, ComplexMLPBlock, ParametricITBlock, MLPBlock
from pynop.core.ops import CartesianEmbedding, sinusoidal_encoding_2d
from pynop.core.loss import ortho_loss
from pynop.core.utils import gs_orthogonalization


# TODO: use Fourier encoding for coordinates and modulation instead of concatenation for non linear kernels


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
        mlp_layers: int = 2,
        mlp_dim: int = 64,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU,
        nonlinear_kernel=True,
        fixed_pos_encoding: bool = True,
        sinus_pe_freq=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fixed_pos_encoding = fixed_pos_encoding

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        self.timstep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
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
                    mlp_act=mlp_act,
                    sinus_pe_freq=sinus_pe_freq,
                    nonlinear=nonlinear_kernel,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, time=None, residual=False):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]

            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x.permute(0, 2, 3, 1))

        if time is not None:
            time = self.timstep_embedding(time).unsqueeze(1).unsqueeze(1)
            x = x + time

        for op in self.ops:
            x = op(x, time)

        x = self.projection(x).permute(0, 3, 1, 2)

        if residual:
            x = x + shortcut

        return x


class SharedModLITNet(nn.Module):
    """Learned Integral Transform Network with a shared basis using 2D+t sinusoidal encoding and signal modulation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: int,
        n_layers: int,
        mlp_layers: int = 1,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU(),
        fixed_pos_encoding: bool = True,
        sinus_pe_freq=None,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fixed_pos_encoding = fixed_pos_encoding
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.sinus_pe_freq = sinus_pe_freq

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm = AdaptiveLayerNorm(hidden_channels, hidden_channels)

        self.trunk_norm = nn.LayerNorm(in_channels)
        self.branch_norm = nn.LayerNorm(in_channels)

        if sinus_pe_freq is not None:
            trunk_in_ch = 4 * sinus_pe_freq
        else:
            trunk_in_ch = 2

        self.trunk = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=trunk_in_ch,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.branch = MLPBlock(
            out_ch=2 * self.m1 * self.m2,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.ops = nn.ModuleList()

        for i in range(n_layers):
            self.ops.append(
                ParametricITBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    activation=activation,
                    complex=False,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

        # self.trunk.apply(self._init_weights)

    def _init_weights(self, module, std=0.0002):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: Tensor, time: Union[None, Tensor], residual: bool = False):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        if residual:
            shortcut = x

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x.permute(0, 2, 3, 1))

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        if self.sinus_pe_freq:
            coords = sinusoidal_encoding_2d(coords, self.sinus_pe_freq).view(H, W, -1)  # [H, W, 4 * F]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        if time is not None:
            time = self.timsetep_embedding(time).unsqueeze(1).unsqueeze(1)
            x = x + time

        x = self.norm(x, time)

        trunk = self.trunk_norm(self.trunk(coords))  # [B, H, W, 2 * m1*m2] depends only on coordinates
        branch = self.branch_norm(self.branch(x))  # [B, H, W, m1*m2] depends on the input signal modulated by time
        gamma, beta = branch.chunk(2, dim=-1)
        basis = trunk * (1 + gamma) + beta
        basis = basis.view(B, H, W, self.m1, self.m2)
        # norm_factor = torch.sqrt(torch.mean(basis**2, dim=(1, 2), keepdim=True) + 1e-6)
        basis = basis / (H * W)

        for op in self.ops:
            x = op(x, fwd_basis=basis, time=time)

        x = self.projection(x).permute(0, 3, 1, 2)

        if residual:
            return x + shortcut

        else:
            return x  # + shortcut


class SharedLITNet(nn.Module):
    """Learned Integral Transform Network with a shared basis using 2D+t sinusoidal encoding and signal modulation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: int,
        n_layers: int,
        mlp_layers: int = 1,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU(),
        fixed_pos_encoding: bool = True,
        nonlinear_kernel=True,
        sinus_pe_freq=None,
        orthogonal_init=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fixed_pos_encoding = fixed_pos_encoding
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.sinus_pe_freq = sinus_pe_freq
        self.nonlinear_kernel = nonlinear_kernel

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm = AdaptiveLayerNorm(hidden_channels, hidden_channels)

        if sinus_pe_freq is not None:
            in_ch = 4 * sinus_pe_freq
        else:
            in_ch = 2

        if nonlinear_kernel:
            in_ch = in_ch + hidden_channels

        self.generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        if orthogonal_init:
            self.ortho_init_weights(self.generator)

        self.ops = nn.ModuleList()

        for i in range(n_layers):
            self.ops.append(
                ParametricITBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    activation=activation,
                    complex=False,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def _init_weights(self, module, std=0.02):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: Tensor, time: Union[None, Tensor], residual: bool = False):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        if residual:
            shortcut = x

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x.permute(0, 2, 3, 1))

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        if self.sinus_pe_freq:
            coords = sinusoidal_encoding_2d(coords, self.sinus_pe_freq).view(H, W, -1)  # [H, W, 4 * F]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        if time is not None:
            time = self.timsetep_embedding(time).unsqueeze(1).unsqueeze(1)
            x = x + time

        x = self.norm(x, time)
        if self.nonlinear_kernel:
            basis_input = torch.cat([x, coords], dim=-1)
        else:
            basis_input = coords
        basis = self.generator(basis_input)
        basis = basis.view(B, H, W, self.m1, self.m2)
        basis = basis / (H * W)

        for op in self.ops:
            x = op(x, fwd_basis=basis, time=time)

        x = self.projection(x).permute(0, 3, 1, 2)

        if residual:
            return x + shortcut

        else:
            return x  # + shortcut
