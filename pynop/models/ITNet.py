import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.norm import AdaptiveLayerNorm, LayerNorm2d
from pynop.core.blocks import ITBlock, ComplexMLPBlock, ParametricITBlock, MLPBlock
from pynop.core.ops import CartesianEmbedding, sinusoidal_encoding_2d, time_encoding
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
        hidden_channels: Sequence[int],
        block: Callable = ITBlock,
        mlp_layers: int = 2,
        mlp_dim: int = 64,
        nonlinear=True,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU,
        norm: Callable = AdaptiveLayerNorm,
        fixed_pos_encoding: bool = True,
        ortho_loss_mode="MSE",
        dim=2,
    ):

        super().__init__()

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        self.fixed_pos_encoding = fixed_pos_encoding

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

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
                    mlp_hidden_dim=mlp_dim,
                    mlp_num_layers=mlp_layers,
                    activation=activation,
                    mlp_act=mlp_act,
                    nonlinear=nonlinear,
                    norm=norm,
                    ortho_loss_mode=ortho_loss_mode,
                    dim=dim,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x, time=None, residual=False):

        if residual:
            shortcut = x

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        for op in self.ops:
            x = op(x, time)

        x = self.projection(x)

        if residual:
            x = x + shortcut

        return x


class SharedLITNet(nn.Module):
    """Learned Integral Transform Network with a shared basis"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Sequence[int],
        mlp_layers: int = 2,
        mlp_dim: int = 64,
        nonlinear=True,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU(),
        norm: Callable = AdaptiveLayerNorm,
        fixed_pos_encoding: bool = True,
        ortho_loss_mode="fro",
        ortho_loss_sampling=2048,
        compute_ortho_loss=True,
        complex=True,
        dim=2,
    ):

        super().__init__()

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        self.fixed_pos_encoding = fixed_pos_encoding
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.nonlinear = nonlinear
        self.ortho_loss = 0.0
        self.ortho_loss_mode = ortho_loss_mode
        self.ortho_loss_sampling = ortho_loss_sampling
        self.compute_ortho_loss = compute_ortho_loss
        self.dim = dim
        self.complex = complex

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        in_ch = dim + hidden_channels[0] if nonlinear else dim

        if complex:
            self.basis_generator = ComplexMLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=in_ch,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )
        else:
            self.basis_generator = MLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=in_ch,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.ops = nn.ModuleList()

        for i, channels in enumerate(hidden_channels):
            in_channels = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            self.ops.append(
                ParametricITBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    activation=activation,
                    complex=complex,
                    norm=norm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x: Tensor, time: Union[None, Tensor] = None, residual: bool = False):

        if residual:
            shortcut = x

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        # Generate kernels
        B, C, H, W = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        coords = coords.reshape(B, H, W, 2).contiguous()

        if time is not None:
            t = time.view(B, 1, 1, 1).expand(-1, H, W, -1)  # VRAM: 0
            coords = torch.cat([coords, t], dim=-1)

        # 3. Generate kernels
        if self.nonlinear:
            x_in = torch.concat([coords, x.permute(0, 2, 3, 1)], dim=-1)  # (B, H, W, dim + Cin)
        else:
            x_in = coords

        basis = self.basis_generator(x_in)  # -> (B*H*W, m1*m2)
        basis = basis.view(B, H, W, self.m1, self.m2)
        norm_factor = torch.sqrt(torch.sum(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        basis = basis / (norm_factor + 1e-6)

        if self.compute_ortho_loss:
            self.ortho_loss = ortho_loss(basis, n_samples=self.ortho_loss_sampling, mode=self.ortho_loss_mode)

        for op in self.ops:
            x = op(x, fwd_basis=basis, time=time)

        x = self.projection(x)

        if residual:
            return x + shortcut

        else:
            return x  # + shortcut


class SharedModLITNet(nn.Module):
    """Learned Integral Transform Network with a shared basis using 2D+t sinusoidal encoding and signal modulation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Sequence[int],
        mlp_layers: int = 1,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act: Callable = nn.GELU(),
        norm: Callable = AdaptiveLayerNorm,
        fixed_pos_encoding: bool = True,
        encoding_freq=6,
        dim=2,
    ):

        super().__init__()

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        self.fixed_pos_encoding = fixed_pos_encoding
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.encoding_freq = encoding_freq

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        self.time_mlp = MLPBlock(
            in_ch=2 * encoding_freq,
            out_ch=self.m1 * self.m2,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        self.trunk = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=4 * encoding_freq,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.branch = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=hidden_channels[0],
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.ops = nn.ModuleList()

        for i, channels in enumerate(hidden_channels):
            in_channels = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            self.ops.append(
                ParametricITBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    m1=modes if isinstance(modes, int) else modes[0],
                    m2=modes if isinstance(modes, int) else modes[1],
                    activation=activation,
                    complex=False,
                    norm=norm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x: Tensor, time: Union[None, Tensor], residual: bool = False):

        if residual:
            shortcut = x

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        # Generate kernels
        B, C, H, W = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = sinusoidal_encoding_2d(coords, self.encoding_freq).view(H, W, -1)  # [H, W, 4 * F]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        t = time_encoding(time, self.encoding_freq)  # [B, 2 * F]
        t = self.time_mlp(t)  # [B, m1*m2]

        trunk = self.trunk(coords)  # [B, H, W, m1*m2]
        branch = self.branch(x.permute(0, 2, 3, 1))  # [B, H, W, m1*m2]

        basis = branch * t.unsqueeze(1).unsqueeze(1) * trunk

        basis = basis.view(B, H, W, self.m1, self.m2)
        norm_factor = torch.sqrt(torch.sum(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        basis = basis / (norm_factor + 1e-6)

        for op in self.ops:
            x = op(x, fwd_basis=basis, time=time)

        x = self.projection(x)

        if residual:
            return x + shortcut

        else:
            return x  # + shortcut
