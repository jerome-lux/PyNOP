import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.norm import LayerNorm2d
from pynop.core.blocks import ITBlock, ComplexMLPBlock, ParametricITBlock
from pynop.core.ops import CartesianEmbedding
from pynop.core.loss import ortho_loss
from pynop.core.utils import gs_orthogonalization


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
        norm: Callable = LayerNorm2d,
        fixed_pos_encoding: bool = True,
        ortho_loss_mode="QR",
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
                    nonlinear=nonlinear,
                    norm=norm,
                    ortho_loss_mode=ortho_loss_mode,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x):

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        for op in self.ops:
            x = op(x)

        x = self.projection(x)

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
        norm: Callable = LayerNorm2d,
        fixed_pos_encoding: bool = True,
        ortho_loss_mode="QR",
        ortho_loss_sampling=2048,
        compute_ortho_loss=True,
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

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding()

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)

        in_ch = 2 + hidden_channels[0] if nonlinear else 2
        self.basis_generator = ComplexMLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
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
                    complex=True,
                    norm=norm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x):

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        # Generate kernels
        B, C, H, W = x.shape

        h_coords_map = torch.linspace(0, 1, H, device=x.device).view(1, H, 1).repeat(1, 1, W)
        w_coords_map = torch.linspace(0, 1, W, device=x.device).view(1, 1, W).repeat(1, H, 1)
        coords_2d_base = torch.concat([h_coords_map, w_coords_map], dim=0)
        # coords_2d: (B, 2, H_basis, W_basis). Add the batch dimension.
        coords_2d = coords_2d_base.unsqueeze(0).repeat(B, 1, 1, 1)  # B C H W

        if self.nonlinear:
            x_in = (
                torch.concat([coords_2d, x], dim=1).permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 2 + C)
            )  # (B, H, W, 2+cin)
        else:
            x_in = coords_2d.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 2)

        basis = self.basis_generator(x_in)  # -> (B*H*W, m1*m2)
        basis = basis.view(B, H, W, self.m1, self.m2)
        basis = gs_orthogonalization(basis)
        # basis = F.normalize(basis, p=2, dim=(1, 2))

        if self.compute_ortho_loss:
            self.ortho_loss = ortho_loss(basis, n_samples=self.ortho_loss_sampling, mode=self.ortho_loss_mode)

        for op in self.ops:
            x = op(x, basis)

        x = self.projection(x)

        return x
