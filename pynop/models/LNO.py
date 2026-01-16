import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import MLPBlock, TransformerBlock
from pynop.core.ops import CartesianEmbedding, sin_positional_encoding_2d
from pynop.core.loss import ortho_loss
from pynop.core.utils import gs_orthogonalization


class ITLNO(nn.Module):
    """
    Forward IT, self attention x N then inverse transform
    Use real basis only
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Sequence[int],
        num_heads: int = 2,
        linear_kernel: bool = True,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        fixed_pos_encoding: bool = True,
        compute_ortho_loss: bool = True,
        orth_loss_sampling: int = 2048,
    ):

        super().__init__()

        assert isinstance(hidden_channels, abc.Sequence), "hidden_channels must be a sequence"

        self.fixed_pos_encoding = fixed_pos_encoding
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.compute_orth_loss = compute_ortho_loss
        self.ortho_loss_sampling = orth_loss_sampling

        if fixed_pos_encoding:
            in_channels += 2
            self.grid_encoding = CartesianEmbedding(minval=-1, maxval=1)

        # will be added to the latent representation
        self.pos_embedding_weights = nn.Parameter(
            torch.randn(
                1,
                self.m1 * self.m2,
                hidden_channels[0],
            )
        )

        self.lifting = nn.Conv2d(in_channels, hidden_channels[0], 1, bias=True)

        in_ch = 2 if linear_kernel else 2 + in_channels
        self.basis_generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
        )
        self.alpha = nn.Parameter(torch.ones(1, 1, 1, out_channels))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i, channels in enumerate(hidden_channels):
            in_channels = hidden_channels[0] if i == 0 else hidden_channels[i - 1]
            self.ops.append(
                TransformerBlock(
                    in_ch=in_channels,
                    out_ch=channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=4 * channels,
                )
            )

        self.projection = nn.Conv2d(hidden_channels[-1], out_channels, 1, bias=True)

    def forward(self, x, cond=None):

        if self.fixed_pos_encoding:
            x = self.grid_encoding(x)

        x = self.lifting(x)

        # Forward transform
        B, C, H, W = x.shape

        H_basis, W_basis = H, W

        h_coords_map = torch.linspace(0, 1, H_basis, device=x.device).view(1, H_basis, 1).repeat(1, 1, W_basis)
        w_coords_map = torch.linspace(0, 1, W_basis, device=x.device).view(1, 1, W_basis).repeat(1, H_basis, 1)
        coords_2d_base = torch.concat([h_coords_map, w_coords_map], dim=0)
        coords_2d = coords_2d_base.unsqueeze(0).repeat(B, 1, 1, 1)  # B C H W

        # 3. Generate kernels
        if self.linear_kernel:
            x_in = coords_2d.permute(0, 2, 3, 1).reshape(B * H * W, 2)
        else:
            x_in = torch.concat([coords_2d, x], dim=1).permute(0, 2, 3, 1).reshape(B * H * W, 2 + C)  # (B, 2+cin, H, W)

        encoder_basis = self.basis_generator(x_in)  # -> (B*H*W, m1*m2)
        encoder_basis = encoder_basis.view(B, H, W, self.m1, self.m2)

        # Normalization of each basis vector
        basis = self.basis_generator(x_in)  # -> (B*H*W, m1*m2)
        basis = basis.view(B, H, W, self.m1, self.m2)
        basis = gs_orthogonalization(basis)

        # just to verify that the basis is nearly orthogonal
        if self.compute_ortho_loss:
            self.ortho_loss = ortho_loss(encoder_basis, n_samples=self.ortho_loss_sampling, mode="MSE")

        # Forward integral transform
        xhat = torch.einsum("bchw,bhwmn->bmnc", x, encoder_basis)
        PE = sin_positional_encoding_2d(C, self.m1, self.m2, device=xhat.device).repeat(B, 1, 1, 1)  # B, C, m1, m2
        PE = PE.permute(0, 2, 3, 1)
        xhat = xhat + PE

        xhat.reshape(B, -1, C)  # -> m1*m2 tokens

        # Add learnable positional encoding
        xhat = xhat + self.pos_embedding_weights.repeat(B, 1, 1)

        # Attention in transformed domain
        for op in self.ops:
            xhat = op(xhat, cond)

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, encoder_basis)
        x_rec = x_rec * self.alpha

        # Mixing channels
        output = self.projection(x_rec)

        return output


class TLNO(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        num_heads: int = 2,
        num_blocks: int = 4,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        dropout=0,
        transformer_mlp_factor: int = 4,
        activation: Callable = nn.GELU,
        dim=2,
    ):

        super().__init__()

        self.trunk_projector = MLPBlock(
            in_ch=dim,
            out_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )
        self.branch_projector = MLPBlock(
            in_ch=in_channels,
            out_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )
        self.attention_projector = MLPBlock(
            in_ch=hidden_channels,
            out_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    in_ch=hidden_channels,
                    out_ch=hidden_channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=transformer_mlp_factor * hidden_channels,
                )
            )

        self.out_mlp = self.attention_projector = MLPBlock(
            in_ch=hidden_channels,
            out_ch=out_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )

    def forward(self, x, cond=None):

        B, C, H, W = x.shape
        h_coords_map = torch.linspace(0, 1, H, device=x.device).view(H, 1, 1).repeat(1, W, 1)
        w_coords_map = torch.linspace(0, 1, W, device=x.device).view(1, W, 1).repeat(H, 1, 1)
        coords_2d_base = torch.concat([h_coords_map, w_coords_map], dim=-1)
        coordinates = coords_2d_base.unsqueeze(0).repeat(B, 1, 1, 1)  # B H W 2
        coordinates = coordinates.contiguous().reshape(B, H * W, 2)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)

        x = self.trunk_projector(coordinates)
        y = self.branch_projector(x)

        score = self.attention_projector(x)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)

        z = torch.einsum("bij,bic->bjc", score_encode, y)

        for block in self.ops:
            z = block(z, cond)

        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        r = r.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return r
