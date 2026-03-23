import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import (
    MLPBlock,
    TransformerBlock,
    LatentTemporalTransformer,
    LinearTransformerBlock,
    GalerkinTransformerBlock,
)

from pynop.core.ops import GalerkinAttention
from pynop.core.norm import AdaptiveLayerNorm, AdaRMSNorm
from pynop.core.activations import gumbel_softmax, Sine
from pynop.core.utils import ChebyshevBasis, print_stats
from .CSTS import AB_coeffs


class ITLNO_v1(nn.Module):
    """Integral Transform-based Latent Nonlinear Operator (ITLNO) using self-attention in transfomed domain.

    This module performs a forward integral transform using learned bases,
    applies self-attention in the transformed domain, and then reconstructs the output
    in the original spatial domain. It uses real basis functions only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (Union[int, Sequence[int]]): Number of modes for the integral transform. If int, uses same
            value for both dimensions. If sequence, uses modes[0] for m1 and modes[1] for m2.
        hidden_channels (Sequence[int]): Sequence of hidden channel dimensions for each transformer block.
        num_heads (int): Number of attention heads in transformer blocks. Default: 2.
        linear_kernel (bool): Whether to use only coordinates as basis generator input (True) or
            include input channels (False). Default: True.
        mlp_layers (int): Number of layers in the basis generator MLP. Default: 2.
        mlp_dim (int): Hidden dimension for the basis generator MLP. Default: 128.
        activation (Callable): Activation function to use. Default: nn.GELU.


        Il manque une projection linéaire à la fin de l'attentiondans ccette implémentation...
    """

    class TransformerBlock(nn.Module):
        def __init__(self, dim, n_heads, activation=nn.GELU, mlp_dim=256, dropout=0.1):
            super().__init__()

            self.attention = ITLNO_v1.deprec_Attention(dim, dim, n_heads)

            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout),
            )

            self.norm1 = AdaptiveLayerNorm(dim, dim)
            self.norm2 = nn.LayerNorm(dim)

        def forward(self, x, cond=None):
            # x: [B, N, C]

            res = x
            x = self.norm1(x, cond)
            x = self.attention(Q=x, V=x, K=x)
            x = x + res
            res = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = x + res

            return x

    class deprec_Attention(nn.Module):
        """Attention Module (can be self or cross attention) depedning on inputs Q, K, V of the forward method"""

        def __init__(self, in_ch, out_ch, num_heads):
            super().__init__()
            assert out_ch % num_heads == 0
            self.d_model = out_ch
            self.num_heads = num_heads
            self.head_dim = out_ch // num_heads

            self.wq = nn.Linear(in_ch, out_ch, bias=True)
            self.wk = nn.Linear(in_ch, out_ch, bias=True)
            self.wv = nn.Linear(in_ch, out_ch, bias=True)

        def split_heads(self, x):
            batch_size, seq_len, d_model = x.shape
            x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            return x.transpose(1, 2)

        def combine_heads(self, x):
            x = x.transpose(1, 2).contiguous()
            batch_size, seq_len, num_heads, head_dim = x.shape
            return x.reshape(batch_size, seq_len, num_heads * head_dim)

        def forward(self, Q, V, K):
            Q = self.split_heads(self.wq(Q))
            K = self.split_heads(self.wk(K))
            V = self.split_heads(self.wv(V))

            attention_scores = torch.matmul(Q, K.transpose(-2, -1))
            attention_scores = attention_scores / (self.head_dim**0.5)

            attention_scores = attention_scores
            attention_weights = F.softmax(attention_scores, dim=-1)

            weighted_output = torch.matmul(attention_weights, V)
            output = self.combine_heads(weighted_output)

            return output

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        linear_kernel: bool = True,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor=4,
        dropout=0,
        dim=3,
        orthogonal_init=True,
        pe=True,
        pe_freqs=32,
        cond_dim=None,
        basis_mode="learned",
        verbose=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.pe = pe
        self.pe_freqs = pe_freqs
        self.basis_mode = basis_mode
        self.verbose = verbose

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm = AdaptiveLayerNorm(hidden_channels, hidden_channels)

        in_ch = dim if not pe else pe_freqs + 1

        if self.basis_mode == "cheb":
            self.coord_generator = ChebyshevBasis(self.m1, self.m2)
        else:
            self.coord_generator = MLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=in_ch,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.signal_generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        if orthogonal_init:
            self.ortho_init_weights(self.coord_generator)
            self.ortho_init_weights(self.signal_generator)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.alpha = nn.Parameter(torch.full((1, hidden_channels, 1, 1), 0.98))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                ITLNO_v1.TransformerBlock(
                    dim=hidden_channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=mlp_factor * hidden_channels,
                    dropout=dropout,
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
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

        # time_scaling = F.softplus(self.time_scaling(time))
        time_scaling = self.time_scaling(time)

        x = self.lifting(x.permute(0, 2, 3, 1))

        # normalisation & time modulation
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]

        if self.basis_mode == "cheb":
            basis = self.coord_generator(coords.unsqueeze(0).expand(B, H, W, -1))

        if self.pe:
            coords = sin_positional_encoding_2d(coords, self.pe_freqs, max_freq=H // 2)
        coords = coords.unsqueeze(0).expand(B, H, W, -1)

        t = time[:, None, None, :].expand(-1, H, W, -1)
        coords = torch.cat([coords, t], dim=-1)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        time = self.timsetep_embedding(time).unsqueeze(1).unsqueeze(1)
        x = self.norm(x + time, time)
        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after norm:")

        time = time.view(B, 1, -1)
        if self.basis_mode != "cheb":
            basis = self.coord_generator(coords)

        if not self.linear_kernel:
            basis += self.signal_generator(x)

        basis = basis.view(B, H, W, self.m1, self.m2)

        basis = basis / math.sqrt(H * W)
        if self.verbose:
            print_stats(basis.view(B, H, W, -1), -1, "basis (along modes):")

        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis)
        if self.verbose:
            print_stats(xhat, -1, "after IT:")
        # Add Positional encoding in latent representation before the self-attention modules
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.latent_pe(PE)

        xhat = xhat + PE
        xhat = xhat.reshape(B, self.m1 * self.m2, -1).contiguous()  # -> m1*m2 tokens

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            xhat = op(xhat, time)
        if self.verbose:
            print_stats(xhat, -1, "after ATT:")

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, basis)

        # ATESTER tanh(x_rec) * alpha
        x_rec = x_rec * self.alpha
        if self.verbose:
            print_stats(x_rec, 1, "after REC:")

        # Mixing channels and multiplying by the time scaling
        output = time_scaling.view(B, self.out_channels, 1, 1) * self.projection(x_rec)

        if not return_derivative:
            return output + shortcut
        else:
            return output


class ITLNO(nn.Module):
    """Latent Nonlinear Operator using self-attention in transfomed domain.

    This module performs a forward integral transform using learned bases,
    applies self-attention in the transformed domain, and then reconstructs the output
    in the original spatial domain. It uses real basis functions only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (Union[int, Sequence[int]]): Number of modes for the integral transform. If int, uses same
            value for both dimensions. If sequence, uses modes[0] for m1 and modes[1] for m2.
        hidden_channels (Sequence[int]): Sequence of hidden channel dimensions for each transformer block.
        num_heads (int): Number of attention heads in transformer blocks. Default: 2.
        linear_kernel (bool): Whether to use only coordinates as basis generator input (True) or
            include input channels (False). Default: True.
        mlp_layers (int): Number of layers in the basis generator MLP. Default: 2.
        mlp_dim (int): Hidden dimension for the basis generator MLP. Default: 128.
        activation (Callable): Activation function to use. Default: nn.GELU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        dt: float = 1,
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        linear_kernel: bool = True,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor: int = 4,
        dropout: float = 0,
        dim: int = 2,
        orthogonal_init: bool = True,
        pe: bool = True,
        cond_dim: Union[int, None] = None,
        basis_mode: str = "learned",
        rmsnorm: bool = True,
        verbose: bool = False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.pe = pe
        self.basis_mode = basis_mode
        self.dt = dt
        self.verbose = verbose

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)
        self.latent_projection = nn.Linear(hidden_channels, hidden_channels, bias=True)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.pelayer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=dim,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)
        # self.norm2 = nn.RMSNorm(hidden_channels)
        # self.norm3 = nn.LayerNorm(hidden_channels)

        in_ch = dim

        if self.basis_mode == "cheb":
            self.coord_generator = ChebyshevBasis(self.m1, self.m2)
        else:
            self.coord_generator = MLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=in_ch,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=Sine,
            )

        self.signal_generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        if orthogonal_init:
            self.ortho_init_weights(self.coord_generator)
            self.ortho_init_weights(self.signal_generator)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        # self.alpha = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.01))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    dim=hidden_channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=mlp_factor * hidden_channels,
                    dropout=dropout,
                    rmsnorm=rmsnorm,
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
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
        x = self.lifting(x.permute(0, 2, 3, 1))

        time_scaling = F.softplus(self.time_scaling(time))
        encoded_time = self.timsetep_embedding(time)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        basis = self.coord_generator(coords)

        if self.pe:  # SIREN -> [B, H, W, D]
            coords = self.pelayer(coords)

        # adds temporal encoding
        x = self.norm1(x + encoded_time[:, None, None, :] + coords, encoded_time)

        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after AdaNorm:")

        if not self.linear_kernel:
            basis += self.signal_generator(x)
            basis = 0.5 * basis

        if self.verbose:
            print_stats(basis.view(B, H, W, -1), -1, "basis before normalization (along modes):")

        basis = basis.view(B, H, W, self.m1, self.m2)
        d_norm = torch.norm(basis, p=2, dim=(1, 2), keepdim=True)
        # d = torch.sum(torch.abs(basis), dim=(1, 2))
        basis = basis / d_norm
        if self.verbose:
            print_stats(basis.view(B, H, W, self.m1 * self.m2), -1, "basis:")
        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis) / math.sqrt(H * W)
        xhat = self.latent_projection(xhat)
        # xhat = self.norm2(xhat)   # normalization degrades performance
        if self.verbose:
            print_stats(xhat.view(B, self.m1, self.m2, -1), -1, "after IT:")

        # Add Positional encoding in latent representation before the self-attention modules (SIREN)
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.latent_pe(PE)

        xhat = xhat + PE
        xhat = xhat.reshape(B, self.m1 * self.m2, -1).contiguous()  # -> m1*m2 tokens

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            xhat = op(xhat, encoded_time)
        if self.verbose:
            print_stats(xhat.view(B, self.m1, self.m2, -1), -1, "after ATT:")

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)  # * self.inverse_scale[None, :, :, None]
        x_rec = torch.einsum("bmnc,bhwmn->bhwc", xhat, basis)

        # x_rec = self.norm3(x_rec)
        if self.verbose:
            print_stats(x_rec.view(B, H * W, -1), -1, "after REC:")
        x_rec = x_rec.permute(0, 3, 1, 2)
        # Mixing channels and multiplying by the time scaling
        output = time_scaling.view(B, self.out_channels, 1, 1) * self.projection(x_rec)
        # output = self.projection(x_rec)

        if self.verbose:
            print_stats(output, 1, "Final:")

        if return_derivative:
            return output
        else:
            return shortcut + output * self.dt


class LatentGalerkin(nn.Module):
    """Projection using Galerkin Cross-Attention to M latent tokens

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (Union[int, Sequence[int]]): Number of modes for the integral transform. If int, uses same
            value for both dimensions. If sequence, uses modes[0] for m1 and modes[1] for m2.
        hidden_channels (Sequence[int]): Sequence of hidden channel dimensions for each transformer block.
        num_heads (int): Number of attention heads in transformer blocks. Default: 2.
        linear_kernel (bool): Whether to use only coordinates as basis generator input (True) or
            include input channels (False). Default: True.
        mlp_layers (int): Number of layers in the basis generator MLP. Default: 2.
        mlp_dim (int): Hidden dimension for the basis generator MLP. Default: 128.
        activation (Callable): Activation function to use. Default: nn.GELU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        dt: float = 1.0,
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor: int = 4,
        dropout: float = 0,
        dim: int = 2,
        cond_dim: Union[int, None] = None,
        latent_attention="galerkin",
        verbose: bool = False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.dt = dt
        self.verbose = verbose

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        self.encoder = GalerkinAttention(dim=hidden_channels, heads=num_heads)
        self.decoder = GalerkinAttention(dim=hidden_channels, heads=num_heads)

        self.latents = nn.Parameter(torch.randn(1, modes, hidden_channels))

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.pelayer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=dim,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        # List of attention modules

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
            )
        else:
            block = partial(
                TransformerBlock,
                dim=hidden_channels,
                n_heads=num_heads,
                activation=activation,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                rmsnorm=True,
            )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
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
        x = self.lifting(x.permute(0, 2, 3, 1))

        time_scaling = F.softplus(self.time_scaling(time))
        time_encoding = self.timsetep_embedding(time)

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        pe = self.pelayer(coords)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        # adds temporal encoding and time conditionning
        x = x + pe
        x = x.view(B, H * W, -1)
        x = self.norm1(x + time_encoding[:, None, :], time_encoding)
        if self.verbose:
            print_stats(x, -1, "after AdaNorm:")

        # Add Positional encoding in latent representation before the self-attention modules (SIREN)
        m_coords = torch.linspace(-1, 1, steps=self.modes, device=x.device).unsqueeze(-1)
        m_coords = m_coords.unsqueeze(0).expand(B, -1, -1)
        latent_pe = self.latent_pe(m_coords)
        latents = self.latents.expand((B, -1, -1))

        # Galerkin Cross Attention -> [B, M, C]
        xhat = self.encoder(x=latents, context=x)

        if self.verbose:
            print_stats(xhat, -1, "after IT:")

        xhat = xhat + latent_pe

        if self.verbose:
            print_stats(xhat, -1, "after xhat + pe:")

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            xhat = op(xhat)
        if self.verbose:
            print_stats(xhat, -1, "after ATT modules:")

        # Go back to spatial domain
        x_rec = self.decoder(x=x, context=xhat)  # [B, H*W, C]

        if self.verbose:
            print_stats(x_rec.view(B, H * W, -1), -1, "after REC:")

        x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Mixing channels and multiplying by the time scaling
        output = time_scaling.view(B, self.out_channels, 1, 1) * self.projection(x_rec)
        # output = self.projection(x_rec)

        if self.verbose:
            print_stats(output, 1, "Final:")

        if return_derivative:
            return output
        else:
            return output * self.dt + shortcut


class TimeAttentionLatentGalerkin(nn.Module):
    """
    This module combines cross-attention mechanisms with temporal transformers to project input data
    onto a set of learnable latent tokens. It uses Galerkin-type attention for efficient integration
    of information across spatial and temporal dimensions.

    The forward pass:
    1. Lifts input channels to a higher-dimensional hidden space
    2. Encodes spatial information through positional encoding (SIREN)
    3. Applies Galerkin cross-attention to project onto latent tokens
    4. Processes latent representations through temporal transformer blocks
    5. Decodes back to the original spatial resolution
    6. Projects to output channels

        modes (int): Number of latent modes/tokens for Galerkin projection.
        dt (float): Time step scaling factor for derivative computation. Default: 1.0.
        num_blocks (int): Number of spatial transformer blocks to apply. Default: 4.
        num_temporal_block (int): Number of layers in the temporal transformer. Default: 2.
        hidden_channels (int): Dimension of hidden representations. Default: 256.
        num_heads (int): Number of attention heads in transformer blocks. Default: 4.
        mlp_layers (int): Number of layers in MLP blocks for embeddings. Default: 2.
        mlp_dim (int): Hidden dimension of MLP blocks. Default: 128.
        activation (Callable): Activation function for transformer blocks. Default: nn.GELU.
        mlp_act (Callable): Activation function for MLP blocks. Default: nn.GELU.
        mlp_factor (int): Multiplier for MLP hidden dimensions. Default: 4.
        dropout (float): Dropout probability. Default: 0.
        dim (int): Spatial dimensionality (2 for 2D grids). Default: 2.
        cond_dim (Union[int, None]): Dimension of conditional input, or None if not used. Default: None.
        latent_attention (str): Type of attention mechanism for latent processing ("galerkin" or "standard"). Default: "galerkin".
        verbose (bool): Whether to print intermediate statistics during forward pass. Default: False.
        memory (int): Number of past time steps to maintain in temporal history. Default: 6.

    Attributes:
        history (list): Buffer storing past latent representations for temporal processing.
        latents (nn.Parameter): Learnable latent token embeddings of shape (1, modes, hidden_channels).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        dt: float = 1.0,
        num_blocks: int = 4,
        num_temporal_block: int = 2,
        hidden_channels: int = 256,
        num_heads: int = 4,
        mlp_layers: int = 2,
        mlp_dim: int = 128,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor: int = 4,
        dropout: float = 0,
        dim: int = 2,
        cond_dim: Union[int, None] = None,
        latent_attention="galerkin",
        verbose: bool = False,
        memory: int = 6,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.dt = dt
        self.verbose = verbose
        self.memory = memory

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)
        self.encoder = GalerkinAttention(dim=hidden_channels, heads=num_heads)
        self.decoder = GalerkinAttention(dim=hidden_channels, heads=num_heads)
        self.latents = nn.Parameter(torch.randn(1, modes, hidden_channels))

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.pelayer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=dim,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        # List of attention modules

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
            )
        else:
            block = partial(
                TransformerBlock,
                dim=hidden_channels,
                n_heads=num_heads,
                activation=activation,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                rmsnorm=True,
            )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        self.temporal_block = LatentTemporalTransformer(
            dim=hidden_channels,
            n_heads=num_heads,
            mlp_factor=mlp_factor,
            num_layers=num_temporal_block,
            max_history=self.memory,
        )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        history: torch.Tensor = None,
        cond: Union[None, torch.Tensor] = None,
        return_derivative: bool = True,
        timeattention: bool = True,
    ):

        if not return_derivative:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                return_derivative = True

        B, C, H, W = x.shape
        x = self.lifting(x.permute(0, 2, 3, 1))

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        pe = self.pelayer(coords)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        x = x + pe
        x = x.view(B, H * W, -1)
        x = self.norm1(x)
        if self.verbose:
            print_stats(x, -1, "after AdaNorm:")

        # Add Positional encoding in latent representation before the self-attention modules (SIREN)
        m_coords = torch.linspace(-1, 1, steps=self.modes, device=x.device).unsqueeze(-1)
        m_coords = m_coords.unsqueeze(0).expand(B, -1, -1)
        latent_pe = self.latent_pe(m_coords)
        latents = self.latents.expand((B, -1, -1))

        # Galerkin Cross Attention -> [B, M, C]
        xhat = self.encoder(x=latents, context=x)

        if self.verbose:
            print_stats(xhat, -1, "after IT:")

        xhat = xhat + latent_pe

        if self.verbose:
            print_stats(xhat, -1, "after xhat + pe:")

        # Multi-head attention
        for op in self.ops:
            xhat = op(xhat)
        if self.verbose:
            print_stats(xhat, -1, "after ATT modules:")

        # if not autoencoder:
        #     xhat, self.history = self.temporal_block(xhat, self.history, coords_t=time.unsqueeze(-1))
        if timeattention:
            xhat, history = self.temporal_block(xhat, history, coords_t=time.unsqueeze(-1))
        # self.history = self.history.detach()

        # Go back to spatial domain
        x_rec = self.decoder(x=x, context=xhat)  # [B, H*W, C]

        if self.verbose:
            print_stats(x_rec.view(B, H * W, -1), -1, "after REC:")

        x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Mixing channels and multiplying by the time scaling
        output = self.projection(x_rec)

        if self.verbose:
            print_stats(output, 1, "Final:")

        if return_derivative:
            return output, history
        else:
            return output * self.dt + shortcut, history


# class LNO(nn.Module):
#     """Transformer-based Latent Neural Operator (TLNO) module.
#     See https://github.com/L-I-M-I-T/LatentNeuralOperator/tree/main

#     A neural operator architecture that combines transformer blocks with latent space
#     projections for learning mappings between function spaces. The model uses attention
#     mechanisms to learn relationships between spatial coordinates and input features,
#     projecting them through a learned latent representation before decoding back to
#     the output space.
#     This implementation uses an adaptive LayerNorm that is conditioned by the current time step

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         d_model (int, optional): Dimension of the model representation. Defaults to 128.
#         modes (int, optional): Number of tokens used to represent the latent space. Defaults to 128.
#         num_heads (int, optional): Number of attention heads in transformer blocks. Defaults to 2.
#         num_blocks (int, optional): Number of transformer blocks to stack. Defaults to 4.
#         mlp_layers (int, optional): Number of layers in MLP blocks. Defaults to 2.
#         mlp_hidden_dim (int, optional): Hidden dimension of MLP blocks. Defaults to 128.
#         dropout (float, optional): Dropout rate. Defaults to 0.
#         transformer_mlp_factor (int, optional): Multiplier for MLP dimension in transformer blocks. Defaults to 4.
#         activation (Callable, optional): Activation function to use. Defaults to nn.GELU.
#         nonlinear: if True, concat the values to the coordinates so that the QK^T kernel is conditionned by the input function
#         dim (int, optional): Spatial dimension (2 for 2D, 3 for 3D, etc.). Defaults to 2.

#     Attributes:
#         trunk_projector (MLPBlock): Projects spatial coordinates to d_model dimension.
#         branch_projector (MLPBlock): Projects input features to d_model dimension.
#         attention_projector (MLPBlock): Projects trunk output to latent dimension for attention scores.
#         ops (nn.ModuleList): Stack of transformer blocks for latent space processing.
#         out_mlp (MLPBlock): Projects decoded output to out_channels dimension."""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         d_model: int = 128,
#         modes: int = 256,
#         num_heads: int = 4,
#         num_blocks: int = 4,
#         mlp_layers: int = 2,
#         mlp_hidden_dim: int = 256,
#         trunk_activation: Callable = nn.GELU,
#         dropout: float = 0,
#         transformer_mlp_factor: int = 4,
#         activation: Callable = nn.GELU,
#         dim: int = 2,
#         cond_dim=None,
#         std_init=0.005,
#     ):

#         super().__init__()

#         self.dim = dim
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.norm = AdaptiveLayerNorm(d_model, d_model)

#         if cond_dim is not None:
#             self.cond_embedding = MLPBlock(
#                 out_ch=d_model,
#                 in_ch=cond_dim,
#                 hidden_dim=mlp_hidden_dim,
#                 num_layers=mlp_layers,
#                 activation=activation,
#             )

#         self.trunk_projector = MLPBlock(
#             in_ch=dim + 1,  # spatial dim + time
#             out_ch=d_model,
#             hidden_dim=mlp_hidden_dim,
#             num_layers=mlp_layers,
#             activation=trunk_activation,
#             norm=None,
#             dropout=dropout,
#         )
#         self.branch_projector = MLPBlock(
#             in_ch=in_channels + dim,
#             out_ch=d_model,
#             hidden_dim=mlp_hidden_dim,
#             num_layers=mlp_layers,
#             activation=activation,
#             norm=None,
#             dropout=dropout,
#         )

#         attention_input_dim = d_model

#         self.attention_projector_forward = MLPBlock(
#             in_ch=attention_input_dim,
#             out_ch=modes,
#             hidden_dim=d_model,
#             num_layers=mlp_layers,
#             activation=trunk_activation,
#             norm=None,
#             dropout=dropout,
#         )

#         self.attention_projector_inverse = MLPBlock(
#             in_ch=attention_input_dim,
#             out_ch=modes,
#             hidden_dim=d_model,
#             num_layers=mlp_layers,
#             activation=trunk_activation,
#             norm=None,
#             dropout=dropout,
#         )

#         # self.proj_temperature = nn.Sequential(
#         #     nn.Linear(d_model, modes), activation(), nn.Linear(modes, 1), activation()
#         # )
#         # self.bias = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)

#         self.ops = nn.ModuleList()

#         for i in range(num_blocks):
#             self.ops.append(
#                 TransformerBlock(
#                     dim=d_model,
#                     n_heads=num_heads,
#                     activation=activation,
#                     mlp_dim=transformer_mlp_factor * d_model,
#                     dropout=dropout,
#                 )
#             )

#         self.out_mlp = MLPBlock(
#             in_ch=d_model,
#             out_ch=out_channels,
#             hidden_dim=mlp_hidden_dim,
#             num_layers=mlp_layers,
#             activation=activation,
#             norm=None,
#             dropout=dropout,
#         )
#         # LNO needs very small weights in the Linear layers to avoid explosion!
#         if std_init is not None:
#             self.apply(partial(self._init_weights, std=std_init))

#     def _init_weights(self, module, std=0.0002):
#         if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=std)
#             if isinstance(module, torch.nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, torch.nn.LayerNorm):
#             module.weight.data.fill_(1.0)
#             module.bias.data.zero_()

#     def forward(
#         self, x: torch.Tensor, time: torch.Tensor, cond: Union[None, torch.Tensor] = None, return_derivative: bool = False
#     ):
#         # takes the last time step
#         if return_derivative:
#             if self.in_channels > self.out_channels:
#                 shortcut = x[:, -self.out_channels :, ...]
#             elif self.in_channels == self.out_channels:
#                 shortcut = x
#             else:
#                 return_derivative = False

#         B, C, H, W = x.shape

#         x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

#         h_coords = torch.linspace(-1, 1, H, device=x.device)
#         w_coords = torch.linspace(-1, 1, W, device=x.device)
#         grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

#         coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
#         coords = coords.view(H * W, -1)
#         coords = coords.unsqueeze(0).expand(B, -1, -1)
#         x = torch.cat([x, coords], dim=-1)

#         t = time.view(B, 1, 1)
#         t = t.expand(B, H * W, 1)
#         coords = torch.cat([coords, t], dim=-1)

#         trunk_output = self.trunk_projector(coords)  # -> [B, H*W, d_model]

#         branch_output = self.branch_projector(x)
#         if cond is not None:
#             cond = self.cond_embedding(cond)
#             if len(cond.shape) == 2:
#                 cond = cond[:, None, :]
#             branch_output = branch_output + cond
#         branch_output = branch_output  # [B, H*W, d_model]

#         # temperature = self.proj_temperature(trunk_output) + self.bias
#         # temperature = torch.clamp(temperature, min=0.01)
#         score_encode = self.attention_projector_forward(trunk_output) / math.sqrt(
#             trunk_output.size(-1)
#         )  # [B, H*W, modes]

#         # score = score / temperature

#         # score_encode = gumbel_softmax(score, temperature, dim=-1)
#         # score_decode = gumbel_softmax(score, temperature, dim=1)
#         score_encode = torch.softmax(score_encode, dim=-1)
#         score_decode = torch.softmax(score_encode, dim=1)
#         z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)

#         for block in self.ops:
#             z = block(z)

#         r = torch.einsum("bij,bjc->bic", score_decode, z)
#         r = self.out_mlp(r)
#         r = r.permute(0, 2, 1).contiguous().reshape(B, self.out_channels, H, W)

#         if return_derivative:
#             return r + shortcut
#         else:
#             return r
