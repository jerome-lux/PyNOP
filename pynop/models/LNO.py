from calendar import c
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import (
    MLPBlock,
    SirenBlock,
    TransformerBlock,
    SpatioTemporalTransformer,
    GalerkinTransformerBlock,
    SlicingBlock,
    DeslicingBlock,
    PEBlock,
)
from pynop.core.ops import GalerkinAttention
from pynop.core.norm import AdaptiveLayerNorm, AdaRMSNorm
from pynop.core.utils import ChebyshevBasis, print_stats, add_noise
from pynop.core.loss import MMDLoss


class ITLNO(nn.Module):
    """Latent Nonlinear Operator using self-attention in transformed domain.

    This module performs a forward integral transform using learned bases,
    applies self-attention in the transformed domain, and then reconstructs the output
    in the original spatial domain. It uses real basis functions only.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (Union[int, Sequence[int]]): Number of modes for the integral transform. If int, uses the same
            value for both dimensions. If sequence, uses modes[0] for m1 and modes[1] for m2.
        dt (float): Time step or scaling factor for the integral transform. Default: 1.
        num_blocks (int): Number of transformer blocks in the latent domain. Default: 4.
        hidden_channels (int): Hidden channel dimension for the latent representation. Default: 256.
        num_heads (int): Number of attention heads in transformer blocks. Default: 4.
        linear_kernel (bool): Whether to use only coordinates as basis generator input (True) or
            include input channels (False). Default: True.
        mlp_layers (int): Number of layers in the basis generator and embedding MLPs. Default: 2.
        mlp_dim (int): Hidden dimension for the basis generator and embedding MLPs. Default: 128.
        activation (Callable): Activation function used in transformer blocks. Default: nn.GELU.
        mlp_act (Callable): Activation function used in MLP embedding blocks. Default: nn.GELU.
        mlp_factor (int): Expansion factor for the MLP hidden dimension. Default: 4.
        dropout (float): Dropout probability. Default: 0.
        dim (int): Spatial dimension of the input domain. Default: 2.
        orthogonal_init (bool): Whether to use orthogonal initialization for basis weights. Default: True.
        cond_dim (Union[int, None]): Dimension of optional conditioning input. Default: None.
        basis_mode (str): Type of basis to use for the integral transform. Default: "learned".
        rmsnorm (bool): Whether to use RMS normalization instead of layer norm. Default: True.
        verbose (bool): Whether to print additional model information. Default: False.
        std_ini (float): Standard deviation used for initial weight scaling. Default: 2e-2.
        use_sin_pe (bool): Whether to use sinusoidal positional encoding. Default: False.
    """

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
        mlp_factor: int = 4,
        dropout: float = 0,
        ndim: int = 2,
        orthogonal_init: bool = True,
        cond_dim: Union[int, None] = None,
        basis_mode: str = "learned",
        rmsnorm: bool = True,
        verbose: bool = False,
        std_init=2e-2,
        pe=None,
        pe_param=10,
        cat_coords: bool = True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.ndim = ndim
        self.linear_kernel = linear_kernel
        self.basis_mode = basis_mode
        self.verbose = verbose
        self.pe = pe
        self.cat_coords = cat_coords

        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm

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

        self.pelayer = PEBlock(
            ndim,
            hidden_channels,
            method=pe,
            max_freq_val=pe_param,
            mlp_layers=pe_param,
            mlp_dim=hidden_channels,
            mlp_act=mlp_act,
        )

        if cat_coords:
            self.mixer = MLPBlock(
                out_ch=hidden_channels,
                in_ch=2 * hidden_channels,
                hidden_dim=hidden_channels,
                num_layers=1,
                activation=mlp_act,
            )

        # self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)
        self.norm1 = norm_layer(hidden_channels)
        self.kernel_norm = norm_layer(self.m1 * self.m2)

        in_ch = hidden_channels

        if self.basis_mode == "cheb":
            self.trunk_generator = ChebyshevBasis(self.m1, self.m2)
        else:
            self.trunk_generator = MLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=in_ch,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.branch_generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        if not linear_kernel:
            self.kernel_mixing = MLPBlock(
                out_ch=self.m1 * self.m2,
                in_ch=2 * self.m1 * self.m2,
                hidden_dim=mlp_dim,
                num_layers=1,
                activation=mlp_act,
            )

        if orthogonal_init:
            self.ortho_init_weights(self.trunk_generator)
            self.ortho_init_weights(self.branch_generator, gain=nn.init.calculate_gain("relu"))
            self.ortho_init_weights(self.kernel_mixing, gain=nn.init.calculate_gain("relu"))

        self.latent_pe = MLPBlock(2, hidden_channels, num_layers=1, activation=activation)

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
                    scaling=num_blocks,
                    std_ini=std_init,
                )
            )

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def ortho_init_weights(self, module, gain=1.0):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: Union[None, torch.Tensor] = None):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x)
        x = self.norm1(x)

        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after lifting:")

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)
        pe = self.pelayer(coords)

        kernel = self.trunk_generator(pe).view(1, H * W, self.m1 * self.m2).expand(B, -1, -1)  # B, H*W, M
        kernel = self.kernel_norm(kernel)

        if self.cat_coords:
            x = torch.cat([x, pe.expand(B, -1, -1, -1)], dim=-1)
            x = self.mixer(x)

        x = x.view(B, H * W, -1)

        if self.verbose:
            print_stats(kernel, -1, "kernel coord")

        if not self.linear_kernel:
            signal_kernel = self.branch_generator(x)
            kernel = torch.concat([kernel, signal_kernel], dim=-1)
            if self.verbose:
                print_stats(signal_kernel, -1, "signal kernel")
                print_stats(kernel, -1, "concat kernel")
            kernel = self.kernel_mixing(kernel)  # [B, N, M]

        if self.verbose:
            print_stats(kernel.view(B, H, W, -1), -1, "kernel after mixing:")

        d = torch.norm(kernel, p=2, dim=1, keepdim=True)  # [B, 1, M]

        if self.verbose:
            print_stats(kernel, -1, "kernel:")

        # Forward integral transform
        # print(x.shape, kernel.shape, d.shape)
        xhat = torch.einsum("bnc,bnm->bmc", x, kernel)
        xhat = xhat / d.permute(0, 2, 1)
        xhat = self.latent_projection(xhat)

        if self.verbose:
            print_stats(xhat, -1, "after IT:")

        # Multi-head attention in transformed domain
        for i, op in enumerate(self.ops):
            xhat = op(xhat)
            if self.verbose:
                print_stats(xhat, -1, f"after ATT: module {i}")

        # Go back to spatial domain
        x_rec = torch.einsum("bmc,bnm->bnc", xhat, kernel) / (self.m1 * self.m2)  # ** F.relu(self.IT_scale_inverse)

        if self.verbose:
            print_stats(x_rec, -1, "after REC:")

        # Mixing channels
        x_rec = self.projection(x_rec)
        x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.verbose:
            print_stats(x_rec, 1, "Final:")

        return x_rec


class LatentGalerkin(nn.Module):
    """
    Projection using Galerkin Cross-Attention to M latent tokens.

    This module lifts the input field to a hidden space, projects spatial features onto a
    set of latent tokens using Galerkin cross-attention, refines the latent representation
    through transformer blocks, and decodes it back to the original spatial domain.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (int): Number of latent modes / tokens used for Galerkin projection.
        dt (float): Time step or scaling factor applied when return_derivative=False. Default: 1.0.
        num_blocks (int): Number of transformer blocks applied in latent space. Default: 4.
        hidden_channels (int): Hidden feature dimension for lifted and latent representations. Default: 256.
        num_heads (int): Number of attention heads in Galerkin/Transformer blocks. Default: 4.
        mlp_layers (int): Number of layers in MLP blocks used for conditional and latent embeddings. Default: 2.
        mlp_dim (int): Hidden dimension for the MLP blocks. Default: 128.
        activation (Callable): Activation function used by Transformer blocks. Default: nn.GELU.
        mlp_act (Callable): Activation function used by MLP blocks. Default: nn.GELU.
        mlp_factor (int): Expansion factor for Transformer MLP hidden dimension. Default: 4.
        dropout (float): Dropout probability inside transformer blocks. Default: 0.
        dim (int): Spatial input dimensionality (typically 2 for 2D grids). Default: 2.
        cond_dim (Union[int, None]): Dimension of optional conditioning vector. If provided, a conditioning embedding is used. Default: None.
        latent_attention (str): Type of latent attention block to use, either "galerkin" or any standard transformer. Default: "galerkin".
        verbose (bool): If True, prints intermediate tensor statistics during forward pass. Default: False.
        galerkin_norm (bool): Whether to apply normalization inside Galerkin attention layers. Default: True.
        pe (str): which type of positional encoding
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        num_blocks: int = 4,
        hidden_channels: int = 256,
        num_heads: int = 4,
        activation: Callable = nn.GELU,
        mlp_act=nn.GELU,
        mlp_factor: int = 4,
        dropout: float = 0,
        dim: int = 2,
        cond_dim: Union[int, None] = None,
        latent_attention="galerkin",
        verbose: bool = False,
        galerkin_norm=False,
        rmsnorm: bool = True,
        pe="random_ipe",
        pe_param=10,
        std_init=1e-2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.hidden_channels = hidden_channels
        self.pe_method = pe
        self.verbose = verbose
        self.pe = pe

        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm

        cdim = cond_dim if cond_dim is not None else 0
        self.lifting = nn.Linear(in_channels + cdim, hidden_channels, bias=True)

        if pe is not None:
            self.pelayer = PEBlock(
                dim,
                hidden_channels,
                method=pe,
                max_freq_val=pe_param,
                mlp_layers=pe_param,
                mlp_dim=hidden_channels,
                mlp_act=mlp_act,
            )

            self.mixer = MLPBlock(
                out_ch=hidden_channels,
                in_ch=2 * hidden_channels,
                hidden_dim=hidden_channels,
                num_layers=1,
                activation=mlp_act,
            )

        self.encoder = GalerkinAttention(
            dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=std_init
        )
        self.decoder = GalerkinAttention(
            dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=std_init
        )

        self.forward_queries = nn.Parameter(torch.randn(1, self.modes, self.hidden_channels))

        self.backward_queries_predictor = MLPBlock(
            out_ch=hidden_channels,
            in_ch=hidden_channels,
            hidden_dim=hidden_channels,
            num_layers=1,
            activation=mlp_act,
        )

        self.norm1 = norm_layer(hidden_channels)
        # self.norm2 = nn.RMSNorm(hidden_channels)

        # List of attention modules

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                std_ini=1e-2,
                scaling=num_blocks,
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
                std_ini=1e-2,
                scaling=num_blocks,
            )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def encode(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        if cond is not None:
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = torch.cat([x, cond], dim=-1)

        x = self.lifting(x).view(B, H * W, -1)
        x = self.norm1(x)

        if self.pe is not None:
            h_coords = torch.linspace(-1, 1, H, device=x.device)
            w_coords = torch.linspace(-1, 1, W, device=x.device)
            grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
            coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)  # 1, H, W, 2
            pe = self.pelayer(coords).view(1, H * W, -1)
            pe = pe.expand(B, -1, -1)
            x = torch.cat([x, pe], dim=-1)
            x = self.mixer(x)

        # Galerkin Cross Attention -> Q [B, N, D] K^TV [B, D, D] -> [B, M, C]
        z = self.encoder(x=self.forward_queries.expand(B, -1, -1), context=x)

        if self.verbose:
            print_stats(z, -1, "after IT:")

        # Multi-head attention
        for i, op in enumerate(self.ops):
            z = op(z)
            if self.verbose:
                print_stats(z, -1, f"after ATT module {i+1}:")

        return z, x

    def decode(self, z, q, out_shape):
        B = z.size(0)
        x = self.decoder(x=self.backward_queries_predictor(q), context=z)  # [B, H*W, C]
        H, W = out_shape
        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after REC:")

        x = self.projection(x)
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):
        B, C, H, W = x.shape
        z, q = self.encode(x, cond)
        x_rec = self.decode(z, q, (H, W))
        return x_rec


class LatentTransolver(nn.Module):
    """Using a transolverv3 transform to projet in the latent space and going back to physical space.
    Self-attention blocs are applied in the latent space
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
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
        rmsnorm: bool = True,
        cond_dim: Union[int, None] = None,
        latent_attention="vanilla",
        verbose: bool = False,
        use_gumbel_softmax: bool = False,
        std_init=2e-2,
        # pe="random_ipe",
        # pe_param=10,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.verbose = verbose
        self.hidden_channels = hidden_channels

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        self.encoder = SlicingBlock(hidden_channels, hidden_channels, modes)
        self.decoder = DeslicingBlock(hidden_channels, hidden_channels)

        self.pe = MLPBlock(2, hidden_channels, hidden_dim=hidden_channels)
        self.mixer = nn.Linear(2 * hidden_channels, hidden_channels)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        norm_layer = nn.RMSNorm if rmsnorm else nn.LayerNorm

        self.norm = norm_layer(hidden_channels)
        self.post_norm = norm_layer(hidden_channels)

        # List of attention modules
        self.latent_preprocessing = MLPBlock(
            out_ch=hidden_channels,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
            norm=norm_layer,
        )

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                std_ini=std_init,
                scaling=num_blocks,
            )
        else:
            block = partial(
                TransformerBlock,
                dim=hidden_channels,
                n_heads=num_heads,
                activation=activation,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                rmsnorm=rmsnorm,
                std_ini=std_init,
                scaling=num_blocks,
            )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        # self.projection = nn.Linear(hidden_channels, out_channels, bias=True)
        self.projection = MLPBlock(
            out_ch=out_channels,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=2,
            activation=mlp_act,
            norm=norm_layer,
        )

    def encode(self, x, cond: Union[None, torch.Tensor] = None):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        # PE original grid
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)
        pe = self.pe(coords).expand(B, -1, -1, -1)

        # concat raw coords to input before lifting
        # print_stats(x, -1, "Input:")
        x = self.lifting(x)
        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond
        x = self.norm(x)  # .view(B, H * W, -1)
        x = self.mixer(torch.cat([x, pe], dim=-1).view(B, H * W, -1))

        # print_stats(x, -1, "After Lifting:")

        z, slice_weights = self.encoder(x)
        # print_stats(x, -1, "After forward transform:")
        z = self.latent_preprocessing(z)
        # print_stats(z, -1, "preprocessed latent:")

        # Multi-head attention in latent space
        for i, op in enumerate(self.ops):
            z = op(z)
            # print_stats(z, -1, f"After attention block {i}:")

        return z, slice_weights

    def decode(self, z, slice_weights, out_shape):

        B = z.shape[0]
        # Deslicing
        x = self.decoder(z, slice_weights)
        # print_stats(x, -1, "After inverse transform")
        x = self.post_norm(x)
        # print_stats(x, -1, "After post_norm")
        x = self.projection(x)
        # print_stats(x, -1, "Final")
        H, W = out_shape
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):

        B, C, H, W = x.shape
        z, w = self.encode(x, cond)
        x = self.decode(z, w, (H, W))

        return x


class LatentNO(nn.Module):
    """Transformer-based Latent Neural Operator (TLNO) module.
    See https://github.com/L-I-M-I-T/LatentNeuralOperator/tree/main

    A neural operator architecture that combines transformer blocks with latent space
    projections for learning mappings between function spaces. The model uses attention
    mechanisms to learn relationships between spatial coordinates and input features,
    projecting them through a learned latent representation before decoding back to
    the output space.
    This implementation uses an adaptive LayerNorm that is conditioned by the current time step

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        d_model (int, optional): Dimension of the model representation. Defaults to 128.
        modes (int, optional): Number of tokens used to represent the latent space. Defaults to 128.
        num_heads (int, optional): Number of attention heads in transformer blocks. Defaults to 2.
        num_blocks (int, optional): Number of transformer blocks to stack. Defaults to 4.
        mlp_layers (int, optional): Number of layers in MLP blocks. Defaults to 2.
        mlp_dim (int, optional): Hidden dimension of MLP blocks. Defaults to 128.
        dropout (float, optional): Dropout rate. Defaults to 0.
        transformer_mlp_factor (int, optional): Multiplier for MLP dimension in transformer blocks. Defaults to 4.
        activation (Callable, optional): Activation function to use. Defaults to nn.GELU.
        nonlinear: if True, concat the values to the coordinates so that the QK^T kernel is conditionned by the input function
        dim (int, optional): Spatial dimension (2 for 2D, 3 for 3D, etc.). Defaults to 2.

    Attributes:
        trunk_projector (MLPBlock): Projects spatial coordinates to d_model dimension.
        branch_projector (MLPBlock): Projects input features to d_model dimension.
        attention_projector (MLPBlock): Projects trunk output to latent dimension for attention scores.
        ops (nn.ModuleList): Stack of transformer blocks for latent space processing.
        out_mlp (MLPBlock): Projects decoded output to out_channels dimension."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int = 128,
        modes: int = 256,
        num_heads: int = 4,
        num_blocks: int = 4,
        mlp_layers: int = 1,
        mlp_dim: int = 256,
        dropout: float = 0,
        transformer_mlp_factor: int = 4,
        activation: Callable = nn.GELU,
        dim: int = 2,
        tau_ini: float = 0.1,
        pe="random_fourier",
        pe_param=10,
        cond_dim=None,
        std_init=None,
        rmsnorm: bool = True,
    ):

        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.modes = modes

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=d_model,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=1,
                activation=activation,
            )

        self.pe = self.pelayer = PEBlock(
            dim,
            d_model,
            method=pe,
            max_freq_val=pe_param,
            mlp_layers=pe_param,
            mlp_dim=d_model,
            mlp_act=activation,
        )

        self.trunk_projector = MLPBlock(
            in_ch=d_model,
            out_ch=d_model,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=self.norm,
            dropout=dropout,
        )
        self.branch_projector = MLPBlock(
            in_ch=in_channels,
            out_ch=d_model,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=self.norm,
            dropout=dropout,
        )

        self.tau1 = nn.Parameter(torch.ones(1, 1, modes) * tau_ini)
        self.tau2 = nn.Parameter(torch.ones(1, 1, modes) * tau_ini)

        self.attention_projector_forward = MLPBlock(
            in_ch=d_model,
            out_ch=modes,
            hidden_dim=d_model,
            num_layers=mlp_layers,
            activation=activation,
            norm=self.norm,
            dropout=dropout,
        )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    dim=d_model,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=transformer_mlp_factor * d_model,
                    dropout=dropout,
                    scaling=1 / math.sqrt(2 * num_blocks),
                    std_ini=1e-2,
                )
            )

        self.out_mlp = MLPBlock(
            in_ch=d_model,
            out_ch=out_channels,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=activation,
            norm=None,
            dropout=dropout,
        )

        if std_init is not None:
            self.apply(partial(self._init_weights, std=std_init))

    def _init_weights(self, module, std=0.0002):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def encode(self, x: torch.Tensor, cond: Union[None, torch.Tensor] = None):
        # takes the last time step

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.view(1, H * W, -1)
        # concatenate the raw grid coordinates to input signal

        # use PE to generate the projection matrix
        pe = self.pe(coords)
        trunk_output = self.trunk_projector(pe)  # -> [B, H*W, d_model]
        # score = self.trunk_projector(pe).expand(B, -1, -1)
        # print_stats(trunk_output, -1, "trunk")

        branch_output = self.branch_projector(x)
        # print_stats(branch_output, -1, "branch")
        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            branch_output = branch_output + cond

        # Only one head. Maybe we can use a multi head prohjection ? also why not using a single MLP to compute score from pe?
        score = self.attention_projector_forward(trunk_output)  # / math.sqrt(trunk_output.size(-1))  # [B, H*W, modes]
        # print_stats(score, -1, "score")

        tau1 = F.relu(self.tau1) + 1e-6
        tau2 = F.relu(self.tau2) + 1e-6
        score_encode = torch.softmax(score / tau1, dim=1).expand(B, -1, -1)
        score_decode = torch.softmax(score / tau2, dim=-1).expand(B, -1, -1)
        z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)
        # print_stats(z, -1, "latent")

        for block in self.ops:
            z = block(z)
        # print_stats(z, -1, "latent final")

        return z, score_decode

    def decode(self, z, score_decode, out_coords):
        B, M, C = z.shape
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        # print_stats(r, -1, "rec")
        r = self.out_mlp(r)
        # print_stats(r, -1, "out")
        H, W = out_coords
        r = r.permute(0, 2, 1).contiguous().reshape(B, self.out_channels, H, W)
        return r

    def forward(self, x: torch.Tensor, cond: Union[None, torch.Tensor] = None):
        B, C, H, W = x.shape
        z, score_decode = self.encode(x, cond)
        x = self.decode(z, score_decode, (H, W))
        return x
