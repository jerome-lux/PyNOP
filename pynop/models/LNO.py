import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import MLPBlock, TransformerBlock, SpatioTemporalTransformer, GalerkinTransformerBlock, FNOBlock
from pynop.core.ops import GalerkinAttention
from pynop.core.norm import AdaptiveLayerNorm, AdaRMSNorm
from pynop.core.activations import gumbel_softmax, Sine
from pynop.core.utils import ChebyshevBasis, print_stats, add_noise
from pynop.core.loss import MMDLoss
from pynop.core.encoding import AdaptivePE2d


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
        cond_dim: Union[int, None] = None,
        basis_mode: str = "learned",
        rmsnorm: bool = True,
        verbose: bool = False,
        std_ini=2e-2,
        use_sin_pe=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.basis_mode = basis_mode
        self.dt = dt
        self.verbose = verbose

        self.IT_scale_forward = nn.Parameter(torch.empty(1))
        self.IT_scale_inverse = nn.Parameter(torch.empty(1))
        with torch.no_grad():
            nn.init.constant_(self.IT_scale_forward, 1.1)
            nn.init.constant_(self.IT_scale_inverse, 1.1)

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

        if use_sin_pe:
            self.sin_pe = AdaptivePE2d(hidden_channels)
            pe_input = hidden_channels
        else:
            self.sin_pe = None
            pe_input = 2

        self.pelayer = MLPBlock(
            out_ch=hidden_channels,
            in_ch=pe_input,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.timsetep_embedding = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)

        if use_sin_pe:
            in_ch = hidden_channels
        else:
            in_ch = dim

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
            num_layers=1,
            activation=mlp_act,
        )

        self.basis_mixing = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=2 * self.m1 * self.m2,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        if orthogonal_init:
            self.ortho_init_weights(self.coord_generator)
            self.ortho_init_weights(self.signal_generator, gain=nn.init.calculate_gain("relu"))
            self.ortho_init_weights(self.basis_mixing, gain=nn.init.calculate_gain("relu"))

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

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
                    scaling=1 / math.sqrt(2 * num_blocks),
                    std_ini=std_ini,
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module, gain=1.0):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
        return_derivative: bool = True,
    ):

        if not return_derivative:
            if self.in_channels > self.out_channels:
                input = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                input = x
            else:
                return_derivative = True

        B, C, H, W = x.shape
        x = self.lifting(x.permute(0, 2, 3, 1))
        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after lifting:")

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
        coords = torch.stack([grid_h, grid_w], dim=-1)
        if self.sin_pe is not None:
            coords = self.sin_pe(coords)

        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        basis = self.coord_generator(coords)  # N D

        # [B, H, W, D]
        coords = self.pelayer(coords)

        # adds positional encoding + time conditioning
        x = self.norm1(x + coords, encoded_time)

        if self.verbose:
            print_stats(x.view(B, H * W, -1), -1, "after AdaNorm:")
            print_stats(basis, -1, "basis coord")

        if not self.linear_kernel:
            signal_basis = self.signal_generator(x)
            # basis = basis * (1 + signal_basis)
            basis = torch.concat([basis, signal_basis], axis=-1)
            if self.verbose:
                print_stats(signal_basis, -1, "signal basis")
                print_stats(basis, -1, "concat basis")
            basis = self.basis_mixing(basis)

        # if self.verbose:
        #     print_stats(basis.view(B, H, W, -1), -1, "basis before normalization (along modes):")

        basis = basis.view(B, H, W, self.m1, self.m2)
        # d_norm = torch.sqrt(torch.mean(basis**2, dim=(1, 2), keepdim=True) + 1e-8)
        # d = torch.sum(torch.abs(basis), dim=(1, 2))
        # basis = basis  # / d_norm
        if self.verbose:
            print_stats(basis, -1, "basis:")
        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis) / (H * W) ** torch.sigmoid(self.IT_scale_forward)
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

        # Multi-head attention in transformed domain
        for i, op in enumerate(self.ops):
            xhat = op(xhat)
            if self.verbose:
                print_stats(xhat.view(B, self.m1, self.m2, -1), -1, f"after ATT: module {i}")

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)  # * self.inverse_scale[None, :, :, None]
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, basis) / (self.m1 * self.m2) ** torch.sigmoid(
            self.IT_scale_inverse
        )

        # x_rec = self.norm3(x_rec)
        if self.verbose:
            print_stats(x_rec, 1, "after REC:")

        # Mixing channels and multiplying by the time scaling
        output = self.projection(x_rec)
        output = time_scaling.view(B, self.out_channels, 1, 1) * output

        if self.verbose:
            print_stats(output, 1, "Final:")

        if return_derivative:
            return output
        else:
            return input + output * self.dt


# class ITLNOv3(nn.Module):
#     """Iterative projection ( N -> M1 -> ...- > Mn then (ATT)*L -> Mn-1 -> ... -> M1 -> N)
#     Latent Nonlinear Operator using self-attention in transfomed domain.

#     This module performs a forward integral transform using learned bases,
#     applies self-attention in the transformed domain, and then reconstructs the output
#     in the original spatial domain. It uses real basis functions only.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         modes (Union[int, Sequence[int]]): Number of modes for the integral transform. If int, uses same
#             value for both dimensions. If sequence, uses modes[0] for m1 and modes[1] for m2.
#         hidden_channels (Sequence[int]): Sequence of hidden channel dimensions for each transformer block.
#         num_heads (int): Number of attention heads in transformer blocks. Default: 2.
#         linear_kernel (bool): Whether to use only coordinates as basis generator input (True) or
#             include input channels (False). Default: True.
#         mlp_layers (int): Number of layers in the basis generator MLP. Default: 2.
#         mlp_dim (int): Hidden dimension for the basis generator MLP. Default: 128.
#         activation (Callable): Activation function to use. Default: nn.GELU.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         modes: Sequence[int],
#         dt: float = 1,
#         num_blocks: int = 4,
#         hidden_channels: int = 256,
#         num_heads: int = 4,
#         mlp_layers: int = 1,
#         mlp_dim: int = 128,
#         activation: Callable = nn.GELU,
#         mlp_act=nn.GELU,
#         mlp_factor: int = 4,
#         dropout: float = 0,
#         dim: int = 2,
#         orthogonal_init: bool = True,
#         pe: bool = True,
#         cond_dim: Union[int, None] = None,
#         rmsnorm: bool = True,
#         verbose: bool = False,
#         std_ini=2e-2,
#     ):

#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes
#         self.dim = dim
#         self.pe = pe
#         self.dt = dt
#         self.verbose = verbose

#         self.IT_scale_forward = nn.Parameter(torch.empty(len(modes)))
#         self.IT_scale_inverse = nn.Parameter(torch.empty(len(modes)))
#         with torch.no_grad():
#             nn.init.constant_(self.IT_scale_forward, 1.1)
#             nn.init.constant_(self.IT_scale_inverse, 1.1)

#         self.lifting = nn.Linear(in_channels + dim, hidden_channels, bias=True)
#         self.latent_projection = nn.Linear(hidden_channels, hidden_channels, bias=True)

#         if cond_dim is not None:
#             self.cond_embedding = MLPBlock(
#                 out_ch=hidden_channels,
#                 in_ch=cond_dim,
#                 hidden_dim=mlp_dim,
#                 num_layers=mlp_layers,
#                 activation=mlp_act,
#             )

#         self.timsetep_embedding = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=1,
#             hidden_dim=mlp_dim,
#             num_layers=1,
#             activation=mlp_act,
#         )

#         self.time_scaling = MLPBlock(
#             out_ch=out_channels,
#             in_ch=1,
#             hidden_dim=mlp_dim,
#             num_layers=1,
#             activation=mlp_act,
#         )

#         self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)

#         # Basis generator: K = K_coord + K_signal
#         self.coord_generator = nn.ModuleList(
#             [
#                 MLPBlock(
#                     out_ch=m**2,
#                     in_ch=dim,
#                     hidden_dim=mlp_dim,
#                     num_layers=mlp_layers,
#                     activation=mlp_act,
#                 )
#                 for m in modes
#             ]
#         )

#         self.signal_generator = nn.ModuleList(
#             [
#                 MLPBlock(
#                     out_ch=m**2,
#                     in_ch=hidden_channels,
#                     hidden_dim=mlp_dim,
#                     num_layers=1,
#                     activation=mlp_act,
#                 )
#                 for m in modes
#             ]
#         )

#         self.basis_mixing = nn.ModuleList(
#             [
#                 MLPBlock(
#                     out_ch=m**2,
#                     in_ch=2 * m**2,
#                     hidden_dim=mlp_dim,
#                     num_layers=1,
#                     activation=mlp_act,
#                 )
#                 for m in modes
#             ]
#         )

#         if orthogonal_init:
#             self.ortho_init_weights(self.coord_generator)
#             self.ortho_init_weights(self.signal_generator, gain=nn.init.calculate_gain("relu"))
#             self.ortho_init_weights(self.basis_mixing, gain=nn.init.calculate_gain("relu"))

#         self.latent_pe = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=2,
#             hidden_dim=mlp_dim,
#             num_layers=mlp_layers,
#             activation=mlp_act,
#         )

#         # List of attention modules
#         self.ops = nn.ModuleList()

#         for i in range(num_blocks):
#             self.ops.append(
#                 TransformerBlock(
#                     dim=hidden_channels,
#                     n_heads=num_heads,
#                     activation=activation,
#                     mlp_dim=mlp_factor * hidden_channels,
#                     dropout=dropout,
#                     rmsnorm=rmsnorm,
#                     scaling=1 / math.sqrt(2 * num_blocks),
#                     std_ini=std_ini,
#                 )
#             )

#         self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

#     def ortho_init_weights(self, module, gain=1.0):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(
#         self,
#         x: torch.Tensor,
#         time: torch.Tensor,
#         cond: Union[None, torch.Tensor] = None,
#         return_derivative: bool = True,
#     ):

#         if not return_derivative:
#             if self.in_channels > self.out_channels:
#                 shortcut = x[:, -self.out_channels :, ...]
#             elif self.in_channels == self.out_channels:
#                 shortcut = x
#             else:
#                 return_derivative = True

#         B, C, H, W = x.shape

#         h_coords = torch.linspace(-1, 1, H, device=x.device)
#         w_coords = torch.linspace(-1, 1, W, device=x.device)
#         grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
#         coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0)  # [B, H, W, 2]

#         x = x.permute(0, 2, 3, 1)  # [B, H, W, Cin]
#         x = self.lifting(torch.cat([x, coords.expand(B, -1, -1, -1)], dim=-1))

#         if self.verbose:
#             print_stats(x.view(B, H * W, -1), -1, "after lifting:")

#         time_scaling = F.softplus(self.time_scaling(time))
#         encoded_time = self.timsetep_embedding(time)

#         if cond is not None:
#             cond = self.cond_embedding(cond)
#             if len(cond.shape) == 2:
#                 cond = cond[:, None, None, :]
#             x = x + cond

#         # adds temporal conditionning
#         xhat = self.norm1(x, encoded_time).view(B, H * W, -1)
#         coords = coords.view(H * W, 2)
#         basis_list = []
#         # Encoding using successive integral trasnforms
#         for i, mode in enumerate(self.modes):
#             coord_basis = self.coord_generator[i](coords).expand(B, -1, -1)  # B (M_{i-1})^2 (M_i)^2
#             signal_basis = self.signal_generator[i](xhat)
#             basis = torch.concat([coord_basis, signal_basis], dim=-1)
#             basis = self.basis_mixing[i](basis)
#             basis_list.append(basis)
#             xhat = torch.einsum("bnc,bnm->bmc", xhat, basis) / (H * W) ** torch.sigmoid(self.IT_scale_forward[i])
#             if self.verbose:
#                 print_stats(signal_basis, -1, "signal basis")
#                 print_stats(coord_basis, -1, "coord_basis")
#                 print_stats(basis, -1, "basis")
#                 print_stats(xhat, -1, f"Ater IT {i+1}")
#             # Update coords
#             h_coords = torch.linspace(-1, 1, mode, device=x.device)
#             w_coords = torch.linspace(-1, 1, mode, device=x.device)
#             grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
#             coords = torch.stack([grid_h, grid_w], dim=-1).view(mode * mode, 2).unsqueeze(0)

#         # Add Positional encoding in latent representation before the self-attention modules
#         PE = self.latent_pe(coords)
#         xhat = xhat + PE
#         xhat = xhat

#         # Multi-head attention in transformed domain
#         for i, op in enumerate(self.ops):
#             xhat = op(xhat)
#             if self.verbose:
#                 print_stats(xhat, -1, f"after ATT: module {i}")

#         # Decoder
#         basis_list.reverse()
#         for i, mode in enumerate(self.modes[::-1]):
#             xhat = torch.einsum("bmc,bnm->bnc", xhat, basis_list[i]) / (mode**2) ** torch.sigmoid(
#                 self.IT_scale_inverse[i]
#             )

#         if self.verbose:
#             print_stats(xhat, -1, "after REC:")

#         # Mixing channels and multiplying by the time scaling
#         output = self.projection(xhat)
#         output = output.view(B, H, W, -1)
#         output = output.permute(0, 3, 1, 2)
#         output = time_scaling.view(B, self.out_channels, 1, 1) * output

#         if self.verbose:
#             print_stats(output, 1, "Final:")

#         if return_derivative:
#             return output
#         else:
#             return shortcut + output * self.dt


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
        scaling (float): Scaling factor passed to latent transformer blocks. Default: 1.
        use_sin_pe (bool): Whether to use sinusoidal positional encoding for query generation. Default: False.
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
        galerkin_norm=True,
        scaling=1,
        use_sin_pe=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.dt = dt
        self.verbose = verbose
        self.hidden_channels = hidden_channels

        self.lifting = nn.Linear(in_channels + dim, hidden_channels, bias=True)

        self.encoder = GalerkinAttention(
            dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=1e-2
        )
        self.decoder = GalerkinAttention(
            dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=1e-2
        )

        if use_sin_pe:
            self.sin_pe = AdaptivePE2d(hidden_channels)
            queries_dim = hidden_channels
        else:
            self.sin_pe = None
            queries_dim = 2

        self.queries_predictor = MLPBlock(
            out_ch=hidden_channels,
            in_ch=queries_dim,
            hidden_dim=hidden_channels,
            num_layers=1,
            activation=mlp_act,
        )

        # self.latents = nn.Parameter(torch.randn(1, modes, hidden_channels))

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
            num_layers=1,
            activation=mlp_act,
        )

        self.time_scaling = MLPBlock(
            out_ch=out_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
        )

        self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)
        self.norm2 = nn.RMSNorm(hidden_channels)

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=1,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        # List of attention modules

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                std_ini=1e-2,
                scaling=scaling,
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
                scaling=scaling,
            )

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        self.rec_projection = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

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
        x = x.permute(0, 2, 3, 1)

        time_scaling = F.softplus(self.time_scaling(time))
        time_encoding = self.timsetep_embedding(time)

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        # adds coords, temporal encoding and time conditionning
        x = self.lifting(torch.cat([x, coords], dim=-1).view(B, H * W, -1))

        if self.verbose:
            print_stats(x, -1, "x after lifting:")
        x = self.norm1(x, time_encoding)
        if self.verbose:
            print_stats(x, -1, "after AdaNorm:")

        # Add Positional encoding in latent representation before the self-attention modules
        m_coords = torch.linspace(-1, 1, steps=self.modes, device=x.device).unsqueeze(-1)
        m_coords = m_coords.unsqueeze(0).expand(B, -1, -1)
        latent_pe = self.latent_pe(m_coords)

        m_coords = torch.linspace(-1, 1, int(math.sqrt(self.modes)), device=x.device)
        m1, m2 = torch.meshgrid(m_coords, m_coords, indexing="ij")
        q_coords = torch.stack([m1, m2], dim=-1)

        if self.sin_pe is not None:
            q_coords = self.sin_pe(q_coords)
            q_coords = q_coords.view(self.modes, self.hidden_channels)
        else:
            q_coords = q_coords.view(self.modes, 2)
        q_coords = q_coords.unsqueeze(0).expand(B, -1, -1)
        queries = self.queries_predictor(q_coords)
        # Galerkin Cross Attention -> [B, M, C]
        xhat = self.encoder(x=queries, context=x)

        if self.verbose:
            print_stats(xhat, -1, "after IT:")

        # xhat = self.norm2(xhat)
        if self.verbose:
            print_stats(xhat, -1, "after xhat norm:")

        xhat = xhat + latent_pe

        if self.verbose:
            print_stats(xhat, -1, "after xhat + lpe:")

        # Multi-head attention
        for i, op in enumerate(self.ops):
            xhat = op(xhat)
            if self.verbose:
                print_stats(xhat, -1, f"after ATT module {i+1}:")

        # Go back to spatial domain
        x_rec = self.decoder(x=self.rec_projection(x), context=xhat)  # [B, H*W, C]

        if self.verbose:
            print_stats(x_rec.view(B, H * W, -1), -1, "after REC:")

        x_rec = self.projection(x_rec) * time_scaling.view(B, 1, self.out_channels)
        x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.verbose:
            print_stats(x_rec, 1, "Final:")

        if return_derivative:
            return x_rec
        else:
            return x_rec * self.dt + shortcut


# class LatentGalerkinv3(nn.Module):
#     """Iterative Projections using Galerkin Cross-Attention to M latent tokens"""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         modes: Sequence,
#         dt: float = 1.0,
#         num_blocks: int = 4,
#         hidden_channels: int = 256,
#         num_heads: int = 4,
#         mlp_layers: int = 2,
#         mlp_dim: int = 128,
#         activation: Callable = nn.GELU,
#         mlp_act=nn.GELU,
#         mlp_factor: int = 4,
#         dropout: float = 0,
#         dim: int = 2,
#         cond_dim: Union[int, None] = None,
#         latent_attention="galerkin",
#         verbose: bool = False,
#         galerkin_norm=True,
#         scaling=1,
#     ):

#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes
#         self.dim = dim
#         self.dt = dt
#         self.verbose = verbose

#         self.lifting = nn.Linear(in_channels + dim, hidden_channels, bias=True)

#         self.encoder = nn.ModuleList(
#             [
#                 GalerkinAttention(dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=1e-1)
#                 for _ in modes
#             ]
#         )
#         self.decoder = nn.ModuleList(
#             [
#                 GalerkinAttention(dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=1e-1)
#                 for _ in modes
#             ]
#         )

#         self.queries_predictors = nn.ModuleList(
#             [
#                 MLPBlock(
#                     out_ch=hidden_channels,
#                     in_ch=dim,
#                     hidden_dim=hidden_channels,
#                     num_layers=1,
#                     activation=mlp_act,
#                 )
#                 for _ in modes
#             ]
#         )

#         if cond_dim is not None:
#             self.cond_embedding = MLPBlock(
#                 out_ch=hidden_channels,
#                 in_ch=cond_dim,
#                 hidden_dim=mlp_dim,
#                 num_layers=mlp_layers,
#                 activation=mlp_act,
#             )

#         self.pelayer = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=dim,
#             hidden_dim=mlp_dim,
#             num_layers=mlp_layers,
#             activation=mlp_act,
#         )

#         self.timsetep_embedding = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=1,
#             hidden_dim=mlp_dim,
#             num_layers=1,
#             activation=mlp_act,
#         )

#         self.norm1 = AdaRMSNorm(hidden_channels, hidden_channels)
#         self.norm2 = nn.RMSNorm(hidden_channels)

#         self.latent_pe = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=2,
#             hidden_dim=mlp_dim,
#             num_layers=mlp_layers,
#             activation=mlp_act,
#         )

#         # List of attention modules

#         if latent_attention.lower() == "galerkin":
#             block = partial(
#                 GalerkinTransformerBlock,
#                 dim=hidden_channels,
#                 heads=num_heads,
#                 mlp_dim=mlp_factor * hidden_channels,
#                 dropout=dropout,
#                 std_ini=1e-2,
#                 scaling=scaling,
#             )
#         else:
#             block = partial(
#                 TransformerBlock,
#                 dim=hidden_channels,
#                 n_heads=num_heads,
#                 activation=activation,
#                 mlp_dim=mlp_factor * hidden_channels,
#                 dropout=dropout,
#                 rmsnorm=True,
#                 std_ini=1e-2,
#                 scaling=scaling,
#             )

#         self.ops = nn.ModuleList()

#         for i in range(num_blocks):
#             self.ops.append(block())

#         self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

#     def ortho_init_weights(self, module, gain=1.0):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def kaiming_init(self, module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def normal_init(self, module, std: float = 1):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=std)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def xavier_init(self, module, gain: float = 1):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(
#         self,
#         x: torch.Tensor,
#         time: torch.Tensor,
#         cond: Union[None, torch.Tensor] = None,
#         return_derivative: bool = True,
#     ):

#         if not return_derivative:
#             if self.in_channels > self.out_channels:
#                 shortcut = x[:, -self.out_channels :, ...]
#             elif self.in_channels == self.out_channels:
#                 shortcut = x
#             else:
#                 return_derivative = True

#         B, C, H, W = x.shape
#         x = x.permute(0, 2, 3, 1)

#         time_encoding = self.timsetep_embedding(time)

#         h_coords = torch.linspace(-1, 1, H, device=x.device)
#         w_coords = torch.linspace(-1, 1, W, device=x.device)
#         grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
#         coords = torch.stack([grid_h, grid_w], dim=-1)
#         coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
#         pe = self.pelayer(coords).view(B, H * W, -1)

#         if cond is not None:
#             cond = self.cond_embedding(cond)
#             if len(cond.shape) == 2:
#                 cond = cond[:, None, None, :]
#             x = x + cond

#         # adds coords, temporal encoding and time conditionning
#         x = self.lifting(torch.cat([x, coords], dim=-1).view(B, H * W, -1))

#         if self.verbose:
#             print_stats(x, -1, "x after lifting:")
#             print_stats(pe, -1, "pe:")
#         x = self.norm1(x + time_encoding[:, None, :], time_encoding)
#         if self.verbose:
#             print_stats(x, -1, "after AdaNorm:")

#         # Encoding
#         xhat = x
#         decoder_queries = []
#         for i, mode in enumerate(self.modes):
#             decoder_queries.append(xhat)
#             # Generate M=mode*mode queries using coordinates
#             m_coords = torch.linspace(-1, 1, mode, device=x.device)
#             m1, m2 = torch.meshgrid(m_coords, m_coords, indexing="ij")
#             q_coords = torch.stack([m1, m2], dim=-1).view(mode**2, 2)
#             q_coords = q_coords.unsqueeze(0)
#             queries = self.queries_predictors[i](q_coords).expand(B, -1, -1)
#             # Galerkin Cross Attention -> [B, M, C]
#             xhat = self.encoder[i](x=queries, context=xhat)
#             if self.verbose:
#                 print_stats(xhat, -1, f"after IT pass {i+1}:")

#         # Attention in Latent space
#         # Add Positional encoding in latent representation before the self-attention modules
#         latent_pe = self.latent_pe(q_coords)
#         xhat = xhat + latent_pe

#         # Multi-head attention
#         for i, op in enumerate(self.ops):
#             xhat = op(xhat)
#             if self.verbose:
#                 print_stats(xhat, -1, f"after ATT module {i+1}:")

#         # decoder
#         decoder_queries.reverse()
#         for i, mode in enumerate(self.modes[::-1]):
#             xhat = self.decoder[i](x=decoder_queries[i], context=xhat)
#             if self.verbose:
#                 print_stats(xhat, -1, f"after REC pass {i+1}:")

#         x_rec = self.projection(xhat)
#         x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         if self.verbose:
#             print_stats(x_rec, 1, "Final:")

#         if return_derivative:
#             return x_rec
#         else:
#             return x_rec * self.dt + shortcut


# class TimeAttentionLatentGalerkin(nn.Module):
#     """
#     This module combines cross-attention mechanisms with temporal transformers to project input data
#     onto a set of learnable latent tokens. It uses Galerkin-type attention for efficient integration
#     of information across spatial and temporal dimensions.

#     The forward pass:
#     1. Lifts input channels to a higher-dimensional hidden space
#     2. Encodes spatial information through positional encoding (SIREN)
#     3. Applies Galerkin cross-attention to project onto latent tokens
#     4. Processes latent representations through spatio-temporal transformer blocks
#     5. Decodes back to the original spatial resolution
#     6. Projects to output channels

#         modes (int): Number of latent modes/tokens for Galerkin projection.
#         dt (float): Time step scaling factor for derivative computation. Default: 1.0.
#         num_blocks (int): Number of spatial transformer blocks to apply. Default: 4.
#         num_temporal_block (int): Number of layers in the temporal transformer. Default: 2.
#         hidden_channels (int): Dimension of hidden representations. Default: 256.
#         num_heads (int): Number of attention heads in transformer blocks. Default: 4.
#         mlp_layers (int): Number of layers in MLP blocks for embeddings. Default: 2.
#         mlp_dim (int): Hidden dimension of MLP blocks. Default: 128.
#         activation (Callable): Activation function for transformer blocks. Default: nn.GELU.
#         mlp_act (Callable): Activation function for MLP blocks. Default: nn.GELU.
#         mlp_factor (int): Multiplier for MLP hidden dimensions. Default: 4.
#         dropout (float): Dropout probability. Default: 0.
#         dim (int): Spatial dimensionality (2 for 2D grids). Default: 2.
#         cond_dim (Union[int, None]): Dimension of conditional input, or None if not used. Default: None.
#         latent_attention (str): Type of attention mechanism for latent processing ("galerkin" or "standard"). Default: "galerkin".
#         verbose (bool): Whether to print intermediate statistics during forward pass. Default: False.
#         memory (int): Number of past time steps to maintain in temporal history. Default: 6.

#     Attributes:
#         history (list): Buffer storing past latent representations for temporal processing.
#         latents (nn.Parameter): Learnable latent token embeddings of shape (1, modes, hidden_channels).
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         modes: int,
#         dt: float = 1.0,
#         num_blocks: int = 4,
#         num_temporal_block: int = 2,
#         hidden_channels: int = 256,
#         num_heads: int = 4,
#         mlp_layers: int = 2,
#         mlp_dim: int = 128,
#         activation: Callable = nn.GELU,
#         mlp_act=nn.GELU,
#         mlp_factor: int = 4,
#         dropout: float = 0,
#         dim: int = 2,
#         cond_dim: Union[int, None] = None,
#         latent_attention="galerkin",
#         verbose: bool = False,
#         memory: int = 6,
#         galerkin_delta=1e-2,
#         std=0.02,
#     ):

#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes = modes
#         self.dim = dim
#         self.dt = dt
#         self.verbose = verbose
#         self.memory = memory
#         self.hidden_channels = hidden_channels
#         self.num_blocks = num_blocks
#         self.std = std

#         self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

#         self.encoder = GalerkinAttention(dim=hidden_channels, heads=num_heads, delta=galerkin_delta, std_ini=2e-2)
#         self.decoder = GalerkinAttention(dim=hidden_channels, heads=num_heads, delta=galerkin_delta, std_ini=2e-2)
#         # nn.init.trunc_normal_(self.decoder.out_proj.weight, std=0.1)
#         # self.latents = nn.Parameter(torch.randn(1, modes, hidden_channels))
#         self.latents = nn.Parameter(torch.empty(1, modes, hidden_channels))
#         nn.init.trunc_normal_(self.latents, std=1)
#         self.pe_scaling = nn.Parameter(torch.empty(1, 1, hidden_channels))
#         self.latent_pe_scaling = nn.Parameter(torch.empty(1, 1, hidden_channels))
#         nn.init.constant_(self.pe_scaling, 1)
#         nn.init.constant_(self.latent_pe_scaling, 1)

#         if cond_dim is not None:
#             self.cond_embedding = MLPBlock(
#                 out_ch=hidden_channels,
#                 in_ch=cond_dim,
#                 hidden_dim=mlp_dim,
#                 num_layers=mlp_layers,
#                 activation=mlp_act,
#             )

#             self.xavier_init(self.cond_embedding)

#         self.pelayer = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=dim,
#             hidden_dim=mlp_dim,
#             num_layers=mlp_layers,
#             activation=Sine,
#         )

#         self.norm1 = nn.RMSNorm(hidden_channels)
#         self.norm2 = nn.RMSNorm(hidden_channels)

#         self.latent_pe = MLPBlock(
#             out_ch=hidden_channels,
#             in_ch=1,
#             hidden_dim=mlp_dim,
#             num_layers=mlp_layers,
#             activation=Sine,
#         )

#         # VAE mu and sigma
#         self.fc_mu = nn.Linear(hidden_channels, hidden_channels)
#         # self.fc_logvar = nn.Linear(hidden_channels, hidden_channels)

#         # List of attention modules
#         # To limit the increase o fthe varia,ce after each block,
#         # the out projection of each attention block is initialized using a very small std

#         if latent_attention.lower() == "galerkin":
#             block = partial(
#                 GalerkinTransformerBlock,
#                 dim=hidden_channels,
#                 heads=num_heads,
#                 mlp_dim=mlp_factor * hidden_channels,
#                 dropout=dropout,
#                 delta=galerkin_delta,
#                 std_ini=1e-2,
#                 scaling=1 / math.sqrt(2 * num_blocks),
#             )
#         else:
#             block = partial(
#                 TransformerBlock,
#                 dim=hidden_channels,
#                 n_heads=num_heads,
#                 activation=activation,
#                 mlp_dim=mlp_factor * hidden_channels,
#                 dropout=dropout,
#                 rmsnorm=True,
#                 std_ini=1e-2,
#                 scaling=1 / math.sqrt(2 * num_blocks),
#             )

#         self.ops = nn.ModuleList()

#         for i in range(num_blocks):
#             self.ops.append(block())

#         self.temporal_block = SpatioTemporalTransformer(
#             dim=hidden_channels,
#             n_heads=num_heads,
#             mlp_factor=mlp_factor,
#             num_layers=num_temporal_block,
#             max_history=self.memory,
#             std_ini=1e-1,
#         )
#         # self.temporal_block = LatentTemporalMLP(dim=hidden_channels, max_history=self.memory)

#         self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

#         self.MMD = MMDLoss()

#         self.xavier_init(self.lifting, gain=1.414)
#         # self.normal_init(self.fc_logvar, std=1e-3)
#         # nn.init.constant_(self.fc_logvar.bias, -5)
#         self.normal_init(self.fc_mu, std=2e-2)

#     def ortho_init_weights(self, module, gain=1.0):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.orthogonal_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def kaiming_init(self, module, std: float = 1):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def normal_init(self, module, std: float = 1):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=std)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def xavier_init(self, module, gain: float = 1):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=gain)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def reparameterize(self, mu, logvar):
#         torch.clamp(logvar, min=-10, max=3)
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(
#         self,
#         x: torch.Tensor,
#         time: torch.Tensor,
#         history: torch.Tensor = None,
#         cond: Union[None, torch.Tensor] = None,
#         timeattention: bool = True,
#         sampling: bool = True,
#         latent_noise: float = 0,
#     ):

#         B, C, H, W = x.shape

#         if self.verbose:
#             print_stats(x, 1, "Input:")

#         x = self.lifting(x.permute(0, 2, 3, 1))

#         if self.verbose:
#             print_stats(x, -1, "after lifting:")

#         h_coords = torch.linspace(-1, 1, H, device=x.device)
#         w_coords = torch.linspace(-1, 1, W, device=x.device)
#         grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
#         coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
#         pe = self.pelayer(coords) * self.pe_scaling

#         if cond is not None:
#             cond = self.cond_embedding(cond)
#             if len(cond.shape) == 2:
#                 cond = cond[:, None, None, :]
#             x = x + cond

#         x = self.norm1(x)
#         x = x + pe
#         x = x.view(B, H * W, -1)

#         if self.verbose:
#             print("pe_scaling", self.pe_scaling.mean())
#             print_stats(pe, -1, "pe:")
#             print_stats(x, -1, "after Norm1:")

#         # Add Positional encoding in latent representation before the self-attention modules (SIREN)
#         m_coords = torch.linspace(-1, 1, steps=self.modes, device=x.device).unsqueeze(-1)
#         m_coords = m_coords.unsqueeze(0).expand(B, -1, -1)
#         latent_pe = self.latent_pe(m_coords)
#         latents = self.latents.expand((B, -1, -1))

#         # Galerkin Cross Attention -> [B, M, C]
#         xhat = self.encoder(x=latents, context=x)
#         # xhat = self.norm2(xhat)

#         if self.verbose:
#             print_stats(xhat, -1, "after IT:")

#         xhat = xhat + self.latent_pe_scaling * latent_pe

#         if self.verbose:
#             print_stats(xhat, -1, "after xhat + pe:")

#         # Multi-head attention
#         for i, op in enumerate(self.ops):
#             xhat = op(xhat)
#             # xhat = op(xhat)
#             if self.verbose:
#                 print_stats(xhat, -1, f"after ATT module {i+1}:")

#         # xhat = self.norm2(xhat)
#         mu = self.fc_mu(xhat)
#         # logvar = self.fc_logvar(xhat)

#         # sampling
#         if sampling:
#             # z = self.reparameterize(mu, logvar)
#             z = mu + self.std * torch.randn_like(mu)
#         else:
#             z = mu

#         if self.verbose:
#             print_stats(mu, -1, "Mu:")
#             # print_stats(logvar, -1, "logvar:")
#             print_stats(z, -1, "Z after sampling:")

#         # Return KL loss
#         if self.training:
#             # ae_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#             # ae_loss = torch.mean(torch.sum(ae_loss, dim=[1, 2])) / (self.modes * self.hidden_channels)
#             ae_loss = self.MMD(z)
#             z_flat = z.view(B, -1)
#             # ae_loss = (z_flat.mean(dim=0) ** 2).mean() + F.mse_loss(z_flat.std(dim=0), torch.ones_like(z_flat.std(dim=0)))
#         else:
#             ae_loss = 0

#         if timeattention:
#             z, history = self.temporal_block(z, history)

#             # if latent_noise > 0:
#             #     noisy_z = add_noise(z, latent_noise, max_val=1e-2)
#             #     noisy_pred, _ =

#         # Go back to spatial domain
#         # x_rec = self.decoder(x, context=z)
#         x_rec = self.decoder(pe.view(B, H * W, -1), context=z)

#         if self.verbose:
#             print_stats(x_rec.view(B, H * W, -1), -1, "after REC:")

#         # Mixing channels and multiplying by the time scaling
#         x_rec = self.projection(x_rec)

#         if self.verbose:
#             print_stats(x_rec, -1, "Final:")
#         x_rec = x_rec.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         return x_rec, history, ae_loss


class LNO(nn.Module):
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
        mlp_hidden_dim (int, optional): Hidden dimension of MLP blocks. Defaults to 128.
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
        mlp_layers: int = 2,
        mlp_hidden_dim: int = 256,
        trunk_activation: Callable = nn.GELU,
        dropout: float = 0,
        transformer_mlp_factor: int = 4,
        activation: Callable = nn.GELU,
        dim: int = 2,
        cond_dim=None,
        std_init=0.005,
    ):

        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = AdaptiveLayerNorm(d_model, d_model)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=d_model,
                in_ch=cond_dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_layers,
                activation=activation,
            )

        self.trunk_projector = MLPBlock(
            in_ch=dim + 1,  # spatial dim + time
            out_ch=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=trunk_activation,
            norm=None,
            dropout=dropout,
        )
        self.branch_projector = MLPBlock(
            in_ch=in_channels + dim,
            out_ch=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )

        attention_input_dim = d_model

        self.attention_projector_forward = MLPBlock(
            in_ch=attention_input_dim,
            out_ch=modes,
            hidden_dim=d_model,
            num_layers=mlp_layers,
            activation=trunk_activation,
            norm=None,
            dropout=dropout,
        )

        self.attention_projector_inverse = MLPBlock(
            in_ch=attention_input_dim,
            out_ch=modes,
            hidden_dim=d_model,
            num_layers=mlp_layers,
            activation=trunk_activation,
            norm=None,
            dropout=dropout,
        )

        # self.proj_temperature = nn.Sequential(
        #     nn.Linear(d_model, modes), activation(), nn.Linear(modes, 1), activation()
        # )
        # self.bias = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    dim=d_model,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=transformer_mlp_factor * d_model,
                    dropout=dropout,
                )
            )

        self.out_mlp = MLPBlock(
            in_ch=d_model,
            out_ch=out_channels,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )
        # LNO needs very small weights in the Linear layers to avoid explosion!
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

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
        return_derivative: bool = False,
    ):
        # takes the last time step
        if return_derivative:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                return_derivative = False

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.view(H * W, -1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, coords], dim=-1)

        t = time.view(B, 1, 1)
        t = t.expand(B, H * W, 1)
        coords = torch.cat([coords, t], dim=-1)

        trunk_output = self.trunk_projector(coords)  # -> [B, H*W, d_model]

        branch_output = self.branch_projector(x)
        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            branch_output = branch_output + cond
        branch_output = branch_output  # [B, H*W, d_model]

        # temperature = self.proj_temperature(trunk_output) + self.bias
        # temperature = torch.clamp(temperature, min=0.01)
        score_encode = self.attention_projector_forward(trunk_output) / math.sqrt(
            trunk_output.size(-1)
        )  # [B, H*W, modes]

        score_encode = torch.softmax(score_encode, dim=-1)
        score_decode = torch.softmax(score_encode, dim=1)
        z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)

        for block in self.ops:
            z = block(z)

        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        r = r.permute(0, 2, 1).contiguous().reshape(B, self.out_channels, H, W)

        if return_derivative:
            return r + shortcut
        else:
            return r
