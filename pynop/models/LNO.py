import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import MLPBlock, TransformerBlock
from pynop.core.ops import CartesianEmbedding, sinusoidal_encoding_2d
from pynop.core.norm import AdaptiveLayerNorm
from pynop.core.activations import gumbel_softmax


class ModITLNO(nn.Module):
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
        mlp_factor=4,
        dropout=0,
        sinus_pe_freq=None,
        dim=2,
    ):

        super().__init__()

        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.sinus_pe_freq = sinus_pe_freq

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)
        self.timsetep_embedding = nn.Linear(1, hidden_channels, bias=True)

        self.norm = AdaptiveLayerNorm(hidden_channels, hidden_channels)

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

        self.pe = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.alpha = nn.Parameter(torch.ones(1, hidden_channels, 1, 1))

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
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

        self.trunk.apply(self._init_weights)

    def _init_weights(self, module, std=0.0002):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor, time: torch.Tensor, residual: bool = False):

        if residual:
            shortcut = x

        B, C, H, W = x.shape

        x = self.lifting(x.permute(0, 2, 3, 1))
        # normalisation & time modulation
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        if self.sinus_pe_freq:
            coords = sinusoidal_encoding_2d(coords, self.sinus_pe_freq).view(H, W, -1)  # [H, W, 4 * F]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        time = self.timsetep_embedding(time).unsqueeze(1).unsqueeze(1)
        x = x + time

        x = self.norm(x, time)

        trunk = self.trunk(coords)  # [B, H, W, m1*m2]
        branch = self.branch(x)  # [B, H, W, m1*m2]

        # the kernel is modulated by the signal
        gamma, beta = branch.chunk(2, dim=-1)
        basis = trunk * (1 + F.softsign(gamma)) + beta

        basis = basis.view(B, H, W, self.m1, self.m2) / (H * W)
        # norm_factor = torch.sqrt(torch.mean(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        # basis = basis / (norm_factor + 1e-6)

        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis)
        # Add Positional encoding in latent representation before the self-attention modules
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.pe(PE)
        xhat = xhat + PE
        xhat = xhat.reshape(B, -1, self.m1 * self.m2).contiguous()  # -> m1*m2 tokens

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            xhat = op(xhat, time)

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, basis)
        x_rec = x_rec * self.alpha

        # Mixing channels
        output = self.projection(x_rec)

        if residual:
            return output + shortcut
        else:
            return output


class ITLNO(nn.Module):
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
        mlp_factor=4,
        dropout=0,
        sinus_pe_freq=None,
        dim=2,
        orthogonal_init=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.sinus_pe_freq = sinus_pe_freq

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

        if not linear_kernel:
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

        self.pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        self.alpha = nn.Parameter(torch.ones(1, hidden_channels, 1, 1))

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
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def _init_weights(self, module, std=0.0002):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor, time: torch.Tensor, residual: bool = False):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        x = self.lifting(x.permute(0, 2, 3, 1))
        # normalisation & time modulation
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        if self.sinus_pe_freq:
            coords = sinusoidal_encoding_2d(coords, self.sinus_pe_freq).view(H, W, -1)  # [H, W, 4 * F]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)

        time = self.timsetep_embedding(time).unsqueeze(1).unsqueeze(1)
        x = self.norm(x + time, time)
        time = time.view(B, 1, -1)

        basis_input = torch.cat([x, coords], dim=-1)

        basis = self.generator(basis_input)

        basis = basis.view(B, H, W, self.m1, self.m2) / (H * W)
        # norm_factor = torch.sqrt(torch.mean(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        # basis = basis / (norm_factor + 1e-6)

        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis)
        # Add Positional encoding in latent representation before the self-attention modules
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.pe(PE)
        # print(PE.shape, xhat.shape)
        xhat = xhat + PE
        xhat = xhat.reshape(B, self.m1 * self.m2, -1).contiguous()  # -> m1*m2 tokens

        # Multi-head attention with time conditioning in transformed domain
        for op in self.ops:
            xhat = op(xhat, time)

        # Go back to spatial domain
        out_ch = xhat.shape[-1]
        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, basis)
        x_rec = x_rec * self.alpha

        # Mixing channels
        output = self.projection(x_rec)

        if residual:
            return output + shortcut
        else:
            return output


class TLNO(nn.Module):
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
        dim (int, optional): Spatial dimension (2 for 2D, 3 for 3D or 2D + time, etc.). Defaults to 2.

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
        sinus_pe_freq=None,
        std_init=0.005,
    ):

        super().__init__()

        self.dim = dim
        self.sinus_pe_freq = sinus_pe_freq
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = AdaptiveLayerNorm(d_model, d_model)
        self.trunk_norm = nn.LayerNorm(d_model)
        self.timsetep_embedding = MLPBlock(
            out_ch=d_model,
            in_ch=1,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
        )

        if sinus_pe_freq is not None:
            trunk_in_ch = 4 * sinus_pe_freq
        else:
            trunk_in_ch = 2

        self.trunk_projector = MLPBlock(
            in_ch=trunk_in_ch,
            out_ch=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=trunk_activation,
            norm=None,
            dropout=dropout,
        )
        self.branch_projector = MLPBlock(
            in_ch=in_channels,
            out_ch=d_model,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=None,
            dropout=dropout,
        )

        attention_input_dim = d_model

        self.attention_projector = MLPBlock(
            in_ch=attention_input_dim,
            out_ch=modes,
            hidden_dim=d_model,
            num_layers=mlp_layers,
            activation=trunk_activation,
            norm=None,
            dropout=dropout,
        )

        self.proj_temperature = nn.Sequential(
            nn.Linear(d_model, modes), activation(), nn.Linear(modes, 1), activation()
        )
        self.bias = nn.Parameter(torch.ones([1, 1, 1]) * 0.5)

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
        # LNO needs very small weight in the Linear layers to avoid explosion!
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

    def forward(self, x: torch.Tensor, time: torch.Tensor, residual: bool = False):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        if self.sinus_pe_freq:
            coords = sinusoidal_encoding_2d(coords, self.sinus_pe_freq)  # [H, W, 4 * F]
        coords = coords.view(H * W, -1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        time = self.timsetep_embedding(time)[:, None, :]

        trunk_output = self.trunk_norm(self.trunk_projector(coords))  # -> [B, H*W, d_model]
        branch_output = self.branch_projector(x)
        branch_output = branch_output + time  # [B, H*W, d_model]
        branch_output = self.norm(branch_output, time)

        temperature = self.proj_temperature(trunk_output) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        score = self.attention_projector(trunk_output) / math.sqrt(trunk_output.size(-1))  # [B, H*W, modes]
        score = score / temperature

        # score_encode = gumbel_softmax(score, temperature, dim=-1)
        # score_decode = gumbel_softmax(score, temperature, dim=1)
        score_encode = torch.softmax(score, dim=-1)
        score_decode = torch.softmax(score, dim=1)
        z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)

        for block in self.ops:
            z = block(z, time)

        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        r = r.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        if residual:
            return r + shortcut
        else:
            return r
