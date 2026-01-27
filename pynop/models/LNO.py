import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import MLPBlock, TransformerBlock
from pynop.core.ops import CartesianEmbedding, sin_positional_encoding_2d
from pynop.core.loss import ortho_loss
from pynop.core.norm import AdaptiveLayerNorm
from pynop.core.activations import gumbel_softmax


class ITLNO(nn.Module):
    """Integral Transform-based Latent Nonlinear Operator (ITLNO) using self-attention in transfomed domain.

    This module performs a forward integral transform using learned bases,
    applies self-attention in the transformed domain, and then reconstructs the output
    in the original spatial domain. It uses real basis functions only.

    Attributes:
        fixed_pos_encoding (bool): Whether to use fixed positional encoding via Cartesian embedding.
        m1 (int): Number of modes in the first spatial dimension.
        m2 (int): Number of modes in the second spatial dimension.
        compute_orth_loss (bool): Whether to compute orthogonality loss during forward pass.
        ortho_loss_sampling (int): Number of samples for orthogonality loss computation.
        grid_encoding (CartesianEmbedding): Cartesian grid embedding layer (if fixed_pos_encoding=True).
        pos_embedding_weights (nn.Parameter): Learnable positional embeddings for transformed domain.
        lifting (nn.Conv2d): 1x1 convolution to lift input to hidden channel dimension.
        basis_generator (MLPBlock): MLP network that generates integral transform basis vectors.
        alpha (nn.Parameter): Learnable scaling parameter for the reconstructed output.
        ops (nn.ModuleList): List of TransformerBlock modules for self-attention operations.
        projection (nn.Conv2d): 1x1 convolution to project from hidden channels to output channels.

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
        fixed_pos_encoding (bool): Whether to use fixed Cartesian positional encoding. Default: True.
        compute_ortho_loss (bool): Whether to compute orthogonality loss for the basis. Default: True.
        orth_loss_sampling (int): Number of samples for orthogonality loss. Default: 2048."""

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
        mlp_activation=nn.GELU,
        dropout=0,
        compute_ortho_loss: bool = True,
        orth_loss_sampling: int = 2048,
        dim=2,
    ):

        super().__init__()

        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.compute_ortho_loss = compute_ortho_loss
        self.ortho_loss_sampling = orth_loss_sampling
        self.dim = dim
        self.linear_kernel = linear_kernel

        self.lifting = nn.Conv2d(in_channels, hidden_channels, 1, bias=True)

        in_ch = dim if linear_kernel else dim + hidden_channels
        self.basis_generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=mlp_activation,
        )
        self.alpha = nn.Parameter(torch.ones(1, hidden_channels, 1, 1))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(
                TransformerBlock(
                    in_ch=hidden_channels,
                    out_ch=hidden_channels,
                    n_heads=num_heads,
                    activation=activation,
                    mlp_dim=4 * hidden_channels,
                    dropout=dropout,
                )
            )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor, time: Union[None, torch.Tensor], residual: bool = False):

        if residual:
            shortcut = x

        x = self.lifting(x)

        # Forward transform
        B, C, H, W = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]
        coords = coords.reshape(B, H, W, 2).contiguous()

        if time is not None:
            t = time.view(B, 1, 1, 1).expand(-1, H, W, -1)
            coords = torch.cat([coords, t], dim=-1)

        # 3. Generate kernels
        if self.linear_kernel:
            x_in = coords
        else:
            x_in = torch.concat([coords, x.permute(0, 2, 3, 1)], dim=-1)  # (B, H, W, 2 + Cin)

        # Normalization of each basis vector
        basis = self.basis_generator(x_in)  # -> (B*H*W, m1*m2)
        basis = basis.view(B, H, W, self.m1, self.m2)
        norm_factor = torch.sqrt(torch.sum(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        basis = basis / (norm_factor + 1e-6)

        if self.compute_ortho_loss:
            self.ortho_loss = ortho_loss(basis, n_samples=self.ortho_loss_sampling, mode="MSE")

        # Forward integral transform
        xhat = torch.einsum("bchw,bhwmn->bmnc", x, basis) / (H * W)
        # Add Positional encoding
        PE = sin_positional_encoding_2d(C, self.m1, self.m2, device=xhat.device).repeat(B, 1, 1)  # B, m1 * m2, C
        xhat = xhat.reshape(B, -1, C).contiguous()  # -> m1*m2 tokens
        xhat = xhat + PE

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
        nonlinear=False,
        dim: int = 2,
    ):

        super().__init__()

        self.dim = dim
        self.nonlinear = nonlinear
        trunk_ch = dim + in_channels if self.nonlinear else dim
        self.trunk_projector = MLPBlock(
            in_ch=trunk_ch,
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
                    in_ch=d_model,
                    out_ch=d_model,
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

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x: torch.Tensor, time: Union[None, torch.Tensor] = None, residual: bool = False):

        if residual:
            shortcut = x  # The network learns the residual r(t+dt): f(t+dt)= r(t+dt) + f(t)

        B, C, H, W = x.shape
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2] - VRAM: 0
        trunk_input = coords.reshape(B, H * W, 2)

        if time is not None:
            t = time.view(B, 1, 1).expand(-1, H * W, -1)  # VRAM: 0
            trunk_input = torch.cat([trunk_input, t], dim=-1)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        # We add the function to the trunk output -> non linear kernel
        if self.nonlinear:
            trunk_input = torch.concat([x, trunk_input], dim=-1)

        trunk_output = self.trunk_projector(trunk_input)  # -> [B, H*W, d_model]
        branch_output = self.branch_projector(x)  # [B, H*W, d_model]

        temperature = self.proj_temperature(trunk_output) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        score = self.attention_projector(trunk_output) / math.sqrt(trunk_output.size(-1))  # [B, H*W, modes]

        score_encode = gumbel_softmax(score, temperature, dim=-1)
        score_decode = gumbel_softmax(score, temperature, dim=1)

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
