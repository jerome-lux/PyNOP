



import math
from functools import partial
from sympy import chebyshevt_poly
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence, Callable
import collections.abc as abc
from pynop.core.blocks import MLPBlock, TransformerBlock, LatentTemporalTransformer, LinearTransformerBlock
from pynop.core.ops import CartesianEmbedding, sin_positional_encoding_2d
from pynop.core.norm import AdaptiveLayerNorm
from pynop.core.activations import gumbel_softmax, Sine
from pynop.core.utils import ChebyshevBasis, Newton_Schulz


class Attention(nn.Module):
    """Attention Module (can be self or cross attention) depedning on inputs Q, K, V of the forward method"""

    def __init__(self, in_ch, out_ch, num_heads):
        super(Attention, self).__init__()
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

        attention_weights = F.softmax(attention_scores, dim=-1)

        weighted_output = torch.matmul(attention_weights, V)
        output = self.combine_heads(weighted_output)

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
        dim=3,
        orthogonal_init=True,
        pe=True,
        pe_freqs=32,
        cond_dim=None,
        basis_mode="learned",
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

    def forward(
        self, x: torch.Tensor, time: torch.Tensor, cond: Union[None, torch.Tensor] = None, residual: bool = False
    ):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape

        time_scaling = F.softplus(self.time_scaling(time))

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

        time = time.view(B, 1, -1)
        if self.basis_mode != "cheb":
            basis = self.coord_generator(coords)

        if not self.linear_kernel:
            basis += self.signal_generator(x)

        basis = basis.view(B, H, W, self.m1, self.m2)

        basis = basis / math.sqrt(H * W)

        # norm_factor = torch.sqrt(torch.mean(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        # basis_norm = torch.linalg.norm(basis, ord=2, dim=(1, 2), keepdim=True)
        # basis = basis / (norm_factor + 1e-6)

        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis)
        # Add Positional encoding in latent representation before the self-attention modules
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.latent_pe(PE)
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
        # ATESTER tanh(x_rec) * alpha
        x_rec = x_rec * self.alpha

        # Mixing channels and multiplying by the time scaling
        output = time_scaling.view(B, self.out_channels, 1, 1) * self.projection(x_rec)

        if residual:
            return output + shortcut
        else:
            return output


class CausalLNO(nn.Module):
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
        num_temporal_block: int = 2,
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
        cond_dim=None,
        max_history=5,
        basis_mode="learned",
        attention="classic",
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = modes if isinstance(modes, int) else modes[0]
        self.m2 = modes if isinstance(modes, int) else modes[1]
        self.dim = dim
        self.linear_kernel = linear_kernel
        self.max_history = max_history
        self.basis_mode = basis_mode

        self.lifting = nn.Linear(in_channels, hidden_channels, bias=True)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.norm = nn.LayerNorm(hidden_channels, hidden_channels)

        in_ch = dim

        if self.basis_mode == "cheb":
            self.coord_generator = ChebyshevBasis(self.m1, self.m2)
        else:  # use SIREN Network here (replace sinusoidal PE)
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

        self.latent_pe = MLPBlock(
            out_ch=hidden_channels,
            in_ch=2,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=Sine,
        )

        self.alpha = nn.Parameter(torch.full((1, hidden_channels, 1, 1), 0.98))

        # List of attention modules
        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            if attention == "linear":
                self.ops.append(
                    LinearTransformerBlock(
                        dim=hidden_channels,
                        n_heads=num_heads,
                        activation=activation,
                        mlp_dim=mlp_factor * hidden_channels,
                        dropout=dropout,
                    )
                )
            else:
                self.ops.append(
                    TransformerBlock(
                        dim=hidden_channels,
                        n_heads=num_heads,
                        activation=activation,
                        mlp_dim=mlp_factor * hidden_channels,
                        dropout=dropout,
                    )
                )

        self.temporal_block = LatentTemporalTransformer(
            dim=hidden_channels,
            n_heads=num_heads,
            mlp_factor=mlp_factor,
            num_layers=num_temporal_block,
            max_history=max_history,
        )

        self.projection = nn.Conv2d(hidden_channels, out_channels, 1, bias=True)

        if orthogonal_init:
            self.ortho_init_weights(self.coord_generator)
            self.ortho_init_weights(self.signal_generator)
            self.ortho_init_weights(self.latent_pe)

    def ortho_init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.orthogonal_(module.weight)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        history: Union[torch.Tensor, None] = None,
        cond: Union[None, torch.Tensor] = None,
        residual: bool = False,
        autoencoder=False,
    ):

        if residual:
            if self.in_channels > self.out_channels:
                shortcut = x[:, -self.out_channels :, ...]
            elif self.in_channels == self.out_channels:
                shortcut = x
            else:
                residual = False

        B, C, H, W = x.shape
        x = self.lifting(x.permute(0, 2, 3, 1))

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]

        if self.basis_mode == "cheb":
            basis = self.coord_generator(coords.unsqueeze(0).expand(B, H, W, -1))
        else:
            basis = self.coord_generator(coords.unsqueeze(0))
            basis = basis.expand(B, -1, -1, -1)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        x = self.norm(x)

        if not self.linear_kernel:
            basis += self.signal_generator(x)

        basis = basis.view(B, H, W, self.m1, self.m2)

        basis = basis / math.sqrt(H * W)

        # Forward integral transform
        xhat = torch.einsum("bhwc,bhwmn->bmnc", x, basis)

        # Add Positional encoding in latent representation before the self-attention modules
        m1_coords = torch.linspace(-1, 1, steps=self.m1, device=xhat.device)
        m2_coords = torch.linspace(-1, 1, steps=self.m2, device=xhat.device)
        grid_m1, grid_m2 = torch.meshgrid(m1_coords, m2_coords, indexing="ij")
        m_coords = torch.stack([grid_m1, grid_m2], dim=-1)
        PE = m_coords.view(self.m1, self.m2, -1).unsqueeze(0).expand(B, -1, -1, -1)
        PE = self.latent_pe(PE)

        xhat = xhat + PE
        xhat = xhat.reshape(B, self.m1 * self.m2, -1).contiguous()  # -> m1*m2 tokens

        # Multi-head attention in latent space
        for op in self.ops:
            xhat = op(xhat)

        # If the network is not in autoencoder mode, compute the temporal attention
        if not autoencoder:
            xhat, history = self.temporal_block(xhat, history, coords_t=time.unsqueeze(-1))

        # Go back to spatial domain
        out_ch = xhat.shape[-1]

        # with torch.no_grad():
        #     latent_variance = xhat.var(dim=1).mean()  # Variance across tokens
        #     print(f"Latent energy distribution: {latent_variance.item():.6f}")

        xhat = xhat.reshape(B, self.m1, self.m2, out_ch)
        x_rec = torch.einsum("bmnc,bhwmn->bchw", xhat, basis)
        # x_rec = x_rec * self.alpha

        # Mixing channels
        output = self.projection(x_rec)

        if residual and not autoencoder:
            return output + shortcut, history
        else:
            return output, history


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
        self, x: torch.Tensor, time: torch.Tensor, cond: Union[None, torch.Tensor] = None, residual: bool = False
    ):
        # takes the last time step
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
        coords = coords.view(H * W, -1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)
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
        score = self.attention_projector(trunk_output)  # / math.sqrt(trunk_output.size(-1))  # [B, H*W, modes]
        # score = score / temperature

        # score_encode = gumbel_softmax(score, temperature, dim=-1)
        # score_decode = gumbel_softmax(score, temperature, dim=1)
        score_encode = torch.softmax(score, dim=-1)
        score_decode = torch.softmax(score, dim=1)
        z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)

        for block in self.ops:
            z = block(z)

        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r)
        r = r.permute(0, 2, 1).contiguous().reshape(B, self.out_channels, H, W)

        if residual:
            return r + shortcut
        else:
            return r


class LNOvanilla(torch.nn.Module):
    # implementation from https://github.com/L-I-M-I-T/LatentNeuralOperator/tree/main

    class MLP(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, n_layer, act):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.n_layer = n_layer
            self.act = act()
            self.input = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.hidden = torch.nn.ModuleList(
                [torch.nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layer)]
            )
            self.output = torch.nn.Linear(self.hidden_dim, self.output_dim)

        def forward(self, x):
            r = self.act(self.input(x))
            for i in range(0, self.n_layer):
                r = r + self.act(self.hidden[i](r))
            r = self.output(r)
            return r

    class SelfAttention(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
            self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
            self.proj = torch.nn.Linear(self.n_dim, self.n_dim)

        def forward(self, x):
            B, N, D = x.size()
            q = self.Wq(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            k = self.Wk(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            v = self.Wv(x).view(B, N, self.n_head, D // self.n_head).permute(0, 2, 1, 3)
            score = torch.softmax(torch.einsum("bhic,bhjc->bhij", q, k) / math.sqrt(k.shape[-1]), dim=-1)
            r = torch.einsum("bhij,bhjc->bhic", score, v)
            r = r.permute(0, 2, 1, 3).contiguous().view(B, N, D)
            r = self.proj(r)
            return r

    class AttentionBlock(torch.nn.Module):
        def __init__(self, n_mode, n_dim, n_head, act):
            super().__init__()
            self.n_mode = n_mode
            self.n_dim = n_dim
            self.n_head = n_head
            self.act = act()

            self.self_attn = LNOvanilla.SelfAttention(self.n_mode, self.n_dim, self.n_head)

            self.ln1 = torch.nn.LayerNorm(self.n_dim)
            self.ln2 = torch.nn.LayerNorm(self.n_dim)
            self.drop = torch.nn.Dropout(0.0)

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_dim, self.n_dim * 2),
                self.act,
                torch.nn.Linear(self.n_dim * 2, self.n_dim),
            )

        def forward(self, y):
            y = y + self.drop(self.self_attn(self.ln1(y)))
            y = y + self.mlp(self.ln2(y))
            return y

    def __init__(self, in_ch, out_ch, c_dims, n_block, n_mode, n_dim, n_head, n_layer, act):
        super().__init__()
        self.n_block = n_block
        self.n_mode = n_mode
        self.n_dim = n_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.act = act

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.c_dims = c_dims

        self.trunk_projector = LNOvanilla.MLP(self.c_dims, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.branch_projector = LNOvanilla.MLP(in_ch, self.n_dim, self.n_dim, self.n_layer, self.act)
        self.out_mlp = LNOvanilla.MLP(self.n_dim, self.n_dim, self.out_ch, self.n_layer, self.act)
        self.attention_projector = LNOvanilla.MLP(self.n_dim, self.n_dim, self.n_mode, self.n_layer, self.act)
        self.attn_blocks = torch.nn.Sequential(
            *[LNOvanilla.AttentionBlock(self.n_mode, self.n_dim, self.n_head, self.act) for _ in range(0, self.n_block)]
        )

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, x, time: Union[None, torch.Tensor] = None, residual: Union[None, torch.Tensor] = None):

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.view(H * W, -1)
        coords = coords.unsqueeze(0).expand(B, -1, -1)

        coords = self.trunk_projector(coords)
        x = self.branch_projector(x)

        score = self.attention_projector(coords)
        score_encode = torch.softmax(score, dim=1)
        score_decode = torch.softmax(score, dim=-1)

        z = torch.einsum("bij,bic->bjc", score_encode, x)

        for block in self.attn_blocks:
            z = block(z)

        r = torch.einsum("bij,bjc->bic", score_decode, z)
        r = self.out_mlp(r).permute(0, 2, 1).contiguous().reshape(B, self.out_ch, H, W)
        return r
