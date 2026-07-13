import torch.nn as nn
from typing import Union, Callable, List, Iterable, Tuple
from functools import partial

from pynop.core.ops import ConvLayer, GalerkinAttention, LinearAttentionELU, LinearAttention
from pynop.core.blocks import (
    MLPBlock,
    GalerkinTransformerBlock,
    TransformerBlock,
    SlicingBlock,
    DeslicingBlock,
    SirenBlock,
    PEBlock,
)
from pynop.core.encoding import GaussianFourierEmbedding
from pynop.core.norm import AdaRMSNorm
from pynop.core.utils import *
from pynop.core.activations import gumbel_softmax


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels,
        latent_dim,
        block: nn.Module,
        downblock=None,
        stem_kernel_size=3,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_activation="silu",
        stem_norm="bn",
    ):
        """in_channels: number of channels of the input image
        latent_dim: number of channel of the latent representation
        block: basic block used to construct the network
        downblock: if None, the downsampling is done in the first block of each stage. If a nn.Module is given, then it is responsible for the downsampling


        """
        super().__init__()

        self.op_list = nn.ModuleList()
        self.stem_conv = ConvLayer(
            in_channels, dims[0], kernel_size=stem_kernel_size, norm=stem_norm, activation=stem_activation
        )

        for stage, repeats in enumerate(depths):

            if downblock is not None:
                input_channels = dims[stage] if stage == 0 else dims[stage - 1]
                self.op_list.append(downblock(in_channels=input_channels, out_channels=dims[stage]))
                input_channels = dims[stage]
            else:
                input_channels = dims[stage] if stage == 0 else dims[stage - 1]

            for i in range(repeats):
                stride = 2 if downblock is None and i == 0 else 1
                self.op_list.append(block(in_channels=input_channels, out_channels=dims[stage], stride=stride))

        # Project to latent space
        self.bottleneck = nn.Conv2d(dims[-1], latent_dim, 1, bias=True)

    def forward(self, x):

        x = self.stem_conv(x)
        for op in self.op_list:
            x = op(x)
        x = self.bottleneck(x)
        return x


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels,
        latent_dim,
        block: nn.Module,
        upblock: nn.Module,
        depths=[3, 6, 3, 3],
        dims=[1024, 512, 256, 1024],
    ):
        """upblock: it must be a nn.Module responsible for the upsampling AND the projection to the stage's number of channels"""

        super().__init__()

        self.op_list = nn.ModuleList()
        for stage, repeats in enumerate(depths):

            input_channels = latent_dim if stage == 0 else dims[stage - 1]
            self.op_list.append(upblock(in_channels=input_channels, out_channels=dims[stage]))

            for i in range(repeats):
                self.op_list.append(block(in_channels=dims[stage], out_channels=dims[stage]))

        # Project back to in_channels
        self.projet = ConvLayer(dims[-1], out_channels, 1, use_bias=True, activation=None, norm=None)

    def forward(self, x):
        for op in self.op_list:
            x = op(x)
        x = self.projet(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LNOAE(nn.Module):
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
        norm: Union[Callable, None] = None,
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
    ):

        super().__init__()

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.RMSNorm(d_model)
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
            norm=norm,
            dropout=dropout,
        )
        self.branch_projector = MLPBlock(
            in_ch=in_channels,
            out_ch=d_model,
            hidden_dim=mlp_dim,
            num_layers=mlp_layers,
            activation=activation,
            norm=nn.RMSNorm,
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
            norm=norm,
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
        print_stats(trunk_output, -1, "trunk")

        branch_output = self.branch_projector(x)
        print_stats(branch_output, -1, "branch")
        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, :]
            branch_output = branch_output + cond

        # Only one head. Maybe we can use a multi head prohjection ? also why not using a single MLP to compute score from pe?
        score = self.attention_projector_forward(trunk_output)  # / math.sqrt(trunk_output.size(-1))  # [B, H*W, modes]
        print_stats(score, -1, "score")

        tau1 = F.relu(self.tau1) + 1e-6
        tau2 = F.relu(self.tau2) + 1e-6
        score_encode = torch.softmax(score / tau1, dim=1).expand(B, -1, -1)
        score_decode = torch.softmax(score / tau2, dim=-1).expand(B, -1, -1)
        z = torch.einsum("bnm,bnc->bmc", score_encode, branch_output)
        print_stats(z, -1, "latent")

        for block in self.ops:
            z = block(z)
        print_stats(z, -1, "latent final")

        z = (z - z.mean(dim=(0, 1), keepdim=True)) / (z.std(dim=(0, 1), keepdim=True) + 1e-6)

        return z, score_decode

    def decode(self, z, score_decode, out_coords):
        B, M, C = z.shape
        r = torch.einsum("bij,bjc->bic", score_decode, z)
        print_stats(r, -1, "rec")
        r = self.out_mlp(r)
        print_stats(r, -1, "out")
        H, W = out_coords
        r = r.permute(0, 2, 1).contiguous().reshape(B, self.out_channels, H, W)
        return r

    def forward(self, x: torch.Tensor, cond: Union[None, torch.Tensor] = None):
        B, C, H, W = x.shape
        z, score_decode = self.encode(x, cond)
        x = self.decode(z, score_decode, (H, W))
        return x


class GalerkinAE(nn.Module):
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
        galerkin_norm=False,
        scaling=1,
        pe="fourier",
        pe_max_freq=256,
        input_conditioning=False,
        std_init=1e-1,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.dim = dim
        self.dt = dt
        self.hidden_channels = hidden_channels
        self.pe_method = pe
        self.verbose = verbose
        self.input_conditioning = input_conditioning

        self.lifting = nn.Linear(in_channels + dim + self.cond_dim, hidden_channels, bias=True)
        # self.lifting = MLPBlock(
        #     out_ch=hidden_channels,
        #     in_ch=in_channels + dim,
        #     hidden_dim=hidden_channels,
        #     num_layers=2,
        #     activation=mlp_act,
        #     norm=nn.RMSNorm,
        # )

        self.encoder = GalerkinAttention(
            dim=hidden_channels,
            heads=num_heads,
            kv_normalization=galerkin_norm,
            std_ini=std_init,
            softmax=False,
        )
        self.decoder = GalerkinAttention(
            dim=hidden_channels,
            heads=num_heads,
            kv_normalization=galerkin_norm,
            std_ini=std_init,
            softmax=False,
        )

        # max_freq val is used only for SINPE2D, mlp kwargs are sued if pe='SIREN' or 'mlp
        self.pe = PEBlock(
            dim, hidden_channels, method=pe, max_freq_val=pe_max_freq, mlp_layers=2, mlp_dim=2, mlp_act=mlp_act
        )

        self.latent_preproc = nn.Linear(dim + hidden_channels, hidden_channels)

        self.backward_queries_predictor = MLPBlock(
            out_ch=hidden_channels,
            in_ch=hidden_channels,
            hidden_dim=hidden_channels,
            num_layers=mlp_layers,
            activation=mlp_act,
        )

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=1,
                activation=mlp_act,
            )

        self.norm1 = nn.RMSNorm(hidden_channels)
        self.norm2 = nn.RMSNorm(hidden_channels)

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

        m = int(math.sqrt(self.modes))
        latent_coords = torch.linspace(-1, 1, m)
        m1, m2 = torch.meshgrid(latent_coords, latent_coords, indexing="ij")
        latent_coords = torch.stack([m1, m2], dim=-1).view(1, self.modes, -1)
        # self.register_buffer("latent_coords", latent_coords)
        latent_pe = PEBlock(dim, hidden_channels, method=pe)(latent_coords)
        self.register_buffer("latent_pe", latent_pe)
        # queries = fourier_kernels_2d(m, m, hidden_channels)
        # self.forward_queries = nn.Parameter(queries)
        self.forward_queries = nn.Parameter(torch.randn(1, modes, hidden_channels))

        self.latent_pe_preproc = nn.Linear(hidden_channels, hidden_channels)
        self.latent_preproc = nn.Linear(2 * hidden_channels, hidden_channels)

        self.ops = nn.ModuleList()

        for i in range(num_blocks):
            self.ops.append(block())

        # self.modulation = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels),
        #     nn.GELU(),
        #     nn.Linear(hidden_channels, 2 * hidden_channels),
        # )
        # self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

        self.projection = MLPBlock(
            out_ch=out_channels,
            in_ch=hidden_channels,
            hidden_dim=hidden_channels,
            num_layers=2,
            norm=nn.RMSNorm,
            activation=mlp_act,
        )

    def encode(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).view(1, H * W, -1)
        pe = self.pe(coords)
        pe = pe.expand(B, -1, -1)

        x = torch.cat([x, coords.expand(B, -1, -1)], dim=-1)

        if cond is not None:
            cond = self.cond_embedding(cond)
            cond = cond.view(B, -1, self.cond_dim)
            x = torch.cat([x, cond], dim=-1)

        x = self.lifting(x).view(B, H * W, -1)

        if self.verbose:
            print_stats(x, -1, "x after lifting:")
        x = self.norm1(x)
        if self.verbose:
            print_stats(x, -1, "after preNorm:")

        # Galerkin Cross Attention -> Q [B, N, D] K^TV [B, D, D] -> [B, M, C]
        z = self.encoder(x=self.forward_queries.expand(B, -1, -1), context=x)

        if self.verbose:
            print_stats(z, -1, "after IT:")

        # latent_pe = self.latent_pe(latent_coords)

        # z = torch.cat([z, self.latent_coords.expand(B, -1, -1)], dim=-1)
        latent_pe = self.latent_pe_preproc(self.latent_pe)
        z = torch.cat([z, latent_pe.expand(B, -1, -1)], dim=-1)
        z = self.latent_preproc(z)

        if self.verbose:
            print_stats(x, -1, "after pe:")

        # Normalize the latent before the attention modules
        # z = self.norm2(z)

        # Multi-head attention
        for i, op in enumerate(self.ops):
            z = op(z)
            if self.verbose:
                print_stats(z, -1, f"after ATT module {i+1}:")

        z = z - z.mean(dim=(0, 1), keepdim=True)

        if self.input_conditioning:
            return z, x
        else:
            return z, pe

    def decode(self, z, q, H, W):
        B = z.size(0)

        x = self.decoder(x=self.backward_queries_predictor(q), context=z)  # [B, H*W, C]
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
        x_rec = self.decode(z, q, H, W)
        return x_rec


class FNOAE(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_channels: int = 256,
        mlp_layers: int = 2,
        activation: Callable = nn.GELU,
        dim: int = 2,
        cond_dim: Union[int, None] = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.cond_dim = cond_dim if cond_dim is not None else 0
        self.dim = dim
        self.hidden_channels = hidden_channels
        self.verbose = verbose

        self.lifting = nn.Linear(in_channels + dim + self.cond_dim, hidden_channels, bias=True)

        m = int(math.sqrt(modes))

        self.spectral_weights = nn.Parameter(torch.empty(hidden_channels, hidden_channels, m, m, dtype=torch.cfloat))

        self.projection = MLPBlock(
            out_ch=out_channels,
            in_ch=hidden_channels,
            hidden_dim=hidden_channels,
            num_layers=mlp_layers,
            norm=nn.RMSNorm,
            activation=activation,
        )

        # Initialize complex parameters (Xavier Uniform)
        self._initialize_parameters_complex_xavier()

    def _initialize_parameters_complex_xavier(self):
        # Apply Xavier uniform separately to the real and imaginary parts
        with torch.no_grad():
            # Initialize the real part
            nn.init.xavier_uniform_(self.spectral_weights.real)
            # Initialize the imaginary part
            nn.init.xavier_uniform_(self.spectral_weights.imag)

    def encode(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, -1, C)

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).view(1, H * W, -1)

        x = torch.cat([x, coords.expand(B, -1, -1)], dim=-1)

        if cond is not None:
            cond = cond.view(B, -1, self.cond_dim)
            x = torch.cat([x, cond], dim=-1)

        x = self.lifting(x)

        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        m = int(math.sqrt(self.modes))
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        x_ft_low_modes = x_ft[:, :, :m, :m]

        y_ft_low_modes_out = torch.einsum(
            "bixy, oixy -> boxy", x_ft_low_modes, self.spectral_weights
        )  # (B, C_out, modes_x, modes_y)

        # Zero-padding of ignored frequencies
        out_ft = torch.zeros(B, self.hidden_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, :m, :m] = y_ft_low_modes_out

        return out_ft, y_ft_low_modes_out

    def decode(self, z, q, H, W):

        x = torch.fft.irfft2(z, s=(H, W), dim=(-2, -1), norm="ortho")  # (B, C_out, H, W), float
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x).permute(0, 3, 1, 2)

        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):
        B, C, H, W = x.shape
        z, q = self.encode(x, cond)
        x_rec = self.decode(z, q, H, W)

        return x_rec


class PhysicsAttentionAE(nn.Module):

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
        cond_dim: Union[int, None] = None,
        latent_attention="galerkin",
        verbose: bool = False,
        scaling=1,
        use_gumbel_softmax: bool = False,
        std_init=1e-2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.dim = dim
        self.verbose = verbose
        self.hidden_channels = hidden_channels

        self.lifting = nn.Linear(in_channels + dim, hidden_channels, bias=True)

        self.encoder = SlicingBlock(hidden_channels, hidden_channels, modes**2, num_heads, use_gumbel_softmax)
        self.decoder = DeslicingBlock(hidden_channels, hidden_channels, modes**2, num_heads)

        # self.sin_pe_in = SinPE2d(hidden_channels, max_freq_val=modes // 2)
        # self.sin_pe_out = SinPE2d(hidden_channels, max_freq_val=max_freq_val)
        self.latent_grid_pe = GaussianFourierEmbedding(2, hidden_channels, scale=10)
        # self.output_grid_pe = GaussianFourierEmbedding(2, hidden_channels, scale=10)

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels,
                in_ch=cond_dim,
                hidden_dim=mlp_dim,
                num_layers=mlp_layers,
                activation=mlp_act,
            )

        self.norm = nn.RMSNorm(hidden_channels)
        self.post_norm = nn.RMSNorm(hidden_channels)

        # List of attention modules
        self.latent_preprocessing = MLPBlock(
            out_ch=hidden_channels,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=1,
            activation=mlp_act,
            norm=nn.RMSNorm,
        )

        if latent_attention.lower() == "galerkin":
            block = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                std_ini=std_init,
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

        # self.projection = nn.Linear(hidden_channels, out_channels, bias=True)
        self.projection = MLPBlock(
            out_ch=out_channels,
            in_ch=hidden_channels,
            hidden_dim=mlp_dim,
            num_layers=2,
            activation=mlp_act,
            norm=nn.RMSNorm,
        )

    def encode(self, x, cond: Union[None, torch.Tensor] = None):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

        # concat coords to input before lifting
        # print_stats(x, -1, "Input:")
        x = self.lifting(torch.cat([x, coords], dim=-1).view(B, H * W, -1))

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        # print_stats(x, -1, "After Lifting:")
        x = self.norm(x)

        # latent coordinates and PE
        latent_coords = torch.linspace(-1, 1, self.modes, device=x.device)
        m1, m2 = torch.meshgrid(latent_coords, latent_coords, indexing="ij")
        latent_coords = torch.stack([m1, m2], dim=-1).view(1, self.modes**2, 2)
        latent_pe = self.latent_grid_pe(latent_coords).expand(B, -1, -1)

        # Galerkin Cross Attention -> [B, M, D] + MLP
        z, slice_weights = self.encoder(x)
        z = z + latent_pe
        # print_stats(x, -1, "After forward transform:")
        z = self.latent_preprocessing(z)
        # print_stats(z, -1, "preprocessed latent:")
        # Multi-head attention in latent space
        for i, op in enumerate(self.ops):
            z = op(z)
            # print_stats(z, -1, f"After attention block {i}:")

        return z, slice_weights

    def decode(self, x, slice_weights, H, W):

        B = x.shape[0]
        x = self.decoder(x, slice_weights)
        # print_stats(x, -1, "After inverse transform")
        x = self.post_norm(x)
        # print_stats(x, -1, "After post_norm")
        x = self.projection(x)
        # print_stats(x, -1, "Final")
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: Union[None, torch.Tensor] = None,
    ):
        B, C, H, W = x.shape
        z, w = self.encode(x, cond)
        x = self.decode(z, w, H, W)

        return x


class GalerkinAEv2(nn.Module):
    """Applies a downsampling step from M_in to M_out modes in the latent space.

    This block uses Galerkin cross-attention to project a larger token context
    onto a smaller set of learned query tokens, followed by a non-linear MLP.

    Args:
        dim: Input and output feature dimension (channels).
        heads: Number of attention heads.
        mlp_dim: Hidden dimension of the non-linear processing MLP.
        dropout: Dropout rate.
        activation: Activation function class for the MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_list: Union[List[int], Tuple[int]],  # e.g., [128, 64, 32]
        num_blocks: int = 4,  # Applied ONLY to the final latent representation
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
        galerkin_norm=False,
        scaling=1,
        max_freq_val=64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_list = modes_list
        self.dim = dim
        self.verbose = verbose
        self.hidden_channels = hidden_channels

        self.lifting = nn.Linear(in_channels + dim, hidden_channels, bias=True)
        self.norm = nn.RMSNorm(hidden_channels)

        # Spatial grid structures (for final decoding)
        self.sin_pe_spatial = SinPE2d(hidden_channels, max_freq_val=max_freq_val)
        self.queries_predictor_spatial = MLPBlock(
            out_ch=hidden_channels, in_ch=hidden_channels, hidden_dim=hidden_channels, num_layers=1, activation=mlp_act
        )
        self.decoder_spatial = GalerkinAttention(
            dim=hidden_channels, heads=num_heads, kv_normalization=galerkin_norm, std_ini=1e-2
        )

        if cond_dim is not None:
            self.cond_embedding = MLPBlock(
                out_ch=hidden_channels, in_ch=cond_dim, hidden_dim=mlp_dim, num_layers=mlp_layers, activation=mlp_act
            )

        # Pyramidal Reductions and Expansions operators
        self.reductions = nn.ModuleList()
        self.expansions = nn.ModuleList()

        self.sin_pes_latent = nn.ModuleList()
        self.latent_pes = nn.ModuleList()
        self.queries_predictors_latent = nn.ModuleList()

        for idx, m_val in enumerate(modes_list):
            # Coordinate tracking maps for each level
            self.sin_pes_latent.append(SinPE2d(hidden_channels, max_freq_val=m_val // 2))
            self.latent_pes.append(
                MLPBlock(out_ch=hidden_channels, in_ch=2, hidden_dim=mlp_dim, num_layers=mlp_layers, activation=mlp_act)
            )
            self.queries_predictors_latent.append(
                MLPBlock(
                    out_ch=hidden_channels,
                    in_ch=hidden_channels,
                    hidden_dim=hidden_channels,
                    num_layers=1,
                    activation=mlp_act,
                )
            )

            # step-by-step reduction block
            self.reductions.append(
                GalerkinReductionBlock(
                    dim=hidden_channels,
                    heads=num_heads,
                    mlp_dim=mlp_factor * hidden_channels,
                    dropout=dropout,
                    activation=activation,
                )
            )
            # step-by-step expansion block (inserted in reverse order for symmetric decoding)
            self.expansions.insert(
                0,
                GalerkinExpansionBlock(
                    dim=hidden_channels,
                    heads=num_heads,
                    mlp_dim=mlp_factor * hidden_channels,
                    dropout=dropout,
                    activation=activation,
                ),
            )

        # Deep processing blocks: Applied ONLY on the deepest latent bottleneck (M_last)
        if latent_attention.lower() == "galerkin":
            block_fn = partial(
                GalerkinTransformerBlock,
                dim=hidden_channels,
                heads=num_heads,
                mlp_dim=mlp_factor * hidden_channels,
                dropout=dropout,
                std_ini=1e-2,
                scaling=scaling,
            )
        else:
            block_fn = partial(
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

        self.ops = nn.ModuleList([block_fn() for _ in range(num_blocks)])
        self.projection = nn.Linear(hidden_channels, out_channels, bias=True)

    def _get_latent_queries_and_pe(self, B: int, level_idx: int, m_val: int, device: torch.device):
        m_coords = torch.linspace(-1, 1, m_val, device=device)
        m1, m2 = torch.meshgrid(m_coords, m_coords, indexing="ij")
        latent_coords = torch.stack([m1, m2], dim=-1)

        latent_sin_pe = self.sin_pes_latent[level_idx](latent_coords).view(-1, self.hidden_channels)
        latent_pe = self.latent_pes[level_idx](latent_coords.unsqueeze(0).expand(B, -1, -1, -1).view(B, -1, 2))

        queries_in = self.queries_predictors_latent[level_idx](latent_sin_pe).unsqueeze(0).expand(B, -1, -1)
        return queries_in, latent_pe

    def encode(self, x, cond: Union[None, torch.Tensor] = None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        if cond is not None:
            cond = self.cond_embedding(cond)
            if len(cond.shape) == 2:
                cond = cond[:, None, None, :]
            x = x + cond

        context = self.lifting(torch.cat([x, coords], dim=-1).view(B, H * W, -1))
        context = self.norm(context)

        # 1. Progressive downsampling cascade (No internal transformer blocks)
        for idx, reduction_op in enumerate(self.reductions):
            m_val = self.modes_list[idx]
            queries_in, latent_pe = self._get_latent_queries_and_pe(B, idx, m_val, x.device)

            # Reduction path
            z = reduction_op(q_latent=queries_in, context=context)
            context = z + latent_pe  # Continuous PE insertion

        # 2. Deep processing stage: applied only to the final latent representation
        for block in self.ops:
            context = block(context)

        return context

    def decode(self, z, H, W):
        B = z.shape[0]
        context = z

        # 1. Deep processing stage at bottleneck (Decoder entry symmetry)
        for block in self.ops:
            context = block(context)

        # 2. Progressive upsampling cascade
        num_levels = len(self.modes_list)
        for idx, expansion_op in enumerate(self.expansions):
            orig_level_idx = num_levels - 1 - idx
            m_val = self.modes_list[orig_level_idx]

            queries_in, latent_pe = self._get_latent_queries_and_pe(B, orig_level_idx, m_val, z.device)

            # Expansion path
            x_lat = expansion_op(q_latent=queries_in, context=context)
            context = x_lat + latent_pe

        # 3. Final spatial projection to (H, W) domain
        h_coords = torch.linspace(-1, 1, H, device=z.device)
        w_coords = torch.linspace(-1, 1, W, device=z.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_h, grid_w], dim=-1)

        sin_pe = self.sin_pe_spatial(coords).view(-1, self.hidden_channels).unsqueeze(0).expand(B, -1, -1)
        queries_out = self.queries_predictor_spatial(sin_pe)

        x_out = self.decoder_spatial(x=queries_out, context=context)
        x_out = self.projection(x_out)
        x_out = x_out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x_out
