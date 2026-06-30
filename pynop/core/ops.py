import math
from warnings import warn
from typing import Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import torch.nn.init as init

from torch.nn.utils import spectral_norm
from .norm import LayerNorm2d
from .utils import get_same_padding_2d

padding_layers = {"zero": nn.ZeroPad2d, "reflect": nn.ReflectionPad2d, "replicate": nn.ReplicationPad2d}


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm=None,
        activation=None,
        padding="same",
        boundary_confitions="zero",
    ):
        """Conv2D Layer with same padding, norm and activation"""

        super(ConvLayer, self).__init__()

        self.pad_mode = padding
        self.padding = padding_layers.get(boundary_confitions, nn.ZeroPad2d)(
            get_same_padding_2d(kernel_size * dilation)
        )
        self.ops = nn.ModuleList(())
        if dropout > 0:
            self.ops.append(nn.Dropout2d(dropout))
        if padding == "same":
            self.ops.append(self.padding)
        self.ops.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
            )
        )

        if norm is not None:
            self.norm = norm(out_channels)
            self.ops.append(self.norm)
        if activation is not None:
            self.activation = activation()
            self.ops.append(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x


class StridedConvDownsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels, factor, kernel_size=1, groups=1):
        super().__init__()
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=factor,
            groups=groups,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MaxPoolConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=factor, padding=math.floor((kernel_size - 1) / 2))
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    """conv + pixel unshuffling to downsample a tensor
    The first conv can be used to reduce the number of channels before the unshuffling
    (*,Cin,H*r,W*r) -> (*,Cout/r**2,H*r,W*r) -> (*, Cout, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    """Downsampling using by averaging the results of
    PixelUnshuffle to get a tensor with C_out channels < C_in * factor**2
    C_in * factor**2 must be divisible by C_out_channels"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    """conv + pixel shuffling to upsample a tensor
    (*, Cin, H, W) -> (*, Cout, H, W) -> (*, Cout // r**2, H*r, W*r)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    """Upsampling layer with interpolation + convolution:
    (*, Cin, H, W) -> (*, Cin, H*r, W*r) -> (*, Cout, H*r, W*r)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        factor: int = 2,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    """Upsampling layer using pixel duplication followed by pixel shuffling
    (*, Cin, H, W) -> (*, Cin, H*r, W*r) -> (*, Cout//r**2, H*r, W*r)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


class LinearLayer(nn.Module):
    """Linear Layer with optional activation and normalization"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        activation=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = norm(out_features)
        self.act = activation()

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class TuckerSpectralConv2d(nn.Module):
    """
    convolution spectrale 2D avec factorisation de Tucker.
    Les facteurs de Tucker (U, V, S, G) sont des tenseurs complexes.
    Le nombres de paramètres est égal à (Cin * r1 + Cout * r2 + (modesx * modesy) * r3 + r1 * r2 * r3)
    vs  (Cin * Cout * modes_x * modes_y sans factorisation
    On peut prendre
    r1 = Cin / k
    r2 = Cout / k
    r3 = modes_x * modes_y / k'
    avec k & k' les facteurs de réduction souhaités (2, 4 etc)
    où encore r1 = r2 = r3 = min(Cin, Cout, modes_x * modes_y)
    Parameters:
    in_channels (int): Nombre de canaux d'entrée.
    out_channels (int): Nombre de canaux de sortie.
    modes (tuple): Tuple (modes_x, modes_y) pour le nombre de modes de basse fréquence à conserver dans x et y.
                   modes_x correspond à la dimension de hauteur (H) dans le domaine spatial.
                   modes_y correspond à la dimension de largeur (W) dans le domaine spatial.
    ranks (tuple): Tuple (r1, r2, r3) pour les rangs de la factorisation de Tucker.
    scaling: Facteur d'échelle pour la taille de sortie. The output shape is (H*scaling, W*scaling).

    """

    def __init__(self, in_channels, out_channels, modes, ranks, scaling: Union[int, float] = 1, channel_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling
        self.channel_last = channel_last

        self.r1, self.r2, self.r3 = ranks

        # Facteurs de la décomposition de Tucker
        self.U = nn.Parameter(torch.empty(in_channels, self.r1, dtype=torch.cfloat))  # (C_in, r1)
        self.V = nn.Parameter(torch.empty(out_channels, self.r2, dtype=torch.cfloat))  # (C_out, r2)
        self.S = nn.Parameter(torch.empty(self.modes_x, self.modes_y, self.r3, dtype=torch.cfloat))  # (mx, my, r3)
        self.G = nn.Parameter(torch.empty(self.r1, self.r2, self.r3, dtype=torch.cfloat))  # (r1, r2, r3)

        # Initialisation des paramètres complexes
        self._initialize_parameters_complex()

    def _initialize_parameters_complex(self):
        with torch.no_grad():
            nn.init.xavier_normal_(self.U.real)
            nn.init.xavier_normal_(self.V.real)
            nn.init.xavier_normal_(self.S.real)
            nn.init.xavier_normal_(self.G.real)

            self.U.imag.copy_(torch.randn_like(self.U.imag) * 0.01)
            self.V.imag.copy_(torch.randn_like(self.V.imag) * 0.01)
            self.S.imag.copy_(torch.randn_like(self.S.imag) * 0.01)
            self.G.imag.copy_(torch.randn_like(self.G.imag) * 0.01)

    def forward(self, x):
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Les canaux d'entrée attendus sont {self.in_channels}, mais reçus {C_in}"

        # Les dimensions spectrales après rfft2(H, W) sont (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            warn(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H. "
            )
            self.modes_x = max_spectral_modes_x
        if self.modes_y > max_spectral_modes_y:
            warn(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )
            self.modes_y = max_spectral_modes_y

        # 1. FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")  # (B, C_in, H, W//2 + 1), cfloat

        # 2. select modes
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]  # (B, C_in, mx, my), cfloat

        # 3. Compute What
        G1 = torch.einsum("ij,jkl->ikl", self.U, self.G)  # (C_in, r2, r3), cfloat
        G2 = torch.einsum("ok,ikl->iol", self.V, G1)  # (C_in, C_out, r3), cfloat
        # S(mx, my, r3) et G2(i, o, r3)
        W_hat = torch.einsum("iol,xyl->ioxy", G2, self.S)  # (C_in, C_out, modes_x, modes_y), cfloat

        # 4. Spectral conv
        # y_ft_low_modes[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * W_hat[i, o, mx, my])
        y_ft_low_modes = torch.einsum("bixy,ioxy->boxy", x_ft_low_modes, W_hat)  # (B, C_out, mx, my), cfloat

        # 5. Zero-padding
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes

        # 6. IFFT
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        if self.channel_last:
            x = x.permute(0, 2, 3, 1)

        return y


class SpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        scaling: Union[int, float] = 1,
        channel_last=False,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (tuple): Tuple (modes_x, modes_y) for the number of low-frequency modes to retain in x and y.
                           modes_x corresponds to the height dimension (H) in the spatial domain.
                           modes_y corresponds to the width dimension (W) in the spatial domain.
            scaling (int): Scaling factor for the output shape. Default is 1, which means no scaling.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling
        self.channel_last = channel_last

        # Spectral convolution kernel (learned parameter)
        # Its shape is (out_channels, in_channels, modes_x, modes_y)
        self.spectral_weights = nn.Parameter(
            torch.empty(out_channels, in_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Expected input channels are {self.in_channels}, but received {C_in}"

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            warn(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H. "
            )
            self.modes_x = max_spectral_modes_x
        if self.modes_y > max_spectral_modes_y:
            warn(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )
            self.modes_y = max_spectral_modes_y

        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]
        y_ft_low_modes_out = torch.einsum(
            "bixy, oixy -> boxy", x_ft_low_modes, self.spectral_weights
        )  # (B, C_out, modes_x, modes_y)

        # Zero-padding of ignored frequencies
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes_out
        # irfft2 to get a real output of size (H, W)
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        if self.channel_last:
            x = x.permute(0, 2, 3, 1)

        return y


class SeparableSpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        scaling: Union[int, float] = 1,
        channel_last=False,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (tuple): Tuple (modes_x, modes_y) for the number of low-frequency modes to retain in x and y.
                           modes_x corresponds to the height dimension (H) in the spatial domain.
                           modes_y corresponds to the width dimension (W) in the spatial domain.
            scaling (int): Scaling factor for the output shape. Default is 1, which means no scaling.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling
        self.channel_last = channel_last

        self.spectral_weights_1 = nn.Parameter(torch.empty(in_channels, self.modes_x, self.modes_y, dtype=torch.cfloat))
        self.spectral_weights_2 = nn.Parameter(torch.empty(out_channels, in_channels, dtype=torch.cfloat))

        # Initialize complex parameters (Xavier Uniform)
        self._initialize_parameters_complex_xavier()

    def _initialize_parameters_complex_xavier(self):
        # Apply Xavier uniform separately to the real and imaginary parts
        with torch.no_grad():
            nn.init.xavier_uniform_(self.spectral_weights_1.real)
            nn.init.xavier_uniform_(self.spectral_weights_1.imag)
            nn.init.xavier_uniform_(self.spectral_weights_2.real)
            nn.init.xavier_uniform_(self.spectral_weights_2.imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Expected input channels are {self.in_channels}, but received {C_in}"

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            warn(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H. "
            )
            self.modes_x = max_spectral_modes_x
        if self.modes_y > max_spectral_modes_y:
            warn(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )
            self.modes_y = max_spectral_modes_y

        xhat = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        xhat = xhat[:, :, : self.modes_x, : self.modes_y]

        # depthwise convolution (i.e. depthwise multiplication) followed by linear channel mixing
        xhat = torch.einsum("bimn, imn -> bimn", xhat, self.spectral_weights_1)
        xhat = torch.einsum("bimn,oi->bomn", xhat, self.spectral_weights_2)

        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, : self.modes_x, : self.modes_y] = xhat
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        if self.channel_last:
            x = x.permute(0, 2, 3, 1)
        return y


class GalerkinAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        delta: float = 1e-2,
        std_ini: float = 1e-2,
        kv_normalization: bool = False,
    ):
        """Linear Galerkin Attention acting as an integral operator.

        Args:
            dim: Input feature dimension.
            heads: Number of attention heads.
            delta: Identity weight for diagonally dominant initialization.
            std_ini: Standard deviation for output projection initialization.
            kv_normalization: If True, applies RMSNorm to Key and Value.
        """
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.kv_norm = kv_normalization

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        if kv_normalization:
            self.k_norm = nn.RMSNorm(self.head_dim)
            self.v_norm = nn.RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Initialize weights
        self.init_weights(delta, std_ini)

    def init_weights(self, delta: float, std_ini: float):
        """Applies specific initializations to projections."""
        for layer in [self.to_q, self.to_k, self.to_v]:
            nn.init.xavier_uniform_(layer.weight)
            with torch.no_grad():
                d_out, d_in = layer.weight.size()
                identity = torch.eye(d_out, d_in, device=layer.weight.device)
                layer.weight.add_(identity, alpha=delta)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.trunc_normal_(self.out_proj.weight, std=std_ini)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for Galerkin Attention.

        Args:
            x: Query tensor of shape [B, M, dim]
            context: Context tensor of shape [B, N, dim]
        """
        if context is None:
            context = x

        b, m, d = x.shape
        n = context.size(1)
        h = self.heads

        # Project and split heads -> [B, h, T, d_h]
        q = self.to_q(x).view(b, m, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, n, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, n, h, self.head_dim).transpose(1, 2)

        if self.kv_norm:
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Kernel matrix computation (K^T * V) / n -> [B, h, d_h, d_h]
        kernel = torch.matmul(k.transpose(-1, -2), v) / n

        # Continuous operator projection Q * Kernel -> [B, h, M, d_h]
        out = torch.matmul(q, kernel)

        # Reshape back to original dim
        out = out.transpose(1, 2).reshape(b, m, d)
        return self.out_proj(out)


class LinearAttentionELU(nn.Module):
    """Linear Cross-Attention for continuous.

    Compatible with high-resolution target queries (N) and latent modes (M).
    Memory complexity: O(N + M) instead of O(N * M).
    """

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections for Queries (from continuous coordinates/MLP)
        self.q_proj = nn.Linear(d_model, d_model)
        # Projections for Latent Modes (K and V)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # phi
        self.phi = nn.ELU(alpha=1.0)

    def forward(self, x, context):
        """Forward pass.

        Args:
            q (Tensor): Continuous queries from target coordinates.
                Shape: [B, N, d_model] where N = H_target * W_target
            context (Tensor): Encoded latent features.  Shape: [B, M,
            d_model]

        Returns:
            Tensor: Projected continuous physical field.
                Shape: [B, N, d_model]
        """
        B, N, _ = x.shape
        _, M, _ = context.shape

        # 1. Project and split into heads
        # Shape: [B, n_heads, N/M, d_head]
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(context).view(B, M, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(context).view(B, M, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # 2. Apply non-negative feature mapping (crucial for stability)
        # Scale factor helps preventing gradient saturation before ELU
        scale = 1.0 / self.d_head**-0.25
        q = self.phi(q * scale) + 1.0
        k = self.phi(k * scale) + 1.0

        # k: [B, heads, M, d_head] -> k.transpose(-2, -1): [B, heads, d_head, M]
        # v: [B, heads, M, d_head]
        context = torch.matmul(k.transpose(-2, -1), v)

        # 4. Compute the normalizer denominator to bind the operator (sum(K))
        # k_sum: [B, heads, d_head, 1]
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        # denom: [B, heads, N, 1]
        denom = torch.matmul(q, k_sum).clamp(1e-6)

        # 5. Project queries onto the context matrix -> O(N) memory
        # num: [B, heads, N, d_head]
        num = torch.matmul(q, context)

        # 6. Normalize and merge heads (Partition of unity)
        # out: [B, heads, N, d_head] -> [B, N, d_model]
        out = num / denom
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)

        return self.out_proj(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        # x: [batch, seq_q, dim], context: [batch, seq_k, dim]
        B, N, C = x.shape

        # if torch.isnan(x).any():
        #     print("NaN detected in x!")

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        scores = torch.matmul(k.transpose(-2, -1), v) / N  # [B, H, D, D]
        att = torch.matmul(q, scores)  # [B, H, Sq, D]

        att = att.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(att)


class Attention(nn.Module):
    """Multi-Head Attention with standardized normal initialization and SDPA optimization."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int,
        std_ini: float = 0.02,
    ):
        super().__init__()
        assert out_ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_ch // num_heads

        # Combined QKV projection can be faster, but keeping separated for RoPE clarity
        self.wq = nn.Linear(in_ch, out_ch)
        self.wk = nn.Linear(in_ch, out_ch)
        self.wv = nn.Linear(in_ch, out_ch)
        self.out_proj = nn.Linear(out_ch, out_ch)

        self._init_weights(std_ini)

    def _init_weights(self, std_ini: float):
        """Standardized normal initialization for attention weights."""
        with torch.no_grad():
            nn.init.trunc_normal_(self.wq.weight, std=std_ini)
            nn.init.trunc_normal_(self.wk.weight, std=std_ini)
            nn.init.trunc_normal_(self.wv.weight, std=std_ini)
            nn.init.trunc_normal_(self.out_proj.weight, std=std_ini)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to [B, num_heads, N, head_dim]."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
        rope: nn.Module = None,
        coords: torch.Tensor = None,
        causal_attn: bool = False,
    ) -> torch.Tensor:
        # Project and split heads -> [B, n_h, N, head_dim]
        q = self.split_heads(self.wq(Q))
        k = self.split_heads(self.wk(K))
        v = self.split_heads(self.wv(V))

        # Apply Rotary Position Embeddings if available (Su et al., 2021)
        if rope is not None and coords is not None:
            q = rope(q, coords)
            k = rope(k, coords)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=causal_attn)

        # Reshape back to [B, N, out_ch]
        out = out.transpose(1, 2).contiguous().view(Q.shape[0], Q.shape[1], -1)

        return self.out_proj(out)


class ComplexAttention(nn.Module):
    """Complex attention Module (can be self or cross attention) depedning on inputs Q, K, V of the forward method"""

    def __init__(self, in_ch, out_ch, num_heads):
        super(ComplexAttention, self).__init__()
        assert out_ch % num_heads == 0
        self.d_model = out_ch
        self.num_heads = num_heads
        self.head_dim = out_ch // num_heads

        self.wq = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)
        self.wk = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)
        self.wv = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)

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

        # Produit scalaire Hermitien pour le score: Q @ K^H
        attention_scores = torch.matmul(Q, K.transpose(-2, -1).conj())
        attention_scores = attention_scores / (self.head_dim**0.5)

        # Softmax appliqué sur la magnitude (valeur absolue) du score
        real_attention_scores = torch.abs(attention_scores)
        attention_weights = F.softmax(real_attention_scores, dim=-1)

        # Application des poids réels à la valeur V complexe
        weighted_output = torch.matmul(attention_weights, V)
        output = self.combine_heads(weighted_output)

        return output
