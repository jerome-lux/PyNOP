import math
from posixpath import sep
from typing import Tuple, Any
from typing import Callable, Union, Sequence, Optional
import numpy as np
from functools import partial
from sympy import ln
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
from einops import rearrange

from .utils import make_tuple
from .ops import *
from .norm import AdaptiveLayerNorm, LayerNorm2d, AdaptiveLayerNorm, AdaRMSNorm, GRN
from .activations import Sine, gumbel_softmax
from .encoding import (
    RoPE,
    IntegratedPositionalEncoding,
    GaussianFourierEmbedding,
    GaussianIntegratedPositionalEncoding,
    FourierEmbedding,
    AdaptiveFourierEmbedding,
)
from .utils import print_stats

# TODO: implement FNO block with  differential kernels (i.e. conv when the kernel is rescaled by 1/h and the mean is substracted)
# It would allows to capture the high frequencies while keeping the resolution invariance, far better and more grounded than  the U-FNO! see neuralop for an implementation

DOWNSAMPLING_LAYERS = {
    "convpixunshuffle": ConvPixelUnshuffleDownSampleLayer,
    "pixunshuffleaveraging": PixelUnshuffleChannelAveragingDownSampleLayer,
    "stridedconv": StridedConvDownsamplingLayer,
    "maxpooling": MaxPoolConv,
}

UPSAMPLING_LAYERS = {
    "convpixshuffle": ConvPixelShuffleUpSampleLayer,
    "duplicatingpixshuffle": ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    "interpolate": InterpolateConvUpSampleLayer,
}


class ResBlock(nn.Module):
    """Residual block with n convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        nconv=2,
        stride=1,
        kernel_size=3,
        groups=1,
        norm=partial(nn.GroupNorm, num_channels=32),
        activation=nn.GELU,
        use_bias=False,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, nconv)
        norm = make_tuple(norm, nconv)
        activation = make_tuple(activation, nconv)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ops = nn.ModuleList()
        for i in range(nconv):
            input_channels = in_channels if i == 0 else out_channels
            conv_stride = stride if i == 0 else 1
            self.ops.append(
                ConvLayer(
                    input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=conv_stride,
                    groups=groups,
                    use_bias=use_bias[i],
                    norm=norm[i],
                    activation=activation[i],
                )
            )

        if stride > 1:
            if downsampling_method == "pixunshuffleaveraging":
                self.shortconv = PixelUnshuffleChannelAveragingDownSampleLayer(
                    in_channels=in_channels, out_channels=out_channels, factor=stride
                )
            else:

                self.shortconv = DOWNSAMPLING_LAYERS.get(downsampling_method, StridedConvDownsamplingLayer)(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=stride
                )
        elif in_channels != out_channels:
            self.shortconv = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_bias=False,
                norm=None,
                activation=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for op in self.ops:
            x = op(x)

        if self.stride > 1:
            shortcut = self.shortconv(shortcut)
        elif self.in_channels != self.out_channels:
            shortcut = self.shortconv(shortcut)

        return x + shortcut


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio=4,
        stride=1,
        kernel_size=3,
        groups=1,
        norm=partial(nn.GroupNorm, num_channels=32),
        activation=nn.GELU,
        se=True,
        se_ratio=4,
        use_bias=False,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, 3)
        norm_list = make_tuple(norm, 3)
        activation_list = make_tuple(activation, 3)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // bottleneck_ratio

        self.proj = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            use_bias=use_bias[0],
            norm=norm_list[0],
            activation=activation_list[0],
        )

        self.conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            use_bias=use_bias[1],
            norm=norm_list[1],
            activation=activation_list[1],
        )
        if se:
            self.se = SqueezeExcitation(mid_channels, mid_channels // se_ratio, activation=activation)
        else:
            self.se = None

        self.expand = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            use_bias=use_bias[2],
            norm=norm_list[2],
            activation=activation_list[2],
        )

        if stride > 1:
            if downsampling_method == "pixunshuffleaveraging":
                self.shortconv = PixelUnshuffleChannelAveragingDownSampleLayer(
                    in_channels=in_channels, out_channels=out_channels, factor=stride
                )
            else:

                self.shortconv = DOWNSAMPLING_LAYERS.get(downsampling_method, StridedConvDownsamplingLayer)(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=stride
                )
        elif in_channels != out_channels:
            self.shortconv = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_bias=False,
                norm=None,
                activation=None,
            )

    def forward(self, x):

        shortcut = x

        x = self.proj(x)
        x = self.conv(x)

        if self.se is not None:
            x = self.se(x)

        x = self.expand(x)

        if self.stride > 1:
            shortcut = self.shortconv(shortcut)
        elif self.in_channels != self.out_channels:
            shortcut = self.shortconv(shortcut)

        x = x + shortcut
        return x


class ConvNextv2Block(nn.Module):
    """https://github.com/facebookresearch/ConvNeXt-V2"""

    def __init__(self, in_channels, activation=nn.GELU, drop_path=0.0, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )  # depthwise conv
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation()
        self.grn = GRN(4 * in_channels)
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class FNOBlock(nn.Module):
    """
    A single block of a Fourier Neural Operator (FNO) with a residual connection.


    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        hidden_channels: Union[int, None] = None,
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        scaling: Union[int, float] = 1,
        channel_last=False,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            hidden_channels:Number of output channels for the SpectralConv
            out_channels (int): Number of output channels for the block.
            modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
            spectral_layer_type (str): Type of spectral layer ('standard', 'separable' or 'tucker').
            Separable is the equivalent of Separable Convolution (depthwise + channel mixing)
            ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
            A way to set (r1, r2, r3) is to fix a compression factor and to set
            r1 = cin / k
            r2 = cout / k
            r3 = np.prod(modes) / k
            For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
            for k=4 it's 10 times, etc.
            even with no compression, (k=1) the number of parameters is 2.5 times less
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            self.hidden_channels = out_channels
        else:
            self.hidden_channels = hidden_channels

        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.scaling = scaling
        self.channel_last = channel_last

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(
                in_channels, self.hidden_channels, modes, scaling=scaling, channel_last=channel_last
            )
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(
                in_channels, self.hidden_channels, modes, ranks, scaling=scaling, channel_last=channel_last
            )
        elif spectral_layer_type == "separable":
            self.spectral_conv = SeparableSpectralConv2d(
                in_channels, self.hidden_channels, modes, scaling=scaling, channel_last=channel_last
            )
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation = activation()
        if normalization is not None:
            self.norm = normalization(out_channels)
        else:
            self.norm = None

        # mix channels
        self.linear = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            if self.channel_last:
                x = x.permute(0, 3, 1, 2)
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)
            if self.channel_last:
                x = x.permute(0, 2, 3, 1)

        # Main branch
        if self.norm is not None:
            x = self.norm(x)  # pre norm
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)

        if self.activation is not None:
            x = self.activation(x)

        return x + x_shortcut  # (B, out_channels, H, W)


class FNOBlockv2(nn.Module):
    """
    Improved  Fourier Neural Operator (FNO) block with a double residual connection.

    see
    Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs,
    Jean Kossaifi, Nikola Kovachki, Kamyar Azizzadenesheli, Anima Anandkumar
    https://arxiv.org/abs/2310.00120

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        mlp_layers=2,
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        scaling: Union[int, float] = 1,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            out_channels (int): Number of output channels for the block.
            modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
            spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
            ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
            A way to set (r1, r2, r3) is to fix a compression factor and to set
            r1 = cin / k
            r2 = cout / k
            r3 = np.prod(modes) / k
            For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
            for k=4 it's 10 times, etc.
            even with no compression, (k=1) the number of parameters is 2.5 times less
            mlp_layers: number of linear layers after the SpectralConv (default 1)
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.scaling = scaling

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes, scaling=scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, out_channels, modes, ranks, scaling=scaling)
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation1 = activation()
        self.norm1 = normalization(out_channels) if normalization is not None else None
        self.activation2 = activation()
        self.norm2 = normalization(out_channels) if normalization is not None else None

        # mix channels with n mlp_layers of 1x1 conv.
        # The hidden dimension is out_channels // 2
        self.mlp = nn.ModuleList()
        for i in range(mlp_layers):
            if mlp_layers == 1:
                self.mlp.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
            else:
                out_mlp = out_channels if i == mlp_layers - 1 else out_channels // 2
                in_mlp = out_channels if i == 0 else out_channels // 2
                self.mlp.append(nn.Conv2d(in_mlp, out_mlp, kernel_size=1))

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        # If in_channels == out_channels, this acts as a trainable identity or scaling
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        x = self.spectral_conv(x)

        if self.norm1 is not None:
            x = self.norm1(x)

        x = x + x_shortcut

        x = self.activation1(x)

        for op in self.mlp:
            x = op(x)

        if self.norm2 is not None:
            x = self.norm2(x)

        x = x + x_shortcut

        x = self.activation2(x)

        return x


class ConvFNOBlock(nn.Module):
    """
    A single block of a Fourier Neural Operator (FNO) with a local convolutional branch using Differential Kernel [1] and a residual connection.
    This block combines the spectral convolution with a local convolutional layer to capture both low and high frequencies in the input data.

    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        hidden_channels: Union[int, None] = None,
        nconv=1,
        kernel_size: int = 3,
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        resampling: Union[str, None] = None,
        grid_width: Union[int, float] = 1.0,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            hidden_channels:Number of output channels for the SpectralConv
            out_channels (int): Number of output channels for the block.
            nconv (int): Number of local convolutional layers in the local branch.
            activation (Callable): Activation function to use after the spectral convolution and local convolution.
            normalization (Callable, optional): Normalization layer to apply after the spectral convolution.
            resampling (Union[str, None]): either 'up' (x2), 'down' (//2) or None (no scaling is applied)
            modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
            spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
            ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
            A way to set (r1, r2, r3) is to fix a compression factor and to set
            r1 = cin / k
            r2 = cout / k
            r3 = np.prod(modes) / k
            For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
            for k=4 it's 10 times, etc.
            even with no compression, (k=1) the number of parameters is 2.5 times less
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            self.hidden_channels = out_channels
        else:
            self.hidden_channels = hidden_channels

        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.resampling = resampling
        self.nconv = nconv
        self.grid_width = grid_width

        if self.resampling is not None:
            if self.resampling == "up":
                self.scaling = 2
                self.resampling_module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            elif self.resampling == "down":
                self.scaling = 0.5
                self.resampling_module = nn.MaxPool2d(kernel_size=2)
            else:
                self.resampling_module = None
                self.scaling = 1
                raise ValueError(f"Unknown resampling method: {self.resampling}. Choose 'up', 'down' or None.")
        else:
            self.scaling = 1
            self.resampling_module = None
        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, self.hidden_channels, modes, scaling=self.scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(
                in_channels, self.hidden_channels, modes, ranks, scaling=self.scaling
            )
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation = activation()
        if normalization is not None:
            self.norm = normalization(out_channels)
        else:
            self.norm = None

        # mix channels
        self.linear = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        self.conv = nn.ModuleList()

        for i in range(nconv):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == nconv - 1 else hidden_channels
            self.conv.append(
                FiniteDifferenceLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    norm=normalization,
                    activation=activation,
                    grid_width=self.grid_width * self.scaling,
                )
            )

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        x_local = x
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        # Main branch computation
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)
        if self.norm is not None:
            x = self.norm(x)

        # local branch computation
        if self.resampling_module is not None:
            x_local = self.resampling_module(x_local)
        for op in self.conv:
            x_local = op(x_local)

        x = x + x_shortcut + x_local  # (B, out_channels, H, W)

        if self.activation is not None:
            x = self.activation(x)

        return x


class UFNOBlock(nn.Module):
    """
    A single block of a U - Fourier Neural Operator (FNO) with a residual connection and a small unet network
    as in (https://doi.org/10.1016/j.advwatres.2022.104180).
    It deals with the high frequencies, but we lose the resolution invariance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        nconv: Union[tuple, list] = (1, 2, 2),
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            out_channels (int): Number of output channels for the block.
            modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
            nconv:  a list which gives the number of conv of each stage of a symetric unet
            spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
            ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
            A way to set (r1, r2, r3) is to fix a compression factor and to set
            r1 = cin / k
            r2 = cout / k
            r3 = np.prod(modes) / k
            For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
            for k=4 it's 10 times, etc.
            even with no compression, (k=1) the number of parameters is 2.5 times less
            nconv: number of convolution layers
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.nconv = nconv

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, out_channels, modes, ranks)
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        if activation is not None:
            self.activation = activation()
        if normalization is not None:
            self.norm = normalization(out_channels)

        # mix channels
        self.linear = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        filters = [in_channels] * len(nconv)
        self.unet = UBlock(
            in_channels,
            out_channels,
            nconv=nconv,
            filters=filters,
            normalization=normalization,
            activation=activation,
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        x_spatial = x

        # spectral branch computation
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        if self.activation is not None:
            x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)
        if self.norm is not None:
            x = self.norm(x)

        # spatial branch
        x_spatial = self.unet(x_spatial)
        x = x + x_shortcut + x_spatial

        if self.activation is not None:
            x = self.activation(x)

        return x


class UBlock(nn.Module):
    """simple implementation of a unet network to be used as a block in other blocks/network"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filters: Sequence,
        nconv: Sequence,
        normalization: Union[Callable, None] = LayerNorm2d,
        activation: Callable = nn.GELU,
    ):
        super().__init__()

        use_bias = True if normalization is None else False
        assert len(nconv) == len(filters), "UBlock: filters must have the same length as nconv"
        self.filters = filters
        self.nconv = nconv
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.shortcut_filters = list(filters[: len(nconv) - 2])[::-1]
        self.shortcut_filters.append(in_channels)
        # Encoder
        for i, n in enumerate(nconv):
            in_ch = in_channels if i == 0 else filters[i - 1]
            for j in range(n):
                in_ch = in_ch if j == 0 else filters[i]
                # Downsample last layer (except for the last stage, which is the bridge)
                if i < len(nconv) - 1:
                    strides = 2 if j == n - 1 else 1
                else:
                    strides = 1
                self.encoder.append(
                    ConvLayer(
                        in_channels=in_ch,
                        out_channels=filters[i],
                        kernel_size=3,
                        stride=strides,
                        norm=normalization,
                        activation=activation,
                        use_bias=use_bias,
                    )
                )
        # Decoder
        upconvs = nconv[:-1][::-1]
        upfilters = filters[:-1][::-1]

        for i, n in enumerate(upconvs):
            in_ch = filters[-1] if i == 0 else upfilters[i - 1]
            self.upblocks.append(InterpolateConvUpSampleLayer(in_ch, upfilters[i]))
            self.proj.append(
                ConvLayer(
                    in_channels=upfilters[i] + self.shortcut_filters[i],
                    out_channels=upfilters[i],
                    kernel_size=1,
                    stride=1,
                    use_bias=True,
                )
            )
            for j in range(n):
                if j == n - 1 and i == len(upconvs) - 1:
                    out_ch = out_channels
                else:
                    out_ch = upfilters[i]

                self.decoder.append(
                    ConvLayer(
                        in_channels=upfilters[i],
                        out_channels=out_ch,
                        kernel_size=3,
                        stride=1,
                        norm=normalization,
                        activation=activation,
                        use_bias=use_bias,
                    )
                )

    def forward(self, x):

        shortcuts = []
        counter = -1
        # print(self.shortcut_filters)

        for i, n in enumerate(self.nconv):
            if i < len(self.nconv) - 1:
                shortcuts.append(x)
                # print("shortcut", i, x.shape)
            for j in range(n):
                counter += 1
                # print("encoder input", i, j, counter, x.shape, self.encoder[counter])
                x = self.encoder[counter](x)
                # print("encoder output", i, j, counter, x.shape)

        upconvs = self.nconv[:-1][::-1]
        counter = -1
        shortcuts = shortcuts[::-1]

        for i, n in enumerate(upconvs):
            # print("upblock before", i, x.shape)
            # print("shortcut", shortcuts[i].shape)
            x = self.upblocks[i](x)
            # print("upblock after", i, x.shape)
            x = torch.concat([shortcuts[i], x], 1)
            x = self.proj[i](x)
            for j in range(n):
                counter += 1
                x = self.decoder[counter](x)

        return x


class CoDABlock2D(nn.Module):
    """Co-domain Attention Block (CODABlock) implement the transformer
    architecture in the operator learning framework, as described in [1]_.
    It is a simplified version of the implementation found in https://github.com/neuraloperator

    We also use and equivariant mixing after the attention layer.
    TODO: add BCs to the attention layer (how to do it?)

    References
    ----------
    .. [1]: M. Rahman, R. George, M. Elleithy, D. Leibovici, Z. Li, B. Bonev,
        C. White, J. Berner, R. Yeh, J. Kossaifi, K. Azizzadenesheli, A. Anandkumar (2024).
        "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs."
        arxiv:2403.12553
    """

    def __init__(
        self,
        modes: tuple[int, int],
        token_dim: int,
        attention_dim: int,
        n_heads: int = 1,
        activation: Callable = nn.GELU,
        temperature: float = 1.0,
        norm: Callable = partial(nn.InstanceNorm2d, affine=True),
        spectral_compression_factor: Sequence = (1, 1, 1),
        scaling: Union[float, int] = 1,
    ):

        super().__init__()

        self.token_dim = token_dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_dim = 2  # only 2d spatial dimensions
        self.ranks = [self.token_dim, n_heads * self.token_dim, np.prod(modes)]
        self.ranks = tuple(np.ceil(np.divide(self.ranks, spectral_compression_factor)).astype(int))
        self.attention_dim = attention_dim

        self.Q = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.attention_dim,
            out_channels=n_heads * self.attention_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
            scaling=scaling,
        )

        self.V = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.attention_dim,
            out_channels=n_heads * self.attention_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
            scaling=scaling,
        )

        self.K = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.attention_dim,
            out_channels=n_heads * self.attention_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
            scaling=scaling,
        )

        # To project back each token from the n heads to token_dim

        self.projection = FNOBlock(
            in_channels=self.n_heads * self.attention_dim,
            hidden_channels=self.token_dim,
            out_channels=self.token_dim,
            modes=modes,
            activation=nn.Identity,
            spectral_layer_type="tucker",
            ranks=self.ranks,
            scaling=1 / scaling,
        )
        mixer_ranks = [self.token_dim, self.token_dim, np.prod(modes)]
        mixer_ranks = np.ceil(np.divide(mixer_ranks, spectral_compression_factor)).astype(int)

        self.mixer = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=self.token_dim,
            out_channels=self.token_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=mixer_ranks,
        )
        self.norm1 = norm(self.token_dim)
        self.norm2 = norm(self.token_dim)
        self.norm3 = norm(self.token_dim)

    def MultiHeadAttention(self, tokens, batch_size):
        """Compute multi-head Attention where each variable latent representation is a token

        input tensor shape (b*t), d, h, w
        The tensor is first transformed into k, q and v with shape (b*t),(n*d), h, w
        where
        b: batch size
        t: number of tokens
        n: number of heads
        d: token dimension (the latent dimension of each variable)
        h, w: spatial dimensions

        Note taht when using per_channel attention, the token dimension is 1, so d=1

        Then k, q and v are reshaped to b, n, t, (d h w)
        as torch.matul multiplies the two last dimensions
        Finally the output is reshaped to b, n, (t*d), h, w

        """
        # k, q, v (b*t, n*d, h, w)
        k = self.K(tokens)
        q = self.Q(tokens)
        v = self.V(tokens)

        spatial_shape = k.shape[-self.n_dim :]

        assert k.size(1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

        # reshape from (b*t) (n*d) h w -> b n t (d*h*w ...)
        t = k.size(0) // batch_size  # Compute the number of tokens `t` (each token is a variable here)
        # n heads with token codimension `d` (in the case of per layer attention, d=1)
        d = k.size(1) // self.n_heads

        # reshape from (b*t) (n*d) h w ... to b n t d h w ...
        k = k.view(batch_size, t, self.n_heads, d, *k.shape[-self.n_dim :])
        q = q.view(batch_size, t, self.n_heads, d, *q.shape[-self.n_dim :])
        v = v.view(batch_size, t, self.n_heads, d, *v.shape[-self.n_dim :])

        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)

        # reshape to flatten the d, h and w dimensions
        k = k.reshape(batch_size, self.n_heads, t, -1)
        q = q.reshape(batch_size, self.n_heads, t, -1)
        v = v.reshape(batch_size, self.n_heads, t, -1)

        # print(q.shape, k.shape, v.shape)

        # attention mechanism
        dprod = torch.matmul(q, k.transpose(-1, -2)) / (np.sqrt(k.shape[-1]) * self.temperature)
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # Reshape from (b, n, t, d * h * w) to (b, n, t, d, h, w, ...)
        attention = attention.view(attention.size(0), attention.size(1), attention.size(2), d, *spatial_shape)
        attention = torch.transpose(attention, 1, 2)  # b t n d h w
        attention = attention.reshape(
            attention.size(0) * attention.size(1), attention.size(2) * d, *spatial_shape
        )  # (b * t) (n * d) h w

        return attention

    def forward(self, x):

        # the input tensor must have a shape b (n_var * hidden_dim) h w
        # if the token_dim is different than the hidden dim,  it means that each token does not represent the full latent embedding of a variable

        batch_size = x.shape[0]
        # spatial_shape = x.shape[-self.n_dim :]

        assert x.shape[1] % self.token_dim == 0, "Number of channels in x should be divisible by token_codimension"

        n_tokens = x.shape[1] // self.token_dim
        # Reshape from shape b (t*d) h w ... to (b*t) d h w
        x = x.view(x.size(0) * n_tokens, self.token_dim, *x.shape[-self.n_dim :])

        attention = self.norm1(x)
        attention = self.MultiHeadAttention(attention, batch_size)
        attention = self.projection(attention)
        attention = self.norm2(attention + x)  # shortcut
        attention = self.mixer(attention)
        attention = self.norm3(attention)

        # reshape to b (n_var * hidden_var_dim // token_dim) h w
        attention = attention.view(batch_size, n_tokens * attention.shape[1], *attention.shape[-self.n_dim :])

        return attention


class ComplexMLPBlock(nn.Module):
    """
    A small MLP that produces (complex) values for a set of basis functions.
    """

    def __init__(self, in_ch, out_ch, hidden_dim=64, num_layers=2, activation=nn.GELU, norm=None, dropout=0.0):
        """
        Initializes the MLP.
        Args:
            n_dim (int): Dimension of the input coordinates (default: 2).
            Note: the kernel can be made non linear by using an input with the evaluation of the funciton at the coordinates.
            num_modes (int): Number of modes to learn.
            hidden_dim (int): Hidden dimension of the MLP.
            num_layers (int): Number of hidden layers in the MLP.
            activation (callable): Activation function to use (default: nn.GELU).
            norm (callable): Normalization layer to use (default: nn.LayerNorm).
            dropout (float): Dropout rate (default 0.0).
        """
        super().__init__()
        self.num_modes = out_ch

        layers = []
        # The first layer takes 1 coordinate (e.g., normalized)
        layers.append(nn.Linear(in_ch, hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if norm is not None:
            layers.append(norm(hidden_dim))
        if activation is not None:
            if type(activation) is Sine:
                layers.append(activation(w0=30))
            else:
                layers.append(activation())  # Or ReLU, LeakyReLU, etc.

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if norm is not None:
                layers.append(norm(hidden_dim))
            if activation is not None:
                layers.append(activation())

        # The last layer produces 2 * num_modes real values (for real and imaginary parts)
        layers.append(nn.Linear(hidden_dim, 2 * out_ch))

        self.mlp = nn.Sequential(*layers)

        # SIREN init for sinusoidal activation
        if type(activation) is Sine:
            with torch.no_grad():
                first_layer_init = True
                for i, m in enumerate(self.mlp.modules()):
                    if isinstance(m, nn.Linear):
                        if first_layer_init:
                            first_layer_init = False
                            m.weight.uniform_(-1 / m.in_features, 1 / m.in_features)
                        else:
                            limit = np.sqrt(6 / m.in_features)
                            m.weight.uniform_(-limit, limit)

    def forward(self, x):
        # x is a tensor of shape (N, n_dim) where N is the number of sampling points and n_dim the dimension of the input
        # for example, N = H or N = W

        # Pass the coordinates through the MLP
        output = self.mlp(x)  # (N, 2 * num_modes)

        # Separate real and imaginary parts
        real_part = output[..., : self.num_modes]  # (N, num_modes)
        imag_part = output[..., self.num_modes :]  # (N, num_modes)

        # Combine to form a complex tensor
        complex_output = torch.complex(real_part, imag_part)  # (N, num_modes)

        return complex_output


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU,
        norm: nn.Module = None,
        dropout: float = 0.0,
    ):
        """
        Standard Multi-Layer Perceptron (MLP) block.
        in_ch:       Number of input channels
        out_ch:      Number of output channels
        hidden_dim:  Hidden dimension size
        num_layers:  Total number of layers including hidden layers
        activation:  Activation layer class (e.g., nn.GELU)
        norm:        Normalization layer class (e.g., nn.LayerNorm)
        dropout:     Dropout probability
        """
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(in_ch, hidden_dim))
        if norm is not None:
            layers.append(norm(hidden_dim))
        if activation is not None:
            layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers (num_layers must be > 1)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm is not None:
                layers.append(norm(hidden_dim))
            if activation is not None:
                layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Output projection layer
        layers.append(nn.Linear(hidden_dim, out_ch))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization optimized for rectifier-like activations."""
        with torch.no_grad():
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Xavier uniform setup for the final linear projection layer
            nn.init.xavier_uniform_(self.mlp[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, ..., in_ch] - Input feature tensor
        Returns: [B, ..., out_ch] - Projected output tensor
        """
        return self.mlp(x)


class SirenBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, hidden_dim: int = 64, num_layers: int = 2, w0: float = 30.0):
        """
        Sinusoidal Representation Network (SIREN) block.
        in_ch:      Number of input channels
        out_ch:     Number of output channels
        hidden_dim: Hidden dimension size
        num_layers: Total number of layers including hidden layers
        w0:         Frequency scaling factor (30.0 is standard for coordinates)
        """
        super().__init__()
        self.w0 = w0

        layers = []

        # First layer
        layers.append(nn.Linear(in_ch, hidden_dim))
        layers.append(Sine(w0=self.w0))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine(w0=self.w0))

        # Output projection layer
        layers.append(nn.Linear(hidden_dim, out_ch))

        self.siren = nn.Sequential(*layers)
        self._init_siren()

    def _init_siren(self):
        """SIREN uniform initialization scheme to preserve distribution variance."""
        with torch.no_grad():
            linears = [m for m in self.siren if isinstance(m, nn.Linear)]
            n = len(linears)

            for i, m in enumerate(linears):
                if i == 0:
                    # Map input domain properly by scaling bounds with w0
                    m.weight.uniform_(-1.0 / m.in_features, 1.0 / m.in_features)
                elif i < n - 1:
                    # Calibrate variance for subsequent sine activations
                    limit = math.sqrt(6.0 / m.in_features) / self.w0
                    m.weight.uniform_(-limit, limit)
                else:
                    # Final layer standard Xavier initialization
                    nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, ..., in_ch] - Input coordinate or feature tensor
        Returns: [B, ..., out_ch] - Reconstructed signal representation
        """
        return self.siren(x)


class SepLITBlock(nn.Module):
    """
    Linear Integral Transform Block (LITBlock) is a
    PyTorch module for a learned 2D integral transform that is resolution-invariant.
    The transform bases are learned as continuous functions via MLPs.
    This is the separable kernels implementation
    Inspired by IAE-NET: INTEGRAL AUTOENCODERS FOR DISCRETIZATION-INVARIANT LEARNING
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=1,
        activation=nn.GELU,
        norm=AdaptiveLayerNorm,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = ComplexMLPBlock(
            out_ch=self.m1, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )
        self.basis_w_fn = ComplexMLPBlock(
            out_ch=self.m2, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(out_channels, self.in_channels, self.m1, self.m2, dtype=torch.cfloat)
        )

        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
        # nn.init.constant_(self.learned_weights_freq, 1.0)  # Initialize to 1+0j
        std = (1.0 / (in_channels + out_channels)) ** 0.5
        with torch.no_grad():
            self.learned_weights_freq.real.normal_(0, std)
            self.learned_weights_freq.imag.normal_(0, std)

        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        B, C, H, W = x.shape

        # Convert input to complex numbers if it is real
        x_complex = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # Generate normalized coordinates for the current height and width.
        # These coordinates are passed to our MLPs to generate the bases dynamically.
        h_coords = torch.linspace(-1, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        w_coords = torch.linspace(-1, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)  # (W, 1)

        # Dynamically generate the basis matrices for the current resolution
        base_values_h = self.basis_h_fn(h_coords)
        base_values_w = self.basis_w_fn(w_coords)

        # Transpose the generated bases so they have shape (m1, H) and (m2, W)
        # for matrix multiplication
        transform_h_basis_runtime = base_values_h.T
        transform_w_basis_runtime = base_values_w.T

        # --- Step 1: Direct transform (projection onto learned modes) ---
        # F_truncated = W'_M * f * W'_N

        # Transform along rows (dimension H -> m1 modes)
        # x_complex: (B, C, H, W)
        # transform_h_basis_runtime: (m1, H)
        # Result: (B, C, m1, W)
        transformed_freq_h = torch.einsum("bchw,mh->bcmw", x_complex, transform_h_basis_runtime)

        # Transform along columns (dimension W -> m2 modes)
        # transformed_freq_h: (B, C, m1, W)
        # transform_w_basis_runtime: (m2, W)
        # Result: (B, C, m1, m2)
        transformed_freq_hw = torch.einsum("bcmw,nw->bcmn", transformed_freq_h, transform_w_basis_runtime)

        # --- Step 2: Multiplication by learned weights in the transformed space ---
        # self.learned_weights_freq: (C, m1, m2)
        # transformed_freq_hw: (B, C, m1, m2)
        # processed_freq_domain = transformed_freq_hw * self.learned_weights_freq.unsqueeze(0)

        processed_freq_domain = torch.einsum("bixy,oixy->boxy", transformed_freq_hw, self.learned_weights_freq)

        # --- Step 3: Inverse transform (projection back) ---
        # f_approx = (W'_M)^H * F_truncated * (W'_N)^H

        # Inverse transform along columns (m2 modes -> W)
        # processed_freq_domain: (B, Cout, m1, m2)
        # transform_w_basis_runtime.H: (W, m2) (conjugate transpose of (m2, W))
        # Result: (B, Cout, m1, W)
        reconstructed_spatial_w = torch.einsum("bcmn,wn->bcmw", processed_freq_domain, transform_w_basis_runtime.H)

        # Inverse transform along rows (m1 modes -> H)
        # reconstructed_spatial_w: (B, Cout, m1, W)
        # transform_h_basis_runtime.H: (H, m1) (conjugate transpose of (m1, H))
        # Result: (B, Cout, H, W)
        reconstructed_spatial_hw = torch.einsum("bcmw,hm->bchw", reconstructed_spatial_w, transform_h_basis_runtime.H)

        # Normalization
        normalisation_factor = 1.0 / (H * W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.real * normalisation_factor

        # Mixing channels
        output = self.mixer(reconstructed_spatial_hw)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = output + self.shortcut(x)
        output = self.activation(output)
        return output


class SepNLITBlock(nn.Module):
    """
    Non Linear Integral Transform Block (SNLITBlock) is a
    PyTorch module for a learned 2D non linear integral transform that is resolution-invariant.
    The kernels are assumed to be separable in space and
    are learned as a function of the input coordinates and values.
    Inspired by IAE-NET: INTEGRAL AUTOENCODERS FOR DISCRETIZATION-INVARIANT LEARNING

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        activation=nn.GELU,
        norm=AdaptiveLayerNorm,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        self.coords_dim = 1 + in_channels

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = ComplexMLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = ComplexMLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        self.learned_weights_freq = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Initialization of learned_weights_freq
        std = (1.0 / (in_channels + out_channels)) ** 0.5
        with torch.no_grad():
            self.learned_weights_freq.real.normal_(0, std)
            self.learned_weights_freq.imag.normal_(0, std)

        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        batch_size, channels, H, W = x.shape

        # Convert input to complex numbers if it is real
        x_complex = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # STEP 1.1 Process coordinates along W dimension (width)
        # Generate normalized coordinates for the current width.
        w_coords = torch.linspace(-1, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)

        # Add function values to condition our kernel to the input
        x_w = x_complex.real.permute(0, 2, 3, 1)  # (B, H, W, Cin)
        # Repeat the coordinates to match batch and height dimensions: (B, H, W, 1)
        w_coords = w_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, H, 1, 1)
        # Concatenate channel values and W coordinates: (B, H, W, Cin + 1)
        w_input = torch.concat([x_w, w_coords], dim=-1)
        # Flatten for MLP input: (B*H*W, Cin + 1)
        # w_input = w_input.view(-1, channels + 1) not needed
        # Generate basis values for W: (B, H, W, m2)
        w_basis_runtime = self.basis_w_fn(w_input)
        # Reshape to (B, H, W, m2)
        # w_basis_runtime = w_basis_runtime.view(batch_size, H, W, self.m2)
        # Perform matrix multiplication for the W transformation
        # x_complex: (B, Cin, H, W) -> permute to (B, H, Cin, W) for matmul with (B, H, W, m2)
        x_w = x_complex.permute(0, 2, 1, 3)  # (B, H, Cin, W)
        # Multiplication: (B, H, Cin, W) @ (B, H, W, m2) -> (B, H, Cin, m2)
        transformed_freq_w = torch.matmul(x_w, w_basis_runtime)
        # Permute to (B, Cin, H, m2) for the next step
        transformed_freq_w = transformed_freq_w.permute(0, 2, 1, 3)  # (B, Cin, H, m2)

        # STEP 1.2: Process coordinates along H dimension (height)
        # Permute to (B, m2, H, Cin) for concatenation
        x_h = transformed_freq_w.real.permute(0, 3, 2, 1)  # (B, m2, H, Cin)

        h_coords = torch.linspace(-1, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        # Repeat the coordinates to match batch and m2 dimensions: (B, m2, H, 1)
        h_coords = h_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.m2, 1, 1)

        # Concatenate channel values and H coordinates: (B, m2, H, Cin + 1)
        h_input = torch.concat([x_h, h_coords], dim=-1)

        # Flatten for MLP input: (B*m2*H, Cin + 1)
        # h_input = h_input.view(-1, channels + 1)

        # Generate basis values for H: (B*m2*H, m1)
        h_basis_runtime = self.basis_h_fn(h_input)

        # Reshape to (B, m2, H, m1) - this is the "kernel" of IAE-Net for the second layer
        h_basis_runtime = h_basis_runtime.view(batch_size, self.m2, H, self.m1)

        # Perform matrix multiplication for the H transformation
        # transformed_freq_w_permuted: (B, Cin, H, m2) -> permute to (B, m2, Cin, H) for matmul with (B, m2, H, m1)
        x_for_h_transform = transformed_freq_w.permute(0, 3, 1, 2)  # (B, m2, Cin, H)

        # Multiplication: (B, m2, Cin, H) @ (B, m2, H, m1) -> (B, m2, Cin, m1)
        transformed_freq_hw = torch.matmul(x_for_h_transform, h_basis_runtime)

        # Permute to (B, Cin, m2, m1)
        transformed_freq_hw = transformed_freq_hw.permute(
            0, 2, 3, 1
        )  # (B, Cin, m1, m2) if m1 is axis 2 and m2 is axis 3

        # --- Step 2: Multiplication by learned weights ---
        # processed_freq_domain = transformed_freq_hw * self.learned_weights_freq.unsqueeze(0)
        # transformed_freq_hw: (B, C_in, m1, m2)
        # self.learned_weights_freq: (C_out, C_in, m1, m2)
        # Résultat: (B, C_out, m1, m2)
        processed_freq_domain = torch.einsum("bixy,oixy->boxy", transformed_freq_hw, self.learned_weights_freq)

        # --- Step 3: Inverse transform with data-dependent bases ---
        # We reverse the order of transformations: first H, then W.

        # 1. Inverse transform along H dimension (Height)
        # Input: processed_freq_domain (B, C_out, m1, m2)
        # Permute to (B, m2, C, m1) for matmul with (B, m2, m1, H)
        x_for_h_inv_transform = processed_freq_domain.permute(0, 3, 1, 2)  # (B, m2, C_out, m1)

        # Conjugate transpose of the H basis: (B, m2, m1, H).H -> (B, m2, H, m1)
        # Multiplication: (B, m2, C_out, m1) @ (B, m2, m1, H) -> (B, m2, C_out, H)
        reconstructed_h = torch.matmul(x_for_h_inv_transform, h_basis_runtime.mH)

        # Permute to (B, C_out, m2, H)
        reconstructed_h_permuted = reconstructed_h.permute(0, 2, 3, 1)  # (B, C_out, H, m2)

        # 2. Inverse transform along W dimension (Width)
        # Input: reconstructed_h_permuted (B, C_out, H, m2)
        # Permute to (B, H, C_out, m2) for matmul with (B, H, m2, W)
        x_for_w_inv_transform = reconstructed_h_permuted.permute(0, 2, 1, 3)  # (B, H, C_out, m2)

        # Conjugate transpose of the W basis: (B, H, m2, W).H -> (B, H, W, m2)
        # Multiplication: (B, H, C_out, m2) @ (B, H, m2, W) -> (B, H, C_out, W)
        reconstructed_spatial_hw = torch.matmul(x_for_w_inv_transform, w_basis_runtime.mH)

        # Permute to (B, C_out, H, W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.permute(0, 2, 1, 3)

        # Dynamic normalization
        normalisation_factor = 1.0 / (H * W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.real * normalisation_factor

        # Mixing channels
        output = self.mixer(reconstructed_spatial_hw)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = output + self.shortcut(x)
        output = self.activation(output)

        return output


class ParametricITBlock(nn.Module):
    """
    ITBlock Module for Non-Linear or linear Integral Transform.

    This module implements a non-linear integral transform that:
        1. Generates learned basis functions K(u, v, x, y, f(x, y)) or K(u, v, x, y) based on input coordinates and values
        2. Transforms the input to a 'spectral' representation via einsum operation
        3. Multiply by learned weights in the transformed space
        4. Applies inverse transform back to spatial domain using conjugate basis functions
        5. Applies channel mixing, normalization, and activation
        6. Adds shortcut connection with optional resampling

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        m1 (int): Number of modes in height dimension (u) in the basis
        m2 (int): Number of modes in the width dimension (v) in the basis
        activation (callable, optional): Activation function to use. Default is nn.GELU.
        norm (callable, optional): Normalization layer to use. Default is AdaptiveLayerNorm.
        dim (int, optional): Spatial dimensionality (default is 2 for 2d problems).

    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.
        basis (torch.Tensor): the kernels to perform the transformation

    Forward Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) (real part)

    """

    def __init__(
        self,
        m1: int,
        m2: int,
        in_channels: int,
        out_channels: int,
        complex: bool = False,
        activation: Callable = nn.GELU,
        dim: int = 2,
    ):

        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.alpha = nn.Parameter(torch.ones(1, 1, 1, out_channels))

        # Learned parameters for multiplication in the transformed space (per channel).
        if complex:
            # self.learned_weights = nn.Parameter(
            #     torch.zeros(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
            # )
            # std = (1.0 / (in_channels + out_channels)) ** 0.5
            # with torch.no_grad():
            #     self.learned_weights.real.normal_(0, std)
            #     self.learned_weights.imag.normal_(0, std)

            # channel wise multiplicaito
            self.learned_weights_1 = nn.Parameter(torch.zeros(self.in_channels, self.m1, self.m2, dtype=torch.cfloat))
            # mixing
            self.learned_weights_2 = nn.Parameter(torch.zeros(self.in_channels, self.out_channels, dtype=torch.cfloat))

            std = (1.0 / (in_channels + out_channels)) ** 0.5
            with torch.no_grad():
                self.learned_weights_1.real.normal_(0, std)
                self.learned_weights_1.imag.normal_(0, std)
                self.learned_weights_2.real.normal_(0, std)
                self.learned_weights_2.imag.normal_(0, std)
        else:
            # self.learned_weights = nn.Parameter(torch.zeros(self.in_channels, out_channels, self.m1, self.m2))
            # std = (2.0 / (in_channels + out_channels)) ** 0.5
            # with torch.no_grad():
            #     self.learned_weights.normal_(0, std)
            # channel wise multiplicaito
            self.learned_weights_1 = nn.Parameter(torch.zeros(self.in_channels, self.m1, self.m2))
            # mixing
            self.learned_weights_2 = nn.Parameter(torch.zeros(self.in_channels, self.out_channels))

            std = (2.0 / (in_channels + out_channels)) ** 0.5
            with torch.no_grad():
                self.learned_weights_1.normal_(0, std)
                self.learned_weights_2.normal_(0, std)

        self.mixer = MLPBlock(out_channels, out_channels, hidden_dim=out_channels, num_layers=1, activation=activation)
        self.norm = AdaptiveLayerNorm(out_channels, out_channels)

    def forward(self, x: Tensor, fwd_basis: Tensor, time: Union[None, Tensor] = None):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).
                                H and W may vary.
            fwd_basis: basis for forward IT tranform
            cond: contionning variable (e.g. time) passed to the norm layer

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out), real part.
        """
        B, H, W, C = x.shape

        # Convert input to complex numbers if it is real
        if fwd_basis.is_complex():
            xin = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x
        else:
            xin = x

        xhat = torch.einsum("bhwc,bhwmn->bmnc", xin, fwd_basis)  # "Spectral" representation

        # Multiply by learned weigths in transformed space
        # xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)
        xhat = torch.einsum("bmni, imn -> bmni", xhat, self.learned_weights_1)
        xhat = torch.einsum("bmni,oi->bmno", xhat, self.learned_weights_2)

        if fwd_basis.is_complex():
            x_rec = torch.einsum("bmnc,bhwmn->bhwc", xhat, fwd_basis.conj()).real
        else:
            x_rec = torch.einsum("bmnc,bhwmn->bhwc", xhat, fwd_basis)

        x_rec = x_rec * self.alpha + x  # 1st shortcut
        shortcut = x_rec
        # Mixing channels
        x_rec = self.norm(x_rec, time)
        x_rec = self.mixer(x_rec)
        x_rec = x_rec + shortcut  # 2nd shortcut"

        return x_rec


class LITBlock(nn.Module):
    """
    Learnable Intregral transform block for Non-Linear or linear Integral Transform.

    This module implements a non-linear integral transform that:
        1. Generates learned basis functions K(u, v, x, y, f(x, y)) or K(u, v, x, y) based on input coordinates and values
        2. Transforms the input to a 'spectral' representation via einsum operation
        3. Multiply by learned weights in the transformed space
        4. Applies inverse transform back to spatial domain using conjugate basis functions
        5. Applies channel mixing, normalization, and activation
        6. Adds shortcut connection with optional resampling

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        m1 (int): Number of modes to keep in the height dimension (u).
        m2 (int): Number of modes to keep in the width dimension (v).
        mlp_hidden_dim (int, optional): Hidden dimension of the MLPs learning the bases. Default is 64.
        mlp_num_layers (int, optional): Number of hidden layers in the MLPs learning the bases. Default is 2.
        activation (callable, optional): Activation function to use. Default is nn.GELU.
        dim (int, optional): Spatial dimensionality (default is 2 for 2d problems).

    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.

    Forward Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) (real part)

    """

    def __init__(
        self,
        in_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=1,
        activation=nn.GELU,
        mlp_act=nn.GELU,
        nonlinear=True,
        separable=True,
        orthogonal_init=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.nonlinear = nonlinear
        self.separable = separable

        self.IT_scale_forward = nn.Parameter(torch.empty(1))
        self.IT_scale_inverse = nn.Parameter(torch.empty(1))
        # To scale the integral transform. sigmoid(1.1)~0.75 so between 1/sqrt(N) and 1/N
        with torch.no_grad():
            nn.init.constant_(self.IT_scale_forward, 1.1)
            nn.init.constant_(self.IT_scale_inverse, 1.1)

        in_ch = in_channels + 2 if nonlinear else 2

        self.pe = MLPBlock(
            out_ch=in_channels,
            in_ch=2,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_act,
        )

        self.generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_act,
        )
        if orthogonal_init:
            self.ortho_init_weights(self.generator)

        # Learned parameters for multiplication in the transformed space (per channel).
        # self.learned_weights = nn.Parameter(torch.zeros(self.in_channels, out_channels, self.m1, self.m2))
        std = (1.0 / (in_channels)) ** 0.5
        if self.separable:  # this is equivalent to a separable convolution
            self.learned_weights_1 = nn.Parameter(torch.zeros(self.in_channels, self.m1, self.m2))
            # mixing
            self.learned_weights_2 = nn.Parameter(torch.zeros(self.in_channels, self.in_channels))
            with torch.no_grad():
                self.learned_weights_1.normal_(0, std)
                self.learned_weights_2.normal_(0, std)
        else:
            self.learned_weights = nn.Parameter(torch.zeros(self.in_channels, in_channels, self.m1, self.m2))
            with torch.no_grad():
                self.learned_weights.normal_(0, std)

        self.alpha = nn.Parameter(torch.ones(1, 1, 1, in_channels))

        self.mixer = MLPBlock(in_channels, in_channels, num_layers=1, activation=activation)

        self.norm1 = nn.RMSNorm(in_channels)
        self.norm2 = nn.RMSNorm(in_channels)

    def ortho_init_weights(self, module):
        with torch.no_grad():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.orthogonal_(module.weight)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def _init_weights(self, module, std=0.02):
        with torch.no_grad():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=std)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: Tensor):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C)
        """

        shortcut = x

        # Generate kernels
        B, H, W, C = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        pe = self.pe(coords)

        x = x + pe

        if self.nonlinear:
            basis_input = torch.cat([x, coords], dim=-1)
        else:
            basis_input = coords

        # print(gamma.shape, beta.shape, x.shape, trunk.shape)
        basis = self.generator(basis_input)

        basis = basis.view(B, H, W, self.m1, self.m2)
        # norm_factor = torch.sqrt(torch.mean(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        # basis = basis / (norm_factor + 1e-6)

        xhat = torch.einsum("bhwc,bhwmn->bcmn", x, basis) / (
            (H * W) ** torch.sigmoid(self.IT_scale_forward)
        )  # "Spectral" representation

        # Multiply by learned weigths in transformed space
        # xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)
        if self.separable:
            xhat = torch.einsum("bimn, imn -> bimn", xhat, self.learned_weights_1)
            xhat = torch.einsum("bimn,oi->bomn", xhat, self.learned_weights_2)
        else:
            xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)

        x_rec = torch.einsum("bcmn,bhwmn->bhwc", xhat, basis) / (
            (self.m1 * self.m2) ** torch.sigmoid(self.IT_scale_inverse)
        )
        x_rec = self.norm1(x_rec) * self.alpha + shortcut  # 1st shortcut
        shortcut = x_rec
        x_rec = self.mixer(self.norm2(x_rec)) + shortcut

        return x_rec


class FNOLITBlock(nn.Module):
    """
    FNO  branch  + Learnable Intregral transform block

    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.

    Forward Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) (real part)

    """

    def __init__(
        self,
        in_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=1,
        separable="True",
        activation=nn.GELU,
        mlp_act=nn.GELU,
        nonlinear=True,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.nonlinear = nonlinear
        self.separable = separable

        self.IT_scale_forward = nn.Parameter(torch.empty(1))
        self.IT_scale_inverse = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.IT_scale_forward, 1)
        nn.init.constant_(self.IT_scale_inverse, 1)

        self.pe = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_channels,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_act,
        )

        in_ch = in_channels + 2 if nonlinear else 2

        self.generator = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=in_ch,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=mlp_act,
        )

        self.ortho_init_weights(self.generator)

        # Learned parameters for multiplication in the transformed space (per channel).
        # self.learned_weights = nn.Parameter(torch.zeros(self.in_channels, out_channels, self.m1, self.m2))
        std = (1.0 / (in_channels)) ** 0.5
        if self.separable:  # this is equivalent to a separable convolution
            self.learned_weights_1 = nn.Parameter(torch.zeros(self.in_channels, self.m1, self.m2))
            # mixing
            self.learned_weights_2 = nn.Parameter(torch.zeros(self.in_channels, self.in_channels))
            with torch.no_grad():
                self.learned_weights_1.normal_(0, std)
                self.learned_weights_2.normal_(0, std)
        else:
            self.learned_weights = nn.Parameter(torch.zeros(self.in_channels, in_channels, self.m1, self.m2))
            with torch.no_grad():
                self.learned_weights.normal_(0, std)

        self.mixer = MLPBlock(in_channels, in_channels, num_layers=1, activation=activation)
        self.in_norm = nn.RMSNorm(in_channels)

    def ortho_init_weights(self, module):
        with torch.no_grad():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.orthogonal_(module.weight)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def _init_weights(self, module, std=0.02):
        with torch.no_grad():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=std)
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: Tensor, time: Union[None, Tensor] = None):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C)
        """

        shortcut = x

        # Generate kernels
        B, H, W, C = x.shape

        h_coords = torch.linspace(-1, 1, H, device=x.device)
        w_coords = torch.linspace(-1, 1, W, device=x.device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")

        coords = torch.stack([grid_h, grid_w], dim=-1)  # [H, W, 2]
        coords = coords.unsqueeze(0).expand(B, -1, -1, -1)
        pe = self.pe(coords)

        x = self.in_norm(x + pe, time)

        if self.nonlinear:
            basis_input = torch.cat([x, coords], dim=-1)
        else:
            basis_input = coords

        # print(gamma.shape, beta.shape, x.shape, trunk.shape)
        basis = self.generator(basis_input)

        basis = basis.view(B, H, W, self.m1, self.m2) / (H * W)
        # norm_factor = torch.sqrt(torch.mean(torch.abs(basis) ** 2, dim=(1, 2), keepdim=True))
        # basis = basis / (norm_factor + 1e-6)

        xhat = torch.einsum("bhwc,bhwmn->bcmn", x, basis) / (
            (H * W) ** torch.sigmoid(self.IT_scale_forward)
        )  # "Spectral" representation

        # Multiply by learned weigths in transformed space
        # xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)
        if self.separable:
            xhat = torch.einsum("bimn, imn -> bimn", xhat, self.learned_weights_1)
            xhat = torch.einsum("bimn,oi->bomn", xhat, self.learned_weights_2)
        else:
            xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)

        x_rec = torch.einsum("bcmn,bhwmn->bhwc", xhat, basis) / (
            (self.m1 * self.m2) ** torch.sigmoid(self.IT_scale_inverse)
        )
        x_rec = x_rec * self.alpha + shortcut  # 1st shortcut
        shortcut = x_rec
        x_rec = self.out_norm(x_rec)
        x_rec = self.mixer(x_rec)
        x_rec = x_rec + shortcut  # 2nd shortcut

        return x_rec


class GalerkinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        mlp_dim=512,
        dropout=0.1,
        delta=0.01,
        activation=nn.GELU(),
        std_ini: float = 1,
        kv_normalization=False,
        scaling: float = 1,
    ):
        super().__init__()
        self.attn = GalerkinAttention(dim, heads, delta, std_ini=std_ini, kv_normalization=kv_normalization)
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        # self.scaling = nn.Parameter(torch.empty(1))
        self.scaling = scaling

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            # nn.init.constant_(self.scaling, scaling)
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x + self.attn(x) * 1 / math.sqrt(2 * self.scaling)
        x = x + self.mlp(x) * 1 / math.sqrt(2 * self.scaling)
        return x


class GalerkinCrossAttentionBlock(nn.Module):
    """ """

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float, activation: Callable):
        super().__init__()
        self.attn = GalerkinAttention(dim=dim, heads=heads, kv_normalization=False, std_ini=1e-2)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, context):
        # Reduction step: context is [B, M_in, D], q_latent is [B, M_out, D]
        x = self.attn(x=q, context=context)
        x = x + self.mlp(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with standardized initialization."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        activation: Callable = nn.GELU,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        rmsnorm: bool = True,
        std_ini: float = 0.02,
        scaling: Union[None, float] = None,
    ):
        super().__init__()
        # Attention layer assumed to handle its own input projection init
        self.attention = Attention(dim, dim, n_heads, std_ini=std_ini)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # RMSNorm (Zhang & Sennrich, 2019) or standard LayerNorm
        self.norm1 = nn.RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim)
        self.norm2 = nn.RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim)

        # Depth-scaled initialization constraint (e.g., Megatron-LM / GPT-2)
        # Scale output projections by 1 / sqrt(2 * n_layers) to prevent variance explosion
        self.out_std = std_ini / math.sqrt(2 * scaling) if scaling is not None else std_ini

        self._init_weights(std_ini)

    def _init_weights(self, std_ini: float):
        """Standardized normal initialization across layers."""
        with torch.no_grad():
            # Input projection: Standard normal variation
            nn.init.trunc_normal_(self.mlp[0].weight, std=std_ini)
            if self.mlp[0].bias is not None:
                nn.init.zeros_(self.mlp[0].bias)

            # Output projection: Scaled by depth to stabilize residual stream
            nn.init.trunc_normal_(self.mlp[3].weight, std=self.out_std)
            if self.mlp[3].bias is not None:
                nn.init.zeros_(self.mlp[3].bias)

    def forward(
        self,
        x: torch.Tensor,
        causal_attn: bool = False,
    ) -> torch.Tensor:
        # x: [B, N, C]
        # Clean Pre-LN residual stream
        x = x + self.attention(Q=self.norm1(x), V=self.norm1(x), K=self.norm1(x), causal_attn=causal_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class LatentTemporalTransformerFullAttention(nn.Module):
    def __init__(self, dim, n_heads, num_layers=2, mlp_factor=2, max_history=5, rmsnorm=True):
        super().__init__()
        self.max_history = max_history
        self.rope = RoPE(dim // n_heads)  # RoPE on head_dim
        if rmsnorm:
            self.norm = nn.RMSNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([RoPETransformerBlock(dim, n_heads, dim * mlp_factor) for _ in range(num_layers)])

    def forward(self, z_current, z_history=None, coords_t=None, update_history=True):
        """
        z_current: [B, M, C]
        z_history: [B, T_past, M, C]
        coords_t:  [B, T_total, 1] -> Real time values
        update_history: if False, the latent history is not updated
        """
        B, M, C = z_current.shape

        # 1. Sequence construction [B, T, M, C]
        if z_history is not None and update_history:
            z_full = torch.cat([z_history, z_current.unsqueeze(1)], dim=1)
            z_full = z_full[:, -self.max_history :]
        else:
            z_full = z_current.unsqueeze(1)

        T = z_full.shape[1]

        # 2. Prepare Time for Flattened Sequence
        # coords_t must be [B, T, 1]. We expand it to [B, T, M, 1]
        if coords_t is None:
            # Fallback to frame indices if no real time provided
            full_time = torch.arange(T, device=z_current.device).view(1, T, 1).expand(B, T, 1)
        else:
            if coords_t.shape[1] < T:
                last_t = coords_t[:, -1, :]
                offsets = torch.arange(-(T - 1), 1, device=z_current.device).view(1, T, 1)
                full_time = last_t.unsqueeze(1) + offsets
            else:
                full_time = coords_t[:, -T:, :]

        # Expand time to match flattened modes: [B, T*M, 1]
        full_time = full_time.unsqueeze(2).expand(-1, -1, M, -1)  # [B, T, M, 1]
        full_time = full_time.contiguous().view(B, T * M, 1)

        # full_time = coords_t[:, :T, :].unsqueeze(2).expand(B, T, M, 1).reshape(B, T * M, 1)
        x_seq = z_full.contiguous().view(B, T * M, C)
        x_seq = self.norm(x_seq)

        # 3. Block Causal Mask [T*M, T*M]
        # Diagonal=1 to block future frames
        t_mask = torch.triu(torch.ones(T, T, device=x_seq.device), diagonal=1).bool()
        mask = t_mask.repeat_interleave(M, dim=0).repeat_interleave(M, dim=1)

        # 4. Iterative pass through custom blocks
        for layer in self.layers:
            x_seq = layer(x_seq, coords=full_time, mask=mask, rope=self.rope)

        # 5. Extract last timestep [B, M, C]
        z_refined = x_seq.view(B, T, M, C)[:, -1]

        return z_refined, z_full


class LatentTemporalMLP(nn.Module):
    def __init__(self, dim, max_history=3, mlp_factor=4, dropout=0.05):
        """
        Simple MLP for latent temporal prediction.
        Each mode M is processed independently (Shared MLP across modes).

        Args:
            dim (int): Dimension of each latent (D).
            max_history (int): Number of past steps used for prediction.
            mlp_factor (int): Expansion factor for the hidden layer.
        """
        super().__init__()
        self.max_history = max_history
        self.dim = dim

        # Input: [B, M, max_history * dim]
        # Output: [B, M, dim] (the residual delta_z)
        input_dim = max_history * dim
        hidden_dim = dim * mlp_factor

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),  # Predicts Delta Z
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_current, z_history=None):
        """
        z_current: [B, M, D]
        z_history: [B, T_past, M, D]
        """
        B, M, D = z_current.shape

        # 1. Pr�paration du contexte temporel
        if z_history is not None:
            # On prend les (max_history - 1) derniers pas de l'historique
            # z_past: [B, T_selected, M, D]
            z_past = z_history[:, -(self.max_history - 1) :]
            # Concat avec le pr�sent: [B, T_total, M, D]
            z_seq = torch.cat([z_past, z_current.unsqueeze(1)], dim=1)
        else:
            # Si pas d'historique, on r�p�te le pr�sent (fallback)
            z_seq = z_current.unsqueeze(1).repeat(1, self.max_history, 1, 1)

        # 2. Reshape pour le MLP [B, M, T_total * D]
        # On veut que chaque mode M voit son propre historique
        # [B, T, M, D] -> [B, M, T, D] -> [B, M, T*D]
        x = z_seq.transpose(1, 2).reshape(B, M, -1)

        # 3. Pr�diction du r�sidu
        delta_z = self.net(x)

        # 4. Int�gration d'Euler (R�siduel)
        z_next = z_current + delta_z

        return z_next


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, dim, n_heads, num_layers=2, mlp_factor=2, max_history=10, rmsnorm=True, std_ini=0.02):
        """
        Spatio-temporal Transformer for latent tokens.
        Alternates between Temporal Attention (mode-wise) and Spatial Attention (cross-mode).
        std_ini: std used in the init of the last projection layer in the attention block
        """
        super().__init__()
        self.max_history = max_history
        self.n_heads = n_heads

        # Rotary Positional Embedding for temporal dimension
        self.rope = RoPE(dim // n_heads, max_period=max_history * 50)

        # self.norm = nn.RMSNorm(dim) if rmsnorm else nn.LayerNorm(dim)

        # We alternate between temporal processing and spatial mixing
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "temporal": RoPETransformerBlock(dim, n_heads, dim * mlp_factor, std_ini=std_ini),
                        "spatial": TransformerBlock(dim, n_heads, mlp_dim=dim * mlp_factor, std_ini=std_ini),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, z_current, z_history=None, coords_t=None):
        """
        Forward pass with axial attention.
        """
        B, M, C = z_current.shape

        # 1. Manage history buffer [B, T, M, C]
        if z_history is not None:
            z_full = torch.cat([z_history, z_current.unsqueeze(1)], dim=1)
            z_full = z_full[:, -self.max_history :]
        else:
            z_full = z_current.unsqueeze(1)

        T = z_full.shape[1]

        # 2. Prepare temporal coordinates [B*M, T, 1]
        if coords_t is None:
            coords_t_seq = torch.arange(T, device=z_current.device).view(1, T, 1).expand(B, T, 1)
        else:
            coords_t_seq = coords_t[:, -T:, :]

        # Replicate for each mode for the temporal RoPE
        full_time = coords_t_seq.repeat_interleave(M, dim=0)

        # 3. Interleaved Axial Attention
        # Initial state x: [B, T, M, C]
        x = z_full

        for layer in self.layers:
            # --- A. Temporal Attention (Independent per mode) ---
            # Reshape to [B*M, T, C]
            x = x.transpose(1, 2).reshape(B * M, T, C)
            # x = self.norm(x)
            x = layer["temporal"](x, coords=full_time, mask=None, rope=self.rope)

            # Reshape back to [B, T, M, C]
            x = x.view(B, M, T, C).transpose(1, 2)

            # --- B. Spatial Attention (Mixing modes) ---
            # Reshape to [B*T, M, C]
            x = x.reshape(B * T, M, C)
            # x = self.norm(x)
            x = layer["spatial"](x)

            # Reshape back to [B, T, M, C]
            x = x.view(B, T, M, C)

        # 4. Final refinement
        # Extract last frame: [B, M, C]
        z_refined = x[:, -1, :, :]

        return z_refined, x


class RoPETransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.1, rmsnorm=True, std_ini=0.02):
        super().__init__()

        if rmsnorm:
            self.norm1 = nn.RMSNorm(dim)  # AdaRMSNorm(dim, dim)
            self.norm2 = nn.RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)  # AdaptiveLayerNorm(dim, dim)
            self.norm2 = nn.LayerNorm(dim)

        self.attn = Attention(dim, dim, n_heads, std_ini=std_ini)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, coords, mask=None, rope=None):
        # x: [B, T*M, C], time: [B, T*M, 1]
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, mask=mask, rope=rope, coords=coords)
        x = x + self.mlp(self.norm2(x))

        return x


class GalerkinTransolverBlock(nn.Module):
    """Projection using a Galerkin Attention then attention then reconstruction using Galerkin Attention"""

    def __init__(
        self, dim, num_heads, modes, mlp_factor=2, dropout=0.1, activation=nn.GELU, kv_normalization=False, scaling=1.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.modes = modes  # M: Number of modes

        self.encoder = GalerkinAttention(dim=dim, heads=num_heads, kv_normalization=kv_normalization, std_ini=1e-2)
        self.decoder = GalerkinAttention(dim=dim, heads=num_heads, kv_normalization=kv_normalization, std_ini=1e-2)
        self.queries_predictor = MLPBlock(out_ch=dim, in_ch=2, hidden_dim=dim, num_layers=1, activation=activation)
        self.scaling = nn.Parameter(torch.empty(1))
        self.latent_pe = MLPBlock(
            out_ch=dim,
            in_ch=2,
            hidden_dim=dim,
            num_layers=1,
            activation=activation,
        )

        self.attention = Attention(
            in_ch=dim,
            out_ch=dim,
            num_heads=num_heads,
            std_ini=1e-2,
        )

        self.out_proj = MLPBlock(
            out_ch=dim,
            in_ch=dim,
            hidden_dim=dim,
            num_layers=1,
            activation=activation,
        )

        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)

        with torch.no_grad():
            nn.init.constant_(self.scaling, scaling)

    def forward(self, x):

        B, N, D = x.shape

        # Add Positional encoding in latent representation before the self-attention modules

        m_coords = torch.linspace(-1, 1, int(math.sqrt(self.modes)), device=x.device)
        m1, m2 = torch.meshgrid(m_coords, m_coords, indexing="ij")
        q_coords = torch.stack([m1, m2], dim=-1)
        q_coords = q_coords.view(self.modes, 2)
        q_coords = q_coords.unsqueeze(0).expand(B, -1, -1)
        queries = self.queries_predictor(q_coords)
        latent_pe = self.latent_pe(q_coords)

        # Galerkin Cross Attention -> [B, M, C]
        xhat = self.encoder(x=queries, context=self.norm1(x))
        xhat = xhat + latent_pe

        # Multi-head attention
        xhat = self.attention(xhat)

        # Go back to spatial domain do we need normalization here?
        x_rec = self.decoder(x=x, context=xhat)
        x_rec = self.scaling * self.out_proj(self.norm2(x)) + x

        return x_rec


class TransolverBlock(nn.Module):
    """
    Base block for Transolver++
    see https://github.com/thuml/Transolver_plus/blob/b2f656d382d0c8454415f004f1c7af1436f90c42/models/Transolver_plus.py#L16
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, mlp_ratio=2, activation=nn.GELU):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num), activation(), nn.Linear(slice_num, 1), activation()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for m in [self.in_project_slice]:
            torch.nn.init.orthogonal_(m.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.proj = MLPBlock(inner_dim, dim, hidden_dim=mlp_ratio * dim, activation=activation)

    def forward(self, x):
        # input B N C
        B, N, C = x.shape

        x_mid = self.ln1(x)
        # --- Physic Attention block
        x_mid = (
            self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        )  # B H N C

        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)
        slice_weights = gumbel_softmax(self.in_project_slice(x_mid), temperature)
        slice_norm = slice_weights.sum(2)  # B H K - K number of physical tokens
        slice_token = torch.einsum("bhnc,bhnk->bhkc", x_mid, slice_weights).contiguous()  # B, num_head, K, dim_head
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_slice_token, k_slice_token, v_slice_token)

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")

        # --- End of Physic Attention block

        out_x = self.proj(self.ln2(out_x)) + x

        return out_x


class LinearNOBlock(nn.Module):
    """
    Implementation of LinearNO.
    Ref: "Transolver Is a Linear Transformer: Revisiting Physics-Attention..." (2026)
    """

    def __init__(self, dim, n_tokens=64, n_heads=8, dropout=0.0, mlp_ratio=2, activation=nn.GELU, rmsnorm=True):
        super().__init__()
        self.n_heads = n_heads
        self.n_tokens = n_tokens  # This is 'M' (number of slices/latent tokens)
        self.dim = dim
        self.head_dim = dim // n_heads

        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        if rmsnorm:
            self.norm1 = nn.RMSNorm(dim)
            self.norm2 = nn.RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

        # Asymmetric learnable projections for linear attention (Eq. 8 & 9)
        # Slicing projections project onto M latent spaces per head
        self.q_proj = nn.Linear(dim, n_heads * n_tokens)
        self.k_proj = nn.Linear(dim, n_heads * n_tokens)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.mlp = MLPBlock(in_ch=dim, out_ch=dim, hidden_dim=dim * mlp_ratio, num_layers=1, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        H = self.n_heads
        M = self.n_tokens
        D = self.head_dim

        # Step 1: Pre-normalization
        x_in = self.norm1(x)

        # Step 2: Projections & Shape transformation
        # q, k shape: [B, N, H, M] -> [B, H, N, M]
        q = self.q_proj(x_in).view(B, N, H, M).permute(0, 2, 1, 3)
        k = self.k_proj(x_in).view(B, N, H, M).permute(0, 2, 1, 3)
        # v shape: [B, N, H, D] -> [B, H, N, D]
        v = self.v_proj(x_in).view(B, N, H, D).permute(0, 2, 1, 3)

        # Step 3: Apply asymmetric kernel normalizations (Eq. 8)
        # phi(Q): Softmax over slice dimension M (row-wise for Q)
        phi_q = torch.softmax(q, dim=-1)  # [B, H, N, M]

        # psi(K)^T: Softmax over grid dimension N (row-wise for K^T)
        # We transpose N and M for K to apply softmax row-by-row on K^T
        k_t = k.permute(0, 1, 3, 2)  # [B, H, M, N]
        psi_k_t = torch.softmax(k_t, dim=-1)  # [B, H, M, N]

        # Step 4: Canonical Linear Attention computation (Eq. 9)
        # Global context formulation: psi(K)^T @ V -> [B, H, M, N] @ [B, H, N, D] = [B, H, M, D]
        global_context = torch.matmul(psi_k_t, v)

        # Deslicing: phi(Q) @ global_context -> [B, H, N, M] @ [B, H, M, D] = [B, H, N, D]
        attn_out = torch.matmul(phi_q, global_context)

        # Step 5: Post-projections & Residual blocks
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        x = x + self.dropout(self.out_proj(attn_out))

        # MLP block
        out = x + self.mlp(self.norm2(x))
        return out


import math
import torch
import torch.nn as nn


class PEBlock(nn.Module):
    """Positional Encoding Block supporting multiple encoding strategies across

    1D, 2D, and 3D space.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        method: str = "fourier",
        **kwargs,
    ):
        super().__init__()

        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if not isinstance(out_features, int) or out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")

        valid_methods = {
            "random_fourier",
            "fourier",
            "ipe",
            "random_ipe",
            "siren",
            "mlp",
            "adaptive",
            None,
        }
        if method not in valid_methods:
            print(f"Method '{method}' not supported. Defaulting to 'fourier'.")
            method = "fourier"

        self.method = method
        self.in_features = in_features
        self.out_features = out_features

        mlp_dim = kwargs.get("mlp_dim", out_features)
        mlp_act = kwargs.get("mlp_act", nn.GELU)
        mlp_layers = kwargs.get("mlp_layers", 2)

        match method:
            case "random_fourier":
                scale = kwargs.get("scale", 10.0)
                self.pe = GaussianFourierEmbedding(in_features, out_features, scale)

            case "fourier":
                max_freq = kwargs.get("max_freq", 128.0)
                self.pe = FourierEmbedding(in_features, out_features, max_freq=max_freq)

            case "ipe":
                max_freq = kwargs.get("max_freq", 128.0)
                self.pe = IntegratedPositionalEncoding(in_features, out_features, max_freq=max_freq)

            case "random_ipe":
                scale = kwargs.get("scale", 10.0)
                self.pe = GaussianIntegratedPositionalEncoding(in_features, out_features, scale=scale)

            case "siren":
                self.pe = SirenBlock(
                    out_ch=out_features,
                    in_ch=in_features,
                    hidden_dim=mlp_dim,
                    num_layers=mlp_layers,
                    w0=30,
                )

            case "mlp":
                self.pe = MLPBlock(
                    out_ch=out_features,
                    in_ch=in_features,
                    hidden_dim=mlp_dim,
                    num_layers=mlp_layers,
                    activation=mlp_act,
                )
            case "adaptive":
                self.pe = AdaptiveFourierEmbedding(
                    in_features=in_features, out_features=out_features, base_max_freq=kwargs.get("max_freq", 128.0)
                )
            case None:
                self.pe = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies positional encoding dynamically based on input coordinates.

        Args:
            x (torch.Tensor): Coordinates of shape [B, N, in_features] or [B,
              H, W, in_features]. We assume that the coordinates are normalized beween 1 and -1

        Returns:
            torch.Tensor: Encoded positions.
        """
        if x.ndim == 3:
            B, N, C = x.shape
            flat = True
        elif x.ndim == 4:
            B, H, W, C = x.shape
            flat = False
        else:
            raise ValueError("Input tensor must be B, N, C (flattened coordinates) or B, H, W, C")

        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input last dim ({x.shape[-1]}) mismatch with in_features ({self.in_features})")

        if self.method in ["ipe", "random_ipe"]:
            # 1. Compute grid spacing under the uniform domain [-1, 1]^C
            if flat:
                # Case [B, N, C]: No explicit spatial structure.
                # We fall back to assuming a hypercube grid: N^(1/C) points per axis.
                grid_res = math.pow(N, 1.0 / C)
                # Cell size is identical for all dimensions in this fallback scenario
                cell_size_values = [2.0 / grid_res] * C
            else:
                # Case [B, H, W, C]: Dimensions are explicit (e.g., C=2 for 2D).
                # Each axis has its own sample density.
                # Assuming domain width is 2.0 (from -1 to 1) for both axes:
                dx = 2.0 / H
                dy = 2.0 / W
                cell_size_values = [dx, dy]

            # 2. Create the tensor aligned with the input device and dtype
            # Shape: [C] (matches the last dimension of x)
            cell_size = torch.tensor(cell_size_values, dtype=x.dtype, device=x.device)

            # Pass coordinates and the axis-specific spacing vector
            return self.pe(x, cell_size=cell_size)

        elif self.method == "adaptive":
            if flat:
                res = tuple([int(N ** (1.0 / C)) for _ in range(C)])
            else:
                res = (H, W)
            return self.pe(x, resolution=res)

        return self.pe(x)


class MHSlicingBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_slices: int,
        num_heads: int,
        temp: float = 1.0,
        use_gumbel: bool = False,
    ):
        """
        Multi-Head Slicing Block inspired by Transolver.
        in_features:  Dimension C of input features from mesh points
        out_features: Total hidden dimension (must be divisible by num_heads)
        num_slices:   Number of physical states/slices M per head
        num_heads:    Number of heads H
        """
        super().__init__()
        self.M = num_slices
        self.H = num_heads
        self.d_k = out_features // num_heads
        self.gumbel = use_gumbel

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        # Separate projections for weights per head: H * M outputs
        self.weight_proj = nn.Linear(in_features, self.H * self.M)

        temp_ini = torch.log(torch.exp(torch.tensor(temp)) - 1.0)
        self.temp = nn.Parameter(temp_ini)

        # Projection of input features before slicing
        self.linear1 = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        x: [B, N, C] - Input tensor from mesh points
        Returns:
            s: [B, M, out_features] - Sliced physics tokens with combined heads
        """
        B, N, C = x.shape

        # --- 1. COMPUTE MULTI-HEAD SLICE WEIGHTS (w) AND DIAGONAL (d) ---
        # Project and reshape to isolate heads: [B, N, H, M]
        weight_logits = self.weight_proj(x).view(B, N, self.H, self.M)
        temp = F.softplus(self.temp) + 0.05
        if self.gumbel:
            w = gumbel_softmax(weight_logits, temp, dim=-1)
        else:
            w = torch.softmax(weight_logits / temp, dim=-1)

        # d represents the sum of weights per slice per head
        # Sum over N -> [B, 1, H, M]
        d = w.sum(dim=1, keepdim=True)

        # --- 2. PREPARE SLICE INPUT AND PROJECTION ---
        # Apply linear1 and reshape to separate heads: [B, N, H, d_k]
        # Then permute to [B, H, N, d_k] for batch matrix multiplication
        x_projected = self.linear1(x).view(B, N, self.H, self.d_k).permute(0, 2, 1, 3)

        # Permute w to [B, H, M, N]
        w_t = w.permute(0, 2, 3, 1)

        # Slicing via batch matrix multiplication: [B, H, M, N] @ [B, H, N, d_k]
        s_raw = torch.matmul(w_t, x_projected)  # Shape: [B, H, M, d_k]

        # --- 3. FASTER SLICE WITH MULTI-HEAD DIAGONAL INVERSION ---
        # Permute d from [B, 1, H, M] to [B, H, M, 1] to align with s_raw
        d_inv = 1.0 / (d.permute(0, 2, 3, 1) + 1e-8)

        # Scale sliced tokens per head
        s_heads = s_raw * d_inv  # Shape: [B, H, M, d_k]

        # --- 4. COMBINE HEADS FOR DOWNSTREAM ATTENTION BLOCKS ---
        # [B, H, M, d_k] -> [B, M, H, d_k] -> [B, M, out_features]
        s = s_heads.permute(0, 2, 1, 3).contiguous().view(B, self.M, -1)

        return s, w


class MHDeslicingBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_slices: int, num_heads: int):
        """
        Args:
            in_features: Latent dimension D [B, M, D], must be divisible by num_heads.
            out_features: Output physical dimension C.
            num_slices: Number of slices M.
            num_heads: Number of heads H.
        """
        super().__init__()
        self.M = num_slices
        self.H = num_heads
        self.d_k = in_features // num_heads
        assert in_features % num_heads == 0, "in_features must be divisible by num_heads"

        # Per-head projection: d_k -> d_k (reduced space, symmetric to linear1)
        self.linear3 = nn.Linear(self.d_k, self.d_k)
        # Final projection after head recombination: D -> out_features
        self.linear_out = nn.Linear(in_features, out_features)

    def forward(self, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Latent tokens from attention layers [B, M, D].
            w: Slicing weights from SlicingBlock [B, N, H, M].
        Returns:
            x_out: Output tensor [B, N, out_features].
        """
        B, M, D = s.shape

        # --- 1. SPLIT INTO HEADS AND PROJECT ---
        # [B, M, D] -> [B, M, H, d_k] -> [B, H, M, d_k]
        s_heads = s.view(B, M, self.H, self.d_k).permute(0, 2, 1, 3)
        # Per-head projection (Faster Deslice)
        s_heads = self.linear3(s_heads)  # [B, H, M, d_k]

        # --- 2. PER-HEAD SPATIAL RECONSTRUCTION ---
        # [B, N, H, M] -> [B, H, N, M]
        w_deslice = w.permute(0, 2, 1, 3)
        # [B, H, N, M] @ [B, H, M, d_k] -> [B, H, N, d_k]
        x_out_heads = torch.matmul(w_deslice, s_heads)

        # --- 3. HEAD RECOMBINATION ---
        # [B, H, N, d_k] -> [B, N, H, d_k] -> [B, N, D]
        x_out = x_out_heads.permute(0, 2, 1, 3).contiguous().view(B, -1, D)

        # --- 4. FINAL PROJECTION ---
        x_out = self.linear_out(x_out)  # [B, N, out_features]

        return x_out


class SlicingBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_slices: int, temp: float = 1.0):
        super().__init__()
        self.M = num_slices
        self.weight_proj = nn.Linear(in_features, num_slices)  # ? w ? R^{N�M}
        self.linear1 = nn.Linear(in_features, out_features)  # x_proj
        temp_ini = torch.log(torch.exp(torch.tensor(temp)) - 1.0)
        self.temp = nn.Parameter(temp_ini)

    def forward(self, x):
        B, N, C = x.shape
        temp = F.softplus(self.temp) + 0.05

        # w ? [B, N, M]
        w = torch.softmax(self.weight_proj(x) / temp, dim=-1)

        # d ? [B, M] � somme des poids par slice
        d = w.sum(dim=1)  # [B, M]

        # s_raw = w? x_proj : [B, M, N] @ [B, N, D] ? [B, M, D]
        s_raw = self.linear1(torch.bmm(w.permute(0, 2, 1), x))

        # Normalisation diagonale
        s = s_raw / (d.unsqueeze(-1) + 1e-8)  # [B, M, D]

        return s, w


class DeslicingBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.linear3 = nn.Linear(in_features, out_features)

    def forward(self, s_prime, w):
        # s_prime : [B, M, D],  w : [B, N, M]

        # Faster deslice (Eq. 3)
        s_out = self.linear3(s_prime)  # [B, M, C_out]

        # Reprojection vers N : w @ s_out ? [B, N, M] @ [B, M, C_out]
        x_out = torch.bmm(w, s_out)  # [B, N, C_out]

        return x_out


class TransolverBlockv3(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_slices: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        temp: float = 1.0,
    ):
        """
        Complete Transolver3 Block combining Slicing, L Attention Layers, and Deslicing.
        in_features:     Input/output physical channels per mesh point (C)
        hidden_features: Latent dimension for attention layers (D)
        num_slices:      Number of physical slices M
        num_heads:       Number of heads H
        num_layers:      Number of attention layers L
        dim_feedforward: Hidden dimension of FFN (defaults to 4 * hidden_features)
        """
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_features

        self.norm = nn.LayerNorm(in_features)

        # Slicing: Mesh space [B, N, C] -> Slice space [B, M, D]
        self.slicing = SlicingBlock(
            in_features=in_features,
            out_features=hidden_features,
            num_slices=num_slices,
            temp=temp,
        )
        # L Layers of Transformer Attention on slice space
        self.attention_layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=hidden_features,
                    n_heads=num_heads,
                    mlp_dim=dim_feedforward,
                    dropout=dropout,
                    activation=nn.GELU,
                )
                for _ in range(num_layers)
            ]
        )
        # Deslicing: Slice space [B, M, D] -> Mesh space [B, N, C]
        self.deslicing = DeslicingBlock(
            in_features=hidden_features,
            out_features=in_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] - Input mesh tensor
        Returns: [B, N, C] - Output mesh tensor
        """
        # Pre-norm + r�siduelle
        residual = x
        x = self.norm(x)

        s, w = self.slicing(x)
        for layer in self.attention_layers:
            s = layer(s)
        x_out = self.deslicing(s, w)

        return residual + x_out


class IAETBlock(nn.Module):
    """
    Non Reversible Non Linear Integral Transform Block is a
    PyTorch module for a learned 2D non linear integral transform that is resolution-invariant.
    The transform bases are learned as continuous functions via MLPs.
    In this implementation the inverse transform is also learnt.
    The kernels are conditinioned on the transformed function.
    Note that the conditionning is not done the same way in the forward transform and the inverse one.
    As the kernels are learnt both in the encoder and the decoder, the transformation is not reversible
    Strongly inspired by IAE-NET: INTEGRAL AUTOENCODERS FOR DISCRETIZATION-INVARIANT LEARNING

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        activation=nn.GELU,
        norm=LayerNorm2d,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            output_shape: (tuple(H,W)): shape of the reconstructed image
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.coords_dim = 1 + in_channels

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = ComplexMLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = ComplexMLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        # Decoder MLPs: bases conditionnées par les valeurs des modes spectraux (fréquentiels)
        # Pour la transformation inverse H (m1 -> H), conditionnée par (C_out, m2) modes
        self.mlp_h_inv = ComplexMLPBlock(
            in_ch=1 + 2 * out_channels,
            out_ch=m1,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.mlp_w_inv = ComplexMLPBlock(
            in_ch=1 + 2 * out_channels,
            out_ch=m2,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        self.compressor_h_inv = nn.Linear(m1, 1)
        self.compressor_w_inv = nn.Linear(m2, 1)

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Initialization of learned_weights_freq
        nn.init.constant_(self.learned_weights_freq, 1.0)  # Initialize to 1+0j

        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        b, c, h, w = x.shape
        # Convert input to complex if it is real
        xc = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x.float()

        # --- Encoder (Direct transform) ---
        # 1. W (width) transformation
        # We predict m2 kernels for each spatial coordinates
        # Input MLP: (B, H, W, C+1)
        w_coords = torch.linspace(-1, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        mlp_w_in = torch.concat([xc.real.permute(0, 2, 3, 1), w_coords], dim=-1)
        b_w = self.basis_w_fn(mlp_w_in)  # (B, H, W, m2)

        # Matmul: (B, H, C, W) @ (B, H, W, m2) -> (B, H, C, m2)
        x_w = torch.matmul(xc.permute(0, 2, 1, 3), b_w)
        x_w = x_w.permute(0, 2, 1, 3)  # (B, C, H, m2)

        # 2. H (height) transformation
        # Input MLP: (B, m2, H, C+1)
        h_coords = (
            torch.linspace(-1, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )
        mlp_h_in = torch.concat([x_w.real.permute(0, 3, 2, 1), h_coords], dim=-1)
        # predict m1 kernels for eachcoordinates (m2, H)
        b_h = self.basis_h_fn(mlp_h_in)  # (B, m2, H, m1)

        # Matmul: (B, m2, C, H) @ (B, m2, H, m1) -> (B, m2, C, m1)
        x_hw = torch.matmul(x_w.permute(0, 3, 1, 2), b_h)
        x_hw = x_hw.permute(0, 2, 3, 1)  # (B, C, m1, m2)

        # --- Frequency multiplication ---
        xf = torch.einsum("bixy,oixy->boxy", x_hw, self.learned_weights_freq)  # (B, C_out, m1, m2)

        # --- Decoder (Inverse transformation) ---
        # 1. Inverse H transformation (height reconstruction)
        # Output spatial coordinates for H
        h_rec_coords = (
            torch.linspace(-1, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )  # B, m2, H, 1

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # xf is (B, C_out, m1, m2)
        # Concatenate real and imaginary: (B, 2*C_out, m1, m2)
        xf_combined_real_imag = torch.concat([xf.real, xf.imag], dim=1)

        # Permute so that m1 is the last dimension for the compressor: (B, 2*C_out, m2, m1)
        cond_h_val_for_comp = xf_combined_real_imag.permute(0, 1, 3, 2)
        # Apply the compressor: (B, 2*C_out, m2, 1)
        cond_h_val_compressed = self.compressor_h_inv(cond_h_val_for_comp)
        # Repeat the compressed value along the output spatial dimension H: (B, 2*C_out, m2, h)
        cond_h_val_repeated = cond_h_val_compressed.repeat(1, 1, 1, h)
        # Permute for the MLP: (B, m2, h, 2*C_out)
        cond_h_val_mlp_in = cond_h_val_repeated.permute(0, 2, 3, 1)

        mlp_h_inv_in = torch.concat([h_rec_coords, cond_h_val_mlp_in], dim=-1)  # (B, m2, h, 1 + 2*C_out)
        b_h_inv = self.mlp_h_inv(mlp_h_inv_in)  # (B, m2, h, m1) - Basis generated by MLP

        # Matmul: (B, m2, C_out, m1) @ (B, m2, m1, h).mH -> (B, m2, C_out, h)
        x_h_rec = torch.matmul(xf.permute(0, 3, 1, 2), b_h_inv.mH)  # Still complex
        x_h_rec = x_h_rec.permute(0, 2, 3, 1)  # (B, C_out, h, m2) - Still complex

        # 2. Inverse W transformation (width reconstruction)
        # Output spatial coordinates for W
        w_rec_coords = (
            torch.linspace(-1, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        )

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # x_h_rec is (B, C_out, h, m2)
        # Concatenate real and imaginary: (B, 2*C_out, h, m2)
        x_h_rec_combined_real_imag = torch.concat([x_h_rec.real, x_h_rec.imag], dim=1)

        # Permute so that m2 is the last dimension for the compressor: (B, 2*C_out, h, m2)
        cond_w_val_for_comp = x_h_rec_combined_real_imag
        # Apply the compressor: (B, 2*C_out, h, 1)
        cond_w_val_compressed = self.compressor_w_inv(cond_w_val_for_comp)
        # Repeat the compressed value along the output spatial dimension W: (B, 2*C_out, h, w)
        cond_w_val_repeated = cond_w_val_compressed.repeat(1, 1, 1, w)
        # Permute for the MLP: (B, h, w, 2*C_out)
        cond_w_val_mlp_in = cond_w_val_repeated.permute(0, 2, 3, 1)

        mlp_w_inv_in = torch.concat([w_rec_coords, cond_w_val_mlp_in], dim=-1)  # (B, h, w, 1 + 2*C_out)
        b_w_inv = self.mlp_w_inv(mlp_w_inv_in)  # (B, h, w, m2) - Basis generated by MLP

        # Matmul: (B, h, C_out, m2) @ (B, h, m2, w).mH -> (B, h, C_out, w)
        x_rec = torch.matmul(x_h_rec.permute(0, 2, 1, 3), b_w_inv.mH).real
        x_rec = x_rec.permute(0, 2, 1, 3)  # (B, C_out, h, w)

        # Dynamic normalization (based on output resolution)
        x_rec = x_rec * (1.0 / (h * w))

        # Mixing channels
        output = self.mixer(x_rec)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = self.activation(output + self.shortcut(x))

        return output
