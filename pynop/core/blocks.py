from re import L
from typing import Callable, Union, Sequence
import numpy as np
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

from .utils import make_tuple
from .ops import *
from .norm import LayerNorm2d

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
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            hidden_channels:Number of output channels for the SpectralConv
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

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, self.hidden_channels, modes, scaling=scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, self.hidden_channels, modes, ranks, scaling=scaling)
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

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        # Main branch computation
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)

        if self.norm is not None:
            x = self.norm(x)

        x = x + x_shortcut  # (B, out_channels, H, W)

        if self.activation is not None:
            x = self.activation(x)

        return x


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
            x = torch.cat([shortcuts[i], x], 1)
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

        print(q.shape, k.shape, v.shape)

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


class DIT(nn.Module):
    """
    PyTorch module for a learned 2D integral transform with flexible initialization.
    It performs a direct transform to a learned mode space,
    applies a multiplication by learned weights in this space,
    then performs an inverse transform to reconstruct the signal.

    The direct and inverse transform matrices are linked by the conjugate transpose.
    The basis initialization is either 'fourier' or 'random'.
    """

    def __init__(self, in_channels, height, width, m1, m2, initialization_method="fourier"):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            height (int): Input image height (H).
            width (int): Input image width (W).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            initialization_method (str): Basis initialization method.
                                         Can be 'fourier' (default) or 'random'.
        """
        super().__init__()

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.m1 = m1
        self.m2 = m2

        # The basis matrices are first created empty, then initialized by _initialize_bases.
        self.transform_h_basis = nn.Parameter(torch.empty(self.m1, self.height, dtype=torch.cfloat))
        self.transform_w_basis = nn.Parameter(torch.empty(self.m2, self.width, dtype=torch.cfloat))

        # Learned parameters for multiplication in the transformed space.
        self.learned_weights_freq = nn.Parameter(torch.randn(self.in_channels, self.m1, self.m2, dtype=torch.cfloat))

        # Normalization factor for the inverse transform.
        self.register_buffer("normalisation_factor", torch.tensor(1.0 / (self.height * self.width), dtype=torch.float))

        # Call the chosen initialization method
        self._initialize_bases(initialization_method)

    def _initialize_bases(self, method):
        """
        Initializes the basis matrices according to the specified method.
        """
        if method == "fourier":
            # Initialize transform_h_basis (m1 x H) with Fourier bases
            u_indices_h = torch.arange(self.m1, dtype=torch.cfloat).unsqueeze(1)
            m_indices_h = torch.arange(self.height, dtype=torch.cfloat).unsqueeze(0)
            transform_h_initial = torch.exp(
                torch.complex(torch.tensor(0.0), -2j * torch.pi * u_indices_h * m_indices_h / self.height)
            )
            self.transform_h_basis.data = transform_h_initial

            # Initialize transform_w_basis (m2 x W) with Fourier bases
            v_indices_w = torch.arange(self.m2, dtype=torch.cfloat).unsqueeze(1)
            n_indices_w = torch.arange(self.width, dtype=torch.cfloat).unsqueeze(0)
            transform_w_initial = torch.exp(
                torch.complex(torch.tensor(0.0), -2j * torch.pi * v_indices_w * n_indices_w / self.width)
            )
            self.transform_w_basis.data = transform_w_initial

        elif method == "random":
            # Random initialization of the basis matrices
            # For complex tensors, `normal_` initializes the real and imaginary parts
            # with a standard normal distribution (mean 0, std 1).
            # We use normal_ (in-place) on .data to modify the existing parameters.
            self.transform_h_basis.data.real.normal_(0, 1)
            self.transform_h_basis.data.imag.normal_(0, 1)
            self.transform_w_basis.data.real.normal_(0, 1)
            self.transform_w_basis.data.imag.normal_(0, 1)

        else:
            raise ValueError(f"Initialization method '{method}' not recognized. Choose 'fourier' or 'random'.")

    def forward(self, x):
        """
        Performs the forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        batch_size, channels, H, W = x.shape

        # Check input dimensions
        if H != self.height or W != self.width:
            raise ValueError(
                f"Input dimensions ({H}, {W}) do not match initialized dimensions ({self.height}, {self.width})."
            )

        # Convert input to complex numbers if it is real
        x_complex = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # --- Step 1: Direct transform (projection onto learned modes) ---
        # Analogous to F_truncated = W'_M * f * W'_N

        # Transform along rows (dimension H -> m1 modes)
        # x_complex: (B, C, H, W)
        # self.transform_h_basis: (m1, H)
        # Result: (B, C, m1, W)
        transformed_freq_h = torch.einsum("bchw,mh->bcmw", x_complex, self.transform_h_basis)

        # Transform along columns (dimension W -> m2 modes)
        # transformed_freq_h: (B, C, m1, W)
        # self.transform_w_basis: (m2, W)
        # Result: (B, C, m1, m2)
        transformed_freq_hw = torch.einsum("bcmw,nw->bcmn", transformed_freq_h, self.transform_w_basis)

        # --- Step 2: Multiplication by learned weights in the transformed space ---
        # self.learned_weights_freq: (C, m1, m2)
        # transformed_freq_hw: (B, C, m1, m2)
        # The unsqueeze(0) allows broadcasting over the batch dimension.
        processed_freq_domain = transformed_freq_hw * self.learned_weights_freq.unsqueeze(0)

        # --- Step 3: Inverse transform (projection back) ---
        # Analogous to f_approx = (W'_M)^H * F_truncated * (W'_N)^H

        # Inverse transform along columns (m2 modes -> W)
        # processed_freq_domain: (B, C, m1, m2)
        # self.transform_w_basis.H: (W, m2) (conjugate transpose of (m2, W))
        # Result: (B, C, m1, W)
        reconstructed_spatial_w = torch.einsum("bcmn,wn->bcmw", processed_freq_domain, self.transform_w_basis.H)

        # Inverse transform along rows (m1 modes -> H)
        # reconstructed_spatial_w: (B, C, m1, W)
        # self.transform_h_basis.H: (H, m1) (conjugate transpose of (m1, H))
        # Result: (B, C, H, W)
        reconstructed_spatial_hw = torch.einsum("bcmw,hm->bchw", reconstructed_spatial_w, self.transform_h_basis.H)

        # Apply the normalization factor of the IDFT
        reconstructed_spatial_hw = reconstructed_spatial_hw * self.normalisation_factor

        # Return the real part of the resulting tensor, as images are usually real.
        return reconstructed_spatial_hw.real


class MLPBlock(nn.Module):
    """
    A small MLP that produces (complex) values for a set of basis functions.
    """

    def __init__(self, in_ch, out_ch, hidden_dim=64, num_layers=2, activation=nn.GELU, norm=nn.LayerNorm, dropout=0.0):
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


class Linear1x1Conv(nn.Module):
    """
    Linear layers implemented with 1x1 convolutions
    Input/Output format: (B, C, H, W).
    """

    def __init__(self, in_ch, out_ch, hidden_dim=64, num_layers=2, activation=nn.GELU, norm=nn.LayerNorm):
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
        self.out_ch = out_ch

        layers = []

        layers.append(nn.Conv2d(in_ch, hidden_dim, kernel_size=1))
        if norm is not None:
            layers.append(norm(hidden_dim))
        if activation is not None:
            layers.append(activation())  # Or ReLU, LeakyReLU, etc.

        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            if norm is not None:
                layers.append(norm(hidden_dim))
            if activation is not None:
                layers.append(activation())

        # The last layer produces 2 * num_modes real values (for real and imaginary parts)
        layers.append(nn.Conv2d(hidden_dim, 2 * out_ch, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 2 + C_in, H_basis, W_basis)
        out = self.net(x)  # (B, 2*m1*m2, H_basis, W_basis)

        # Split Real and Imaginary parts (along the channel dimension)
        out_real = out[:, : self.out_ch, :, :]
        out_imag = out[:, self.out_ch :, :, :]

        return torch.complex(out_real, out_imag)


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
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(out_channels, self.in_channels, self.m1, self.m2, dtype=torch.cfloat)
        )

        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
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

        # Generate normalized coordinates for the current height and width.
        # These coordinates are passed to our MLPs to generate the bases dynamically.
        h_coords = torch.linspace(0, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        w_coords = torch.linspace(0, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)  # (W, 1)

        # Dynamically generate the basis matrices for the current resolution
        # base_values_h: (H, m1)
        # base_values_w: (W, m2)
        base_values_h = self.basis_h_fn(h_coords)
        base_values_w = self.basis_w_fn(w_coords)

        # Transpose the generated bases so they have shape (m1, H) and (m2, W)
        # for matrix multiplication
        # transform_h_basis_runtime: (m1, H)
        # transform_w_basis_runtime: (m2, W)
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


class LITBlock(nn.Module):
    """
    Linear Integral Transform Block (LITBlock) is a
    PyTorch module for a learned 2D integral transform that is resolution-invariant.
    The transform bases are learned as continuous functions via MLPs.
    This is the non-separable kernels implementation
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
        norm=LayerNorm2d,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            resampling: "down" or "up": resample the output by a factor 2.
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.mlp = MLPBlock(
            out_ch=self.m1 * self.m2,
            in_ch=2,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        # Learned parameters for multiplication in the transformed space (per channel).
        self.learned_weights = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Initialization of learned_weights for a good starting point (e.g., all to 1+0j)
        nn.init.constant_(self.learned_weights, 1.0)

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
        xc = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # if self.resample == "down":
        # elif self.resample == "up":
        # else:
        h_coords = torch.linspace(0, 1, H, device=x.device).unsqueeze(1).repeat(1, W)
        w_coords = torch.linspace(0, 1, W, device=x.device).unsqueeze(0).repeat(H, 1)
        coords_2d = torch.stack([h_coords, w_coords], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        coords_2d = coords_2d.view(B * H * W, 2)

        encoder_basis = self.mlp(coords_2d)  # (B*H*W, m1*m2)

        # 4. Reshape kernels for Einsum
        encoder_basis = encoder_basis.view(B, H, W, self.m1, self.m2)

        # Einsum: (B, C_in, H, W) @ (B, H, W, m1, m2) -> (B, C_in, m1, m2)
        xhat = torch.einsum("bchw,bhwmn->bcmn", xc, encoder_basis)  # "Spectral" representation

        # Multiply by learned weigths in transformed space
        xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)

        # ---2D Inverse Transform ---
        # Einsum: (B, C_out, m1, m2) @ (B, m1, m2, H_out, W_out)* -> (B, C_out, H_out, W_out)
        # We use .conj() for the inverse (adjoint) operation.
        x_rec = torch.einsum("bcmn,bhwmn->bchw", xhat, encoder_basis.conj())

        # Normalization
        x_rec = x_rec.real * (1.0 / (H * W))

        # Mixing channels
        output = self.mixer(x_rec)
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
        norm=LayerNorm2d,
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
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
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
        w_coords = torch.linspace(0, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)

        # Add function values to condition our kernel to the input
        x_w = x_complex.real.permute(0, 2, 3, 1)  # (B, H, W, Cin)
        # Repeat the coordinates to match batch and height dimensions: (B, H, W, 1)
        w_coords = w_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, H, 1, 1)
        # Concatenate channel values and W coordinates: (B, H, W, Cin + 1)
        w_input = torch.cat([x_w, w_coords], dim=-1)
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

        h_coords = torch.linspace(0, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        # Repeat the coordinates to match batch and m2 dimensions: (B, m2, H, 1)
        h_coords = h_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.m2, 1, 1)

        # Concatenate channel values and H coordinates: (B, m2, H, Cin + 1)
        h_input = torch.cat([x_h, h_coords], dim=-1)

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

        # Permute to (B, Cin, m2, m1) - our final format for the frequency domain
        # Note: The order of modes (m1, m2) or (m2, m1) depends on convention.
        # Here, we transformed W (m2) then H (m1), so (B, Cin, m2, m1) makes sense.
        # However, for compatibility with learned_weights_freq (Cin, m1, m2), we permute.
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


class NLITBlock(nn.Module):
    """NLITBlock is a PyTorch module implementing a Non Linear Integral Transform Block for learned 2D non-linear
    integral transforms that are resolution-invariant.

    This block learns kernels K(u, v, x, y, f(x, y)) as a function of input coordinates (x, y) and values (f(x, y)),
    Then it perfoms a multiplication by a learned weigth in the transformed space and use conjugate kernels to go back to original space
    During the inverse transform, it is possible to resample the image by a factor 2 (up or down)

    inspired by IAE-NET: INTEGRAL AUTOENCODERS FOR DISCRETIZATION-INVARIANT LEARNING.

        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mlp_hidden_dim (int, optional): Hidden dimension of the MLPs learning the bases. Default is 64.
        mlp_num_layers (int, optional): Number of hidden layers in the MLPs learning the bases. Default is 2.
        activation (callable, optional): Activation function to use. Default is nn.GELU.
        norm (callable, optional): Normalization layer to use. Default is LayerNorm2d.
        resampling (str, optional): If "down" or "up", resample the output by a factor of 2. Default is None.
        dim (int, optional): Dimensionality of the transform (default is 2 for 2d problems).

    Attributes:
        basis_generator: Module to generate learned basis functions from input coordinates and values.
        learned_weights: Learnable parameters for multiplication in the transformed space.
        mixer: 1x1 convolution for mixing output channels.
        activation: Activation function.
        norm: Normalization layer.
        shortcut: 1x1 convolution for shortcut connection.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.

    Forward Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) (real part)

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
        resampling=None,
        dim=2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.resampling = resampling

        # We use faster 1x1conv layers instead of nn.Linear to implement the linear layer
        # to generate the basis functions from input coordinates and values.
        self.basis_generator = Linear1x1Conv(
            out_ch=self.m1 * self.m2,
            in_ch=2 + in_channels,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        # Learned parameters for multiplication in the transformed space (per channel).
        self.learned_weights = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Initialization of learned_weights for a good starting point (e.g., all to 1+0j)
        nn.init.constant_(self.learned_weights, 1.0)

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
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out), real part.
        """
        B, C, H, W = x.shape

        if self.resampling == "up":
            H_basis, W_basis = H * 2, W * 2
            x_cond = F.interpolate(x, size=(H_basis, W_basis), mode="bilinear", align_corners=False)
        else:
            H_basis, W_basis = H, W
            x_cond = x

        h_coords_map = torch.linspace(0, 1, H_basis, device=x.device).view(1, H_basis, 1).repeat(1, 1, W_basis)
        w_coords_map = torch.linspace(0, 1, W_basis, device=x.device).view(1, 1, W_basis).repeat(1, H_basis, 1)
        coords_2d_base = torch.cat([h_coords_map, w_coords_map], dim=0)
        # coords_2d: (B, 2, H_basis, W_basis). Add the batch dimension.
        coords_2d = coords_2d_base.unsqueeze(0).repeat(B, 1, 1, 1)  # B C H W

        x_in = torch.cat([coords_2d, x_cond], dim=1)  # (B, 2+cin, H, W)

        # 3. Generate kernels
        encoder_basis = self.basis_generator(x_in)  # (B, m1*m2, H, W)

        # 4. Reshape kernels for Einsum
        # encoder_basis = encoder_basis.view(B, H_basis, W_basis, self.m1, self.m2)
        encoder_basis = encoder_basis.view(B, self.m1, self.m2, H_basis, W_basis)

        if self.resampling == "up":
            # Subsample the generated basis back to H_in x W_in (taking every second point)
            fwd_basis = encoder_basis[:, :, :, ::2, ::2]
        else:
            fwd_basis = encoder_basis

        # Convert input to complex numbers if it is real
        xc = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x
        # Einsum: (B, C_in, H, W) @ (B, H, W, m1, m2) -> (B, C_in, m1, m2)
        xhat = torch.einsum("bchw,bmnhw->bcmn", xc, fwd_basis)  # "Spectral" representation

        # Multiply by learned weigths in transformed space
        xhat = torch.einsum("bixy,oixy->boxy", xhat, self.learned_weights)

        # ---2D Inverse Transform ---
        if self.resampling == "down":
            # Subsample the basis generated at H_in x W_in (H_basis) to H/2 x W/2 (every second point)
            inv_basis = encoder_basis[:, :, :, ::2, ::2]
            H_out, W_out = H // 2, W // 2
        else:
            # Use the full generated basis (H_in x W_in or H_out x W_out)
            inv_basis = encoder_basis
            H_out, W_out = H_basis, W_basis

        # Einsum: (B, C_out, m1, m2) @ (B, m1, m2, H_out, W_out)* -> (B, C_out, H_out, W_out)
        # We use .conj() for the inverse (adjoint) operation.
        x_rec = torch.einsum("bcmn,bmnhw->bchw", xhat, inv_basis.conj())

        # Normalization
        x_rec = x_rec.real * (1.0 / (H_out * W_out))

        # Mixing channels
        output = self.mixer(x_rec)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        if self.resampling == "up" or self.resampling == "down":
            x = F.interpolate(x, size=(H_out, W_out), mode="bilinear")

        output = output + self.shortcut(x)
        output = self.activation(output)

        return output


class AttentionITBlock(nn.Module):
    """AttentionITBlock is a PyTorch module implementing a Non Linear Integral Transform Block for learned 2D non-linear
    integral transforms that are resolution-invariant.

    This block learns complex kernels K(u, v, x, y, f(x, y)) as a function of input coordinates (x, y) and values (f(x, y)),
    1 - It transforms the input into a learned space using these kernels.
    2 - It apply a self-attention layer in the transformed space (complex attention).
    3 - It transforms back to the original space using conjugate kernels.
    4 - It mixes the output channels and add a shortcut connection.
    5 - A final Normalization and activation are applied
    During the inverse transform, it is possible to resample the image by a factor 2 (up or down)

    inspired by IAE-NET: INTEGRAL AUTOENCODERS FOR DISCRETIZATION-INVARIANT LEARNING.

        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mlp_hidden_dim (int, optional): Hidden dimension of the MLPs learning the bases. Default is 64.
        mlp_num_layers (int, optional): Number of hidden layers in the MLPs learning the bases. Default is 2.
        num_hads (int): Number of attention heads in the complex attention block
        activation (callable, optional): Activation function to use. Default is nn.GELU.
        norm (callable, optional): Normalization layer to use. Default is LayerNorm2d.
        resampling (str, optional): If "down" or "up", resample the output by a factor of 2. Default is None.
        dim (int, optional): Dimensionality of the transform (default is 2 for 2d problems).

    Attributes:
        basis_generator: Module to generate learned basis functions from input coordinates and values.
        learned_weights: Learnable parameters for multiplication in the transformed space.
        mixer: 1x1 convolution for mixing output channels.
        activation: Activation function.
        norm: Normalization layer.
        shortcut: 1x1 convolution for shortcut connection.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.

    Forward Returns:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W) (real part)

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        num_heads=4,
        activation=nn.GELU,
        norm=LayerNorm2d,
        resampling=None,
        dim=2,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.resampling = resampling

        # We use faster 1x1conv layers instead of nn.Linear to implement the linear layer
        # to generate the basis functions from input coordinates and values.
        self.basis_generator = Linear1x1Conv(
            out_ch=self.m1 * self.m2,
            in_ch=2 + in_channels,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        # Self-Attention block in the transformed space
        self.attention = ComplexAttention(
            in_ch=2 + in_channels,
            out_ch=out_channels,
            num_heads=num_heads,
        )

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
            torch.Tensor: Output tensor of shape (B, C, H_out, W_out), real part.
        """
        B, C, H, W = x.shape

        if self.resampling == "up":
            H_basis, W_basis = H * 2, W * 2
            x_cond = F.interpolate(x, size=(H_basis, W_basis), mode="bilinear", align_corners=False)
        else:
            H_basis, W_basis = H, W
            x_cond = x

        h_coords_map = torch.linspace(0, 1, H_basis, device=x.device).view(1, H_basis, 1).repeat(1, 1, W_basis)
        w_coords_map = torch.linspace(0, 1, W_basis, device=x.device).view(1, 1, W_basis).repeat(1, H_basis, 1)
        coords_2d_base = torch.cat([h_coords_map, w_coords_map], dim=0)
        # coords_2d: (B, 2, H_basis, W_basis). Add the batch dimension.
        coords_2d = coords_2d_base.unsqueeze(0).repeat(B, 1, 1, 1)  # B C H W

        x_in = torch.cat([coords_2d, x_cond], dim=1)  # (B, 2+cin, H, W)

        # 3. Generate kernels
        encoder_basis = self.basis_generator(x_in)  # (B, m1*m2, H, W)

        # 4. Reshape kernels for Einsum
        # encoder_basis = encoder_basis.view(B, H_basis, W_basis, self.m1, self.m2)
        encoder_basis = encoder_basis.view(B, self.m1, self.m2, H_basis, W_basis)

        if self.resampling == "up":
            # Subsample the generated basis back to H_in x W_in (taking every second point)
            fwd_basis = encoder_basis[:, :, :, ::2, ::2]
        else:
            fwd_basis = encoder_basis

        # Convert input to complex numbers if it is real
        xc = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x
        # Einsum: (B, C_in, H, W) @ (B, H, W, m1, m2) -> (B, C_in, m1, m2)
        xhat = torch.einsum("bchw,bmnhw->bcmn", xc, fwd_basis)  # "Spectral" representation

        # Attention in transformed space (uses comple conv2D)
        xhat = xhat.permute(0, 2, 3, 1).view(-1, 2 + C)  # (B, m1, m2, C_in)
        xhat = self.attention(xhat)
        xhat = xhat.permute(0, 3, 1, 2)  # (B, C_out, m1, m2)
        xhat = xhat.view(B, self.out_channels, self.m1, self.m2)

        # ---2D Inverse Transform ---
        if self.resampling == "down":
            # Subsample the basis generated at H_in x W_in (H_basis) to H/2 x W/2 (every second point)
            inv_basis = encoder_basis[:, :, :, ::2, ::2]
            H_out, W_out = H // 2, W // 2
        else:
            # Use the full generated basis (H_in x W_in or H_out x W_out)
            inv_basis = encoder_basis
            H_out, W_out = H_basis, W_basis

        # Einsum: (B, C_out, m1, m2) @ (B, m1, m2, H_out, W_out)* -> (B, C_out, H_out, W_out)
        # We use .conj() for the inverse (adjoint) operation.
        x_rec = torch.einsum("bcmn,bmnhw->bchw", xhat, inv_basis.conj())

        # Normalization
        x_rec = x_rec.real * (1.0 / (H_out * W_out))

        # Mixing channels
        output = self.mixer(x_rec)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        if self.resampling == "up" or self.resampling == "down":
            x = F.interpolate(x, size=(H_out, W_out), mode="bilinear")

        output = output + self.shortcut(x)
        output = self.activation(output)

        return output


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
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        # Decoder MLPs: bases conditionnées par les valeurs des modes spectraux (fréquentiels)
        # Pour la transformation inverse H (m1 -> H), conditionnée par (C_out, m2) modes
        self.mlp_h_inv = MLPBlock(
            in_ch=1 + 2 * out_channels,
            out_ch=m1,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.mlp_w_inv = MLPBlock(
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
        w_coords = torch.linspace(0, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        mlp_w_in = torch.cat([xc.real.permute(0, 2, 3, 1), w_coords], dim=-1)
        b_w = self.basis_w_fn(mlp_w_in)  # (B, H, W, m2)

        # Matmul: (B, H, C, W) @ (B, H, W, m2) -> (B, H, C, m2)
        x_w = torch.matmul(xc.permute(0, 2, 1, 3), b_w)
        x_w = x_w.permute(0, 2, 1, 3)  # (B, C, H, m2)

        # 2. H (height) transformation
        # Input MLP: (B, m2, H, C+1)
        h_coords = (
            torch.linspace(0, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )
        mlp_h_in = torch.cat([x_w.real.permute(0, 3, 2, 1), h_coords], dim=-1)
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
            torch.linspace(0, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )  # B, m2, H, 1

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # xf is (B, C_out, m1, m2)
        # Concatenate real and imaginary: (B, 2*C_out, m1, m2)
        xf_combined_real_imag = torch.cat([xf.real, xf.imag], dim=1)

        # Permute so that m1 is the last dimension for the compressor: (B, 2*C_out, m2, m1)
        cond_h_val_for_comp = xf_combined_real_imag.permute(0, 1, 3, 2)
        # Apply the compressor: (B, 2*C_out, m2, 1)
        cond_h_val_compressed = self.compressor_h_inv(cond_h_val_for_comp)
        # Repeat the compressed value along the output spatial dimension H: (B, 2*C_out, m2, h)
        cond_h_val_repeated = cond_h_val_compressed.repeat(1, 1, 1, h)
        # Permute for the MLP: (B, m2, h, 2*C_out)
        cond_h_val_mlp_in = cond_h_val_repeated.permute(0, 2, 3, 1)

        mlp_h_inv_in = torch.cat([h_rec_coords, cond_h_val_mlp_in], dim=-1)  # (B, m2, h, 1 + 2*C_out)
        b_h_inv = self.mlp_h_inv(mlp_h_inv_in)  # (B, m2, h, m1) - Basis generated by MLP

        # Matmul: (B, m2, C_out, m1) @ (B, m2, m1, h).mH -> (B, m2, C_out, h)
        x_h_rec = torch.matmul(xf.permute(0, 3, 1, 2), b_h_inv.mH)  # Still complex
        x_h_rec = x_h_rec.permute(0, 2, 3, 1)  # (B, C_out, h, m2) - Still complex

        # 2. Inverse W transformation (width reconstruction)
        # Output spatial coordinates for W
        w_rec_coords = (
            torch.linspace(0, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        )

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # x_h_rec is (B, C_out, h, m2)
        # Concatenate real and imaginary: (B, 2*C_out, h, m2)
        x_h_rec_combined_real_imag = torch.cat([x_h_rec.real, x_h_rec.imag], dim=1)

        # Permute so that m2 is the last dimension for the compressor: (B, 2*C_out, h, m2)
        cond_w_val_for_comp = x_h_rec_combined_real_imag
        # Apply the compressor: (B, 2*C_out, h, 1)
        cond_w_val_compressed = self.compressor_w_inv(cond_w_val_for_comp)
        # Repeat the compressed value along the output spatial dimension W: (B, 2*C_out, h, w)
        cond_w_val_repeated = cond_w_val_compressed.repeat(1, 1, 1, w)
        # Permute for the MLP: (B, h, w, 2*C_out)
        cond_w_val_mlp_in = cond_w_val_repeated.permute(0, 2, 3, 1)

        mlp_w_inv_in = torch.cat([w_rec_coords, cond_w_val_mlp_in], dim=-1)  # (B, h, w, 1 + 2*C_out)
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


# class CoDABlock2Dv2(nn.Module):
#     """Co-domain Attention Block (CODABlock) implement the transformer
#     architecture in the operator learning framework, as described in [1]_.
#     It is a simplified version of the implementation found in https://github.com/neuraloperator
#     A Cross Attention module between the tokens representing the variables and the tokens representing the boundary conditions is
#     added after the Self-Attention module


#     References
#     ----------
#     .. [1]: M. Rahman, R. George, M. Elleithy, D. Leibovici, Z. Li, B. Bonev,
#         C. White, J. Berner, R. Yeh, J. Kossaifi, K. Azizzadenesheli, A. Anandkumar (2024).
#         "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs."
#         arxiv:2403.12553
#     """

#     def __init__(
#         self,
#         modes: tuple[int, int],
#         token_dim: int,
#         bc_dim: int,
#         n_heads: int = 1,
#         activation: Callable = nn.GELU,
#         temperature: float = 1.0,
#         norm: Callable = partial(nn.InstanceNorm2d, affine=True),
#         spectral_compression_factor: Sequence = (1, 1, 1),
#     ):

#         super().__init__()

#         self.token_dim = token_dim
#         self.n_heads = n_heads
#         self.temperature = temperature
#         self.n_dim = 2  # only 2d spatial dimensions
#         self.ranks = [self.token_dim, n_heads * self.token_dim, np.prod(modes)]
#         self.ranks = tuple(np.ceil(np.divide(self.ranks, spectral_compression_factor)).astype(int))
#         self.bc_dim = bc_dim

#         self.Q = FNOBlock(
#             in_channels=self.token_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         self.V = FNOBlock(
#             in_channels=self.token_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         self.K = FNOBlock(
#             in_channels=self.token_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         # To project back each token from the n heads to token_dim
#         self.projection = FNOBlock(
#             in_channels=self.n_heads * self.token_dim,
#             hidden_channels=self.token_dim,
#             out_channels=self.token_dim,
#             modes=modes,
#             activation=nn.Identity,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         # Cross Attention with the boundary conditions

#         self.Q2 = FNOBlock(
#             in_channels=self.token_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         self.V2 = FNOBlock(
#             in_channels=self.bc_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         self.K2 = FNOBlock(
#             in_channels=self.bc_dim,
#             hidden_channels=n_heads * self.token_dim,
#             out_channels=n_heads * self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         # To project back each token from the n heads to token_dim
#         self.projection2 = FNOBlock(
#             in_channels=self.n_heads * self.token_dim,
#             hidden_channels=self.token_dim,
#             out_channels=self.token_dim,
#             modes=modes,
#             activation=nn.Identity,
#             spectral_layer_type="tucker",
#             ranks=self.ranks,
#         )

#         mixer_ranks = [self.token_dim, self.token_dim, np.prod(modes)]
#         mixer_ranks = np.ceil(np.divide(mixer_ranks, spectral_compression_factor)).astype(int)
#         self.mixer = FNOBlock(
#             in_channels=self.token_dim,
#             hidden_channels=self.token_dim,
#             out_channels=self.token_dim,
#             modes=modes,
#             activation=activation,
#             spectral_layer_type="tucker",
#             ranks=mixer_ranks,
#         )
#         self.norm1 = norm(self.token_dim)
#         self.norm2 = norm(self.token_dim)
#         self.norm3 = norm(self.token_dim)
#         self.norm4 = norm(self.token_dim)

#     def MultiHeadAttention(self, tokens, batch_size):
#         """Compute multi-head Attention where each variable latent representation is a token

#         input tensor shape (b*t), d, h, w
#         The tensor is first transformed into k, q and v with shape (b*t),(n*d), h, w
#         where
#         b: batch size
#         t: number of tokens
#         n: number of heads
#         d: token dimension (the latent dimension of each variable)
#         h, w: spatial dimensions

#         Then k, q and v are reshaped to b, n, t, (d h w)
#         as torch.matul multiplies the two last dimensions
#         Finally the output is reshaped to b, n, (t*d), h, w

#         """
#         # k, q, v (b*t, n*d, h, w)
#         k = self.K(tokens)
#         q = self.Q(tokens)
#         v = self.V(tokens)

#         assert k.size(1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

#         # reshape from (b*t) (n*d) h w -> b n t (d*h*w ...)
#         t = k.size(0) // batch_size  # Compute the number of tokens `t` (each token is a variable here)
#         # n heads with token codimension `d` (in the case of per layer attention, d=1)
#         d = k.size(1) // self.n_heads

#         # reshape from (b*t) (n*d) h w ... to b n t d h w ...
#         k = k.view(batch_size, t, self.n_heads, d, *k.shape[-self.n_dim :])
#         q = q.view(batch_size, t, self.n_heads, d, *q.shape[-self.n_dim :])
#         v = v.view(batch_size, t, self.n_heads, d, *v.shape[-self.n_dim :])

#         k = torch.transpose(k, 1, 2)
#         q = torch.transpose(q, 1, 2)
#         v = torch.transpose(v, 1, 2)

#         # reshape to flatten the d, h and w dimensions
#         k = k.reshape(batch_size, self.n_heads, t, -1)
#         q = q.reshape(batch_size, self.n_heads, t, -1)
#         v = v.reshape(batch_size, self.n_heads, t, -1)

#         # attention mechanism
#         dprod = torch.matmul(q, k.transpose(-1, -2)) / (np.sqrt(k.shape[-1]) * self.temperature)
#         dprod = F.softmax(dprod, dim=-1)

#         attention = torch.matmul(dprod, v)

#         # Reshape from (b, n, t, d * h * w) to (b, n, t, d, h, w, ...)
#         attention = attention.view(
#             attention.size(0), attention.size(1), attention.size(2), d, *tokens.shape[-self.n_dim :]
#         )
#         attention = torch.transpose(attention, 1, 2)  # b t n d h w
#         attention = attention.reshape(
#             attention.size(0) * attention.size(1), attention.size(2) * d, *tokens.shape[-self.n_dim :]
#         )  # (b * t) (n * d) h w

#         return attention

#     def MultiHeadCrossAttention(self, tokens, bc_tokens, batch_size):
#         """Compute multi-head Attention where each variable latent representation is a token

#         input tensor shape (b*t), d, h, w
#         The tensor is first transformed into k, q and v with shape (b*t),(n*d), h, w
#         where
#         b: batch size
#         t: number of tokens
#         n: number of heads
#         d: token dimension (the latent dimension of each variable)
#         h, w: spatial dimensions

#         Then k, q and v are reshaped to b, n, t, (d h w)
#         as torch.matul multiplies the two last dimensions
#         Finally the output is reshaped to b, n, (t*d), h, w

#         """
#         # k, q, v (b*t, n*d, h, w)
#         q = self.Q2(tokens)
#         k = self.K2(bc_tokens)
#         v = self.V2(bc_tokens)

#         assert k.size(1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

#         # reshape from (b*t) (n*d) h w -> b n t (d*h*w ...)
#         t = k.size(0) // batch_size  # Compute the number of tokens `t` (each token is a variable here)
#         # n heads with token codimension `d` (in the case of per layer attention, d=1)
#         d = k.size(1) // self.n_heads

#         # reshape from (b*t) (n*d) h w ... to b n t d h w ...
#         k = k.view(batch_size, t, self.n_heads, d, *k.shape[-self.n_dim :])
#         q = q.view(batch_size, t, self.n_heads, d, *q.shape[-self.n_dim :])
#         v = v.view(batch_size, t, self.n_heads, d, *v.shape[-self.n_dim :])

#         k = torch.transpose(k, 1, 2)
#         q = torch.transpose(q, 1, 2)
#         v = torch.transpose(v, 1, 2)

#         # reshape to flatten the d, h and w dimensions
#         k = k.reshape(batch_size, self.n_heads, t, -1)
#         q = q.reshape(batch_size, self.n_heads, t, -1)
#         v = v.reshape(batch_size, self.n_heads, t, -1)

#         # attention mechanism
#         dprod = torch.matmul(q, k.transpose(-1, -2)) / (np.sqrt(k.shape[-1]) * self.temperature)
#         dprod = F.softmax(dprod, dim=-1)

#         attention = torch.matmul(dprod, v)

#         # Reshape from (b, n, t, d * h * w) to (b, n, t, d, h, w, ...)
#         attention = attention.view(
#             attention.size(0), attention.size(1), attention.size(2), d, *tokens.shape[-self.n_dim :]
#         )
#         attention = torch.transpose(attention, 1, 2)  # b t n d h w
#         attention = attention.reshape(
#             attention.size(0) * attention.size(1), attention.size(2) * d, *tokens.shape[-self.n_dim :]
#         )  # (b * t) (n * d) h w

#         return attention

#     def forward(self, x):

#         # the input tensor must have a shape b (n_var * hidden_dim) h w
#         # if the token_dim is different than the hidden dim,  it means that each token does not represent the full latent embedding of a variable

#         batch_size = x.shape[0]
#         # spatial_shape = x.shape[-self.n_dim :]

#         assert x.shape[1] % self.token_dim == 0, "Number of channels in x should be divisible by token_codimension"

#         n_tokens = x.shape[1] // self.token_dim
#         # Reshape from shape b (t*d) h w ... to (b*t) d h w
#         x = x.view(x.size(0) * n_tokens, self.token_dim, *x.shape[-self.n_dim :])

#         attention = self.norm1(x)
#         attention = self.MultiHeadAttention(attention, batch_size)  # it ouptputs (b * t) (n * d) h w
#         attention = self.projection(attention)  # now it's projected to (b * t) d h w
#         attention = self.norm2(attention + x)  # shortcut
#         shortcut = attention

#         attention = self.MultiHeadCrossAttention(attention, bc_tokens=x, batch_size=batch_size)
#         attention = self.projection2(attention)
#         attention = self.norm3(attention + shortcut)

#         attention = self.mixer(attention)
#         attention = self.norm4(attention)

#         # reshape to b (n_var * hidden_var_dim // token_dim) h w
#         attention = attention.view(batch_size, n_tokens * attention.shape[1], *attention.shape[-self.n_dim :])

#         return attention
