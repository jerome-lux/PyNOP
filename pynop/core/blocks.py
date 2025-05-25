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
        activation=nn.GELU(),
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
        activation=nn.GELU(),
        se=False,
        se_ratio=4,
        use_bias=False,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, 3)
        norm = make_tuple(norm, 3)
        activation = make_tuple(activation, 3)

        self.se = se
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
            norm=norm[0],
            activation=activation[0],
        )

        self.conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            use_bias=use_bias[1],
            norm=norm[1],
            activation=activation[1],
        )
        if se:
            self.se = SqueezeExcitation(mid_channels, mid_channels // se_ratio, activation=activation)

        self.expand = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            use_bias=use_bias[2],
            norm=norm[2],
            activation=activation[2],
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

        if self.se:
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
        ranks: Union[tuple[int, int, int], None] = None,
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

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, self.hidden_channels, modes)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, self.hidden_channels, modes, ranks)
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
        ranks: Union[tuple[int, int, int], None] = None,
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
        self.activation1 = activation()
        self.norm1 = normalization(out_channels)
        self.activation2 = activation()
        self.norm2 = normalization(out_channels)

        # mix channels with n mlp_layers of 1x1 conv.
        # The hidden dimension is out_channels // 2
        self.mlp = nn.ModuleList()
        for i in mlp_layers:
            if mlp_layers == 1:
                self.mlp.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
            else:
                out_mlp = out_channels if i == len(mlp_layers) - 1 else out_channels // 2
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
        ranks: Union[tuple[int, int, int], None] = None,
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
        n_heads: int = 1,
        activation: Callable = nn.GELU,
        temperature: float = 1.0,
        norm: Callable = partial(nn.InstanceNorm2d, affine=True),
        spectral_compression_factor: Sequence = (1, 1, 1),
    ):

        super().__init__()

        self.token_dim = token_dim
        self.n_heads = n_heads
        self.temperature = temperature
        self.n_dim = 2  # only 2d spatial dimensions
        self.ranks = [self.token_dim, n_heads * self.token_dim, np.prod(modes)]
        self.ranks = tuple(np.ceil(np.divide(self.ranks, spectral_compression_factor)).astype(int))

        self.Q = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.token_dim,
            out_channels=n_heads * self.token_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
        )

        self.V = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.token_dim,
            out_channels=n_heads * self.token_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
        )

        self.K = FNOBlock(
            in_channels=self.token_dim,
            hidden_channels=n_heads * self.token_dim,
            out_channels=n_heads * self.token_dim,
            modes=modes,
            activation=activation,
            spectral_layer_type="tucker",
            ranks=self.ranks,
        )

        # To project back each token from the n heads to token_dim

        self.projection = FNOBlock(
            in_channels=self.n_heads * self.token_dim,
            hidden_channels=self.token_dim,
            out_channels=self.token_dim,
            modes=modes,
            activation=nn.Identity,
            spectral_layer_type="tucker",
            ranks=self.ranks,
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

        Then k, q and v are reshaped to b, n, t, (d h w)
        as torch.matul multiplies the two last dimensions
        Finally the output is reshaped to b, n, (t*d), h, w

        """
        # k, q, v (b*t, n*d, h, w)
        k = self.K(tokens)
        q = self.Q(tokens)
        v = self.V(tokens)

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

        # attention mechanism
        dprod = torch.matmul(q, k.transpose(-1, -2)) / (np.sqrt(k.shape[-1]) * self.temperature)
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # Reshape from (b, n, t, d * h * w) to (b, n, t, d, h, w, ...)
        attention = attention.view(
            attention.size(0), attention.size(1), attention.size(2), d, *tokens.shape[-self.n_dim :]
        )
        attention = torch.transpose(attention, 1, 2)
        attention = attention.reshape(
            attention.size(0) * attention.size(1), attention.size(2) * d, *tokens.shape[-self.n_dim :]
        )

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
