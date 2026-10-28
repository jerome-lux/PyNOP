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
from .norm import AdaptiveLayerNorm, LayerNorm2d, AdaRMSNorm, ComplexLayerNorm, GRN
from .activations import Sine, gumbel_softmax
from .encoding import (
    RoPE,
    AdaptiveRoPE2D,
    IntegratedPositionalEncoding,
    GaussianFourierEmbedding,
    GaussianIntegratedPositionalEncoding,
    FourierEmbedding,
    AdaptiveFourierEmbedding,
)
from .utils import print_stats


class StridedConvDownsamplingLayer(nn.Module):
    """Strided-convolution downsampling: reduce the spatial size by ``factor``.

    (B, C_in, H, W) -> (B, C_out, H // factor, W // factor)

    A single learnable convolution with ``stride = factor`` performs the
    downsampling while (optionally) mixing the channels. It is the default
    shortcut/downsampler of the residual blocks and is **not** pixel-shuffle
    based, so the stride is a fixed ratio independent of the grid size.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        factor: Downsampling stride; the spatial size is divided by this.
        kernel_size: Convolution kernel size (default ``1``).
        groups: Number of convolution groups (default ``1``).
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int, kernel_size: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.factor = factor
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        """Downsample ``x`` with the strided convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, H // factor, W // factor).
        """
        return self.conv(x)


class InterpolateConvUpSampleLayer(nn.Module):
    """Upsampling by grid interpolation followed by a refinement convolution.

    (B, C_in, H, W) -> (B, C_in, H*factor, W*factor) -> (B, C_out, H*factor, W*factor)

    ``F.interpolate`` (bilinear by default) upsamples the grid and a
    ``ConvLayer`` then refines / mixes the channels. Bilinear interpolation
    introduces a mild low-pass filter (unlike the pixel-shuffle upsamplers),
    which the refinement conv partially compensates. This is the decoder
    upsampler used by :class:`UBlock`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default ``1``).
        factor: Upsampling factor; the spatial size is multiplied by this
            (default ``2``).
        mode: Interpolation mode (default ``"bilinear"``).
    """

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
        self.in_channels = in_channels
        self.out_channels = out_channels
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
        """Upsample ``x`` by interpolation, then apply the refinement conv.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, H * factor, W * factor).
        """
        x = F.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


DOWNSAMPLING_LAYERS = {
    "stridedconv": StridedConvDownsamplingLayer,
    "maxpooling": MaxPoolConv,
}

UPSAMPLING_LAYERS = {
    "interpolate": InterpolateConvUpSampleLayer,
}


def make_downsampler(
    downsampling_method: str, in_channels: int, out_channels: int, factor: int, block_name: str
) -> nn.Module:
    """Instantiate a named downsampling layer from :data:`DOWNSAMPLING_LAYERS`.

    Args:
        downsampling_method: Registry key (``"stridedconv"`` or ``"maxpooling"``).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        factor: Downsampling stride / factor.
        block_name: Name of the calling block, used in the error message.

    Returns:
        An instantiated downsampling :class:`nn.Module`.

    Raises:
        ValueError: If ``downsampling_method`` is not a valid registry key.
    """
    if downsampling_method not in DOWNSAMPLING_LAYERS:
        raise ValueError(
            f"{block_name}: unknown downsampling_method {downsampling_method!r}; "
            f"choose one of {sorted(DOWNSAMPLING_LAYERS)}."
        )
    return DOWNSAMPLING_LAYERS[downsampling_method](
        in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=factor
    )


class ResBlock(nn.Module):
    """Residual block made of ``nconv`` convolutions with a shortcut connection.

    Pipeline::

        x -> [Conv]*nconv -> out
        x -> shortcut (1x1 conv / downsampler) -> added to ``out``

    When ``stride > 1`` the first conv downsamples the spatial size by
    ``stride`` and the shortcut is a :func:`make_downsampler` layer (one of
    :data:`DOWNSAMPLING_LAYERS`). When ``stride == 1`` but the channel count
    changes, the shortcut is a plain 1x1 ``ConvLayer``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        nconv: Number of stacked convolutions (default ``2``).
        stride: Stride of the first convolution; ``> 1`` downsamples.
        kernel_size: Convolution kernel size (default ``3``).
        groups: Convolution groups (default ``1``).
        norm: Normalization class (or per-layer list) applied after each conv;
            ``None`` disables it (default :class:`LayerNorm2d`).
        activation: Activation class (or per-layer list); ``None`` disables it
            (default :class:`nn.GELU`).
        use_bias: Conv bias flag (scalar or per-layer list).
        downsampling_method: Shortcut downsampler for ``stride > 1``; one of
            :data:`DOWNSAMPLING_LAYERS` (default ``"maxpooling"``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nconv: int = 2,
        stride: int = 1,
        kernel_size: int = 3,
        groups: int = 1,
        norm=LayerNorm2d,
        activation=nn.GELU,
        use_bias: bool = False,
        downsampling_method: str = "maxpooling",
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
            self.shortconv = make_downsampler(
                downsampling_method, in_channels, out_channels, stride, block_name="ResBlock"
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
    """Residual bottleneck block: 1x1 -> 3x3 -> 1x1 with an optional SE gate.

    Pipeline::

        x -> Conv1x1 (expand to mid) -> Conv(k) -> [SE] -> Conv1x1 (out) -> out
        x -> shortcut (1x1 conv / downsampler) -> added to ``out``

    ``mid = out_channels // bottleneck_ratio``. When ``stride > 1`` the 3x3
    conv downsamples and the shortcut is a :func:`make_downsampler` layer (one
    of :data:`DOWNSAMPLING_LAYERS`); otherwise, if the channel count changes,
    the shortcut is a plain 1x1 ``ConvLayer``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bottleneck_ratio: Reduction factor of the mid channel width
            (default ``4``).
        stride: Stride of the 3x3 convolution; ``> 1`` downsamples.
        kernel_size: Kernel size of the 3x3 convolution (default ``3``).
        groups: Convolution groups (default ``1``).
        norm: Normalization class (or per-layer list) applied after each conv
            (default :class:`LayerNorm2d`).
        activation: Activation class (or per-layer list) (default
            :class:`nn.GELU`).
        se: If True, insert a Squeeze-and-Excitation gate after the 3x3 conv.
        se_ratio: Reduction ratio of the SE gate.
        use_bias: Conv bias flag (scalar or per-layer list).
        downsampling_method: Shortcut downsampler for ``stride > 1``; one of
            :data:`DOWNSAMPLING_LAYERS` (default ``"maxpooling"``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: int = 4,
        stride: int = 1,
        kernel_size: int = 3,
        groups: int = 1,
        norm=LayerNorm2d,
        activation=nn.GELU,
        se: bool = True,
        se_ratio: int = 4,
        use_bias: bool = False,
        downsampling_method: str = "maxpooling",
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
            self.shortconv = make_downsampler(
                downsampling_method, in_channels, out_channels, stride, block_name="BottleneckBlock"
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
    """A single block of a Fourier Neural Operator (FNO).

    The forward follows the ordering of the reference implementations
    (neuralop, and Li et al. 2021)::

        x -> [norm] -> spectral_conv -> [activation] -> pointwise (1x1) -> + shortcut

    * ``spectral_conv`` -- one of the three supported spectral convolutions,
      selected by ``spectral_layer_type``:

      - ``"standard"``  -> :class:`SpectralConv2d` (full low-mode weight tensor).
      - ``"separable"`` -> :class:`SeparableSpectralConv2d` (depthwise spectral
        multiplication + linear channel mixing, the spectral analogue of a
        depthwise-separable convolution).
      - ``"tucker"``    -> :class:`TuckerSpectralConv2d` (Tucker factorization of
        the weight tensor, requires ``ranks``).

    * ``norm`` and ``activation`` are **optional** (pass ``None`` to disable).
      The norm is applied as a pre-norm on the block input (sized for
      ``in_channels``); the activation is applied exactly once, between the
      spectral conv and the pointwise conv.
    * ``shortcut`` -- a 1x1-conv residual (identity-like when
      ``in_channels == out_channels``), added after the pointwise conv.

    The spectral convolution is resolution-invariant: if the input grid is
    smaller than the configured modes, the low-frequency subspace of the
    learned weights is used (see :class:`SpectralConv2d`).

    * ``local_conv`` -- an optional stack of :class:`FiniteDifferenceLayer`
      (differential convolutions, NeuralOperator-style) added in parallel
      with the shortcut. This captures local high-frequency information that
      the global spectral branch (a structural low-pass) cannot represent,
      while remaining resolution-invariant via the ``1/h`` rescaling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: Union[int, Sequence[int]],
        hidden_channels: Union[int, None] = None,
        activation: Union[Callable, None] = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[Sequence[int], np.ndarray, None] = None,
        scaling: Union[int, float] = 1,
        channel_last: bool = False,
        local_conv: int = 0,
        local_kernel_size: int = 3,
        local_grid_width: float = 2.0,
    ):
        """
        Args:
            in_channels (int): Number of input channels for the block.
            out_channels (int): Number of output channels for the block.
            modes: Number of Fourier modes to retain in each dimension. An int is
                broadcast to both dimensions; a 2-sequence is ``(modes_x, modes_y)``.
            hidden_channels (int, optional): Width of the spectral-conv output.
                Defaults to ``out_channels``. The pointwise conv then maps
                ``hidden_channels -> out_channels``.
            activation: Activation callable (e.g. ``nn.GELU``) or ``None`` to
                disable. Applied once, between the spectral conv and the pointwise
                conv.
            normalization: Normalization callable (e.g. ``LayerNorm2d``) or
                ``None`` to disable. Applied as a pre-norm on the block input.
            spectral_layer_type (str): ``"standard"``, ``"separable"`` or
                ``"tucker"``. See the class docstring.
            ranks (tuple, optional): Ranks ``(r1, r2, r3)`` for the Tucker
                factorization; **required** when ``spectral_layer_type="tucker"``.
                A common choice with a compression factor ``k`` is
                ``r1 = cin / k``, ``r2 = cout / k``, ``r3 = prod(modes) / k``;
                ``k = 1`` (no compression) already uses ~2.5x fewer parameters
                than the standard spectral conv.
            scaling (int or float): Output spatial scaling factor (``> 1`` upsamples
                via the inverse FFT; the shortcut is interpolated to match).
            channel_last (bool): If True, expect/produce channels-last layout.
            local_conv (int): Number of :class:`FiniteDifferenceLayer` layers
                in the local high-frequency branch (``0`` = disabled, default).
                The branch is added in parallel with the shortcut, operating on
                the raw block input. Each layer is a differential convolution
                (kernel centered + rescaled by ``1/h``), giving resolution
                invariance and a high-pass character that complements the
                global spectral branch.
            local_kernel_size (int): Spatial kernel size for the local conv
                layers (odd integer, default ``3``).
            local_grid_width (float): Physical width of the domain for the
                local conv layers (default ``2.0`` for ``[-1, 1]``). Used to
                compute the grid spacing ``h = grid_width / (N - 1)``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = out_channels if hidden_channels is None else hidden_channels
        if isinstance(modes, int):
            modes = (modes, modes)
        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.scaling = scaling
        self.channel_last = channel_last

        # Core spectral convolution (one of the three supported types)
        common = dict(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            modes=modes,
            scaling=scaling,
            channel_last=channel_last,
        )
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(**common)
        elif spectral_layer_type == "separable":
            self.spectral_conv = SeparableSpectralConv2d(**common)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("ranks must be provided for spectral_layer_type='tucker'.")
            self.spectral_conv = TuckerSpectralConv2d(ranks=ranks, **common)
        else:
            raise ValueError(
                f"Unknown spectral_layer_type: {spectral_layer_type!r}. "
                "Choose one of 'standard', 'separable', 'tucker'."
            )

        # Optional pre-norm (sized for in_channels, applied to the block input)
        self.norm = normalization(in_channels) if normalization is not None else None
        # Optional activation, applied exactly once (between spectral and pointwise)
        self.activation = activation() if activation is not None else None

        # Pointwise 1x1 channel mixing (spectral-conv output -> out_channels)
        self.linear = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        # Residual shortcut (1x1 conv; identity-like when in_channels == out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Optional local high-frequency branch (stack of FiniteDifferenceLayer)
        self.local_conv_depth = local_conv
        if local_conv > 0:
            self.local_conv = nn.Sequential(
                *[
                    FiniteDifferenceLayer(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=local_kernel_size,
                        grid_width=local_grid_width,
                    )
                    for i in range(local_conv)
                ]
            )
        else:
            self.local_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape ``(B, C_in, H, W)`` (or channels-last when
                ``channel_last=True``).

        Returns:
            torch.Tensor: Output of shape ``(B, C_out, H*scaling, W*scaling)``.
        """
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch (residual)
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            # The spectral conv upsamples by `scaling`; match the shortcut to it.
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        # Local high-frequency branch (parallel to shortcut, operates on raw input)
        if self.local_conv is not None:
            x_local = self.local_conv(x)  # (B, out_channels, H, W)
            if self.scaling != 1:
                x_local = F.interpolate(x_local, scale_factor=self.scaling, mode="bilinear", align_corners=False)
            x_shortcut = x_shortcut + x_local

        # Main branch: [norm] -> spectral_conv -> [activation] -> pointwise
        if self.norm is not None:
            x = self.norm(x)  # pre-norm on the block input
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        if self.activation is not None:
            x = self.activation(x)  # optional activation, applied exactly once
        x = self.linear(x)  # (B, out_channels, H, W)

        return x + x_shortcut  # (B, out_channels, H, W)


class PixelShuffleUpsample(nn.Module):
    """A parameter-efficient pixel-shuffle upsampling block (sub-pixel conv).

    Upsamples by **rearranging channels into spatial sub-pixels** (a 1x1
    "rearrange" conv to ``factor**2`` channels + :class:`nn.PixelShuffle`)
    instead of bilinear interpolation, so it introduces **no low-pass** loss of
    high-frequency detail. A **depthwise-separable** 3x3 conv (depthwise 3x3 +
    pointwise 1x1) then mixes spatial neighbours -- giving the spatial
    refinement a plain 1x1 merge conv lacks, at a fraction of a full 3x3 conv's
    parameters. An optional skip connection (the U-Net decoder's encoder
    feature at the target resolution) is concatenated before the refinement.

    Pipeline (``r = factor``, e.g. ``r = 2``)::

        x (B, Cin, H, W)
        -> Conv1x1(Cin, r²·Cout)            # rearrange channels for the shuffle
        -> PixelShuffle(r)                  # -> (B, Cout, rH, rW); no low-pass
        -> [concat skip (B, Cskip, rH, rW)] # decoder form (skip_channels set)
        -> Depthwise Conv3x3 (groups=C)     # cheap spatial mixing
        -> Conv1x1(C -> Cout)               # channel mixing / merge
        -> (B, Cout, rH, rW)

    This is the standard "sub-pixel convolution" upsample (Shinya et al., 2017)
    refined with a depthwise-separable conv (the MobileNet-style cheap 3x3). It
    is the U-Net decoder's upsampling step, and is strictly more expressive than
    the bilinear + 1x1-conv :class:`UpsampleBlock` (no low-pass, plus spatial
    mixing) while using fewer parameters than a full 3x3 merge conv.

    Args:
        in_channels: Channels of the (low-resolution) input ``x``.
        out_channels: Channels of the output (and of the post-shuffle feature).
        factor: Upsampling factor ``r`` (default ``2``; must be ``>= 2``).
        skip_channels: Channels of the skip connection to concatenate after the
            shuffle (default ``None`` = no skip; in the U-Net decoder this is
            the channels of the corresponding encoder output).
        kernel_size: Depthwise convolution kernel size (default ``3``; must be
            odd).

    Raises:
        ValueError: If ``factor`` or ``kernel_size`` is invalid, or ``skip`` is
            missing at forward time when ``skip_channels`` is set.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 2,
        skip_channels: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if factor < 2:
            raise ValueError(f"PixelShuffleUpsample expects factor >= 2 (it upsamples), got {factor}.")
        if kernel_size % 2 == 0:
            raise ValueError("PixelShuffleUpsample expects an odd kernel_size.")
        self.factor = factor
        self.skip_channels = skip_channels

        # Rearrange conv: expand to r²·out_channels so the shuffle can fold them
        # into an r x r spatial block (this is the "1x1 conv" of the design).
        self.rearrange = nn.Conv2d(in_channels, factor * factor * out_channels, 1)
        self.shuffle = nn.PixelShuffle(factor)

        # Depthwise-separable refinement: cheap spatial mixing (depthwise 3x3)
        # + channel mixing / merge (pointwise 1x1) on the (skip-merged) feature.
        mix_in = out_channels + skip_channels if skip_channels is not None else out_channels
        self.depthwise = nn.Conv2d(mix_in, mix_in, kernel_size, padding=kernel_size // 2, groups=mix_in)
        self.pointwise = nn.Conv2d(mix_in, out_channels, 1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upsample ``x`` (pixel shuffle), merge the optional skip, and refine.

        Args:
            x: Feature map ``(B, Cin, H, W)``.
            skip: Skip connection ``(B, Cskip, H*factor, W*factor)``; required
                when ``skip_channels`` is set, ignored otherwise.

        Returns:
            Feature map ``(B, Cout, H*factor, W*factor)``.
        """
        x = self.rearrange(x)  # (B, r²·Cout, H, W)
        x = self.shuffle(x)  # (B, Cout, rH, rW)
        if self.skip_channels is not None:
            if skip is None:
                raise ValueError("PixelShuffleUpsample with skip_channels expects a `skip` tensor.")
            x = torch.cat([x, skip], dim=1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PatchEmbed(nn.Module):
    """ViT-style patch embedding: a strided ``Conv2d`` mapping a map to tokens.

    ``(B, C_in, H, W) -> (B, (H/p)*(W/p), dim)``. This is the standard ViT patch
    embedding (a ``p x p`` convolution with stride ``p``).

    Args:
        in_ch: Number of input channels.
        dim: Output token dimension.
        patch_size: Patch side length ``p``.
        bias: If ``True``, the projection conv has a bias.
    """

    def __init__(self, in_ch: int, dim: int, patch_size: int = 4, bias: bool = True):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}.")
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Feature map ``(B, C_in, H, W)`` (spatial size divisible by ``p``).

        Returns:
            Token tensor ``(B, (H/p)*(W/p), dim)``.
        """
        x = self.proj(x)  # (B, dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the patch feature *map* ``(B, dim, H/p, W/p)`` (no flattening)."""
        return self.proj(x)


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
        activation: Optional[Callable] = nn.GELU,
        norm: Optional[nn.Module] = None,
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
            norm (callable or None): Normalization layer (channels-first, i.e. normalizes
                over the channel dim of the ``(B, C, H, W)`` layout). Default ``LayerNorm2d``.
        """
        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # Learnable exponent of the integral-transform normalization.
        # The inverse transform accumulates over the spatial domain, so the raw output
        # magnitude grows with the resolution. The legacy hard factor ``1 / (H * W)``
        # assumed a DFT-like (unit-energy) basis; the learned basis is *not* unit-energy,
        # so that constant over-suppressed the transform branch by ~100-400x (making it
        # dead relative to the shortcut). We instead normalize by ``(H * W) ** sigmoid(a)``
        # and let the model learn ``a``. Initialized so sigmoid(a) ~ 0.25, which yields an
        # O(1) transform branch at initialization (comparable to the shortcut).
        self.IT_scale = nn.Parameter(torch.tensor(-1.1))

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

        # Normalization: learnable exponent (see self.IT_scale) instead of a hard 1/(H*W).
        normalisation_factor = 1.0 / ((H * W) ** torch.sigmoid(self.IT_scale))
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
            norm (callable or None): Normalization layer (channels-first, i.e. normalizes
                over the channel dim of the ``(B, C, H, W)`` layout). Default ``LayerNorm2d``.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # Learnable exponent of the integral-transform normalization (see SepLITBlock).
        # Replaces the legacy hard ``1 / (H * W)`` that over-suppressed the (non
        # unit-energy) learned transform branch. Initialized so sigmoid(a) ~ 0.25 -> O(1)
        # transform branch at initialization.
        self.IT_scale = nn.Parameter(torch.tensor(-1.1))

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

        # Shape (out_channels, in_channels, m1, m2) to match the einsum
        # ``"bixy,oixy->boxy"`` (output channel ``o`` first, input channel ``i`` second),
        # consistent with SepLITBlock. (A prior (in, out, m1, m2) shape only worked when
        # in_channels == out_channels and mixed the wrong channels otherwise.)
        self.learned_weights_freq = nn.Parameter(
            torch.randn(out_channels, self.in_channels, self.m1, self.m2, dtype=torch.cfloat)
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

        # Dynamic normalization: learnable exponent (see self.IT_scale) instead of a hard 1/(H*W).
        normalisation_factor = 1.0 / ((H * W) ** torch.sigmoid(self.IT_scale))
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


class GalerkinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        mlp_dim=512,
        dropout=0.1,
        delta=0.01,
        activation: Callable = nn.GELU,
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

        # Accept either an activation class (nn.GELU) or an instance (nn.GELU())
        act = activation() if isinstance(activation, type) else activation

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            act,
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
    """Pre-LN Transformer block with standardized initialization and optional RoPE.

    Positional information can be provided two ways:

    - **additively**, by the caller (e.g. a ``PEBlock`` encoding added to the
      tokens before the block) -- the default behavior;
    - **rotarily**, through an optional
      :class:`~pynop.core.encoding.AdaptiveRoPE2D` enabled with ``rope=True``:
      the inner :class:`~pynop.core.ops.Attention` owns a 2-D rotary embedding
      applied to the query/key projections, driven by the absolute 2-D
      coordinates passed at forward time (``coords``). It is a no-op when no
      coordinates are passed, or when ``rope`` is disabled.

    The rotary embedding is defined at **initialization** time (``rope: bool``);
    it is not injected per call.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads (``dim`` must be divisible by it).
        activation: Activation class used in the MLP.
        mlp_dim: MLP expansion dimension.
        dropout: Dropout rate in the MLP.
        rmsnorm: If True use ``nn.RMSNorm`` (default), else ``nn.LayerNorm``.
        std_ini: Std of the truncated-normal init of the projections.
        scaling: Depth-scaled output-projection init (Megatron-LM style).
        rope: If True, the inner attention builds an
            :class:`~pynop.core.encoding.AdaptiveRoPE2D` (``dim // n_heads``
            must then be divisible by 4) and applies it to q/k whenever
            ``coords`` is passed at forward time.
        rope_kwargs: Extra keyword arguments forwarded to ``AdaptiveRoPE2D``
            (e.g. ``{"domain_size": 1.0, "base": 8.0}``).
    """

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
        rope: bool = False,
        rope_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        head_dim = dim // n_heads
        if rope and head_dim % 4 != 0:
            raise ValueError(
                f"TransformerBlock: rope=True requires dim // n_heads == {head_dim} to be "
                f"divisible by 4 for 2-D RoPE (dim={dim}, n_heads={n_heads})."
            )

        # Attention layer; owns the (optional) rotary positional embedding.
        self.attention = Attention(dim, dim, n_heads, std_ini=std_ini, rope=rope, rope_kwargs=rope_kwargs)

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

    @property
    def rope(self) -> Optional["nn.Module"]:
        """The inner attention's rotary embedding (``None`` when ``rope=False``).

        Kept as a convenience alias so the rotary module can be inspected at
        ``block.rope`` even though it is owned by :attr:`attention`.
        """
        return self.attention.rope

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
        coords: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        causal_attn: bool = False,
    ) -> torch.Tensor:
        """Forward pass (clean pre-LN residual stream).

        Args:
            x: Token sequence of shape ``[B, N, C]``.
            coords: Optional 2-D coordinates for the rotary embedding,
                shape ``[B, N, 2]`` or ``[B, H, W, 2]`` (the latter disambiguates
                non-square grids). Ignored when ``rope`` is disabled.
            mask: Optional additive attention mask for
                ``scaled_dot_product_attention``.
            causal_attn: If True, use causal (autoregressive) attention.

        Returns:
            Tensor of shape ``[B, N, C]``.
        """
        x_norm = self.norm1(x)
        x = x + self.attention(
            Q=x_norm,
            K=x_norm,
            V=x_norm,
            mask=mask,
            coords=coords,
            causal_attn=causal_attn,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class TransolverBlock(nn.Module):
    """Physics-Attention block of Transolver / Transolver++.

    Grid points are softly assigned to ``slice_num`` physical slices per head via a
    Gumbel-Softmax weight matrix, standard attention is run on the (few) slice tokens,
    and the result is desliced back with the same weights. Follows the official
    implementation: https://github.com/thuml/Transolver_plus/blob/b2f656d382d0c8454415f004f1c7af1436f90c42/models/Transolver_plus.py

    The block is pre-normalized, mirroring the official
    ``fx = Attn(ln_1(fx)) + fx`` / ``fx = mlp(ln_2(fx)) + fx`` structure: ``ln1`` is
    applied to the input *before* the attention sub-layer and ``ln2`` before the MLP
    projection.

    Args:
        dim: Input/output feature dimension (the residual stream width).
        heads: Number of attention heads.
        dim_head: Dimension per head; the attention inner dim is ``heads * dim_head``.
        dropout: Dropout rate applied to the projected attention output before the
            residual addition.
        slice_num: Number of physical slices K per head.
        mlp_ratio: Expansion ratio of the output MLP (hidden = ``mlp_ratio * dim``).
        activation: Activation class used in the temperature MLP and the output MLP.
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
        """Forward pass.

        Args:
            x: Token sequence of shape ``[B, N, C]`` with ``C == dim``.

        Returns:
            Tensor of shape ``[B, N, C]`` (residual connection included).
        """
        # input B N C
        B, N, C = x.shape

        # --- Physics Attention block (pre-norm: ln1 before the projection)
        x_mid = (
            self.in_project_x(self.ln1(x)).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
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

        # --- End of Physics Attention block

        out_x = self.dropout(self.proj(self.ln2(out_x))) + x

        return out_x


class LinearNOBlock(nn.Module):
    """Canonical linear attention block — the LinearNO formulation of physics attention.

    Reference: "Transolver is a Linear Transformer: Revisiting Physics-Attention
    through the Lens of Linear Attention" (Hu, Liu, Qiao, Sun & Dou, arXiv:2511.06294).

    Instead of an explicit soft assignment matrix ``w`` with diagonal normalization
    (the Transolver family mechanism — see :class:`TransolverBlock` and
    :class:`TransolverBlockv3`), this block rewrites physics attention as canonical
    linear attention with *asymmetric* kernel normalizations (paper Eqs. 8-9):

        phi(Q)   = softmax over the slice dim M  (per grid point, per head)
        psi(K)^T = softmax over the grid dim N   (per slice, per head)
        out      = phi(Q) @ (psi(K)^T @ V)       # O(N·M + M·N·D), no N×N attention

    The block is pre-normalized: ``x += dropout(out_proj(Attn(norm1(x))))`` followed
    by an MLP sublayer ``x += mlp(norm2(x))``.
    """

    def __init__(
        self,
        dim,
        n_tokens=64,
        n_heads=8,
        dropout=0.0,
        mlp_ratio=2,
        activation: Optional[Callable] = nn.GELU,
        rmsnorm=True,
    ):
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
                scale = kwargs.get("scale", kwargs.get("max_freq_val", 10.0))
                self.pe = GaussianFourierEmbedding(in_features, out_features, scale)

            case "fourier":
                max_freq = kwargs.get("max_freq", kwargs.get("max_freq_val", 128.0))
                self.pe = FourierEmbedding(in_features, out_features, max_freq=max_freq)

            case "ipe":
                max_freq = kwargs.get("max_freq", kwargs.get("max_freq_val", 128.0))
                self.pe = IntegratedPositionalEncoding(in_features, out_features, max_freq=max_freq)

            case "random_ipe":
                scale = kwargs.get("scale", kwargs.get("max_freq_val", 10.0))
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
                    in_features=in_features,
                    out_features=out_features,
                    base_max_freq=kwargs.get("max_freq", kwargs.get("max_freq_val", 256.0)),
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


class SlicingBlock(nn.Module):
    """Slicing operator of the Transolver family (single weight-set form).

    Softly assigns the ``N`` grid points to ``M`` slices with a temperature-scaled
    softmax weight matrix and computes each slice as the *diagonally normalized*
    weighted average of the input features:

        s_j = linear1( sum_i w_ij x_i / d_j ),   d_j = sum_i w_ij

    This is the "faster slicing" mechanism of Transolver-3 (arXiv:2602.04940, Sec. 3;
    physics attention originally from arXiv:2402.02366): the weighted average is taken
    in the mesh domain and the linear projection only touches the ``M`` slices. Because
    each row of ``w / d`` sums to 1, projecting before or after the weighted sum is
    mathematically identical (the bias is added once per slice either way); this class
    projects first.

    The full Transolver-3 block uses one weight set *per head* — see
    :class:`TransolverBlockv3`. This single-set variant is used by
    ``LatentTransolver`` (``pynop/models/LNO.py``), which processes the ``M`` slice
    tokens with its own attention stack.

    Note: "Transolver Is a Linear Transformer" (arXiv:2511.06294) reinterprets this
    physics-attention slicing as canonical linear attention; that formulation is
    implemented by :class:`LinearNOBlock`, not here.

    Args:
        in_features: Input feature dimension C.
        out_features: Projected (slice) feature dimension D.
        num_slices: Number of slices M.
        temp: Initial temperature; the effective one is ``softplus(param) + 0.05``.
    """

    def __init__(self, in_features: int, out_features: int, num_slices: int, temp: float = 1.0):
        super().__init__()
        self.M = num_slices
        self.weight_proj = nn.Linear(in_features, num_slices)  # w : R^{N x M}
        self.linear1 = nn.Linear(in_features, out_features)  # x_proj
        temp_ini = torch.log(torch.exp(torch.tensor(temp)) - 1.0)
        self.temp = nn.Parameter(temp_ini)

    def forward(self, x):
        """Slice the input tokens.

        Args:
            x: Input of shape ``[B, N, C]``.

        Returns:
            Tuple ``(s, w)`` with sliced tokens ``s`` of shape ``[B, M, D]`` and
            slice weights ``w`` of shape ``[B, N, M]`` (rows sum to 1).
        """
        B, N, C = x.shape
        temp = F.softplus(self.temp) + 0.05

        # w : [B, N, M] — rows sum to 1 (softmax over the slice dimension)
        w = torch.softmax(self.weight_proj(x) / temp, dim=-1)

        # d : [B, M] — per-slice weight sums (diagonal normalization, Eq. 3)
        d = w.sum(dim=1)  # [B, M]

        # Project first, then slice: s_raw = W^T . x_proj  ([B, M, N] @ [B, N, D])
        x_proj = self.linear1(x)
        s_raw = torch.bmm(w.permute(0, 2, 1), x_proj)  # [B, M, D]

        # Diagonal normalization
        s = s_raw / (d.unsqueeze(-1) + 1e-8)  # [B, M, D]

        return s, w


class DeslicingBlock(nn.Module):
    """Deslicing operator of the Transolver family (single weight-set form).

    Projects the ``M`` slice tokens to the output dimension and scatters them back
    onto the ``N`` grid points with the slicing weights:

        x_i = sum_j w_ij linear3(s'_j)

    This is the "faster deslice" mechanism of Transolver-3 (arXiv:2602.04940, Sec. 3).
    Because each row of ``w`` sums to 1, projecting the slices *before* deslicing is
    exactly equivalent to deslicing first and then applying the same linear layer —
    while only touching M tokens instead of N. The full Transolver-3 block applies a
    per-head projection before scattering and concatenates the heads; see
    :class:`TransolverBlockv3`.

    Args:
        in_features: Slice feature dimension D.
        out_features: Output feature dimension C_out.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.linear3 = nn.Linear(in_features, out_features)

    def forward(self, s_prime, w):
        """Deslice the slice tokens back to mesh space.

        Args:
            s_prime: Slice tokens of shape ``[B, M, D]``.
            w: Slicing weights of shape ``[B, N, M]`` (rows sum to 1).

        Returns:
            Tensor of shape ``[B, N, C_out]``.
        """
        # Faster deslice: project the M slices, then scatter with w
        s_out = self.linear3(s_prime)  # [B, M, C_out]

        # Reprojection onto N : w @ s_out — [B, N, M] @ [B, M, C_out]
        x_out = torch.bmm(w, s_out)  # [B, N, C_out]

        return x_out


class TransolverBlockv3(nn.Module):
    """Transolver-3 encoder block: per-head physics attention in the slice domain.

    Reference: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale
    Geometries" (arXiv:2602.04940), Sections 3.1-3.3, following the official
    implementation https://github.com/lbruand-db/distributed-transolver3
    (``PhysicsAttentionV3`` + ``Transolver3Block``).

    Block composition (pre-norm, two residual sublayers):

        fx = Attn(ln_1(fx)) + fx      # physics attention in the slice domain
        fx = mlp(ln_2(fx))  + fx      # pointwise MLP sublayer

    Physics attention operates on ``M`` *slices* per head instead of the ``N`` mesh
    tokens. Each head owns an **independent** soft assignment of the grid points to
    its slices (this is what distinguishes it from the single-weight-set
    :class:`SlicingBlock` used by LatentTransolver):

        w_h  = softmax(in_project_slice(x)[:, :, h, :] / T_h, dim=M)   # [B, N, M]
        s_h  = slice_linear1( sum_n w_h[n, m] x_n / d_h[m] )           # [B, H, M, D]
        s'_h = SDPA(to_q(s_h), to_k(s_h), to_v(s_h))                   # attention over the M slices
        out  = concat_h( sum_m w_h[n, m] slice_linear3(s'_h[m]) )      # [B, N, C], C = H * D

    The linear projections live in the slice domain (``M`` tokens) rather than the
    mesh domain (``N`` tokens): the "faster slicing / deslicing" of the paper.
    Because each row of ``w / d`` sums to 1, aggregating the raw features before
    projecting is mathematically identical to projecting first (the bias is added
    once per slice either way).

    The geometry-slice tiling and physical-state caching of the reference are
    intentionally **not** implemented here: they are memory / inference
    optimizations that are mathematically identical to this standard path.

    Note: ``in_project_slice`` is orthogonally initialized, mirroring the
    reference; when used inside :class:`~pynop.models.transolver.Transolver` the
    wrapper's global truncated-normal init overwrites it (same as in the
    reference model).

    Args:
        dim: Residual stream width C. Must equal ``heads * dim_head`` because the
            per-head outputs are concatenated back to C without a final projection.
        heads: Number of attention heads H.
        dim_head: Per-head dimension D (defaults to ``dim // heads``).
        slice_num: Number of slices M per head.
        mlp_ratio: Expansion ratio of the MLP sublayer (hidden = ``dim * mlp_ratio``);
            the reference model uses 1.
        dropout: Dropout rate applied inside the scaled-dot-product attention
            (training only) and to the desliced output before the residual addition.
        activation: Activation class used in the MLP sublayer.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: Optional[int] = None,
        slice_num: int = 64,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        activation: Callable = nn.GELU,
    ):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads
        assert dim == heads * dim_head, (
            f"Transolver-3 requires dim == heads * dim_head ({dim} != {heads} * {dim_head}): "
            "per-head outputs are concatenated back to the residual stream without a final projection."
        )
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        # --- Physics attention (slice domain) -----------------------------------
        # Per-head soft assignment of the N grid points to M slices.
        self.in_project_slice = nn.Linear(dim, heads * slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)  # as in the reference
        # Learnable per-head temperature, clamped to [0.1, 5] (paper).
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # "Faster slicing": aggregate the raw features first, then project the M slices.
        self.slice_linear1 = nn.Linear(dim, dim_head)
        # Attention over the M slice tokens; q/k/v shared across heads (standard MHA layout).
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        # "Faster deslicing": project the M slices back to per-head features before scattering.
        self.slice_linear3 = nn.Linear(dim_head, dim_head)
        self.out_dropout = nn.Dropout(dropout)

        # --- Block sublayers (pre-norm + residual, as in Transolver3Block) ------
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(in_ch=dim, out_ch=dim, hidden_dim=dim * mlp_ratio, num_layers=2, activation=activation)

    def _slice_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head slice weights.

        Args:
            x: Input of shape ``[B, N, C]``.

        Returns:
            Tensor ``w`` of shape ``[B, H, N, M]``; each row sums to 1 over the
            slice dimension (a grid point distributes its mass across slices).
        """
        B, N, _ = x.shape
        logits = self.in_project_slice(x).reshape(B, N, self.heads, self.slice_num)
        logits = logits.permute(0, 2, 1, 3).contiguous()  # [B, H, N, M]
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        return torch.softmax(logits / temp, dim=-1)

    def _physics_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Physics attention in the slice domain (paper Eqs. 3-4).

        Args:
            x: Input of shape ``[B, N, C]`` (already normalized by ``ln_1``).

        Returns:
            Tensor of shape ``[B, N, C]`` (dropout applied, residual **not** added).
        """
        B, N, _ = x.shape
        w = self._slice_weights(x)  # [B, H, N, M]
        d = w.sum(dim=2)  # [B, H, M] — per-slice mass (diagonal normalization)

        # Faster slicing: aggregate the raw features (O(N·M·C)), then project the M slices.
        # Per-head matmul with broadcast (official FIX #2): avoids materializing (B, H, N, C).
        s_raw = torch.matmul(w.transpose(2, 3), x.unsqueeze(1))  # [B,H,M,N] @ [B,1,N,C] -> [B,H,M,C]
        s = self.slice_linear1(s_raw / (d.unsqueeze(-1) + 1e-5))  # [B, H, M, D]

        # Standard attention among the slice tokens of each head.
        q, k, v = self.to_q(s), self.to_k(s), self.to_v(s)
        s_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.out_dropout.p if self.training else 0.0)

        # Faster deslicing: project the slices, scatter with w, concatenate heads.
        s_out = self.slice_linear3(s_out)  # [B, H, M, D]
        x_out = torch.matmul(w, s_out)  # [B, H, N, D]
        x_out = rearrange(x_out, "b h n d -> b n (h d)")  # [B, N, C]
        return self.out_dropout(x_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input mesh tensor of shape ``[B, N, C]`` with ``C == dim``.

        Returns:
            Output mesh tensor of shape ``[B, N, C]`` (residuals included).
        """
        fx = self._physics_attention(self.ln_1(x)) + x
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class IAETBlock(nn.Module):
    """Non-reversible non-linear integral transform block (IAE-Net style).

    A learned 2-D *non-linear* integral transform that is resolution-invariant,
    strongly inspired by IAE-NET ("Integral Autoencoders for Discretization-Invariant
    Learning", Ong et al., arXiv:2203.05142): the encoder and decoder kernels are
    learned as continuous functions via MLPs and are *conditioned on the data*, which
    makes the transform non-linear in its input. The inverse transform uses its own
    learned kernels (not the conjugate of the forward ones), so the block is **not
    reversible** -- hence "IAET".

    Architecture (channels-first ``(B, C_in, H, W)``, real or complex input):

    1. **Encoder** (direct transform, data-conditioned kernels):
       - width pass: kernel ``b_w`` = MLP(real values at the point + x-coordinate),
         applied along W -> ``(B, C_in, H, m2)``;
       - height pass: kernel ``b_h`` = MLP(values after the width pass + y-coordinate),
         applied along H -> spectral representation ``(B, C_in, m1, m2)``.
    2. **Spectral normalization**: complex-safe RMS over the channel dim of the
       spectral representation (the official IAE-Net applies a LayerNorm ``ln1`` at
       this point). The raw Riemann sums grow with the resolution; without it, the
       data-conditioned decoder kernels below would be driven by magnitude instead of
       content at high resolutions.
    3. **Frequency mixing**: per-mode complex weights ``(C_out, C_in, m1, m2)``.
    4. **Decoder** (inverse transform, *separately learned* data-conditioned kernels):
       - height reconstruction: kernel MLP conditioned on the output y-coordinate and a
         per-(channel, width-mode) compressed summary of the spectral coefficients;
       - width reconstruction: same pattern with the remaining mode dimension.
    5. **Learnable final scale** ``1 / (H*W)**sigmoid(IT_scale)`` (family convention, see
       :class:`SepLITBlock`), channel mixing (1x1 conv), channels-first norm, activation,
       and a 1x1-conv shortcut: ``out = act(act(mixer(norm(x_rec))) + shortcut(x))``.

    Input handling: the block operates on the **real part** of its input -- a real
    tensor is embedded as a complex field with zero imaginary part, a complex tensor is
    reduced to its real part (uniformly for the transform and the shortcut). The output
    is always real.

    Differences from the official IAE-Net implementation:
    - complex-valued kernels/bases (:class:`ComplexMLPBlock`) instead of real ones;
    - coordinates in ``[-1, 1]`` (the official uses ``[0, 1]``);
    - non-square grids are supported (the official 2-D code assumes square grids);
    - single branch: no parallel Fourier-IAE path and no dense connections (those live
      in the official model wrapper, not in the block itself).

    Note: with ``norm=None`` the output magnitude is set by the learnable final scale;
    with the default channels-first norm it is renormalized to O(1) per spatial point.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        m1: int,
        m2: int,
        mlp_hidden_dim: int = 64,
        mlp_num_layers: int = 2,
        activation: Callable = nn.GELU,
        norm: Optional[Callable] = LayerNorm2d,
    ) -> None:
        """Initialize the module.

        Args:
            in_channels: Number of input channels (C_in).
            out_channels: Number of output channels (C_out); may differ from ``in_channels``.
            m1: Number of modes kept for the height dimension (u).
            m2: Number of modes kept for the width dimension (v).
            mlp_hidden_dim: Hidden dimension of the MLPs learning the bases.
            mlp_num_layers: Number of hidden layers in the MLPs learning the bases.
            activation: Activation class used by the mixer and the basis MLPs.
            norm: Channels-first normalization class applied after the channel mixer
                (normalizes over C of the ``(B, C, H, W)`` layout), or ``None`` to disable.
                Default ``LayerNorm2d``.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.coords_dim = 1 + in_channels

        # Learnable exponent of the final integral-transform normalization.
        # The inverse transform accumulates over the spatial domain, so the raw output
        # magnitude grows with the resolution. The legacy hard factor ``1 / (H * W)``
        # assumed a DFT-like (unit-energy) basis; the learned basis is *not* unit-energy,
        # so that constant was inconsistent across resolutions. We instead normalize by
        # ``(H * W) ** sigmoid(a)`` and let the model learn ``a``. Initialized so
        # sigmoid(a) ~ 0.25 (same convention as SepLITBlock / SepNLITBlock).
        self.IT_scale = nn.Parameter(torch.tensor(-1.1))

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

        # Spectral normalization (the official IAE-Net applies a LayerNorm "ln1" after the
        # encoder's second transform). The raw Riemann sums grow with the resolution; without
        # it, the data-conditioned decoder kernels below would be driven by magnitude instead
        # of content at high resolutions. Complex-safe RMS over the channel dim (F.layer_norm
        # does not support complex tensors); well behaved even for a single channel.
        self.ln_freq = ComplexLayerNorm(-1)

        # Learned parameters for multiplication in the transformed space (per channel and mode).
        # Shape (out_channels, in_channels, m1, m2) to match the einsum "bixy,oixy->boxy"
        # (output channel first, input channel second), consistent with SepLITBlock /
        # SepNLITBlock. A prior (in, out, m1, m2) shape crashed when in != out and mixed
        # the wrong channels otherwise.
        self.learned_weights_freq = nn.Parameter(
            torch.empty(out_channels, in_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the module.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``; ``H`` and ``W`` may vary
                (resolution invariance). Real or complex -- the block operates on the
                real part (see the class docstring).

        Returns:
            Output tensor of shape ``(B, C_out, H, W)``, always real.
        """
        b, c, h, w = x.shape
        # The block operates on the real part of its input: a real tensor is embedded as a
        # complex field with zero imaginary part; a complex tensor is reduced to its real
        # part (the legacy ``x.float()`` discarded the imaginary part *and* crashed the first
        # matmul against the complex bases). The shortcut uses the same real input.
        x_real = x.real if x.is_complex() else x
        xc = torch.complex(x_real, torch.zeros_like(x_real))

        # --- Encoder (Direct transform) ---
        # 1. W (width) transformation: m2 kernels predicted per spatial coordinate from
        #    (real values at the point + x-coordinate). MLP input: (B, H, W, C+1)
        w_coords = torch.linspace(-1, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        mlp_w_in = torch.concat([xc.real.permute(0, 2, 3, 1), w_coords], dim=-1)
        b_w = self.basis_w_fn(mlp_w_in)  # (B, H, W, m2)

        # Matmul: (B, H, C, W) @ (B, H, W, m2) -> (B, H, C, m2)
        x_w = torch.matmul(xc.permute(0, 2, 1, 3), b_w)
        x_w = x_w.permute(0, 2, 1, 3)  # (B, C, H, m2)

        # 2. H (height) transformation: kernels conditioned on the values after the W pass.
        #    MLP input: (B, m2, H, C+1)
        h_coords = (
            torch.linspace(-1, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )
        mlp_h_in = torch.concat([x_w.real.permute(0, 3, 2, 1), h_coords], dim=-1)
        # predict m1 kernels for each coordinate (m2, H)
        b_h = self.basis_h_fn(mlp_h_in)  # (B, m2, H, m1)

        # Matmul: (B, m2, C, H) @ (B, m2, H, m1) -> (B, m2, C, m1)
        x_hw = torch.matmul(x_w.permute(0, 3, 1, 2), b_h)
        x_hw = x_hw.permute(0, 2, 3, 1)  # (B, C, m1, m2)

        # Spectral normalization (official IAE-Net "ln1"): keeps the spectral representation
        # O(1) at every resolution so that the data-conditioned decoder kernels below are
        # driven by content, not magnitude. RMS over the channel dim (complex-safe).
        x_hw = self.ln_freq(x_hw.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, C, m1, m2)

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

        # Dynamic normalization: learnable exponent (see self.IT_scale) instead of a hard
        # 1/(H*W), which over-suppressed the non unit-energy learned transform branch.
        normalisation_factor = 1.0 / ((h * w) ** torch.sigmoid(self.IT_scale))
        x_rec = x_rec * normalisation_factor

        # Mixing channels
        output = self.mixer(x_rec)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection (real input, see above)
        output = self.activation(output + self.shortcut(x_real))

        return output


class WindowAttention(nn.Module):
    """Window-based self-attention layer with rotary positional encoding.

    The attention is computed with the fused
    :func:`torch.nn.functional.scaled_dot_product_attention` kernel (flash /
    memory-efficient where available, with a graceful fallback to the math backend
    when an additive mask is used). Positional information is injected on the
    query/key vectors with a 2-D rotary embedding (:class:`AdaptiveRoPE2D`) indexed
    by each token's *absolute* coordinates, the frequency band being set from the
    grid Nyquist (resolution-adaptive).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """Initialize the attention layer.

        Args:
            dim: Size of each token embedding.
            num_heads: Number of attention heads. The embedding size must be divisible by this value.
            qkv_bias: If True, learn a bias in the fused QKV projection.
            attn_dropout: Dropout probability on the attention weights (training only).
            proj_dropout: Dropout probability on the output projection.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.scale = self.d_head**-0.5
        self.attn_dropout = attn_dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.rope = AdaptiveRoPE2D(self.d_head)

    def forward(
        self,
        x: torch.Tensor,  # [B*nW, N, dim]
        coords: torch.Tensor,  # [B*nW, win_h, win_w, 2]  absolute coordinates(y,x) in [-1, 1]
        mask: Optional[torch.Tensor] = None,
        resolution: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Apply window attention to a batch of local tokens.

        Args:
            x: Input tokens with shape [B*nW, N, dim].
            coords: Absolute coordinates (of original grid) reshaped as [B*nW, win_h, win_w, 2]
            mask: Optional attention mask with shape [num_windows, win_h*win_w, win_h*win_w].
            resolution: (H_grid, W_grid) of the underlying grid, used to set the RoPE
                frequency band (Nyquist). Pass the full grid size, not the window size.

        Returns:
            A tensor of shape [B*nW, N, dim] containing the attended features.
        """

        b_win, n, c = x.shape
        h = self.num_heads
        # [3, b*nwin, heads, n, d_head] where n is the number of token in a window
        qkv = self.qkv(x).reshape(b_win, n, 3, h, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*nW, Heads, N, d_head]

        # Apply RoPE to queries and keys only.
        q = self.rope.rotate(q, coords, resolution=resolution)
        k = self.rope.rotate(k, coords, resolution=resolution)

        if mask is not None:
            # mask is (nW, N, N); expand to (b_win, 1, N, N) so it broadcasts over the
            # window batch (dim 0) and the heads (dim 1) expected by SDPA.
            nw = mask.shape[0]
            b = b_win // nw
            mask = mask.view(1, nw, 1, n, n).expand(b, nw, 1, n, n).reshape(b_win, 1, n, n)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).reshape(b_win, n, c)
        return self.proj_drop(self.proj(out))


class WindowCrossAttention(nn.Module):
    """Local cross-attention module that compresses tokens into a small learnable query grid."""

    def __init__(self, in_ch: int, dim: int, num_heads: int, m: int = 1):
        """Initialize the cross-attention module.

        Args:
            in_ch: Input feature dimension for the keys and values.
            dim: Hidden dimension used for the queries, keys, values, and output projection.
            num_heads: Number of attention heads. The dimension must be divisible by this value.
            m: Side length of the learnable query grid, yielding $m^2$ queries per window.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.d_head = dim // num_heads
        self.m = m
        self.scale = self.d_head**-0.5

        # Learnable queries defined in the target latent space.
        self.latent_queries = nn.Parameter(torch.randn(1, m * m, dim) * 0.02)

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(in_ch, dim, bias=True)
        self.v_proj = nn.Linear(in_ch, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Compress a set of local tokens into a learned query grid.

        Args:
            x: Input tokens with shape [B_win, N_in, in_ch].
            coords: Absolute coordinates for the input tokens with shape [B_win, N_in, 2].

        Returns:
            Aggregated tokens with shape [B_win, m*m, dim].
        """
        b_win, n, _ = x.shape
        h = self.num_heads

        # Project queries
        q = self.latent_queries.expand(b_win, -1, -1)
        q = self.q_proj(q).reshape(b_win, self.m * self.m, h, self.d_head).transpose(1, 2)

        # Project keys and values
        k = self.k_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(b_win, n, h, self.d_head).transpose(1, 2)

        # Position encoding on keys can be added here if needed.

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_win, self.m * self.m, self.dim)
        return self.out_proj(out)


class AdaptiveSwinBlock(nn.Module):
    """A Swin-style transformer block with local window attention and optional shifted windows."""

    def __init__(
        self,
        dim: int,
        grid_windows: tuple[int, int],
        num_heads: int,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
        shift: bool = False,
    ):
        """Initialize the Swin block.

        Args:
            dim: Feature dimension of each token.
            grid_windows: Number of windows along the height and width axes, provided as (nw_h, nw_w).
            num_heads: Number of attention heads in the local window attention module.
            activation: Activation class used by the MLP subnetwork.
            mlp_factor: Expansion factor for the hidden size of the MLP.
            shift: If True, use shifted windows in the attention operation.
        """
        super().__init__()
        self.dim = dim
        self.nw_h, self.nw_w = grid_windows
        self.num_heads = num_heads
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_factor * dim),
            activation(),
            nn.Linear(mlp_factor * dim, dim),
        )

    def _create_mask(self, h, w, win_h, win_w, shift_h, shift_w, device):
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None))
        w_slices = (slice(0, -win_w), slice(-win_w, -shift_w), slice(-shift_w, None))
        cnt = 0
        for h_sl in h_slices:
            for w_sl in w_slices:
                img_mask[:, h_sl, w_sl, :] = cnt
                cnt += 1
        mask_windows = img_mask.view(1, self.nw_h, win_h, self.nw_w, win_w, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _build_coords(self, h, w, win_h, win_w, shift_h, shift_w, b, device):
        """Build absolute coordinates for tokens after window partitioning.

        Returns a tensor of shape [B*nW, win_h*win_w, 2] in the range [-1, 1].
        """
        # Global grid before shifting.
        gy = (torch.arange(h, device=device).float() / h) * 2 - 1  # [-1, 1)
        gx = (torch.arange(w, device=device).float() / w) * 2 - 1
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H, W]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H, W, 2]

        # Apply the same shift to the coordinates as to the feature map.
        if shift_h > 0 or shift_w > 0:
            coords_img = torch.roll(coords_img, shifts=(-shift_h, -shift_w), dims=(0, 1))

        # Partition the coordinates exactly like the feature tokens.
        coords_win = coords_img.view(self.nw_h, win_h, self.nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(self.nw_h * self.nw_w, win_h, win_w, 2)

        # Repeat over the batch dimension.
        coords_win = coords_win.unsqueeze(0).expand(b, -1, -1, -1, -1)  # [B, nW, win_h, win_w, 2]
        return coords_win.reshape(b * self.nw_h * self.nw_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor, use_mask: bool = True) -> torch.Tensor:
        """Apply the Swin block to a 4D feature map.

        Args:
            x: Input tensor of shape [B, H, W, C].
            use_mask: If True, generate an attention mask when shifted windows are enabled.

        Returns:
            A tensor with the same shape as the input, containing the updated features.
        """
        b, h, w, d = x.shape
        if h % self.nw_h != 0 or w % self.nw_w != 0:
            raise ValueError(
                f"AdaptiveSwinBlock: input spatial size ({h}x{w}) must be divisible by "
                f"grid_windows ({self.nw_h}x{self.nw_w}); got remainder ({h % self.nw_h}, {w % self.nw_w})."
            )
        shortcut = x
        x = self.norm1(x)

        win_h, win_w = h // self.nw_h, w // self.nw_w
        shift_h, shift_w = (win_h // 2, win_w // 2) if self.shift else (0, 0)

        if self.shift and use_mask:
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            mask = self._create_mask(h, w, win_h, win_w, shift_h, shift_w, x.device)
        else:
            mask = None

        # Absolute coordinates with shape [B * num_win, win_h, win_w, 2].
        coords = self._build_coords(h, w, win_h, win_w, shift_h, shift_w, b, x.device)

        x_win = x.view(b, self.nw_h, win_h, self.nw_w, win_w, d)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

        # Pass the full grid resolution so the RoPE frequency band is set from the grid Nyquist
        attn_win = self.attn(x_win, coords=coords, mask=mask, resolution=(h, w))

        x = attn_win.view(b, self.nw_h, self.nw_w, win_h, win_w, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, d)

        if self.shift:
            x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

        return shortcut + x + self.mlp(self.norm2(x))


class PaddedSwinBlock(nn.Module):
    """A Swin-style block with optional padding and boundary-aware window attention."""

    def __init__(
        self,
        dim: int,
        grid_windows: tuple[int, int],
        num_heads: int,
        activation: Callable = nn.GELU,
        pad: bool = False,
        mlp_factor: int = 4,
        bc_mode: str = "circular",
        rope_max_freq: float = 8.0,
    ):
        """Initialize the padded Swin block.

        Args:
            dim: Feature dimension of each token.
            grid_windows: Number of windows along height and width, provided as (nw_h, nw_w).
            num_heads: Number of attention heads in the local window attention module.
            activation: Activation class used by the MLP subnetwork.
            pad: If True, pad the input tensor before computing window attention.
            mlp_factor: Expansion factor for the hidden size of the MLP.
            bc_mode: Boundary padding mode used for the padded feature map.
            rope_max_freq: Maximum frequency used by the positional encoding component.
        """
        super().__init__()
        self.dim = dim
        self.nw_h, self.nw_w = grid_windows
        self.num_heads = num_heads
        self.pad = pad
        self.bc_mode = bc_mode
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_factor * dim),
            activation(),
            nn.Linear(mlp_factor * dim, dim),  # corrig� : �tait 4*dim
        )

    def _build_coords(
        self,
        h: int,
        w: int,
        h_pad: int,
        w_pad: int,
        pad_top: int,
        pad_left: int,
        win_h: int,
        win_w: int,
        current_nw_h: int,
        current_nw_w: int,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build absolute coordinates for padded tokens in the range [-1, 1].

        ``pad_top`` / ``pad_left`` are the number of padded rows/columns *before* the
        original region (the coordinate offset). The original ``h x w`` grid spans
        ``[-1, 1)`` (cell-centred); the padded tokens lie outside that range.

        Returns a tensor of shape [B * current_nw_h * current_nw_w, win_h * win_w, 2].
        """
        # Uniform step based on the original grid.
        step_y = 2.0 / h
        step_x = 2.0 / w

        # Coordinates of the padded tensor: original tokens lie in [-1, 1), while padded tokens lie outside that range.
        gy = (torch.arange(h_pad, device=device).float() - pad_top) * step_y - 1.0 + step_y / 2
        gx = (torch.arange(w_pad, device=device).float() - pad_left) * step_x - 1.0 + step_x / 2

        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [H_pad, W_pad]
        coords_img = torch.stack([cy, cx], dim=-1)  # [H_pad, W_pad, 2]

        # Partition the coordinates like the flattened window tokens.
        coords_win = coords_img.view(current_nw_h, win_h, current_nw_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(current_nw_h * current_nw_w, win_h, win_w, 2)

        return (
            coords_win.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(b * current_nw_h * current_nw_w, win_h, win_w, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the padded Swin block to a 4D feature map.

        Args:
            x: Input tensor of shape [B, H, W, C].

        Returns:
            A tensor with the same shape as the input, containing the updated features.
        """
        b, h, w, d = x.shape
        if h % self.nw_h != 0 or w % self.nw_w != 0:
            raise ValueError(
                f"PaddedSwinBlock: input spatial size ({h}x{w}) must be divisible by "
                f"grid_windows ({self.nw_h}x{self.nw_w}); got remainder ({h % self.nw_h}, {w % self.nw_w})."
            )
        shortcut = x
        x = self.norm1(x)

        win_h, win_w = h // self.nw_h, w // self.nw_w

        if self.pad:
            # Pad a total of exactly ``win`` per axis so the padded size stays divisible
            # by the window size (h = nw * win). Split it as evenly as possible to keep at
            # least ``win // 2`` rows/cols of boundary context on each side; this is
            # symmetric (and identical to the previous behaviour) when ``win`` is even,
            # and adds one extra padded row/col when ``win`` is odd (the fix).
            pad_top, pad_bottom = win_h // 2, win_h - win_h // 2
            pad_left, pad_right = win_w // 2, win_w - win_w // 2

            x_padded = F.pad(
                x.permute(0, 3, 1, 2),
                (pad_left, pad_right, pad_top, pad_bottom),
                mode=self.bc_mode,
            ).permute(
                0, 2, 3, 1
            )  # [B, H_pad, W_pad, D]

            h_pad, w_pad = x_padded.shape[1], x_padded.shape[2]
            current_nw_h = h_pad // win_h
            current_nw_w = w_pad // win_w

            coords = self._build_coords(
                h,
                w,
                h_pad,
                w_pad,
                pad_top,
                pad_left,
                win_h,
                win_w,
                current_nw_h,
                current_nw_w,
                b,
                x.device,
            )

            x_win = x_padded.view(b, current_nw_h, win_h, current_nw_w, win_w, d)
            x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

            attn_win = self.attn(x_win, coords=coords, resolution=(h, w))

            x_merged = attn_win.view(b, current_nw_h, current_nw_w, win_h, win_w, d)
            x_merged = x_merged.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h_pad, w_pad, d)
            x = x_merged[:, pad_top : pad_top + h, pad_left : pad_left + w, :]

        else:
            coords = self._build_coords(h, w, h, w, 0, 0, win_h, win_w, self.nw_h, self.nw_w, b, x.device)

            x_win = x.view(b, self.nw_h, win_h, self.nw_w, win_w, d)
            x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w, d)

            attn_win = self.attn(x_win, coords=coords, resolution=(h, w))

            x = attn_win.view(b, self.nw_h, self.nw_w, win_h, win_w, d)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, d)

        return shortcut + x + self.mlp(self.norm2(x))


class SwinBlock(nn.Module):
    """Classical Swin Transformer block (Liu et al., 2021) built on :class:`WindowAttention`.

    Unlike :class:`AdaptiveSwinBlock` (which fixes the *number* of windows and lets the
    window size adapt to the resolution) and :class:`PaddedSwinBlock` (which fixes the
    number of windows and pads the *boundary*), this block fixes the *window size* in
    pixels (``window_size``, default 7). It therefore works with **any** spatial size:
    the input is padded on the right/bottom to the next multiple of the window, the
    window attention is run, and the result is cropped back to the original ``(H, W)``.
    The number of windows grows with the resolution, so a window always covers the same
    physical patch -- the standard Swin design.

    It reuses the single :class:`WindowAttention` core (shared by all the Swin blocks):
    positional information comes from ``AdaptiveRoPE2D`` applied to the *absolute*
    coordinates of each token (frequency band set from the grid Nyquist), rather than a
    learned relative-position bias table. When ``shift`` is enabled, the features and the
    coordinates are both rolled by half a window and a standard Swin attention mask blocks
    cross-window attention in the mixed windows.

    Operates on a channels-last map ``(B, H, W, C)`` like the other Swin blocks.

    Args:
        dim: Feature dimension ``C`` (divisible by ``num_heads`` and, for the 2-D RoPE,
            by ``4 * num_heads``).
        num_heads: Number of attention heads.
        window_size: Window side length(s) ``[Wh, Ww]`` (default 7).
        shift: If ``True``, use shifted windows (shift of ``window_size // 2`` per axis,
            only when that axis holds more than one window).
        mlp_factor: MLP expansion factor (default 4).
        activation: Activation class for the MLP (default :class:`nn.GELU`).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Union[int, Tuple[int, int]] = 7,
        shift: bool = False,
        mlp_factor: int = 4,
        activation: Callable = nn.GELU,
    ):
        super().__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_factor * dim),
            activation(),
            nn.Linear(mlp_factor * dim, dim),
        )

    def _create_mask(
        self, h: int, w: int, win_h: int, win_w: int, shift_h: int, shift_w: int, device: torch.device
    ) -> torch.Tensor:
        """Build the shifted-window attention mask (standard Swin construction).

        Args:
            h, w: (padded) spatial size.
            win_h, win_w: window size.
            shift_h, shift_w: shift along height / width.
            device: target device.

        Returns:
            Mask ``(nW, win_h*win_w, win_h*win_w)`` (0 where attention is allowed,
            ``-100`` otherwise), with windows ordered ``(row, col)``.
        """
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None))
        w_slices = (slice(0, -win_w), slice(-win_w, -shift_w), slice(-shift_w, None))
        cnt = 0
        for h_sl in h_slices:
            for w_sl in w_slices:
                img_mask[:, h_sl, w_sl, :] = cnt
                cnt += 1
        nW_h, nW_w = h // win_h, w // win_w
        mask_windows = img_mask.view(1, nW_h, win_h, nW_w, win_w, 1)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_h * win_w)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def _build_coords(
        self,
        h: int,
        w: int,
        h_pad: int,
        w_pad: int,
        win_h: int,
        win_w: int,
        nW_h: int,
        nW_w: int,
        shift_h: int,
        shift_w: int,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build absolute coordinates for the (padded) tokens, in the range ``[-1, 1)``.

        The original ``h x w`` grid spans ``[-1, 1)`` with the original cell step
        (``2/h``, ``2/w``); the right/bottom padding extends beyond ``1``. When shifted,
        the coordinates are rolled exactly like the feature map so that each token keeps
        its true absolute position for the RoPE.

        Returns:
            Tensor ``[B*nW, win_h, win_w, 2]`` of ``(y, x)`` coordinates.
        """
        step_y = 2.0 / h
        step_x = 2.0 / w
        gy = torch.arange(h_pad, device=device).float() * step_y - 1.0 + step_y / 2
        gx = torch.arange(w_pad, device=device).float() * step_x - 1.0 + step_x / 2
        cy, cx = torch.meshgrid(gy, gx, indexing="ij")  # [h_pad, w_pad]
        coords_img = torch.stack([cy, cx], dim=-1)  # [h_pad, w_pad, 2]

        if shift_h > 0 or shift_w > 0:
            coords_img = torch.roll(coords_img, shifts=(-shift_h, -shift_w), dims=(0, 1))

        coords_win = coords_img.view(nW_h, win_h, nW_w, win_w, 2)
        coords_win = coords_win.permute(0, 2, 1, 3, 4).contiguous()  # [nW_h, nW_w, win_h, win_w, 2]
        coords_win = coords_win.view(nW_h * nW_w, win_h, win_w, 2)
        coords_win = coords_win.unsqueeze(0).expand(b, -1, -1, -1, -1)
        return coords_win.reshape(b * nW_h * nW_w, win_h, win_w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classical Swin block on a channels-last feature map.

        Args:
            x: Feature map ``(B, H, W, C)`` (any positive spatial size).

        Returns:
            Feature map ``(B, H, W, C)`` (same spatial size as the input).
        """
        b, h, w, d = x.shape
        shortcut = x
        x = self.norm1(x)

        ws_h, ws_w = self.window_size
        # Pad on the right/bottom to the next multiple of the window size.
        pad_h = (-h) % ws_h
        pad_w = (-w) % ws_w
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # channels-last: (L, R, T, B)
        hp, wp = h + pad_h, w + pad_w

        nW_h, nW_w = hp // ws_h, wp // ws_w

        sh = sw = 0
        if self.shift:
            # Shift by half a window, only when that axis holds more than one window.
            if hp > ws_h:
                sh = ws_h // 2
            if wp > ws_w:
                sw = ws_w // 2

        if sh or sw:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
            mask = self._create_mask(hp, wp, ws_h, ws_w, sh, sw, x.device)
        else:
            mask = None

        coords = self._build_coords(h, w, hp, wp, ws_h, ws_w, nW_h, nW_w, sh, sw, b, x.device)

        # Partition into windows: (B, hp, wp, C) -> (B*nW, ws_h*ws_w, C).
        x_win = x.view(b, nW_h, ws_h, nW_w, ws_w, d)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws_h * ws_w, d)

        # Pass the *original* grid resolution so the RoPE band is set from the grid Nyquist.
        attn_win = self.attn(x_win, coords=coords, mask=mask, resolution=(h, w))

        # Merge windows back: (B*nW, ws_h*ws_w, C) -> (B, hp, wp, C).
        x = attn_win.view(b, nW_h, nW_w, ws_h, ws_w, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, d)

        if sh or sw:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
        if pad_h or pad_w:
            x = x[:, :h, :w, :]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class BlockCrossAttentionUpsampler(nn.Module):
    """Upsampler doing local cross-attention by grouping high-res queries into blocks."""

    def __init__(
        self,
        latent_dim: int,
        shortcut_dim: int,
        num_heads: int = 4,
        activation: Callable = nn.GELU,
        mlp_factor: int = 4,
    ):
        """
        Args:
            latent_dim:   Dimension of the latent grid features (D).
            shortcut_dim: Dimension of the high-res shortcut features (C).
            num_heads:    Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (latent_dim // num_heads) ** -0.5

        self.q_proj = nn.Linear(shortcut_dim, latent_dim)
        self.kv_proj = nn.Linear(latent_dim, latent_dim * 2)
        self.out_proj = nn.Linear(latent_dim, shortcut_dim)
        self.mlp = nn.Sequential(
            nn.Linear(shortcut_dim, mlp_factor * shortcut_dim),
            activation(),
            nn.Linear(mlp_factor * shortcut_dim, shortcut_dim),
        )
        self.q_norm = nn.RMSNorm(shortcut_dim)
        self.latent_norm = nn.RMSNorm(latent_dim)
        self.out_norm = nn.RMSNorm(shortcut_dim)

    def forward(self, z_grid: torch.Tensor, shortcut: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_grid:   [B, M1, M2, latent_dim]   Low-res latent grid.
            shortcut: [B, H, W, shortcut_dim]   High-res encoder features.

        Returns:
            [B, H, W, shortcut_dim]             Upsampled features.
        """
        B, M1, M2, D = z_grid.shape
        _, H, W, C = shortcut.shape
        h = self.num_heads
        d_head = D // h

        # Compute upsampling ratios per axis
        rh, rw = H // M1, W // M2
        n_queries = rh * rw  # Number of high-res pixels per latent token

        # Normalize the shortcut features.
        shortcut = self.q_norm(shortcut)
        z_grid = self.latent_norm(z_grid)

        # Reshape and permute shortcut into local blocks
        # [B, H, W, C] -> [B, M1, rh, M2, rw, C] -> [B, M1, M2, rh*rw, C]
        q_local = (
            shortcut.view(B, M1, rh, M2, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, M1, M2, n_queries, C)
        )

        # Project Queries, Keys, and Values
        # Q: [B, M1, M2, h, n_queries, d_head]
        q = self.q_proj(q_local).view(B, M1, M2, n_queries, h, d_head).permute(0, 1, 2, 4, 3, 5)

        # KV: [B, M1, M2, 1, 2*D] -> K, V: [B, M1, M2, h, 1, d_head]
        k, v = self.kv_proj(z_grid).view(B, M1, M2, 1, 2, h, d_head).unbind(dim=4)
        k = k.permute(0, 1, 2, 4, 3, 5)
        v = v.permute(0, 1, 2, 4, 3, 5)

        # Compute Attention over the block: [B, M1, M2, h, n_queries, 1]
        # Each of the n_queries attends to the single local latent token
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Context aggregation and output projection
        # out_local: [B, M1, M2, n_queries, shortcut_dim]
        out_local = torch.matmul(attn, v).permute(0, 1, 2, 4, 3, 5).reshape(B, M1, M2, n_queries, D)
        out_local = self.out_proj(out_local)

        # Reconstruct original high-res grid layout [B, H, W, shortcut_dim]
        out = out_local.view(B, M1, M2, rh, rw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

        out = out + shortcut

        return out + self.mlp(self.out_norm(out))
