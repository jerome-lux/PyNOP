import math
from typing import Union, Optional, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import torch.nn.init as init

from torch.nn.utils import spectral_norm
from .norm import LayerNorm2d
from .utils import get_same_padding_2d
from .encoding import AdaptiveRoPE2D

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


class FiniteDifferenceLayer(nn.Module):
    """Differential convolution layer (NeuralOperator-style, Liu-Schiaffini et al. ICML 2024).

    Learns a base integral kernel K and, at forward time, converts it to a
    differential kernel:

        K_diff = (K - mean(K)) / h

    where h is the grid spacing. Two key properties:

    1. **Resolution invariance**: the 1/h rescaling ensures the operator
       converges to the same continuous differential operator as the grid is
       refined (h → 0), unlike a standard conv whose effect vanishes.
    2. **High-pass / zero-DC**: subtracting the kernel mean removes the DC
       component, so the operator annihilates spatially constant fields and
       captures local variations (gradients, curvature) rather than averaging.

    This complements global/low-rank operators (FNO, Transolver, Galerkin)
    which structurally act as low-pass filters and cannot represent high
    frequencies.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial kernel size (odd integer, e.g. 3 or 5).
        grid_width: Physical width of the domain (e.g. 2.0 for [-1, 1]).
            Used to compute h = grid_width / (N - 1).
        use_bias: Whether to include a bias term.
        norm: Optional normalization class (e.g. nn.RMSNorm, LayerNorm2d).
        activation: Optional activation class (e.g. nn.GELU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        grid_width: float = 2.0,
        stride: int = 1,
        padding: Union[int, str] = "same",
        use_bias: bool = True,
        norm: Optional[Callable] = None,
        activation: Optional[Callable] = None,
    ):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.grid_width = grid_width
        self.stride = stride
        # "same" padding (symmetric) for odd kernels
        self.padding = kernel_size // 2 if padding == "same" else padding

        # Base (integral) kernel — the learned coefficients before differential conversion
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Kaiming init (standard for conv layers)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.norm = norm(out_channels) if norm is not None else None
        self.activation = activation() if activation is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the differential convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        _, _, H, _ = x.shape
        # Grid spacing (assumes uniform grid over domain of width `grid_width`)
        h = self.grid_width / max(H - 1, 1)

        # Convert base kernel to differential kernel: center + 1/h rescale
        kernel = (self.weight - self.weight.mean(dim=(-2, -1), keepdim=True)) / h

        out = F.conv2d(x, kernel, self.bias, padding=self.padding, stride=self.stride, groups=1)

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, grid_width={self.grid_width}"
        )


class MaxPoolConv(nn.Module):
    """Max-pooling downsampling followed by a 1x1 channel-mixing convolution.

    (B, C_in, H, W) -> (B, C_in, H // factor, W // factor) -> (B, C_out, H // factor, W // factor)

    The pooling uses ``stride = factor`` with ``same`` padding (so the spatial
    size is exactly ``H // factor``, ``W // factor``). This is a simple,
    parameter-light downsampling that is **not** pixel-shuffle based.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Max-pool kernel size (default odd; padding is computed for
            ``same`` output size).
        factor: Downsampling stride (spatial size is divided by this).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool and mix channels.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Tensor of shape (B, C_out, H // factor, W // factor).
        """
        x = self.pool(x)
        x = self.conv(x)
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
        self.norm = norm(out_features) if norm is not None else None
        self.act = activation() if activation is not None else None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Tucker-factorized spectral convolution.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)`` (or channels-last
                ``(B, H, W, C_in)`` when ``channel_last=True``).

        Returns:
            torch.Tensor: Real output of shape ``(B, C_out, H*scaling, W*scaling)``.

        Resolution handling: the ``S`` factor spans the *configured* mode count.
        When the input grid is smaller than the configured modes, the low-
        frequency subspace of ``S`` is used (``min`` of configured and available
        modes) instead of mutating module state — so the layer stays valid for
        any input resolution (resolution invariance).
        """
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Les canaux d'entrée attendus sont {self.in_channels}, mais reçus {C_in}"

        # Les dimensions spectrales après rfft2(H, W) sont (H, W//2 + 1) ;
        # les modes effectifs sont plafonnés à ce qui est disponible (pas de
        # mutation d'état du module).
        modes_x = min(self.modes_x, H)
        modes_y = min(self.modes_y, W // 2 + 1)

        # 1. FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")  # (B, C_in, H, W//2 + 1), cfloat

        # 2. select modes
        x_ft_low_modes = x_ft[:, :, :modes_x, :modes_y]  # (B, C_in, mx, my), cfloat
        S = self.S[:modes_x, :modes_y, :]  # (mx, my, r3)

        # 3. Compute What
        G1 = torch.einsum("ij,jkl->ikl", self.U, self.G)  # (C_in, r2, r3), cfloat
        G2 = torch.einsum("ok,ikl->iol", self.V, G1)  # (C_in, C_out, r3), cfloat
        # S(mx, my, r3) et G2(i, o, r3)
        W_hat = torch.einsum("iol,xyl->ioxy", G2, S)  # (C_in, C_out, modes_x, modes_y), cfloat

        # 4. Spectral conv
        # y_ft_low_modes[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * W_hat[i, o, mx, my])
        y_ft_low_modes = torch.einsum("bixy,ioxy->boxy", x_ft_low_modes, W_hat)  # (B, C_out, mx, my), cfloat

        # 5. Zero-padding
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=x_ft.dtype, device=device)
        out_ft[:, :, :modes_x, :modes_y] = y_ft_low_modes

        # 6. IFFT
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

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
        """Apply the spectral convolution.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)`` (or channels-last
                ``(B, H, W, C_in)`` when ``channel_last=True``).

        Returns:
            torch.Tensor: Real output of shape ``(B, C_out, H*scaling, W*scaling)``.

        Resolution handling: the learned weights span the *configured* mode
        count. When the input grid is smaller than the configured modes, the
        low-frequency subspace of the weights is used (``min`` of configured and
        available modes) instead of mutating module state — so the layer stays
        valid for any input resolution (resolution invariance).
        """
        if self.channel_last:
            x = x.permute(0, 3, 1, 2)
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Expected input channels are {self.in_channels}, but received {C_in}"

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1); clamp the
        # *effective* modes to what is available (no state mutation).
        modes_x = min(self.modes_x, H)
        modes_y = min(self.modes_y, W // 2 + 1)

        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_ft_low_modes = x_ft[:, :, :modes_x, :modes_y]
        spectral_weights = self.spectral_weights[:, :, :modes_x, :modes_y]
        y_ft_low_modes_out = torch.einsum(
            "bixy, oixy -> boxy", x_ft_low_modes, spectral_weights
        )  # (B, C_out, modes_x, modes_y)

        # Zero-padding of ignored frequencies
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=x_ft.dtype, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, :modes_x, :modes_y] = y_ft_low_modes_out
        # irfft2 to get a real output of size (H, W)
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

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

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1); clamp the
        # *effective* modes to what is available (no state mutation).
        modes_x = min(self.modes_x, H)
        modes_y = min(self.modes_y, W // 2 + 1)

        xhat = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        xhat = xhat[:, :, :modes_x, :modes_y]

        # depthwise convolution (i.e. depthwise multiplication) followed by linear channel mixing
        spectral_weights_1 = self.spectral_weights_1[:, :modes_x, :modes_y]
        xhat = torch.einsum("bimn, imn -> bimn", xhat, spectral_weights_1)
        xhat = torch.einsum("bimn,oi->bomn", xhat, self.spectral_weights_2)

        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=xhat.dtype, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, :modes_x, :modes_y] = xhat
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        return y


class GalerkinAttention(nn.Module):
    """Linear Galerkin Attention approximating integral operators without softmax.

    Unlike standard softmax attention which computes exp(QK^T/sqrt(d))V, this uses the
    linear approximation: out = Q(K^T V / n). This is NOT a replacement for softmax
    attention but rather implements a continuous operator discretized via Galerkin
    quadrature weights 1/n. The kernel matrix K^T V represents the integral of basis
    functions over the domain, giving O(N) complexity instead of O(N^2).

    Args:
        dim: Input feature dimension (must be divisible by heads).
        heads: Number of attention heads (default 8).
        delta: Identity weight for diagonally dominant initialization (default 1e-2).
            Ensures gradients flow well during early training.
        std_ini: Standard deviation for trunc_normal output projection initialization.
            Uses small value to stabilize deep stacks (default 1e-2).
        kv_normalization: If True, applies RMSNorm to K and V before kernel computation
            (helps numerical stability when values have large magnitude).
    """

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

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for Galerkin Attention.

        Args:
            x: Query tensor of shape [B, M, dim].
            context: Context tensor of shape [B, N, dim] (defaults to x if None).

        Returns:
            Output tensor of shape [B, M, dim] containing attended features.
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
    """Kernelized linear cross-attention with ELU feature mapping (Performer-style).

    Computes attention using the approximation: softmax(Q K^T / sqrt(d)) V ≈
    φ(Q)(φ(K)^T V) / Z, where φ(x) = max(0,x) + min(e^x - 1, eps) is a positive-
    definite feature map (Katharopoulos et al., "Transformers are RNNs"). This gives 
    O(N+M) time/complexity instead of O(N*M), making it suitable for high-resolution
    target queries (N = H_target * W_target tokens).

    Args:
        d_model: Input feature dimension.
        n_heads: Number of attention heads (default 4).
            Use smaller values than standard softmax attention because the kernelized
            approach works best with fewer, wider heads that capture global structure.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
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

        # Apply non-negative feature mapping (crucial for stability). Use the
        # standard Performer-style sqrt(1/d) scaling to keep inner products O(1).
        q = self.phi(q / math.sqrt(self.d_head)) + 1.0
        k = self.phi(k / math.sqrt(self.d_head)) + 1.0

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
    """Simplified linear attention variant using low-rank factorization.

    Unlike standard softmax attention which computes exp(QK^T/sqrt(d))V, this uses a
    different approach: first normalize Q over its features (dim=-1) and K over 
    sequence (dim=-2), then compute scores = K^T V / N, output = Q @ scores. This is
    NOT true kernelized linear attention but rather an alternative normalization approach
    that can be useful for certain architectures where full softmax is computationally expensive.

    Args:
        dim: Input feature dimension (must be divisible by num_heads).
        num_heads: Number of attention heads (default 4).
            Use smaller values than standard softmax attention because this low-rank
            factorization works best with fewer, wider heads that capture global structure.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for LinearAttention with feature-wise softmax normalization.

        Args:
            x: Query tensor of shape [B, M, dim].
            context: Context tensor of shape [B, N, dim] (defaults to x if None).

        Returns:
            Output tensor of shape [B, M, dim] containing attended features via low-rank factorization.
        """
        B, N, C = x.shape
        if context is None:
            context = x

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
    """Standard scaled dot-product multi-head attention (self or cross).

    Structurally coherent with :class:`~pynop.core.blocks.WindowAttention`: the
    same fused :func:`torch.nn.functional.scaled_dot_product_attention` kernel
    (flash / memory-efficient backend where available), the same explicit
    ``1 / sqrt(d_head)`` scale, the same ``qkv_bias`` / ``attn_dropout`` /
    ``proj_dropout`` controls, and the same rotary embedding
    (:class:`~pynop.core.encoding.AdaptiveRoPE2D`) applied to q/k. The
    differences are:

    - **general** -- separate Q, K, V inputs, so it supports both self- and
      cross-attention (and optional causal masking);
    - **optional rotary embedding** -- ``rope=True`` builds it (default off).
      :class:`WindowAttention` always builds it. When disabled the rotation is
      a no-op and :class:`Attention` reduces to plain scaled dot-product
      attention.

    Args:
        in_ch: Input feature dimension for the Q, K, V projections.
        out_ch: Output feature dimension (must be divisible by ``num_heads``).
        num_heads: Number of attention heads.
        rope: If True, build an internal :class:`~pynop.core.encoding.AdaptiveRoPE2D`
            (``out_ch // num_heads`` must then be divisible by 4) and apply it
            to q/k whenever ``coords`` is passed at forward time.
        rope_kwargs: Extra keyword arguments forwarded to ``AdaptiveRoPE2D``
            (e.g. ``{"domain_size": 1.0, "base": 8.0}``).
        qkv_bias: If True, learn a bias in the Q, K, V projections.
        attn_dropout: Dropout probability on the attention weights (training only).
        proj_dropout: Dropout probability on the output projection.
        std_ini: Standard deviation of the truncated-normal init of the
            projections (a small value stabilizes deep stacks).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int,
        rope: bool = False,
        rope_kwargs: Optional[dict] = None,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        std_ini: float = 0.02,
    ):
        super().__init__()
        if out_ch % num_heads != 0:
            raise ValueError(f"out_ch ({out_ch}) must be divisible by num_heads ({num_heads}).")
        head_dim = out_ch // num_heads
        if rope and head_dim % 4 != 0:
            raise ValueError(
                f"Attention: rope=True requires out_ch // num_heads == {head_dim} to be "
                f"divisible by 4 for 2-D RoPE (out_ch={out_ch}, num_heads={num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.attn_dropout = attn_dropout

        # Separate projections so Q, K, V may come from different sources
        # (self- and cross-attention); ``WindowAttention`` fuses them because it
        # is self-attention only.
        self.wq = nn.Linear(in_ch, out_ch, bias=qkv_bias)
        self.wk = nn.Linear(in_ch, out_ch, bias=qkv_bias)
        self.wv = nn.Linear(in_ch, out_ch, bias=qkv_bias)
        self.out_proj = nn.Linear(out_ch, out_ch)
        self.proj_drop = nn.Dropout(proj_dropout)

        # Optional rotary positional embedding (always on in WindowAttention).
        self.rope = AdaptiveRoPE2D(head_dim, **(rope_kwargs or {})) if rope else None

        self._init_weights(std_ini)

    def _init_weights(self, std_ini: float):
        """Standardized normal initialization for the projections."""
        with torch.no_grad():
            for layer in (self.wq, self.wk, self.wv, self.out_proj):
                nn.init.trunc_normal_(layer.weight, std=std_ini)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, N, out_ch]`` to ``[B, num_heads, N, head_dim]``."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        resolution: Optional[tuple[int, int]] = None,
        causal_attn: bool = False,
    ) -> torch.Tensor:
        """Compute multi-head scaled dot-product attention.

        Args:
            Q: Query tensor of shape ``[B, N_q, in_ch]``.
            K: Key tensor of shape ``[B, N_k, in_ch]``.
            V: Value tensor of shape ``[B, N_k, in_ch]``.
            mask: Optional additive attention mask broadcastable to
                ``[N_q, N_k]`` (or ``[B, 1, N_q, N_k]``).
            coords: Absolute coordinates in ``[-1, 1]`` (``[B, N_q, 2]`` or
                ``[B, H, W, 2]``); only used when ``rope=True``.
            resolution: Optional ``(H_grid, W_grid)`` used by the RoPE to set
                its Nyquist-adaptive frequency band.
            causal_attn: If True, use causal (autoregressive) attention. This
                takes precedence over ``mask`` (the two are mutually exclusive
                in ``scaled_dot_product_attention``; a causal mask is a specific
                lower-triangular mask, so it wins when requested).

        Returns:
            Tensor of shape ``[B, N_q, out_ch]``.
        """
        # Project and split heads -> [B, n_h, N, head_dim]
        q = self.split_heads(self.wq(Q))
        k = self.split_heads(self.wk(K))
        v = self.split_heads(self.wv(V))

        # Optional rotary positional embedding (no-op when rope is disabled).
        if self.rope is not None and coords is not None:
            q = self.rope.rotate(q, coords, resolution=resolution)
            k = self.rope.rotate(k, coords, resolution=resolution)

        dropout_p = self.attn_dropout if self.training else 0.0
        if causal_attn:
            # Causal masking (specific lower-triangular mask); must not be
            # combined with an explicit attn_mask.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, scale=self.scale, is_causal=True
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.scale
            )

        # Combine heads -> [B, N_q, out_ch]
        out = out.transpose(1, 2).contiguous().view(Q.shape[0], Q.shape[1], -1)

        return self.proj_drop(self.out_proj(out))


class ComplexAttention(nn.Module):
    """Complex-valued multi-head attention with Hermitian inner product.

    Uses the standard complex inner product (Hermitian) to compute attention scores:
    score = Q @ K^H / sqrt(d), where ^H denotes conjugate transpose. Softmax is
    then applied to |score| magnitudes, and those real weights are multiplied by
    complex values V. This preserves phase information in the attended output while
    maintaining a valid convex optimization objective for training.

    Args:
        in_ch: Input feature dimension for Q, K, V projections (must be divisible by num_heads).
        out_ch: Output feature dimension (same as input for this implementation).
        num_heads: Number of attention heads.
    """

    def __init__(self, in_ch, out_ch, num_heads):
        super(ComplexAttention, self).__init__()
        assert out_ch % num_heads == 0
        self.d_model = out_ch
        self.num_heads = num_heads
        self.head_dim = out_ch // num_heads

        self.wq = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)
        self.wk = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)
        self.wv = nn.Linear(in_ch, out_ch, bias=True, dtype=torch.cfloat)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to [B, num_heads, N, head_dim]."""
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape back from [B, num_heads, N, head_dim] to [B, N, d_model]."""
        B, _, N, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute complex-valued attention with Hermitian inner product.

        Args:
            Q: Query tensor of shape [B, N_q, d_model] (complex).
            K: Key tensor of shape [B, N_k, d_model] (complex).
            V: Value tensor of shape [B, N_k, d_model] (complex).

        Returns:
            Output tensor of shape [B, N_q, d_model] (complex) with phase information preserved.
        """
        Q = self.split_heads(self.wq(Q))
        K = self.split_heads(self.wk(K))
        V = self.split_heads(self.wv(V))

        # Produit scalaire Hermitien pour le score: Q @ K^H
        attention_scores = torch.matmul(Q, K.transpose(-2, -1).conj())
        attention_scores = attention_scores / (self.head_dim**0.5)

        # Softmax applied to the magnitude (absolute value) of the scores
        real_attention_scores = torch.abs(attention_scores)
        attention_weights = F.softmax(real_attention_scores, dim=-1)

        # Apply the (magnitude-only) real weights to the complex values. Cast
        # weights to complex64 because PyTorch matmul on complex tensors requires
        # both operands to be complex (weights become purely real in the complex domain).
        weighted_output = torch.matmul(attention_weights.to(torch.cfloat), V)
        output = self.combine_heads(weighted_output)

        return output
