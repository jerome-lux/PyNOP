import math
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import torch.nn.init as init

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


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


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

    def __init__(self, in_channels, out_channels, modes, ranks, scaling: Union[int, float] = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling

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
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Les canaux d'entrée attendus sont {self.in_channels}, mais reçus {C_in}"

        # Les dimensions spectrales après rfft2(H, W) sont (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            raise ValueError(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H."
            )
        if self.modes_y > max_spectral_modes_y:
            raise ValueError(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )

        # 1. FFT -> domaine spectral
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")  # (B, C_in, H, W//2 + 1), cfloat

        # 2. Sélection des modes bas
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]  # (B, C_in, mx, my), cfloat

        # 3. Reconstruire les poids spectraux W_hat
        G1 = torch.einsum("ij,jkl->ikl", self.U, self.G)  # (C_in, r2, r3), cfloat
        G2 = torch.einsum("ok,ikl->iol", self.V, G1)  # (C_in, C_out, r3), cfloat
        # S(mx, my, r3) et G2(i, o, r3)
        W_hat = torch.einsum("iol,xyl->ioxy", G2, self.S)  # (C_in, C_out, modes_x, modes_y), cfloat

        # 4. Appliquer la convolution spectrale (multiplication de tenseurs complexes)
        # y_ft_low_modes[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * W_hat[i, o, mx, my])
        y_ft_low_modes = torch.einsum("bixy,ioxy->boxy", x_ft_low_modes, W_hat)  # (B, C_out, mx, my), cfloat

        # 5. Zero-padding
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes

        # 6. Retour au domaine spatial
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        return y


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int], scaling: Union[int, float] = 1):
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
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Expected input channels are {self.in_channels}, but received {C_in}"

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            raise ValueError(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H."
            )
        if self.modes_y > max_spectral_modes_y:
            raise ValueError(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )

        # 1. FFT -> spectral domain (x_ft is complex)
        # (B, C_in, H, W//2 + 1) after rfft2 of a (B, C_in, H, W) input
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        # 2. Select low modes (keep mx, my dimensions)
        # (B, C_in, modes_x, modes_y)
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]

        # 3. Apply spectral convolution (complex tensor multiplication)
        # This is element-wise multiplication per mode (mx, my)
        # followed by a summation over input channels (C_in) to get output channels (C_out).
        # y_ft_low_modes_out[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * self.spectral_weights[o, i, mx, my])
        # einsum('bixy, oixy -> boxy')
        # b: batch, i: in_channels, o: out_channels, x: modes_x, y: modes_y
        y_ft_low_modes_out = torch.einsum(
            "bixy, oixy -> boxy", x_ft_low_modes, self.spectral_weights
        )  # (B, C_out, modes_x, modes_y)

        # 4. Zero-padding of ignored frequencies
        # Create a tensor of zeros with the full spectral size after rfft2
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes_out

        # 5. Return to spatial domain
        # irfft2 to get a real output of size (H, W)
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        return y


class CartesianEmbedding(nn.Module):
    """
    Generates and concatenates normalized Cartesian coordinates (x, y) as additional channels.
    Coordinates are normalized to the range [-1, 1] using torch.meshgrid.
    """

    def __init__(self, minval=-1, maxval=1):
        super().__init__()
        # No learnable parameters needed for this embedding layer
        self.minval = minval
        self.maxval = maxval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates normalized (x, y) coordinates using meshgrid and concatenates them to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Input tensor concatenated with normalized x and y coordinates.
                          Shape (B, C + 2, H, W).
        """
        batch_size, _, height, width = x.shape
        device = x.device

        # Generate 1D coordinate vectors normalized to [-1, 1]
        x_lin = torch.linspace(self.minval, self.maxval, steps=width, device=device)  # Shape (W,)
        y_lin = torch.linspace(self.minval, self.maxval, steps=height, device=device)  # Shape (H,)

        # Use meshgrid to create 2D grids of coordinates
        # grid_y will have shape (H, W), grid_x will have shape (H, W)
        # Use indexing='ij' to match the (H, W) spatial dimensions ordering
        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        # Permute dimensions to get shape (2, H, W) and add batch dimension (1, 2, H, W)
        # Permute moves the coordinate dimension (originally last) to the first position
        # unsqueeze(0) adds the batch dimension at the start
        coords = coords.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 2, H, W)

        # Expand for the batch size to match the input tensor batch size
        coords = coords.expand(batch_size, -1, -1, -1)  # Shape (B, 2, H, W)

        # Concatenate with the input tensor along the channel dimension
        # Output shape (B, C_in + 2, H, W)
        output = torch.cat([x, coords], dim=1)

        return output


class SinusoidalEmbedding(nn.Module):
    """
    Generates and concatenates sinusoidal positional embeddings (x, y) as additional channels.
    Uses sine and cosine function pairs at multiple frequencies.
    Coordinates are first normalized to the range [0, 1].
    frequencies are multiple of 2 * math.pi * (2 ** d) where d=0 to num_frequency-1
    The embeddings are periodic in the image (useful for PDEs with periodic BCs!)
    **Note**: This is NOT the same as the positionnal embeddings found in vision transformer, where the
    frequencies are given by 1 / (10000 ** ((2 * i // 2) / num_freq)), where i goes from 0 to num_freq-1
    """

    def __init__(self, num_frequencies: int = 10):
        """
        Args:
            num_frequencies (int): The number of sinusoidal frequency pairs (sin/cos)
                                   per spatial dimension (x and y).
                                   Total added channels = num_frequencies * 2 (sin/cos) * 2 (x/y).
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # Define frequencies. Commonly powers of 2 multiplied by 2*pi.
        # E.g., frequencies = [2*pi*2^0, 2*pi*2^1, ..., 2*pi*2^(num_frequencies-1)]
        self.frequencies = 2 * math.pi * (2 ** torch.arange(num_frequencies))
        # Frequencies are not learnable parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal positional embeddings and concatenates them to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Input tensor concatenated with sinusoidal positional embeddings.
                          Shape (B, C + num_frequencies * 4, H, W).
        """
        batch_size, _, height, width = x.shape
        device = x.device

        # Move frequencies to the correct device
        frequencies = self.frequencies.to(device)  # Shape (num_frequencies,)

        # Generate x and y coordinates normalized to [0, 1]
        # Used to calculate the arguments for the sinusoidal functions
        x_coords_normalized = torch.linspace(0, 1, steps=width, device=device)  # Shape (W,)
        y_coords_normalized = torch.linspace(0, 1, steps=height, device=device)  # Shape (H,)

        # Create a full grid of normalized coordinates
        # grid_y (H, W), grid_x (H, W)
        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing="ij")

        # Stack coordinates to get shape (H, W, 2)
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        # Apply frequencies. Broadcast frequencies (1, 1, 1, num_frequencies)
        # over coordinates (H, W, 2, 1) -> (H, W, 2, num_frequencies)
        grid_frequencies = frequencies.view(1, 1, 1, self.num_frequencies)
        grid_coords_freq = grid_coords.unsqueeze(-1) * grid_frequencies  # Shape (H, W, 2, num_frequencies)

        # Apply sin and cos. Along the last dimension, we have [x_freq1, y_freq1, x_freq2, y_freq2, ...]
        # We want [sin(x_freq1), cos(x_freq1), sin(y_freq1), cos(y_freq1), ...]
        # We can concatenate sin and cos applied separately.
        sin_vals = torch.sin(grid_coords_freq)  # (H, W, 2, num_frequencies)
        cos_vals = torch.cos(grid_coords_freq)  # (H, W, 2, num_frequencies)

        # Concatenate sin and cos for each coordinate and each frequency
        # Resulting shape (H, W, 2, num_frequencies * 2)
        grid_embeddings = torch.cat([sin_vals, cos_vals], dim=-1)

        grid_embeddings = grid_embeddings.permute(2, 3, 0, 1).reshape(
            1, -1, height, width
        )  # Shape (1, num_frequencies * 4, H, W)

        grid_embeddings = grid_embeddings.expand(batch_size, -1, -1, -1)  # Shape (B, num_frequencies * 4, H, W)

        # Concatenate with the input tensor
        # The output will have shape (B, C_in + num_frequencies * 4, H, W)
        output = torch.cat([x, grid_embeddings], dim=1)

        return output


def sin_positional_encoding_2d(d_model, H, W, basis=10000.0, device="cpu"):
    if d_model % 4 != 0:
        raise ValueError("d_model doit être divisible par 4")
    d_half = d_model // 2

    y_coords = torch.arange(H, device=device).float().unsqueeze(1)
    x_coords = torch.arange(W, device=device).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_half, 2, device=device).float() * -(math.log(basis) / d_half))

    pe_y = torch.zeros(d_half, H, W, device=device)
    pe_y[0::2, :, :] = torch.sin(y_coords * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    pe_y[1::2, :, :] = torch.cos(y_coords * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)

    pe_x = torch.zeros(d_half, H, W, device=device)
    pe_x[0::2, :, :] = torch.sin(x_coords * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    pe_x[1::2, :, :] = torch.cos(x_coords * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)

    pe = torch.cat([pe_y, pe_x], dim=0)  # [d_model, H, W]
    pe = pe.view(d_model, -1).transpose(0, 1)  # [H*W, d_model]
    return pe.unsqueeze(0)  # [1, H*W, d_model]


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


class FiniteDifferenceConvolution(nn.Module):
    """Finite Difference Convolution Layer introduced in [1]_.
    "Neural Operators with Localized Integral and Differential Kernels" (ICML 2024)
        https://arxiv.org/abs/2402.16845

    Computes a finite difference convolution on a regular grid,
    which converges to a directional derivative as the grid is refined.

    Parameters
    ----------
    in_channels : int
        number of in_channels
    out_channels : int
        number of out_channels
    n_dim : int
        number of dimensions in the input domain
    kernel_size : int
        odd kernel size used for convolutional finite difference stencil
    groups : int
        splitting number of channels
    padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
        mode of padding to use on input.
        See `torch.nn.functional.padding`.

    References
    ----------
    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_dim=2,
        kernel_size=3,
        groups=1,
        stride=1,
        padding="same",
    ):

        super().__init__()

        self.conv_function = getattr(F, f"conv{n_dim}d")

        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.groups = groups
        self.n_dim = n_dim
        self.padding = padding
        self.stride = stride

        # init kernel weigths
        self.weights = torch.rand((out_channels, in_channels // groups, kernel_size, kernel_size))
        k = torch.sqrt(torch.tensor(groups / (in_channels * kernel_size**2)))
        self.weights = self.weights * 2 * k - k

    def forward(self, x, grid_width: float = 1.0) -> torch.Tensor:
        """FiniteDifferenceConvolution's forward pass.

        Parameters
        ----------
        x : torch.tensor
            input tensor, shape (batch, in_channels, d_1, d_2, ...d_n)
        grid_width : float
            discretization size of input grid
        """

        self.weights = self.weights.to(x.device)
        x = (
            self.conv_function(
                x,
                (self.weights - torch.mean(self.weights)),
                groups=self.groups,
                stride=self.stride,
                padding=self.padding,
            )
            / grid_width
        )
        return x


class FiniteDifferenceLayer(nn.Module):
    """Finite Difference Layer introduced in [1]_.
    "Neural Operators with Localized Integral and Differential Kernels" (ICML 2024)
        https://arxiv.org/abs/2402.16845

    Computes a finite difference convolution on a regular grid,
    which converges to a directional derivative as the grid is refined.

    Parameters
    ----------
    in_channels : int
        number of in_channels
    out_channels : int
        number of out_channels
    n_dim : int
        number of dimensions in the input domain
    kernel_size : int
        odd kernel size used for convolutional finite difference stencil
    groups : int
        splitting number of channels
    padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
        mode of padding to use on input.
        See `torch.nn.functional.padding`.

    References
    ----------
    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_dim=2,
        kernel_size=3,
        groups=1,
        stride=1,
        padding="same",
        norm=None,
        activation=None,
        grid_width: float = 1.0,
    ):
        super().__init__()
        self.fdc = FiniteDifferenceConvolution(
            in_channels,
            out_channels,
            n_dim=n_dim,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=padding,
        )
        self.normalization = norm(out_channels) if norm is not None else None
        self.activation = activation() if activation is not None else None
        self.grid_width = grid_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fdc(x, self.grid_width)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
