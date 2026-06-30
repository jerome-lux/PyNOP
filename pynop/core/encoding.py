import math
import torch
import torch.nn as nn
from typing import Optional


class AdaptiveRoPE2D(nn.Module):
    """
    2D Continuous RoPE with independent coordinates for queries and keys.
    Use rotate() independently on q and k when needed.

    Args:
        d_head (int): Dimension of the head (must be divisible by 4).
        domain_size (float): Total length of the domain (e.g., 2.0 for [-1, 1]).
    """

    def __init__(self, d_head: int, domain_size: float = 2.0):
        super().__init__()
        assert d_head % 4 == 0, "d_head must be divisible by 4 for 2D RoPE"
        self.d_head = d_head
        self.domain_size = domain_size

        exponent = torch.arange(0, d_head // 4, dtype=torch.float32) / (d_head // 4 - 1)
        self.register_buffer("exponent", exponent, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_to_half(self, t: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        sin2 = sin.repeat_interleave(2, dim=-1)
        cos2 = cos.repeat_interleave(2, dim=-1)
        return t * cos2 + self._rotate_half(t) * sin2

    def _compute_sincos(
        self,
        coords: torch.Tensor,  # [B, N, 2]
        h_spatial: int,
        w_spatial: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device, dtype = coords.device, coords.dtype
        exponent = self.exponent.to(device=device, dtype=dtype)

        w_min = (2 * math.pi) / self.domain_size
        freqs_y = w_min * float(h_spatial) ** (-exponent)  # [d_head//4]
        freqs_x = w_min * float(w_spatial) ** (-exponent)

        phases_y = coords[..., 0:1] * freqs_y  # [B, N, d_head//4]
        phases_x = coords[..., 1:2] * freqs_x

        sin_y = phases_y.sin().unsqueeze(1)  # [B, 1, N, d_head//4]
        cos_y = phases_y.cos().unsqueeze(1)
        sin_x = phases_x.sin().unsqueeze(1)
        cos_x = phases_x.cos().unsqueeze(1)

        return sin_y, cos_y, sin_x, cos_x

    def rotate(
        self,
        t: torch.Tensor,  # [B, H_head, N, d_head]
        coords: torch.Tensor,  # [B, H, W, 2]  ou [B, N, 2]
    ) -> torch.Tensor:
        """Applique RoPE 2D sur un unique tenseur (q ou k)."""
        if coords.dim() == 4:
            h_spatial, w_spatial = coords.shape[1:3]
            coords = coords.flatten(1, 2)
        else:
            # Format [B, N, 2] : on ne peut inférer h/w que si carré
            # Préférer le format [B, H, W, 2] pour grilles non carrées
            n = coords.shape[1]
            h_spatial = w_spatial = int(n**0.5)

        sin_y, cos_y, sin_x, cos_x = self._compute_sincos(coords, h_spatial, w_spatial)

        d = t.shape[-1]
        t_y, t_x = t[..., : d // 2], t[..., d // 2 :]
        return torch.cat(
            [
                self._apply_to_half(t_y, sin_y, cos_y),
                self._apply_to_half(t_x, sin_x, cos_x),
            ],
            dim=-1,
        )


class ContinuousRoPE2D(nn.Module):
    """
    RoPE for continuous 2D coordinates scaled in [-1, 1].
    The d_head parameter must be divisible by 4.

    Args:
        d_head (int): Dimension of the head.
        domain_size (float): Total length of the domain (2.0 for [-1, 1]).
        base (float): base frequency (should be approx equal the number of grid cells)
    """

    def __init__(self, d_head: int, domain_size: float = 2.0, base: float = 16.0):
        super().__init__()
        assert d_head % 4 == 0, "d_head must be divisible by 4 for 2D RoPE"

        # Base theta calculated so the largest wavelength matches the domain size
        # Lambda_max = domain_size -> w_min = 2 * pi / domain_size
        w_min = (2 * math.pi) / domain_size

        # Standard RoPE geometric decay for frequencies
        # High frequencies are bounded by d_head to distribute features smoothly
        exponent = torch.arange(0, d_head // 4, dtype=torch.float32) / (d_head // 4 - 1)

        freqs = w_min * (1.0 / (base**exponent))

        self.register_buffer("freqs", freqs)  # [d_head // 4]

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Swaps and negates halves of the last dimension.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_to_half(self, t: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D RoPE rotation to a half-feature tensor.
        """
        sin2 = sin.repeat_interleave(2, dim=-1)
        cos2 = cos.repeat_interleave(2, dim=-1)
        return t * cos2 + self._rotate_half(t) * sin2

    def forward(self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 2D continuous RoPE to queries and keys.

        Args:
            q (torch.Tensor): Queries of shape [B_win, H, N, d_head].
            k (torch.Tensor): Keys of shape [B_win, H, N, d_head].
            coords (torch.Tensor): Coordinates of shape [B_win, N, 2] in [-1, 1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        d = q.shape[-1]

        y = coords[..., 0:1]  # [B_win, N, 1]
        x = coords[..., 1:2]  # [B_win, N, 1]

        # Phase computation: [B_win, N, d_head // 4]
        freqs_y = y * self.freqs
        freqs_x = x * self.freqs

        sin_y, cos_y = freqs_y.sin(), freqs_y.cos()
        sin_x, cos_x = freqs_x.sin(), freqs_x.cos()

        def unsq(t):
            return t.unsqueeze(1)  # [B_win, 1, N, d_head // 4]

        # Split features: [y_half | x_half]
        q_y, q_x = q[..., : d // 2], q[..., d // 2 :]
        k_y, k_x = k[..., : d // 2], k[..., d // 2 :]

        q_rot = torch.cat(
            [
                self._apply_to_half(q_y, unsq(sin_y), unsq(cos_y)),
                self._apply_to_half(q_x, unsq(sin_x), unsq(cos_x)),
            ],
            dim=-1,
        )

        k_rot = torch.cat(
            [
                self._apply_to_half(k_y, unsq(sin_y), unsq(cos_y)),
                self._apply_to_half(k_x, unsq(sin_x), unsq(cos_x)),
            ],
            dim=-1,
        )

        return q_rot, k_rot


class RoPE(nn.Module):
    """1D RoPE"""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequencies
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, coords):
        # x:      [B, n_h, L, head_dim]
        # coords: [B, L, 1]

        angles = coords.unsqueeze(1) * self.inv_freq.view(1, 1, 1, -1)
        # [B, L, 1] → unsqueeze(1) → [B, 1, L, 1]
        # inv_freq: [1, 1, 1, dim//2]
        # angles:   [B, 1, L, dim//2] ✓ broadcast sur n_h

        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # [B, 1, L, head_dim]
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)

        def rotate_half(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape(x.shape)

        return (x * cos) + (rotate_half(x) * sin)


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

        x_lin = torch.linspace(self.minval, self.maxval, steps=width, device=device)  # Shape (W,)
        y_lin = torch.linspace(self.minval, self.maxval, steps=height, device=device)  # Shape (H,)

        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        coords = coords.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 2, H, W)

        # Expand for the batch size to match the input tensor batch size
        coords = coords.expand(batch_size, -1, -1, -1)  # Shape (B, 2, H, W)

        # Output shape (B, C_in + 2, H, W)
        output = torch.cat([x, coords], dim=1)

        return output


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :, :1])], dim=-1)
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):

        super().__init__()
        self.dim = dim
        self.max_period = max_period

        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, timesteps):
        """
        :param timesteps: Tensor 1-D de taille [N]
        :return: Tensor de taille [N, dim]
        """
        # timesteps[:, None] transforme [N] en [N, 1]
        args = timesteps[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2:
            zero_pad = torch.zeros(timesteps.shape[0], 1, device=timesteps.device)
            embedding = torch.cat([embedding, zero_pad], dim=-1)

        return embedding


class GaussianFourierEmbedding(nn.Module):
    """Gaussian Fourier Features for multidimensional continuous coordinates."""

    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        """
        Args:
            in_features: Number of input dimensions (e.g., 2 for 2D, 3 for 3D).
            out_features: Total embedding dimension (must be divisible by 2).
            scale: Standard deviation (sigma) of the Gaussian distribution.
        """
        super().__init__()
        assert out_features % 2 == 0, f"out_features ({out_features}) must be divisible by 2"

        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = out_features // 2

        # Random Gaussian projection matrix: [in_features, num_frequencies]
        # Inspired by Random Fourier Features (Rahimi & Recht, 2007)
        b_matrix = torch.randn(in_features, self.num_frequencies) * scale
        self.register_buffer("b_matrix", b_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps coordinates to random Fourier features.

        Args:
            x: Continuous coordinates, shape [..., in_features]

        Returns:
            Mapped features, shape [..., out_features]
        """
        # Project coordinates onto random directions: [..., num_frequencies]
        x_proj = torch.matmul(x, self.b_matrix)

        # Concatenate sine and cosine components
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierEmbedding(nn.Module):
    """Deterministic axial Fourier features for multidimensional continuous coordinates."""

    def __init__(self, in_features: int, out_features: int, max_freq: float = 256.0):
        """
        Args:
            in_features: Number of input dimensions (e.g., 1 for 1D, 2 for 2D, 3 for 3D).
            out_features: Total embedding dimension (must be divisible by 2 * in_features).
            max_freq: Absolute maximum spatial frequency to capture.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        denom = 2 * in_features
        assert (
            out_features % denom == 0
        ), f"out_features ({out_features}) must be divisible by 2 * in_features ({denom})"
        self.num_freqs = out_features // denom

        # Generate deterministic log-linear frequencies: [num_freqs]
        frequencies = torch.exp(
            torch.arange(self.num_freqs).float() * (math.log(max_freq) / max(1, self.num_freqs - 1))
        )

        # Build a block-diagonal projection matrix: [in_features, in_features * num_freqs]
        # Maps each input dimension to its respective independent frequency band
        b_matrix = torch.zeros(in_features, in_features * self.num_freqs)
        for i in range(in_features):
            b_matrix[i, i * self.num_freqs : (i + 1) * self.num_freqs] = frequencies

        self.register_buffer("b_matrix", b_matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps coordinates to deterministic axial Fourier features.

        Args:
            x: Continuous coordinates, shape [..., in_features]

        Returns:
            Mapped features, shape [..., out_features]
        """
        # Project coordinates onto independent axial frequencies: [..., in_features * num_freqs]
        x_proj = torch.matmul(x, self.b_matrix) * math.pi

        # Compute sine and cosine tracking components
        sin_emb = torch.sin(x_proj)
        cos_emb = torch.cos(x_proj)

        # Interleave sin and cos components: [..., 2 * in_features * num_freqs]
        emb = torch.stack([sin_emb, cos_emb], dim=-1).flatten(start_dim=-2)
        return emb


class IntegratedPositionalEncoding(nn.Module):
    """Integrated Positional Encoding with deterministic log-linear frequencies."""

    def __init__(self, in_features: int, out_features: int, max_freq: float = 256.0):
        """
        Args:
            in_features: Number of input dimensions (e.g., 2 for 2D, 3 for 3D).
            out_features: Total embedding dimension (must be divisible by 2 *
              in_features).
            max_freq: Absolute maximum spatial frequency to capture.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        denom = 2 * in_features
        assert out_features % denom == 0, f"out_features ({out_features}) must be divisible by {denom}"
        self.num_freqs = out_features // denom

        # Generate deterministic log-linear frequencies: [num_freqs]
        # In papers like NeRF, this matches the standard multi-scale power-of-two spacing
        frequencies = torch.exp(
            torch.arange(self.num_freqs).float() * (math.log(max_freq) / max(1, self.num_freqs - 1))
        )

        # Construct projection matrix to apply frequencies per axis: [in_features, in_features * num_freqs]
        # This eliminates the need for Python loops during the forward pass
        b_matrix = torch.zeros(in_features, in_features * self.num_freqs)
        for i in range(in_features):
            b_matrix[i, i * self.num_freqs : (i + 1) * self.num_freqs] = frequencies

        self.register_buffer("b_matrix", b_matrix)

    def forward(self, x: torch.Tensor, cell_size: torch.Tensor) -> torch.Tensor:
        """Computes anti-aliased deterministic positional embeddings.

        Args:
            x: [..., in_features] - Continuous coordinate centers.
            cell_size: [..., in_features] - Local cell width (dx, dy, dz).

        Returns:
            [..., out_features] - Anti-aliased deterministic embedding.
        """
        # 1. Project coordinates and cell sizes into the frequency space
        # Output shape: [..., in_features * num_freqs]
        scaled_x = torch.matmul(x, self.b_matrix) * math.pi

        # 2. Compute analytical variance integration
        # Box variance multiplier (1/12) derived from Mip-NeRF (Barron et al., 2021)
        # Output shape: [..., in_features * num_freqs]
        scaled_var = torch.matmul(cell_size**2, self.b_matrix**2) * (math.pi**2) * (1.0 / 12.0)

        # 3. Compute analytical Gaussian attenuation
        attenuation = torch.exp(-0.5 * scaled_var)

        # 4. Compute modulated waveforms
        sin_embedding = attenuation * torch.sin(scaled_x)
        cos_embedding = attenuation * torch.cos(scaled_x)

        # 5. Interleave or concatenate sin and cos
        # Shape: [..., 2 * in_features * num_freqs] which equals [..., out_features]
        return torch.cat([sin_embedding, cos_embedding], dim=-1)


class GaussianIntegratedPositionalEncoding(nn.Module):
    """Integrated Positional Encoding (IPE) with random Gaussian features and cell variance correction."""

    def __init__(self, in_features: int, out_features: int, scale: float = 10.0):
        """
        Args:
            in_features: Number of input dimensions (e.g., 2 for 2D, 3 for 3D).
            out_features: Output embedding dimension (must be divisible by 2 * in_features).
            scale: Standard deviation (sigma) of the Gaussian distribution.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        denom = 2 * in_features
        assert out_features % denom == 0, f"out_features must be divisible by {denom}"
        num_frequencies = out_features // 2

        # Random Gaussian projection matrix: [in_features, num_frequencies]
        # Each coordinate will be projected onto 'num_frequencies' random directions
        b_matrix = torch.randn(in_features, num_frequencies) * scale
        self.register_buffer("b_matrix", b_matrix)

    def forward(self, x: torch.Tensor, cell_size: torch.Tensor) -> torch.Tensor:
        """Computes anti-aliased random Gaussian positional embeddings.

        Args:
            x: [B, N, in_features] - Normalized coordinate centers.
            cell_size: [B, N, in_features] - Local cell width (dx, dy).
        Returns:
            [B, N, out_features] - Anti-aliased Gaussian embedding.
        """
        # 1. Project coordinates and cell sizes into the random frequency space
        # Output shapes: [B, N, num_frequencies]
        scaled_x = torch.matmul(x, self.b_matrix)

        # Linear transformation of variance: var_pro_axis = (cell_size^2) * (b_matrix^2) / 12
        # Standard deviation along the random directions:
        scaled_var = torch.matmul(cell_size**2, self.b_matrix**2) * (1.0 / 12.0)
        # scaled_var = torch.sqrt(scaled_var)

        # 2. Compute analytical IPE Gaussian attenuation
        attenuation = torch.exp(-0.5 * (scaled_var))

        # 3. Compute modulated waveforms
        sin_embedding = attenuation * torch.sin(scaled_x)
        cos_embedding = attenuation * torch.cos(scaled_x)

        # 4. Concatenate sin and cos: [B, N, 2 * num_frequencies]
        return torch.cat([sin_embedding, cos_embedding], dim=-1)


class AdaptiveFourierEmbedding(nn.Module):

    def __init__(self, in_features: int, out_features: int, base_max_freq: float = 256.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_max_freq = base_max_freq
        denom = 2 * in_features
        assert out_features % denom == 0
        self.num_freqs = out_features // denom

    def _build_b_matrix(self, max_freq: float, device, dtype) -> torch.Tensor:
        frequencies = torch.exp(
            torch.arange(self.num_freqs, device=device, dtype=dtype) * (math.log(max_freq) / max(1, self.num_freqs - 1))
        )
        b = torch.zeros(self.in_features, self.in_features * self.num_freqs, device=device, dtype=dtype)
        for i in range(self.in_features):
            b[i, i * self.num_freqs : (i + 1) * self.num_freqs] = frequencies
        return b

    def forward(self, x: torch.Tensor, resolution: tuple[int, ...] | None = None) -> torch.Tensor:
        """
        Args:
            x: Continuous coordinates, shape [..., in_features].
                The leading dimensions are arbitrary and preserved in the output.
                Common cases:
                  - [B, H*W, in_features]  (flattened spatial tokens)
                  - [B, H, W, in_features] (2D spatial grid, channels-last)
                  - [H*W, in_features]     (no batch dim)
            resolution: (H, W, ...) of the current input grid, one entry per axis.
                Used to set per-axis Nyquist frequency: nyquist_i = resolution[i] / 2.
                If None, falls back to base_max_freq for all axes.

        Returns:
            Embedding, shape [..., out_features].
            The leading dimensions mirror those of x exactly:
              - [B, H*W, out_features]  if x is [B, H*W, in_features]
              - [B, H, W, out_features] if x is [B, H, W, in_features]
        """

        device, dtype = x.device, x.dtype

        if resolution is not None:
            assert len(resolution) == self.in_features
            nyquists = [r / 2.0 for r in resolution]
        else:
            nyquists = [self.base_max_freq] * self.in_features

        # Une bande de frequences independante par axe
        b = torch.zeros(self.in_features, self.in_features * self.num_freqs, device=device, dtype=dtype)
        for i, max_f in enumerate(nyquists):
            freqs = torch.exp(
                torch.arange(self.num_freqs, device=device, dtype=dtype)
                * (math.log(max_f) / max(1, self.num_freqs - 1))
            )
            b[i, i * self.num_freqs:(i + 1) * self.num_freqs] = freqs

        x_proj = torch.matmul(x, b) * math.pi
        return torch.stack([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).flatten(start_dim=-2)


class LocalPositionalEncoding(nn.Module):
    """Generates resolution-invariant 2D Fourier features for local window grids."""

    def __init__(self, d_model: int, scale: float = 10.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2"
        self.mapping_channels = d_model // 2

        # Fixed Random Fourier Features projection matrix
        self.register_buffer("b_matrix", torch.randn(2, self.mapping_channels) * scale)

    def forward(self, win_h: int, win_w: int, device: torch.device) -> torch.Tensor:
        # Coordonnées normalisées dans [-0.5, 0.5], invariantes à la résolution
        # align_corners=False : pas constant = 1/N quelle que soit la taille
        grid_y = (torch.arange(win_h, device=device, dtype=torch.float32) + 0.5) / win_h - 0.5
        grid_x = (torch.arange(win_w, device=device, dtype=torch.float32) + 0.5) / win_w - 0.5

        coords_y, coords_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        coords = torch.stack([coords_y.flatten(), coords_x.flatten()], dim=-1)

        proj = 2 * torch.pi * coords @ self.b_matrix
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
