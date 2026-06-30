from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F

""" Note sur LayerNorm vs InstanceNorm:
- LayerNorm: gamma aet beta sont de dimension normalized_shape
- InstanceNorm: gamma etd beta sont de dimesnions C (1 paramètre par canal)
"""


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm telle que definie dans ConvNext: moyenne et std calculé le long des canaux"""

    # normalized_shape MUST be defined when intanciating the Layer
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm"""

    def __init__(self, feature_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(feature_dim))
        self.to_gamma = nn.Linear(feature_dim, feature_dim)

        # Initialize to zero so modulation starts as an identity mapping
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        # Compute RMS normalization
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * rms

        if condition is not None:
            gamma = self.to_gamma(condition)
            for _ in range(x.ndim - 2):
                gamma = gamma.unsqueeze(1)
            # Apply base weight modulated by adaptive scaling
            return x_normed * self.weight * (1 + gamma)

        return x_normed * self.weight


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm using chunk splitting to prevent batch resizing bugs."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        # Output 2 * channels for scale (gamma) and shift (beta)
        self.cond_mlp = nn.Linear(channels, channels * 2)

        nn.init.zeros_(self.cond_mlp.weight)
        nn.init.zeros_(self.cond_mlp.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        x_norm = F.layer_norm(x, (self.channels,), weight=None, bias=None, eps=self.eps)

        if cond is None:
            return self.weight * x_norm + self.bias

        gamma, beta = self.cond_mlp(cond).chunk(2, dim=-1)  # Both are [B, channels]

        # Dynamic broadcasting matching tensor dimensions
        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        # Standard scale and shift conditioning setup
        return self.weight * (1 + gamma) * x_norm + (self.bias + beta)


class ComplexLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, z):
        # z: (B, tokens, d_model) complex
        power = z.real**2 + z.imag**2
        rms = torch.sqrt(power.mean(dim=self.dim, keepdim=True) + self.eps)
        return z / rms


class RMSNorm2d(nn.RMSNorm):
    # Normalize each channel separately (normalized_shape MUST be defined when intanciating the Layer)
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


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


NORM_DICT = {"ln": LayerNorm2d, "gn": nn.GroupNorm, "bn": nn.BatchNorm2d, "rmsn": RMSNorm2d}


def build_norm(norm: str, num_features: int = None, **kwargs):

    if norm is None:
        return None
    if norm.lower() in NORM_DICT:
        if norm == "gn":
            try:
                groups = kwargs.pop("groups")
            except KeyError:
                groups = 32
            return nn.GroupNorm(num_groups=groups, num_channels=num_features, **kwargs)
        elif norm == "bn":
            return nn.BatchNorm2d(num_features=num_features, **kwargs)
        elif norm in ["ln", "rmsn"]:
            return NORM_DICT[norm](normalized_shape=num_features, **kwargs)
    else:
        return None
