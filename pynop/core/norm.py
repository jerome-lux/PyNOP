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
    def __init__(self, feature_dim, condition_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Standard learnable gain for the fallback case
        self.weight = nn.Parameter(torch.ones(feature_dim))

        # Adaptive gain projection
        self.to_gamma = nn.Linear(condition_dim, feature_dim)

        # Zero-init ensures we start with identity mapping
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)

    def forward(self, x, condition=None):
        # Calculate RMS
        # x: (batch, seq_len, feature_dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * rms

        if condition is not None:
            # Generate adaptive gain: (batch, feature_dim)
            gamma = self.to_gamma(condition)
            for i in range(x.ndim - 2):
                gamma = gamma.unsqueeze(1)
            # Add 1 to ensure stability and broadcast to (batch, 1, feature_dim)
            return x_normed * (1 + gamma)
        else:
            # Fallback to standard RMSNorm with learned weight
            return x_normed * self.weight


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, channels, cond_dim=1):
        super().__init__()
        self.channels = channels
        self.eps = 1e-5

        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, channels * 2))

        nn.init.zeros_(self.cond_mlp[0].weight)
        nn.init.zeros_(self.cond_mlp[0].bias)

    def forward(self, x, cond):
        """
        x: [B, ..., C] (peut �tre [B, N, C] ou [B, H, W, C])
        cond: [B, cond_dim] (temps ou signal global)
        """

        x_norm = F.layer_norm(x, (self.channels,), weight=None, bias=None, eps=self.eps)
        if cond is None:

            return self.weight * x_norm + self.bias

        # [B, C*2] -> [B, 2, C]
        ada_params = self.cond_mlp(cond).view(-1, 2, self.channels)
        gamma = ada_params[:, 0, :]  # [B, C]
        beta = ada_params[:, 1, :]  # [B, C]

        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (self.weight + gamma) * x_norm + (self.bias + beta)


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
