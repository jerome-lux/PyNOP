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


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        cond_dim=1,
        channel_first=True,
        eps=1e-5,
        hidden_dim=64,
        activation=nn.GELU,
    ):
        """
        LayerNorm for tensors with 2 spatial dimensions [B, C, H, W]
        Args:
            normalized_shape (int): dimension C à normaliser
            cond_dim (int): dimension du conditionneur
            eps (float): epsilon LayerNorm
            affine (bool): si False, désactive gamma/beta adaptatifs
        """
        super().__init__()
        self.channel_first = channel_first

        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, 2 * normalized_shape),
        )

    def forward(self, x, cond=None):
        """
        x: [B, ..., C]
        cond: [B, D] if cond is None, it returns classic LayerNorm
        """
        if not self.channel_first:
            x = x.moveaxis(1, -1)

        x_norm = self.norm(x)

        if cond is None:
            return x_norm

        gamma_beta = self.mlp(cond)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        while gamma.dim() < x_norm.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        x_norm = (1 + gamma) * x_norm + beta
        if not self.channel_first:
            x_norm = x_norm.moveaxis(-1, 1)

        return x_norm


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
