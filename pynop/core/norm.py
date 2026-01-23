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
    def __init__(self, normalized_shape, cond_dim=1):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = 1e-5

        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

        num_features = 1
        for s in self.normalized_shape:
            num_features *= s

        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, num_features * 2))
        nn.init.zeros_(self.cond_mlp[0].weight)
        nn.init.zeros_(self.cond_mlp[0].bias)

    def forward(self, x, cond=None):

        x_norm = F.layer_norm(x, self.normalized_shape, eps=self.eps)

        if cond is None:
            return self.weight * x_norm + self.bias
        else:

            ada_params = self.cond_mlp(cond)
            gamma, beta = torch.chunk(ada_params, 2, dim=-1)

            for _ in range(x.dim() - gamma.dim()):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

            return (1 + gamma) * x_norm + beta


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
