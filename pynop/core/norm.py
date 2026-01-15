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


class TimeConditionedLayerNorm(nn.Module):
    """Conditional LayerNorm, to be used with time conditioning for example.
    inspired from https://github.com/camlab-ethz/poseidon/blob/main/scOT/model.py#L135"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Linear(1, dim)
        self.bias = nn.Linear(1, dim)

    def forward(self, x, time):
        mean = x.mean(dim=1, keepdim=True)
        var = (x**2).mean(dim=1, keepdim=True) - mean**2
        x = (x - mean) / (var + self.eps).sqrt()
        weight = self.weight(time)
        bias = self.bias(time)
        dims_to_unsqueeze = x.dim() - weight.dim()
        for _ in range(dims_to_unsqueeze):
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        return weight * x + bias


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


class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        cond_dim,
        eps=1e-5,
        hidden_dim=None,
        activation=nn.GELU,
    ):
        """
        Args:
            normalized_shape (int): dimension C à normaliser
            cond_dim (int): dimension du conditionneur
            eps (float): epsilon LayerNorm
            affine (bool): si False, désactive gamma/beta adaptatifs
        """
        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=False)

        hidden_dim = hidden_dim or cond_dim
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
        x_norm = self.norm(x)

        if cond is None:
            return x_norm

        gamma_beta = self.mlp(cond)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        while gamma.dim() < x_norm.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return (1 + gamma) * x_norm + beta


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
