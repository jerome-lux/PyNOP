from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):
    # Normalize each channel separately (normalized_shape MUST be defined when intanciating the Layer)
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


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
