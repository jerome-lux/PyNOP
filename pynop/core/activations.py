from typing import Optional
from functools import partial
from torch import nn
import torch
import torch.nn.functional as F


ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "leakyrelu": nn.LeakyReLU,
}


def build_activation(name: str, **kwargs) -> Optional[nn.Module]:

    if name in ACT_DICT:
        act_cls = ACT_DICT[name]
        return act_cls(**kwargs)
    else:
        return None


class ModReLU(nn.Module):
    """A ReLU for complex tensors"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, z):
        mag = torch.abs(z)
        scale = F.relu(mag + self.bias) / (mag + self.eps)
        return z * scale
