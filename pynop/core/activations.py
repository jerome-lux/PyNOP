from typing import Optional, Union
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

class TaylorSoftmax(nn.Module):
    def __init__(self, dim=-1, order=2):
        super().__init__()
        self.dim = dim
        self.order = order

    def forward(self, x):
        # x: [Batch, N, D]
        # Taylor approximation of exp(x) = 1 + x + x^2/2! + ...

        # 1. Ensure x is centered or positive if needed,
        # but Taylor Softmax is often used directly on logits.
        # We add 1 for the first term (1 + x + x^2/2)

        if self.order == 2:
            poly = 1.0 + x + (x**2) / 2.0
        elif self.order == 3:
            poly = 1.0 + x + (x**2) / 2.0 + (x**3) / 6.0
        else:
            raise NotImplementedError("Order not implemented")

        # 2. Prevent negative values due to poly approximation
        # Even with order 2, it's safer to use ReLU or clamp
        poly = torch.clamp(poly, min=1e-6)

        # 3. Normalize
        return poly / poly.sum(dim=self.dim, keepdim=True)

class Sine(nn.Module):
    """Sinusoidal activation with scaling"""

    def __init__(self, w0: float = 1) -> None:
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w0 * x)


def gumbel_softmax(logits: torch.Tensor, tau: Union[torch.Tensor, float] = 1.0, dim=-1, hard=False):

    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau

    y = F.softmax(y, dim=dim)

    if hard:
        _, y_hard = y.max(dim=dim)
        y_one_hot = torch.zeros_like(y).scatter_(dim, y_hard.unsqueeze(dim), 1.0)
        y = (y_one_hot - y).detach() + y  # detach because argmax is non differentiable
    return y
