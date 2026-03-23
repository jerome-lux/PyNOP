# coding=utf-8
from collections.abc import Iterable
from typing import Optional, Any, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_stats(x, dim=-1, text=""):
    with torch.no_grad():
        var = x.var(dim=dim).mean()
        mean = x.mean(dim=dim).mean()
        print(f"{text} mu={mean.item():.2e}, var={var.item():.2e}")
class ChebyshevBasis(nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1  # Degree max in x
        self.m2 = m2  # Degree max in y

    def forward(self, coords):
        """
        coords: [B, H, W, 2] in range [-1, 1]
        returns: [B, H, W, m1 * m2]
        """
        B, H, W, _ = coords.shape
        x = coords[..., 0]  # [B, H, W]
        y = coords[..., 1]  # [B, H, W]

        def get_polys(val, degree):
            polys = [torch.ones_like(val), val]
            for n in range(2, degree):
                polys.append(2 * val * polys[-1] - polys[-2])
            return torch.stack(polys, dim=-1)  # [B, H, W, degree]

        poly_x = get_polys(x, self.m1)
        poly_y = get_polys(y, self.m2)

        # Outer product of all x and y degrees to get 2D basis
        # [B, H, W, m1, 1] * [B, H, W, 1, m2] -> [B, H, W, m1, m2]
        basis = poly_x.unsqueeze(-1) * poly_y.unsqueeze(-2)
        return basis.reshape(B, H, W, self.m1 * self.m2)


def Newton_Schulz(basis, iterations=5):
    """
    Orthogonalize learned bases using Newton-Schulz iteration.

    Args:
        basis: Tensor of shape [B, H, W, M1, M2]
        iterations: Number of orthogonalization steps

    Returns:
        Orthogonalized basis of the same shape [B, H, W, M1, M2]
    """
    B, H, W, M1, M2 = basis.shape
    M = M1 * M2
    N = H * W

    # 1. Reshape to [B, N, M] -> M columns (modes) of size N (space)
    # We want columns to be orthonormal: B^T * B = I
    V = basis.view(B, N, M)

    # 2. Initial scaling to ensure spectral norm < sqrt(3) for convergence
    # Using a safe spectral norm estimate
    V_norm = torch.linalg.norm(V, ord=2, dim=(1, 2), keepdim=True)
    V = V / (V_norm + 1e-6)

    # 3. Newton-Schulz Iteration
    # Formula: V_{n+1} = V_n * (1.5 * I - 0.5 * V_n^T * V_n)
    I = torch.eye(M, device=basis.device).expand(B, M, M)

    for _ in range(iterations):
        # G = V^T * V (Gram matrix [B, M, M])
        G = torch.bmm(V.transpose(1, 2), V)
        # Update V
        V = torch.bmm(V, 1.5 * I - 0.5 * G)

    # 4. Reshape back to original dimensions
    return V.view(B, H, W, M1, M2)


def gs_orthogonalization(X, n_iter=5, eps=1e-6):
    """
    X: [B, HW, M]
    """
    B, H, W, m1, m2 = X.shape
    M = m1 * m2
    HW = H * W
    X = X.reshape(B, HW, M)

    X = F.normalize(X, dim=1)

    # Gram
    if X.is_complex():
        G = torch.matmul(X.conj().transpose(-2, -1), X) / HW  # [B, M, M]
    else:
        G = torch.matmul(X.transpose(-2, -1), X) / HW  # [B, M, M]

    norm_factor = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    # norm_factor = torch.linalg.norm(G, ord="fro", dim=(-2, -1), keepdim=True)
    # norm_factor = torch.linalg.matrix_norm(G, ord=2).view(B, 1, 1)
    G = G / (norm_factor + eps)

    Identity = torch.eye(M, device=X.device, dtype=X.dtype).unsqueeze(0)

    Y = G
    Z = Identity

    for _ in range(n_iter):
        T = 0.5 * (3 * Identity - Z @ Y)
        Y = Y @ T
        Z = T @ Z

    G = Z / torch.sqrt(norm_factor + eps)
    X_out = X @ G
    norm = torch.sqrt(torch.sum(torch.abs(X_out) ** 2, dim=1, keepdim=True))
    X_out = X_out / (norm + eps)
    X_out = X_out.reshape(B, H, W, m1, m2)
    return X_out


def add_noise(u, noise_level=1e-3, positive=True):
    if noise_level <= 0:
        return u
    noise = torch.randn_like(u) * noise_level
    u_noisy = u + noise
    if positive:
        return torch.clamp(u_noisy, min=0.0)
    else:
        return u_noisy


def make_tuple(value, n):
    # should test if "value" is iterable instead?
    if not isinstance(value, Sequence) or isinstance(value, str):
        return (value,) * n
    elif len(value) < n:
        value.append(None)
    return value


def get_same_padding_1d(kernel_size: int) -> tuple[int, ...]:
    pad_total = kernel_size - 1
    pad_1 = math.ceil(pad_total / 2)
    pad_2 = math.ceil(pad_total) - pad_1
    return (pad_1, pad_2)


def get_same_padding_2d(kernel_size):
    if not isinstance(kernel_size, Iterable):
        kernel_size = [kernel_size] * 2
    pad_left, pad_right = get_same_padding_1d(kernel_size[0])
    pad_top, pad_bottom = get_same_padding_1d(kernel_size[1])

    return (pad_left, pad_right, pad_top, pad_bottom)


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)
