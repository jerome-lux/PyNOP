# coding=utf-8
from collections.abc import Iterable
from typing import Optional, Any, Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_stats(x, dim=-1, text=""):
    """
    Print the mean and variance of a tensor along a specified dimension.

    Args:
        x (torch.Tensor): Tensor to compute statistics for.
        dim (int): Dimension along which to compute the mean and variance.
        text (str): Optional prefix text to print before statistics.
    """
    with torch.no_grad():
        var = x.var(dim=dim).mean()
        mean = x.mean(dim=dim).mean()
        print(f"{text} mu={mean.item():.2e}, var={var.item():.2e}")


class ChebyshevBasis(nn.Module):
    """
    Generate a 2D Chebyshev polynomial basis from normalized coordinates.

    The basis is computed for coordinate values in the range [-1, 1] and
    flattened into a vector of polynomial coefficients.
    """

    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1  # Degree max in x
        self.m2 = m2  # Degree max in y

    def forward(self, coords):
        """
        Compute the Chebyshev basis for 2D coordinates.

        Args:
            coords (torch.Tensor): Tensor of shape [B, H, W, 2] with values in [-1, 1].

        Returns:
            torch.Tensor: Basis tensor of shape [B, H, W, m1 * m2].
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
        basis (torch.Tensor): Input tensor of shape [B, H, W, M1, M2].
        iterations (int): Number of orthogonalization steps.

    Returns:
        torch.Tensor: Orthogonalized basis of the same shape [B, H, W, M1, M2].
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
    Perform Gram-Schmidt-like orthogonalization on a learned basis tensor.

    Args:
        X (torch.Tensor): Input tensor of shape [B, H, W, m1, m2].
        n_iter (int): Number of refinement iterations.
        eps (float): Small epsilon for numerical stability.

    Returns:
        torch.Tensor: Orthogonalized tensor of the same shape as X.
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


def add_noise(x, noise_level, max_amplitude=None, max_val=None, positive=False):
    """
    Add random noise to a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        noise_level (float): Standard deviation of the Gaussian noise.
        max_amplitude (Optional[float]): Maximum noise amplitude as a fraction of |x|.
        max_val (Optional[float]): Hard clipping bound for the noise.
        positive (bool): If True, clamp the final tensor to be non-negative.

    Returns:
        torch.Tensor: Noisy tensor.
    """

    noise = torch.randn_like(x) * noise_level
    if max_amplitude is not None:
        max_allowed_noise = max_amplitude * torch.abs(x)
        noise = torch.clamp(noise, min=-max_allowed_noise, max=max_allowed_noise)
    elif max_val is not None:
        noise = torch.clamp(noise, min=-max_val, max=max_val)

    noisy_x = x + noise

    if positive:
        return torch.clamp(noisy_x, min=0)

    return noisy_x


def make_tuple(value, n):
    """
    Convert a scalar or sequence into a tuple of length n.

    Args:
        value: Scalar or sequence value.
        n (int): Desired tuple length.

    Returns:
        tuple: If value is not a sequence or is a string, returns (value,) repeated n times.
               If value is a sequence shorter than n, appends None elements until length n.
    """
    # should test if "value" is iterable instead?
    if not isinstance(value, Sequence) or isinstance(value, str):
        return (value,) * n
    elif len(value) < n:
        value.append(None)
    return value


def get_same_padding_1d(kernel_size: int) -> tuple[int, ...]:
    """
    Compute symmetric padding for 1D "same" convolution.

    Args:
        kernel_size (int): Convolution kernel size.

    Returns:
        tuple[int, int]: Left and right padding.
    """
    pad_total = kernel_size - 1
    pad_1 = math.ceil(pad_total / 2)
    pad_2 = math.ceil(pad_total) - pad_1
    return (pad_1, pad_2)


def get_same_padding_2d(kernel_size):
    """
    Compute symmetric padding for 2D "same" convolution.

    Args:
        kernel_size: Integer or iterable of two integers representing kernel size.

    Returns:
        tuple[int, int, int, int]: (pad_left, pad_right, pad_top, pad_bottom).
    """
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
    """
    Resize a tensor using interpolation.

    Args:
        x (torch.Tensor): Input tensor.
        size (Optional[Any]): Output spatial size.
        scale_factor (Optional[list[float]]): Multiplicative scale factor.
        mode (str): Interpolation mode. Supported: 'bilinear', 'bicubic', 'nearest', 'area'.
        align_corners (Optional[bool]): Align corners for bilinear/bicubic modes.

    Returns:
        torch.Tensor: Resized tensor.

    Raises:
        NotImplementedError: If an unsupported mode is requested.
    """
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
    """
    Convert a value to a list.

    Args:
        x: Input scalar, list, or tuple.
        repeat_time (int): Number of times to repeat scalar values when x is not a sequence.

    Returns:
        list: If x is a list or tuple, returns a converted list. Otherwise returns [x] repeated repeat_time times.
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    """
    Convert a value to a tuple of minimum length min_len.

    Args:
        x: Input scalar, list, or tuple.
        min_len (int): Minimum tuple length.
        idx_repeat (int): Index of the element to repeat when extending shorter sequences.

    Returns:
        tuple: The input converted to a tuple and padded with repeated values if necessary.
    """
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def sample_square_crop_boxes(images, crop_ratio=0.5):
    """
    Sample random square crop coordinates for a batch of images.

    Args:
        images (torch.Tensor): Batch tensor with shape [B, C, H, W].
        crop_ratio (float): Fraction of the smaller image dimension used for crop size.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]: (top, left, crop_size) for the sampled crop boxes.
    """
    _, _, height, width = images.shape
    crop_size = max(1, int(min(height, width) * crop_ratio))
    max_top = height - crop_size
    max_left = width - crop_size
    top = torch.randint(max_top + 1, (images.size(0),), device=images.device)
    left = torch.randint(max_left + 1, (images.size(0),), device=images.device)
    return top, left, crop_size


def apply_square_crop(images, top, left, crop_size):
    """
    Apply a square crop to a batch of images using grid sampling.

    Args:
        images (torch.Tensor): Batch of images with shape [B, C, H, W].
        top (torch.Tensor): Vertical crop start positions of shape [B].
        left (torch.Tensor): Horizontal crop start positions of shape [B].
        crop_size (int): Size of the square crop.

    Returns:
        torch.Tensor: Cropped images sampled with bilinear interpolation.

    Raises:
        ValueError: If crop_size is not a positive integer.
    """
    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")

    ys = torch.linspace(0, crop_size - 1, height, device=device, dtype=dtype)
    xs = torch.linspace(0, crop_size - 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    top = top.to(device=device, dtype=dtype).view(batch_size, 1, 1)
    left = left.to(device=device, dtype=dtype).view(batch_size, 1, 1)
    sample_y = top + grid_y.unsqueeze(0)
    sample_x = left + grid_x.unsqueeze(0)

    if height > 1:
        sample_y = (sample_y / (height - 1)) * 2 - 1
    else:
        sample_y = torch.zeros_like(sample_y)
    if width > 1:
        sample_x = (sample_x / (width - 1)) * 2 - 1
    else:
        sample_x = torch.zeros_like(sample_x)

    grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)


def make_inpainting_mask(images, mask_ratio=0.35):
    """
    Create a random square inpainting mask for a batch of images.

    Args:
        images (torch.Tensor): Batch of images with shape [B, C, H, W].
        mask_ratio (float): Fraction of image area to mask.

    Returns:
        torch.Tensor: Binary mask of shape [B, 1, H, W], where 1 marks unmasked pixels and 0 marks the masked hole.

    Raises:
        ValueError: If mask_ratio is outside the open interval (0, 1).
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

    batch_size, _, height, width = images.shape
    mask_side = max(1, int(round((mask_ratio ** 0.5) * min(height, width))))
    max_top = height - mask_side
    max_left = width - mask_side
    top = torch.randint(max_top + 1, (batch_size,), device=images.device)
    left = torch.randint(max_left + 1, (batch_size,), device=images.device)

    ys = torch.arange(height, device=images.device).view(1, height, 1)
    xs = torch.arange(width, device=images.device).view(1, 1, width)
    hole_y = (ys >= top.view(batch_size, 1, 1)) & (ys < (top + mask_side).view(batch_size, 1, 1))
    hole_x = (xs >= left.view(batch_size, 1, 1)) & (xs < (left + mask_side).view(batch_size, 1, 1))
    hole = hole_y & hole_x
    return (~hole).to(dtype=images.dtype).unsqueeze(1)


def fill_mask_with_image_average(images, mask):
    """
    Fill masked pixels with the per-image average color.

    Args:
        images (torch.Tensor): Batch of images with shape [B, C, H, W].
        mask (torch.Tensor): Binary mask of shape [B, 1, H, W].

    Returns:
        torch.Tensor: Images where masked regions are replaced by the average pixel value of each image.

    Raises:
        ValueError: If mask shape does not match [B, 1, H, W].
    """
    mask = mask.to(device=images.device, dtype=images.dtype)
    if mask.shape != images.shape[:1] + (1,) + images.shape[2:]:
        raise ValueError(
            f"expected mask shape {(images.size(0), 1, images.size(2), images.size(3))}, got {tuple(mask.shape)}"
        )
    image_average = images.mean(dim=(-2, -1), keepdim=True)
    return images * mask + image_average * (1.0 - mask)


def make_inpainted_input(images, mask_ratio=0.35):
    """
    Create an inpainted input by masking a region and filling it with the image average.

    Args:
        images (torch.Tensor): Batch of images with shape [B, C, H, W].
        mask_ratio (float): Fraction of image area to mask.

    Returns:
        torch.Tensor: Inpainted images with masked regions filled by the per-image mean.
    """
    mask = make_inpainting_mask(images, mask_ratio=mask_ratio)
    return fill_mask_with_image_average(images, mask)