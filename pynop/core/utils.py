# coding=utf-8
from collections.abc import Iterable
from typing import Optional, Any
import math
import torch
import torch.nn.functional as F


def make_tuple(value, n):
    # should test if "value" is iterable instead?
    if not isinstance(value, Iterable):
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
