"""Tests for the unified :class:`~pynop.models.resnet.Resnet`.

:class:`Resnet` now selects its residual block through the ``block_type``
argument:

* ``"basic"`` (default) — :class:`~pynop.core.blocks.ResBlock`;
* ``"bottleneck"`` — :class:`~pynop.core.blocks.BottleneckBlock`, whose
  mid-channel width is set by ``bottleneck_ratio`` and which optionally
  inserts a Squeeze-and-Excitation gate (``se``).

The former :class:`Resnetb` class has been removed; ``resnet50`` /
``resnext50`` are thin factories over ``Resnet(block_type="bottleneck", ...)``.

Run from the repository root::

    python -m pytest tests/test_resnet.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pynop.models.resnet import Resnet, resnet18, resnet34, resnet50, resnext50


def _assert_train_step(model: nn.Module, x: torch.Tensor, expected_shape: tuple) -> None:
    """Forward + backward: check output shape, finiteness, and gradients.

    Args:
        model: The module under test.
        x: Input tensor ``(B, C, H, W)``.
        expected_shape: Expected output shape.
    """
    y = model(x)
    assert tuple(y.shape) == expected_shape, f"got {tuple(y.shape)}, expected {expected_shape}"
    assert torch.isfinite(y).all(), "non-finite values in the output"

    loss = (y**2).mean()
    loss.backward()

    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not dead, f"parameters without a finite gradient: {dead}"


def _num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _first_conv(module: nn.Module) -> nn.Conv2d:
    """Return the first :class:`nn.Conv2d` inside ``module`` (e.g. a ``ConvLayer``)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            return m
    raise AssertionError(f"no Conv2d found in {type(module).__name__}")


# --------------------------------------------------------------- block_type routing
@pytest.mark.parametrize("method", ["maxpooling", "stridedconv"])
def test_basic_block_forward_backward(method: str) -> None:
    torch.manual_seed(0)
    model = Resnet(
        in_channels=1, dims=[16, 32], depths=[2, 2], stem_filters=8,
        build_head=True, downsampling_method=method,
    )
    _assert_train_step(model, torch.randn(2, 1, 32, 32), (2, 1000))


@pytest.mark.parametrize("method", ["maxpooling", "stridedconv"])
def test_bottleneck_block_forward_backward(method: str) -> None:
    torch.manual_seed(0)
    model = Resnet(
        in_channels=1, dims=[32, 64], depths=[2, 2], stem_filters=16,
        build_head=True, block_type="bottleneck", downsampling_method=method,
    )
    _assert_train_step(model, torch.randn(2, 1, 32, 32), (2, 1000))


def test_bottleneck_ratio_controls_capacity() -> None:
    """A smaller ``bottleneck_ratio`` keeps more mid-channels => more params."""
    torch.manual_seed(0)
    base = dict(in_channels=1, dims=[32, 64], depths=[2, 2], stem_filters=16, block_type="bottleneck")
    wide = _num_params(Resnet(bottleneck_ratio=2, **base))
    narrow = _num_params(Resnet(bottleneck_ratio=8, **base))
    assert wide > narrow, f"expected ratio=2 ({wide}) > ratio=8 ({narrow})"


def test_se_gate_adds_parameters() -> None:
    torch.manual_seed(0)
    base = dict(in_channels=1, dims=[32, 64], depths=[2, 2], stem_filters=16, block_type="bottleneck")
    without = _num_params(Resnet(se=False, **base))
    with_se = _num_params(Resnet(se=True, **base))
    assert with_se > without, f"expected SE ({with_se}) > no-SE ({without})"


def test_blocks_use_the_right_submodule() -> None:
    """Each ``block_type`` must instantiate the matching block class."""
    from pynop.core.blocks import BottleneckBlock, ResBlock

    basic = Resnet(in_channels=1, dims=[16, 32], depths=[1, 1], stem_filters=8)
    assert all(isinstance(b, ResBlock) for b in basic.ops)

    bottleneck = Resnet(in_channels=1, dims=[16, 32], depths=[1, 1], stem_filters=8, block_type="bottleneck")
    assert all(isinstance(b, BottleneckBlock) for b in bottleneck.ops)


def test_invalid_block_type_rejected() -> None:
    with pytest.raises(ValueError, match="block_type"):
        Resnet(block_type="nope")


# ------------------------------------------------------------------- factory fns
def test_resnet18_34_are_basic() -> None:
    from pynop.core.blocks import ResBlock

    for factory in (resnet18, resnet34):
        model = factory(in_channels=1)
        assert all(isinstance(b, ResBlock) for b in model.ops)


def test_resnet50_resnext50_are_bottleneck() -> None:
    from pynop.core.blocks import BottleneckBlock

    for factory in (resnet50, resnext50):
        model = factory(in_channels=1)
        assert all(isinstance(b, BottleneckBlock) for b in model.ops)


def test_resnext50_groups_and_ratio() -> None:
    model = resnext50(in_channels=1)
    # ResNeXt-style: grouped 3x3 conv (32 groups) and a 2x bottleneck reduction.
    for b in model.ops:
        assert _first_conv(b.conv).groups == 32
        assert _first_conv(b.proj).out_channels == b.out_channels // 2


def test_resnet50_forward_backward() -> None:
    torch.manual_seed(0)
    model = resnet50(in_channels=1, build_head=True)
    _assert_train_step(model, torch.randn(1, 1, 64, 64), (1, 1000))


def test_resnext50_forward_backward() -> None:
    torch.manual_seed(0)
    model = resnext50(in_channels=1, build_head=True)
    _assert_train_step(model, torch.randn(1, 1, 64, 64), (1, 1000))


# --------------------------------------------------------------------- cleanup
def test_resnetb_removed() -> None:
    """The standalone ``Resnetb`` class must no longer exist."""
    import pynop.models.resnet as resnet_mod

    assert not hasattr(resnet_mod, "Resnetb")
