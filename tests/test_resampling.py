"""Tests for the resampling cleanup of ``pynop.core.ops`` / ``pynop.core.blocks``.

After the cleanup:

* the four pixel-shuffle / pixel-unshuffle classes
  (``ConvPixelUnshuffleDownSampleLayer``, ``PixelUnshuffleChannelAveragingDownSampleLayer``,
  ``ConvPixelShuffleUpSampleLayer``, ``ChannelDuplicatingPixelUnshuffleUpSampleLayer``)
  are **removed** from the package;
* ``StridedConvDownsamplingLayer`` and ``InterpolateConvUpSampleLayer`` now live in
  ``pynop.core.blocks`` (moved out of ``ops``) and are still exported at the top level;
* ``MaxPoolConv`` stays in ``pynop.core.ops``;
* :data:`DOWNSAMPLING_LAYERS` exposes only ``"stridedconv"`` and ``"maxpooling"``,
  :data:`UPSAMPLING_LAYERS` only ``"interpolate"``;
* :class:`ResBlock` / :class:`BottleneckBlock` (and hence :class:`Resnet`) build their
  shortcut downsampler from that registry and reject removed methods.

Run from the repository root::

    python -m pytest tests/test_resampling.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pynop.core import blocks, ops
from pynop.core.blocks import (
    DOWNSAMPLING_LAYERS,
    UPSAMPLING_LAYERS,
    BottleneckBlock,
    InterpolateConvUpSampleLayer,
    PixelShuffleUpsample,
    ResBlock,
    StridedConvDownsamplingLayer,
    make_downsampler,
)
from pynop.core.ops import MaxPoolConv
from pynop.models.resnet import Resnet

# The pixel-shuffle / unshuffle classes that were removed from the package.
REMOVED_CLASSES = [
    "ConvPixelUnshuffleDownSampleLayer",
    "PixelUnshuffleChannelAveragingDownSampleLayer",
    "ConvPixelShuffleUpSampleLayer",
    "ChannelDuplicatingPixelUnshuffleUpSampleLayer",
]

VALID_DOWNSAMPLING = ["maxpooling", "stridedconv"]
REMOVED_DOWNSAMPLING = ["convpixunshuffle", "pixunshuffleaveraging"]
REMOVED_UPSAMPLING = ["convpixshuffle", "duplicatingpixshuffle"]


def _assert_train_step(model: nn.Module, x: torch.Tensor, expected_shape: tuple) -> None:
    """Forward + backward: check the output shape, finiteness, and gradients.

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


# -------------------------------------------------------------------------- registries
def test_downsampling_registry_only_valid_entries() -> None:
    assert set(DOWNSAMPLING_LAYERS) == {"maxpooling", "stridedconv"}
    for key, cls in DOWNSAMPLING_LAYERS.items():
        assert key in ("maxpooling", "stridedconv")
        # both registry entries accept the (in_channels, out_channels, kernel_size, factor) call
        d = DOWNSAMPLING_LAYERS[key](in_channels=8, out_channels=16, kernel_size=1, factor=2)
        out = d(torch.randn(1, 8, 16, 16))
        assert tuple(out.shape) == (1, 16, 8, 8)


def test_upsampling_registry_only_valid_entries() -> None:
    assert set(UPSAMPLING_LAYERS) == {"interpolate"}
    u = UPSAMPLING_LAYERS["interpolate"](in_channels=16, out_channels=8, factor=2)
    out = u(torch.randn(1, 16, 8, 8))
    assert tuple(out.shape) == (1, 8, 16, 16)


# --------------------------------------------------------------------------- removals
@pytest.mark.parametrize("name", REMOVED_CLASSES)
def test_shuffle_classes_removed(name: str) -> None:
    """The pixel-shuffle/unshuffle classes must be gone from both modules."""
    assert not hasattr(ops, name), f"{name} should no longer be in pynop.core.ops"
    assert not hasattr(blocks, name), f"{name} should no longer be in pynop.core.blocks"
    import pynop

    assert not hasattr(pynop, name), f"{name} should no longer be exported at the top level"


@pytest.mark.parametrize("name", REMOVED_DOWNSAMPLING + REMOVED_UPSAMPLING)
def test_removed_registry_keys_gone(name: str) -> None:
    assert name not in DOWNSAMPLING_LAYERS
    assert name not in UPSAMPLING_LAYERS


# -------------------------------------------------------------------------- moved classes
def test_strided_conv_downsampling_shape_and_grad() -> None:
    torch.manual_seed(0)
    layer = StridedConvDownsamplingLayer(8, 16, factor=2)
    _assert_train_step(layer, torch.randn(2, 8, 32, 32), (2, 16, 16, 16))


def test_interpolate_conv_upsample_shape_and_grad() -> None:
    torch.manual_seed(0)
    layer = InterpolateConvUpSampleLayer(16, 8, factor=2)
    _assert_train_step(layer, torch.randn(2, 16, 16, 16), (2, 8, 32, 32))


def test_maxpool_conv_shape_and_grad() -> None:
    torch.manual_seed(0)
    layer = MaxPoolConv(8, 16, kernel_size=3, factor=2)
    _assert_train_step(layer, torch.randn(2, 8, 32, 32), (2, 16, 16, 16))


def test_moved_classes_live_in_blocks() -> None:
    """The two moved classes must be defined in ``blocks`` (not ``ops``)."""
    assert StridedConvDownsamplingLayer.__module__ == "pynop.core.blocks"
    assert InterpolateConvUpSampleLayer.__module__ == "pynop.core.blocks"
    assert not hasattr(ops, "StridedConvDownsamplingLayer")
    assert not hasattr(ops, "InterpolateConvUpSampleLayer")
    # MaxPoolConv stays in ops.
    assert MaxPoolConv.__module__ == "pynop.core.ops"


def test_moved_classes_still_exported_top_level() -> None:
    import pynop

    assert pynop.StridedConvDownsamplingLayer is StridedConvDownsamplingLayer
    assert pynop.InterpolateConvUpSampleLayer is InterpolateConvUpSampleLayer
    assert pynop.MaxPoolConv is MaxPoolConv


# --------------------------------------------------------------------- round-trip
def test_down_up_roundtrip_restores_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 32, 32)
    down = StridedConvDownsamplingLayer(8, 16, factor=2)
    up = InterpolateConvUpSampleLayer(16, 8, factor=2)
    y = up(down(x))
    assert tuple(y.shape) == tuple(x.shape)


def test_pixel_shuffle_upsample_kept_in_blocks() -> None:
    """``PixelShuffleUpsample`` (the pixel-shuffle upsampler) must be retained."""
    assert hasattr(blocks, "PixelShuffleUpsample")
    torch.manual_seed(0)
    up = PixelShuffleUpsample(8, 16, factor=2)
    _assert_train_step(up, torch.randn(2, 8, 16, 16), (2, 16, 32, 32))


# ------------------------------------------------------------------ blocks / Resnet
@pytest.mark.parametrize("method", VALID_DOWNSAMPLING)
def test_resblock_downsampling(method: str) -> None:
    torch.manual_seed(0)
    block = ResBlock(8, 16, stride=2, downsampling_method=method)
    _assert_train_step(block, torch.randn(2, 8, 32, 32), (2, 16, 16, 16))


@pytest.mark.parametrize("method", VALID_DOWNSAMPLING)
def test_bottleneckblock_downsampling(method: str) -> None:
    torch.manual_seed(0)
    block = BottleneckBlock(8, 16, stride=2, downsampling_method=method)
    _assert_train_step(block, torch.randn(2, 8, 32, 32), (2, 16, 16, 16))


@pytest.mark.parametrize("method", VALID_DOWNSAMPLING)
def test_resnet_downsampling(method: str) -> None:
    torch.manual_seed(0)
    model = Resnet(
        in_channels=1, dims=[16, 32], depths=[2, 2], stem_filters=8, build_head=True,
        downsampling_method=method,
    )
    _assert_train_step(model, torch.randn(2, 1, 32, 32), (2, 1000))


@pytest.mark.parametrize("method", REMOVED_DOWNSAMPLING)
def test_blocks_reject_removed_downsampling_method(method: str) -> None:
    with pytest.raises(ValueError):
        ResBlock(8, 16, stride=2, downsampling_method=method)
    with pytest.raises(ValueError):
        BottleneckBlock(8, 16, stride=2, downsampling_method=method)
    with pytest.raises(ValueError):
        Resnet(in_channels=1, dims=[16, 32], depths=[2, 2], stem_filters=8, downsampling_method=method)


def test_make_downsampler_helper() -> None:
    layer = make_downsampler("maxpooling", 8, 16, factor=2, block_name="test")
    assert isinstance(layer, MaxPoolConv)
    layer = make_downsampler("stridedconv", 8, 16, factor=2, block_name="test")
    assert isinstance(layer, StridedConvDownsamplingLayer)
    with pytest.raises(ValueError):
        make_downsampler("convpixunshuffle", 8, 16, factor=2, block_name="test")
