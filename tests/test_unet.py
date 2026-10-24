"""Tests for the classic (non-operator) U-Net -- ``pynop.models.unet``.

Covers the four block types the class must accept out of the box:

1. ``ResBlock``            (conv stage)
2. ``BottleneckBlock``     (conv stage)
3. ``ConvNextv2Block``     (conv stage)
4. Transformer bottleneck  (stack of ``TransformerBlock``, positional info via
   additive ``PEBlock`` PE or the block's built-in ``AdaptiveRoPE2D``)

plus the optional ``PatchEmbed`` stem, the ``concat``/``add`` skip fusion,
the transformer-bottleneck on/off switch, the RoPE option of the single
canonical ``TransformerBlock``, and input validation.

Run from the repository root::

    python -m pytest tests/test_unet.py -v
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
import torch.nn as nn

from pynop.core.blocks import BottleneckBlock, ConvNextv2Block, ResBlock, TransformerBlock
from pynop.core.norm import LayerNorm2d
from pynop.models import TransformerBottleneck, UNet
from pynop.models.unet import unet as unet_alias

FILTERS = (32, 64, 128, 256)
REPEATS = (2, 2, 2, 2)

BLOCKS = [
    pytest.param(ResBlock, id="ResBlock"),
    pytest.param(BottleneckBlock, id="BottleneckBlock"),
    pytest.param(ConvNextv2Block, id="ConvNextv2Block"),
]
FUSIONS = [pytest.param("concat", id="concat"), pytest.param("add", id="add")]


def assert_train_step(model: nn.Module, x: torch.Tensor, out_channels: int = 1) -> None:
    """Forward + backward on ``x``: check output shape, finiteness, and gradients.

    Args:
        model: The model under test.
        x: Input tensor ``(B, C, H, W)``.
        out_channels: Expected number of output channels.
    """
    y = model(x)
    assert y.shape == (x.shape[0], out_channels, x.shape[2], x.shape[3])
    assert torch.isfinite(y).all(), "non-finite values in the output"

    loss = (y**2).mean()
    loss.backward()

    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    assert not dead, f"parameters without a finite gradient: {dead}"


# --------------------------------------------------------------------- blocks
@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("fusion", FUSIONS)
def test_unet_forward_backward(block, fusion) -> None:
    """All three conv blocks x both fusions: correct shape, finite output, gradients."""
    torch.manual_seed(0)
    model = UNet(1, 1, block=block, filters=FILTERS, repeats=REPEATS, fusion=fusion)
    x = torch.randn(2, 1, 32, 32)
    assert_train_step(model, x)


def test_unet_partial_block_kwargs() -> None:
    """Notebook-style block instantiation via ``functools.partial``."""
    torch.manual_seed(0)
    block = partial(ResBlock, norm=[None, LayerNorm2d], activation=[nn.GELU, None])
    model = UNet(1, 1, block=block, filters=FILTERS, repeats=REPEATS)
    x = torch.randn(2, 1, 32, 32)
    assert_train_step(model, x)


def test_unet_3_stages() -> None:
    """A shallower U-Net (3 stages) at two resolutions."""
    torch.manual_seed(0)
    model = UNet(1, 1, block=ConvNextv2Block, filters=(32, 64, 128), repeats=(2, 2, 2))
    for res in (16, 64):
        x = torch.randn(1, 1, res, res)
        assert_train_step(model, x)


# --------------------------------------------------------------- transformer
def test_unet_without_transformer_bottleneck() -> None:
    torch.manual_seed(0)
    model = UNet(1, 2, filters=FILTERS, repeats=REPEATS, use_transformer_bottleneck=False)
    assert model.bottleneck is None
    x = torch.randn(2, 1, 32, 32)
    assert_train_step(model, x, out_channels=2)


def test_transformer_bottleneck_alone() -> None:
    """The bottleneck module standalone: shape, finiteness, gradients."""
    torch.manual_seed(0)
    bottleneck = TransformerBottleneck(dim=64, n_layers=2, n_heads=8)
    x = torch.randn(2, 64, 16, 16)
    y = bottleneck(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    dead = [name for name, p in bottleneck.named_parameters() if p.requires_grad and p.grad is None]
    assert not dead, f"parameters without gradient: {dead}"


def test_unet_pe_is_applied() -> None:
    """Zeroing the positional encoding must change the output.

    The PE is what breaks the permutation invariance of the self-attention; if
    it were not applied the bottleneck would be (nearly) position blind.
    """
    torch.manual_seed(0)
    model = UNet(1, 1, filters=FILTERS, repeats=REPEATS)
    x = torch.randn(1, 1, 32, 32)
    with_pe = model(x)

    class ZeroPE(nn.Module):
        """PE stand-in returning zeros of the right shape."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            _, H, W, _ = coords.shape
            return torch.zeros(1, H, W, self.dim)

    model.bottleneck.pe = ZeroPE(FILTERS[-1])
    without_pe = model(x)
    assert not torch.allclose(with_pe, without_pe), "the PE does not seem to be applied"


# ------------------------------------------------------------------ patch stem
@pytest.mark.parametrize("patch_size", [2, 4])
def test_unet_patch_embed_stem(patch_size) -> None:
    """PatchEmbed stem: output size must match the input size (un-patch at the head)."""
    torch.manual_seed(0)
    model = UNet(1, 1, filters=FILTERS, repeats=REPEATS, patch_size=patch_size)
    x = torch.randn(2, 1, 64, 64)
    assert_train_step(model, x)


# ---------------------------------------------------------------------- errors
def test_unet_invalid_configs() -> None:
    with pytest.raises(ValueError):
        UNet(1, 1, fusion="mul")
    with pytest.raises(ValueError):
        UNet(1, 1, factor=1)
    with pytest.raises(ValueError):
        UNet(1, 1, patch_size=1)
    with pytest.raises(ValueError):
        UNet(1, 1, filters=(32, 64), repeats=(2,))
    with pytest.raises(ValueError):
        UNet(1, 1, filters=(32, 64), n_heads=3)  # 64 % 3 != 0
    with pytest.raises(ValueError):
        TransformerBottleneck(dim=64, n_layers=0)
    with pytest.raises(ValueError):
        TransformerBottleneck(dim=64, n_layers=2, n_heads=8, pe_method=None)
    # RoPE requires a head dimension divisible by 4
    with pytest.raises(ValueError):
        TransformerBlock(dim=64, n_heads=32, rope=True)  # head dim 2
    with pytest.raises(ValueError):
        TransformerBottleneck(dim=64, n_heads=32, rope=True)  # head dim 2
    with pytest.raises(ValueError):
        UNet(1, 1, filters=FILTERS, repeats=REPEATS, rope=True, n_heads=128)  # 256 // 128 = 2


def test_unet_invalid_input() -> None:
    torch.manual_seed(0)
    model = UNet(1, 1, filters=FILTERS, repeats=REPEATS)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 30, 30))  # not divisible by 2 ** 3
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 32, 32))  # wrong channel count
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 32))  # not 4D


# --------------------------------------------------------------------- RoPE
def _grid_coords(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Normalized 2-D coordinate grid of shape ``(B, H, W, 2)``."""
    y = torch.linspace(-1.0, 1.0, H, device=device)
    x = torch.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)


def test_transformer_block_rope() -> None:
    """Single ``TransformerBlock`` with built-in ``AdaptiveRoPE2D``: forward + backward with coords."""
    torch.manual_seed(0)
    block = TransformerBlock(dim=64, n_heads=8, mlp_dim=128, rope=True)
    assert block.rope is not None
    x = torch.randn(2, 16, 64, requires_grad=True)
    coords = _grid_coords(2, 4, 4, x.device)  # (B, H, W, 2)
    y = block(x, coords=coords)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_rope_without_coords() -> None:
    """A rope-enabled block without coordinates must still work (rotation skipped)."""
    torch.manual_seed(0)
    block = TransformerBlock(dim=64, n_heads=8, mlp_dim=128, rope=True)
    x = torch.randn(2, 16, 64)
    y = block(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_transformer_block_rope_is_applied() -> None:
    """The rotary embedding must have a measurable effect (positions distinguishable)."""
    torch.manual_seed(0)
    block = TransformerBlock(dim=64, n_heads=8, mlp_dim=128, rope=True)
    block.eval()
    x = torch.randn(1, 16, 64)
    with torch.no_grad():
        y_real = block(x, coords=_grid_coords(1, 4, 4, x.device))
        y_zero = block(x, coords=torch.zeros(1, 4, 4, 2, device=x.device))
    assert not torch.allclose(y_real, y_zero), "RoPE does not seem to be applied (real and zero coords give the same output)"


def test_unet_rope_bottleneck() -> None:
    """UNet with ``rope=True``: square and non-square inputs (bottleneck grids 4x4 and 8x4)."""
    torch.manual_seed(0)
    model = UNet(1, 1, filters=FILTERS, repeats=REPEATS, rope=True)
    assert model.bottleneck.pe is None
    assert model.bottleneck.layers[0].rope is not None
    for shape in ((2, 1, 32, 32), (1, 1, 64, 32)):
        x = torch.randn(*shape)
        assert_train_step(model, x)


def test_unet_rope_is_applied() -> None:
    """The bottleneck's RoPE must have a measurable effect on the output."""
    torch.manual_seed(0)
    model = UNet(1, 1, filters=FILTERS, repeats=REPEATS, rope=True)
    x = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        y_real = model(x)
        # Zero coordinates => zero phases => identity rotation.
        model.bottleneck._coords = staticmethod(lambda H, W, device: torch.zeros(1, H, W, 2, device=device))
        y_zero = model(x)
    assert not torch.allclose(y_real, y_zero), "RoPE does not seem to be applied in the bottleneck"


# ----------------------------------------------------------------------- alias
def test_unet_backwards_compatible_alias() -> None:
    assert unet_alias is UNet
