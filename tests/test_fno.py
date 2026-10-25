"""Tests for the Fourier Neural Operator blocks and model -- ``pynop.core`` / ``pynop.models``.

Covers the three spectral convolutions (``SpectralConv2d``, ``SeparableSpectralConv2d``,
``TuckerSpectralConv2d``), the unified :class:`~pynop.core.blocks.FNOBlock` (including
the local :class:`~pynop.core.ops.FiniteDifferenceLayer` branch), and the
:class:`~pynop.models.FNO` model.

Regression focus
----------------
* The spectral convolutions used to **mutate module state** when the input grid was
  smaller than the configured modes (clamping ``self.modes_x/modes_y``), which both
  crashed the einsum and permanently corrupted the layer for *all* subsequent inputs.
  They now slice the learned weights to the effective (``min``) modes -- no state
  mutation, valid for any input resolution (resolution invariance).
* :class:`FNOBlock` used to apply its activation **twice** (a dead ``if
  self.activation is not None`` guard around a second application). It now follows the
  reference ordering ``[norm] -> spectral -> [activation] -> pointwise -> + shortcut``
  with norm and activation optional.
* The :class:`FNO` model used to crash on its very first forward
  (``AttributeError: 'FNO' object has no attribute 'trainable_pos_encoding'``) and to
  crash when building a Tucker layer with ``spectral_compression_factor=None``. Both are
  fixed; unknown block names now raise ``ValueError`` instead of silently falling back.

Local high-frequency branch
---------------------------
* :class:`FNOBlock` now supports an optional stack of
  :class:`~pynop.core.ops.FiniteDifferenceLayer` (differential convolutions,
  NeuralOperator-style) added in parallel with the shortcut. This captures local
  high-frequency information that the global spectral branch (a structural low-pass)
  cannot represent, while remaining resolution-invariant via the ``1/h`` rescaling.
  The depth is controlled by ``local_conv`` (0 = disabled, default).

Run from the repository root::

    python -m pytest tests/test_fno.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pynop import FNO
from pynop.core.blocks import FNOBlock
from pynop.core.norm import LayerNorm2d
from pynop.core.ops import (
    FiniteDifferenceLayer,
    SeparableSpectralConv2d,
    SpectralConv2d,
    TuckerSpectralConv2d,
)


def _train_step(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run forward + backward and assert the output is finite with no dead parameters."""
    y = model(x)
    assert torch.isfinite(y).all(), "non-finite output"
    y.sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or (p.grad == 0).all()]
    assert not dead, f"dead parameters (no gradient): {dead}"
    return y


# --------------------------------------------------------------------------- spectral convs


@pytest.mark.parametrize("conv_cls", [SpectralConv2d, SeparableSpectralConv2d])
def test_spectral_conv_smaller_than_modes_no_state_mutation(conv_cls) -> None:
    """Input smaller than the configured modes must not mutate module state."""
    conv = conv_cls(4, 4, (8, 8))
    assert conv.modes_x == 8 and conv.modes_y == 8

    # smaller input: must run and must NOT change the configured modes
    y = conv(torch.randn(1, 4, 4, 4))
    assert tuple(y.shape) == (1, 4, 4, 4)
    assert conv.modes_x == 8 and conv.modes_y == 8, "module state was mutated by forward"

    # a larger input must still work (the old code would have stayed clamped)
    y = conv(torch.randn(1, 4, 32, 32))
    assert tuple(y.shape) == (1, 4, 32, 32)
    assert conv.modes_x == 8 and conv.modes_y == 8


def test_tucker_conv_smaller_than_modes_no_state_mutation() -> None:
    """Tucker conv: small then large input, no state mutation, gradients flow."""
    conv = TuckerSpectralConv2d(4, 4, (8, 8), (2, 2, 2))
    y = conv(torch.randn(1, 4, 4, 4))
    assert tuple(y.shape) == (1, 4, 4, 4) and conv.modes_x == 8
    y = conv(torch.randn(1, 4, 32, 32))
    assert tuple(y.shape) == (1, 4, 32, 32) and conv.modes_x == 8
    _train_step(conv, torch.randn(1, 4, 16, 16))


@pytest.mark.parametrize("conv_cls", [SpectralConv2d, SeparableSpectralConv2d])
def test_spectral_conv_gradient_flow(conv_cls) -> None:
    conv = conv_cls(4, 8, (4, 4))
    _train_step(conv, torch.randn(2, 4, 16, 16))


# --------------------------------------------------------------------------- FNOBlock


@pytest.mark.parametrize(
    "spectral_layer_type,ranks",
    [
        ("standard", None),
        ("separable", None),
        ("tucker", (2, 2, 2)),
    ],
)
def test_fno_block_all_spectral_types(spectral_layer_type, ranks) -> None:
    """FNOBlock builds and trains with each of the three spectral convolutions."""
    torch.manual_seed(0)
    block = FNOBlock(
        in_channels=4,
        out_channels=8,
        modes=(4, 4),
        spectral_layer_type=spectral_layer_type,
        ranks=ranks,
    )
    y = _train_step(block, torch.randn(2, 4, 16, 16))
    assert tuple(y.shape) == (2, 8, 16, 16)


def test_fno_block_int_modes() -> None:
    """A scalar `modes` is broadcast to both dimensions."""
    block = FNOBlock(4, 8, 4)
    assert block.modes == (4, 4)
    _train_step(block, torch.randn(2, 4, 16, 16))


def test_fno_block_optional_norm_and_activation() -> None:
    """norm and activation can each be disabled independently."""
    for norm in (LayerNorm2d, None):
        for act in (nn.GELU, None):
            block = FNOBlock(4, 4, (4, 4), activation=act, normalization=norm)
            assert (block.norm is None) == (norm is None)
            assert (block.activation is None) == (act is None)
            _train_step(block, torch.randn(2, 4, 16, 16))


def test_fno_block_hidden_channels_width() -> None:
    """The spectral-conv output width is configurable via `hidden_channels`."""
    block = FNOBlock(4, 8, (4, 4), hidden_channels=16)
    assert block.hidden_channels == 16
    assert block.spectral_conv.out_channels == 16
    _train_step(block, torch.randn(2, 4, 16, 16))


def test_fno_block_resolution_invariance() -> None:
    """The same block is valid for inputs both smaller and larger than its modes."""
    block = FNOBlock(4, 4, (8, 8))
    assert tuple(block(torch.randn(1, 4, 8, 8)).shape) == (1, 4, 8, 8)
    assert tuple(block(torch.randn(1, 4, 32, 32)).shape) == (1, 4, 32, 32)
    # configured modes are unchanged (no state mutation)
    assert block.modes == (8, 8)


def test_fno_block_tucker_requires_ranks() -> None:
    with pytest.raises(ValueError, match="ranks"):
        FNOBlock(4, 4, (4, 4), spectral_layer_type="tucker", ranks=None)


def test_fno_block_unknown_spectral_type() -> None:
    with pytest.raises(ValueError, match="spectral_layer_type"):
        FNOBlock(4, 4, (4, 4), spectral_layer_type="bogus")


def test_fno_block_reference_ordering() -> None:
    """The forward is [norm] -> spectral -> [activation] -> pointwise -> + shortcut.

    With the activation disabled and a fixed norm, the main branch reduces to
    ``linear(spectral_conv(x))``; verify the shortcut is added on top by checking that
    the output changes when the shortcut weight is perturbed.
    """
    torch.manual_seed(0)
    block = FNOBlock(4, 4, (4, 4), activation=None, normalization=None)
    x = torch.randn(2, 4, 16, 16)
    y0 = block(x)
    with torch.no_grad():
        block.shortcut.weight.zero_()
        block.shortcut.bias.zero_()
    y1 = block(x)
    # zeroing the shortcut must change the output (the residual is additive)
    assert not torch.allclose(y0, y1)


# --------------------------------------------------------------------------- FNO local conv branch


@pytest.mark.parametrize("n_local", [1, 2, 3])
def test_fno_block_local_conv_depth(n_local: int) -> None:
    """FNOBlock with a local FiniteDifferenceLayer branch of various depths."""
    torch.manual_seed(0)
    block = FNOBlock(4, 8, (4, 4), local_conv=n_local)
    assert block.local_conv is not None
    assert len(block.local_conv) == n_local
    _train_step(block, torch.randn(2, 4, 16, 16))


def test_fno_block_local_conv_disabled_by_default() -> None:
    """local_conv=0 (default) means no local branch."""
    block = FNOBlock(4, 4, (4, 4))
    assert block.local_conv is None
    assert block.local_conv_depth == 0


def test_fno_block_local_conv_in_out() -> None:
    """Local branch works with in_channels != out_channels."""
    torch.manual_seed(0)
    block = FNOBlock(4, 12, (4, 4), local_conv=2)
    y = _train_step(block, torch.randn(2, 4, 16, 16))
    assert tuple(y.shape) == (2, 12, 16, 16)


def test_fno_block_local_conv_kernel_size() -> None:
    """Local branch respects the kernel_size parameter."""
    torch.manual_seed(0)
    for ks in (3, 5):
        block = FNOBlock(4, 4, (4, 4), local_conv=1, local_kernel_size=ks)
        for layer in block.local_conv:
            assert layer.kernel_size == ks
        _train_step(block, torch.randn(2, 4, 16, 16))


def test_fno_block_local_conv_gradient_flow() -> None:
    """Gradients flow through every layer of the local conv branch."""
    torch.manual_seed(0)
    block = FNOBlock(4, 4, (4, 4), local_conv=3)
    x = torch.randn(2, 4, 16, 16)
    y = block(x)
    y.sum().backward()
    for i, layer in enumerate(block.local_conv):
        assert layer.weight.grad is not None, f"local_conv[{i}] weight has no grad"
        assert layer.bias.grad is not None, f"local_conv[{i}] bias has no grad"


def test_fno_block_local_conv_is_parallel_to_shortcut() -> None:
    """Zeroing the local branch must change the output (it is additive)."""
    torch.manual_seed(0)
    block = FNOBlock(4, 4, (4, 4), local_conv=1)
    x = torch.randn(2, 4, 16, 16)
    y0 = block(x)
    with torch.no_grad():
        for layer in block.local_conv:
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
    y1 = block(x)
    assert not torch.allclose(y0, y1)


def test_fno_block_local_conv_resolution_invariance() -> None:
    """The local branch (1/h rescaling) is resolution-invariant."""
    torch.manual_seed(0)
    block = FNOBlock(4, 4, (8, 8), local_conv=1)
    assert tuple(block(torch.randn(1, 4, 8, 8)).shape) == (1, 4, 8, 8)
    assert tuple(block(torch.randn(1, 4, 32, 32)).shape) == (1, 4, 32, 32)


# --------------------------------------------------------------------------- FNO model


def _make_model(**over) -> FNO:
    base = dict(in_channels=1, out_channels=1, modes=(4, 4), hidden_channels=(8, 8), blocks=["FNO", "FNO"])
    base.update(over)
    return FNO(**base)


def test_fno_model_forward_no_crash() -> None:
    """Regression: the first forward used to raise AttributeError (trainable_pos_encoding)."""
    torch.manual_seed(0)
    model = _make_model()
    _train_step(model, torch.randn(2, 1, 16, 16))


def test_fno_model_trainable_pos_encoding() -> None:
    torch.manual_seed(0)
    model = _make_model(trainable_pos_encoding=True)
    y = _train_step(model, torch.randn(2, 1, 16, 16))
    assert tuple(y.shape) == (2, 1, 16, 16)
    # the learnable embedding must receive a gradient
    assert model.pos_embedding_weights.grad is not None
    assert not (model.pos_embedding_weights.grad == 0).all()


def test_fno_model_no_fixed_pos_encoding() -> None:
    torch.manual_seed(0)
    model = _make_model(fixed_pos_encoding=False)
    assert not hasattr(model, "grid_encoding")
    _train_step(model, torch.randn(2, 1, 16, 16))


def test_fno_model_tucker_factor_none_and_set() -> None:
    """Regression: tucker with spectral_compression_factor=None used to crash at build."""
    torch.manual_seed(0)
    for factor in (None, 2):
        model = _make_model(spectral_layer_type="tucker", spectral_compression_factor=factor)
        _train_step(model, torch.randn(2, 1, 16, 16))


def test_fno_model_separable() -> None:
    torch.manual_seed(0)
    model = _make_model(spectral_layer_type="separable")
    _train_step(model, torch.randn(2, 1, 16, 16))


@pytest.mark.parametrize(
    "modes,hidden,blocks",
    [
        (4, (8, 8), ["FNO", "FNO"]),
        ((4, 4), 8, ["FNO", "FNO"]),
        (4, 8, "FNO"),
    ],
)
def test_fno_model_scalar_inputs(modes, hidden, blocks) -> None:
    """Scalar `modes` / `hidden_channels` / `blocks` are accepted."""
    torch.manual_seed(0)
    model = FNO(in_channels=1, out_channels=1, modes=modes, hidden_channels=hidden, blocks=blocks)
    _train_step(model, torch.randn(2, 1, 16, 16))


@pytest.mark.parametrize("local_conv", [0, 1, 2])
def test_fno_model_local_conv(local_conv: int) -> None:
    """FNO model with local conv branch (0 = disabled, 1-2 = active)."""
    torch.manual_seed(0)
    model = _make_model(local_conv=local_conv)
    if local_conv > 0:
        for op in model.ops:
            assert op.local_conv is not None
    else:
        for op in model.ops:
            assert op.local_conv is None
    _train_step(model, torch.randn(2, 1, 16, 16))


def test_fno_model_local_conv_in_out() -> None:
    """FNO model with local conv and non-square in/out channels."""
    torch.manual_seed(0)
    model = FNO(
        in_channels=1, out_channels=3, modes=(4, 4),
        hidden_channels=(8, 8), blocks=["FNO", "FNO"], local_conv=2,
    )
    y = _train_step(model, torch.randn(2, 1, 16, 16))
    assert tuple(y.shape) == (2, 3, 16, 16)


def test_fno_model_cond() -> None:
    """A conditioning field of `cond_dim` channels is concatenated before the lifting."""
    torch.manual_seed(0)
    model = _make_model(cond_dim=3)
    x = torch.randn(2, 1, 16, 16)
    cond = torch.randn(2, 3, 16, 16)
    y = model(x, cond=cond)
    assert tuple(y.shape) == (2, 1, 16, 16)
    y.sum().backward()


def test_fno_model_unknown_block_rejected() -> None:
    """Unknown block names must raise (no silent fallback to a default block)."""
    with pytest.raises(ValueError, match="Unknown block type"):
        FNO(in_channels=1, out_channels=1, modes=(4, 4), hidden_channels=(8, 8), blocks=["UFNO", "FNO"])


def test_fno_model_resolution_invariance() -> None:
    """The same model is valid for inputs both smaller and larger than its modes."""
    torch.manual_seed(0)
    model = _make_model(modes=(8, 8))
    assert tuple(model(torch.randn(1, 1, 8, 8)).shape) == (1, 1, 8, 8)
    assert tuple(model(torch.randn(1, 1, 32, 32)).shape) == (1, 1, 32, 32)


def test_fno_model_non_square_grid() -> None:
    torch.manual_seed(0)
    model = _make_model(modes=(4, 6))
    x = torch.randn(2, 1, 24, 32)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 24, 32)


def test_fno_model_invalid_blocks_length() -> None:
    with pytest.raises(ValueError):
        FNO(in_channels=1, out_channels=1, modes=(4, 4), hidden_channels=(8, 8), blocks=["FNO"])
