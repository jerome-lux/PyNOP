"""Tests for the learned Integral-Transform blocks -- ``pynop.core.blocks``.

Covers the three IT blocks and, in particular, the magnitude of the integral
transform branch at initialization:

* ``SepLITBlock``  (separable *linear* IT, channels-first ``(B, C, H, W)``)
* ``SepNLITBlock`` (separable *non-linear* IT, channels-first ``(B, C, H, W)``)
* ``LITBlock``     (full IT, channels-last ``(B, H, W, C)``)

Regression focus
----------------
The ``Sep*`` blocks used to (a) apply a channels-last ``AdaptiveLayerNorm`` to
channels-first data (crash unless ``out_channels == W``) and (b) over-suppress
the transform branch with a hard ``1 / (H * W)`` factor, which -- because the
learned basis is not unit-energy -- left the IT branch ~100-400x smaller than
the shortcut (effectively dead). Both are fixed: a channels-first norm
(``LayerNorm2d``) and a learnable normalization exponent (``IT_scale``).

Run from the repository root::

    python -m pytest tests/test_lit_blocks.py -v
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pynop.core.blocks import LITBlock, SepLITBlock, SepNLITBlock
from pynop.core.norm import LayerNorm2d


def _first_activation_input(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return the input of the block's first activation call (the IT branch, post-norm)."""
    captured: dict = {}

    def hook(mod, inp, out):  # noqa: ARG001
        captured.setdefault("inp", inp[0])

    h = block.activation.register_forward_hook(hook)
    with torch.no_grad():
        block(x)
    h.remove()
    return captured["inp"]


def _shortcut_std(block: nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        return float(block.shortcut(x).std())


def _assert_train_step(model: nn.Module, x: torch.Tensor) -> None:
    y = model(x)
    assert y.shape == x.shape, f"shape mismatch: {tuple(y.shape)} != {tuple(x.shape)}"
    assert torch.isfinite(y).all(), "non-finite output"
    y.sum().backward()
    dead = [n for n, p in model.named_parameters() if p.grad is None or (p.grad == 0).all()]
    assert not dead, f"dead parameters (no gradient): {dead}"


def _std(t: torch.Tensor) -> float:
    return float(t.std())


# --------------------------------------------------------------------- basic forward/backward


def test_sep_lit_block_forward_backward() -> None:
    """SepLITBlock (channels-first): forward + backward, no dead params, O(1) output."""
    torch.manual_seed(0)
    block = SepLITBlock(8, 8, 8, 8)
    _assert_train_step(block, torch.randn(2, 8, 32, 32))
    # out_channels (8) != W (32): would crash with a channels-last norm.
    _assert_train_step(SepLITBlock(8, 8, 8, 8), torch.randn(2, 8, 32, 32))
    # changing the number of output channels must not crash (norm is channels-first).
    b2 = SepLITBlock(8, 16, 8, 8)
    y = b2(torch.randn(2, 8, 32, 32))
    assert y.shape == (2, 16, 32, 32) and torch.isfinite(y).all()
    # O(1) magnitude (not over-suppressed, not exploding).
    assert 0.05 < _std(b2(torch.randn(2, 8, 32, 32))) < 10.0


def test_sep_nlit_block_forward_backward() -> None:
    """SepNLITBlock (channels-first): forward + backward, no dead params."""
    torch.manual_seed(0)
    block = SepNLITBlock(8, 8, 8, 8)
    _assert_train_step(block, torch.randn(2, 8, 32, 32))
    b2 = SepNLITBlock(8, 16, 8, 8)
    y = b2(torch.randn(2, 8, 32, 32))
    assert y.shape == (2, 16, 32, 32) and torch.isfinite(y).all()


def test_lit_block_forward_backward() -> None:
    """LITBlock (channels-last): forward + backward, no dead params."""
    torch.manual_seed(0)
    block = LITBlock(8, 8, 8)
    x = torch.randn(2, 32, 32, 8)
    _assert_train_step(block, x)
    # separable=False path (full learned_weights) also runs and learns.
    block_full = LITBlock(8, 8, 8, separable=False)
    _assert_train_step(block_full, torch.randn(2, 32, 32, 8))


# --------------------------------------------------------------------- normalization fix


def test_sep_blocks_use_channels_first_norm() -> None:
    """The norm must be channels-first (normalize over C), not channels-last."""
    assert isinstance(SepLITBlock(8, 8, 8, 8).norm, LayerNorm2d)
    assert isinstance(SepNLITBlock(8, 8, 8, 8).norm, LayerNorm2d)


def test_it_branch_is_meaningful_and_not_suppressed() -> None:
    """The IT branch (post-norm) must be O(1), comparable to the shortcut.

    This is the regression guard for the legacy hard ``1/(H*W)`` over-suppression
    that left the transform branch ~100-400x smaller than the shortcut.
    """
    torch.manual_seed(0)
    for cls in (SepLITBlock, SepNLITBlock):
        block = cls(8, 8, 8, 8)
        x = torch.randn(2, 8, 32, 32)
        it_std = _std(_first_activation_input(block, x))
        sc_std = _shortcut_std(block, x)
        ratio = it_std / sc_std
        # The IT branch must be in the same ballpark as the shortcut (not ~1e-2).
        assert 0.1 < ratio < 10.0, f"{cls.__name__}: IT/shortcut ratio {ratio:.3f} (IT branch suppressed or exploding)"


def test_it_scale_is_learnable_and_effective() -> None:
    """``IT_scale`` must be a learnable parameter that affects the output."""
    torch.manual_seed(0)
    block = SepLITBlock(8, 8, 8, 8)
    x = torch.randn(2, 8, 32, 32, requires_grad=True)
    out = block(x)
    out.sum().backward()
    assert block.IT_scale.grad is not None and block.IT_scale.grad.abs().sum() > 0

    # Changing the scale must change the output (it is not a dead parameter).
    with torch.no_grad():
        block.IT_scale.add_(0.5)
    with torch.no_grad():
        out_changed = block(x)
    assert not torch.allclose(out, out_changed), "IT_scale has no effect on the output"


def test_resolution_invariance_of_it_branch() -> None:
    """The IT/shortcut ratio must stay in the same ballpark across resolutions."""
    torch.manual_seed(0)
    for cls in (SepLITBlock, SepNLITBlock):
        ratios = []
        for H in (16, 32, 64):
            block = cls(8, 8, 8, 8)
            x = torch.randn(2, 8, H, H)
            ratios.append(_std(_first_activation_input(block, x)) / _shortcut_std(block, x))
        # All ratios should be of the same order (no resolution blow-up or collapse).
        assert max(ratios) / min(ratios) < 10.0, f"{cls.__name__}: IT/shortcut not resolution-invariant: {ratios}"


def test_lit_block_resolution_invariance() -> None:
    """LITBlock output must stay O(1) and finite across resolutions (channels-last)."""
    torch.manual_seed(0)
    for H in (16, 32, 64):
        block = LITBlock(8, 8, 8)
        x = torch.randn(2, H, H, 8)
        with torch.no_grad():
            y = block(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert 0.05 < _std(y) < 20.0, f"LITBlock H={H}: output magnitude {_std(y):.3f} out of range"
