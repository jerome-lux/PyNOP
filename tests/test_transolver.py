"""Tests for the Transolver family (``pynop/models/transolver.py``, ``pynop/core/blocks.py``).

Covers the three block modes of :class:`~pynop.models.transolver.Transolver`
(``"linear"``, ``"t3"``, ``"physics"``), the slicing / deslicing operators, and
regression tests for the bugs fixed in session §17:

- mode dispatch (the always-true ``elif "t3":`` made other modes unreachable);
- dead ``ln1`` and unused dropout module in :class:`TransolverBlock`;
- :class:`SlicingBlock` projecting *after* the weighted sum, which mishandles the
  linear bias relative to the Transolver-3 faster slicing (arXiv:2602.04940);
- ``dropout`` / ``activation`` silently ignored by the model in ``"t3"`` mode;
- :class:`TransolverBlockv3` fidelity to the Transolver-3 reference implementation
  (per-head slice weights, clamped per-head temperature, head concatenation).
"""

import pytest
import torch
import torch.nn as nn

from pynop.core.blocks import (
    DeslicingBlock,
    LinearNOBlock,
    SlicingBlock,
    TransolverBlock,
    TransolverBlockv3,
)
from pynop.models.transolver import Transolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_without_grad(model: nn.Module):
    """Names of parameters that received no gradient after ``backward()``."""
    return [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]


def _make_model(mode: str, **overrides) -> Transolver:
    """Small Transolver for fast tests (32 hidden channels, 4 heads)."""
    kwargs = dict(in_ch=2, out_ch=1, slice_num=8, n_layers=2, n_hidden=32, n_head=4, mode=mode)
    kwargs.update(overrides)
    return Transolver(**kwargs)


# ---------------------------------------------------------------------------
# Mode dispatch (regression: `elif "t3":` was always true)
# ---------------------------------------------------------------------------

def test_mode_dispatch_selects_block_class():
    assert all(isinstance(b, LinearNOBlock) for b in _make_model("linear").blocks)
    assert all(isinstance(b, TransolverBlockv3) for b in _make_model("t3").blocks)
    assert all(isinstance(b, TransolverBlock) for b in _make_model("physics").blocks)


def test_mode_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown Transolver mode"):
        _make_model("bogus")


# ---------------------------------------------------------------------------
# Forward pass / shapes (all modes, with and without conditioning)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "t3", "physics"])
def test_forward_shapes(mode):
    model = _make_model(mode).eval()
    x = torch.randn(2, 2, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 16, 16)


@pytest.mark.parametrize("mode", ["linear", "t3", "physics"])
def test_forward_with_condition(mode):
    model = _make_model(mode, cond_dim=3).eval()
    x = torch.randn(2, 2, 8, 8)
    cond = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y = model(x, cond=cond)
    assert y.shape == (2, 1, 8, 8)


# ---------------------------------------------------------------------------
# Gradient flow (regression: dead ln1 / unused dropout in TransolverBlock)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "t3", "physics"])
def test_gradient_flow_reaches_every_parameter(mode):
    model = _make_model(mode, dropout=0.1)
    y = model(torch.randn(2, 2, 16, 16))
    y.pow(2).mean().backward()
    assert _params_without_grad(model) == []


def test_transolverblock_ln1_receives_gradient():
    """ln1 must be applied before the projection (official `Attn(ln_1(x))` design)."""
    block = TransolverBlock(dim=32, heads=4, dim_head=8, slice_num=8)
    x = torch.randn(2, 64, 32, requires_grad=True)
    out = block(x)
    out.pow(2).mean().backward()
    assert block.ln1.weight.grad is not None and block.ln1.bias.grad is not None
    assert block.ln1.weight.grad.abs().sum() > 0


def test_transolverblock_dropout_is_used():
    """The dropout module must sit on the forward path (it was dead code before)."""
    block = TransolverBlock(dim=32, heads=4, dim_head=8, slice_num=8, dropout=1.0)
    # With p=1.0 every attention-output channel is zeroed: the residual stream must
    # then equal the input exactly (proj output fully dropped).
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        out = block(x)
    assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# SlicingBlock / DeslicingBlock (regression: project-before-sum ordering)
# ---------------------------------------------------------------------------

def test_slicingblock_matches_eq3_reference():
    """s_j must equal the diagonally normalized average of the *projected* input."""
    torch.manual_seed(0)
    block = SlicingBlock(in_features=8, out_features=6, num_slices=4).eval()
    x = torch.randn(2, 7, 8)

    s, w = block(x)
    assert s.shape == (2, 4, 6)
    assert w.shape == (2, 7, 4)
    # Rows of w sum to 1 (softmax over the slice dimension).
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 7), atol=1e-5)

    # Manual reference following Eq. (3): v_i = linear1(x_i) first, then weighted avg.
    temp = nn.functional.softplus(block.temp) + 0.05
    w_ref = torch.softmax(block.weight_proj(x) / temp, dim=-1)
    d = w_ref.sum(dim=1)  # [B, M]
    ref = torch.bmm(w_ref.permute(0, 2, 1), block.linear1(x)) / (d.unsqueeze(-1) + 1e-8)
    assert torch.allclose(s, ref, atol=1e-5)


def test_slicingblock_bias_differs_from_buggy_ordering():
    """Projecting after the sum scales the bias by 1/d_j — must NOT be what we do."""
    torch.manual_seed(0)
    block = SlicingBlock(in_features=8, out_features=6, num_slices=4).eval()
    x = torch.randn(2, 7, 8)

    s, w = block(x)
    d = w.sum(dim=1)
    buggy = block.linear1(torch.bmm(w.permute(0, 2, 1), x)) / (d.unsqueeze(-1) + 1e-8)
    assert not torch.allclose(s, buggy, atol=1e-4)


def test_deslicingblock_project_before_or_after_equivalent():
    """With row-stochastic w, deslice-then-project == project-then-deslice."""
    torch.manual_seed(0)
    block = DeslicingBlock(in_features=6, out_features=5).eval()
    B, N, M, D = 2, 7, 4, 6
    s = torch.randn(B, M, D)
    w = torch.softmax(torch.randn(B, N, M), dim=-1)

    with torch.no_grad():
        out = block(s, w)
        ref = block.linear3(torch.bmm(w, s))
    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# LinearNOBlock math (Eq. 8–9 of arXiv:2511.06294)
# ---------------------------------------------------------------------------

def test_linearnoblock_matches_manual_linear_attention():
    torch.manual_seed(0)
    block = LinearNOBlock(dim=16, n_heads=2, n_tokens=4).eval()
    B, N, C = 2, 9, 16
    H, M, D = 2, 4, 8
    x = torch.randn(B, N, C)

    with torch.no_grad():
        out = block(x)

        # Manual reference (asymmetric kernel normalizations + linear attention).
        x_in = block.norm1(x)
        q = block.q_proj(x_in).view(B, N, H, M).permute(0, 2, 1, 3)
        k = block.k_proj(x_in).view(B, N, H, M).permute(0, 2, 1, 3)
        v = block.v_proj(x_in).view(B, N, H, D).permute(0, 2, 1, 3)
        phi_q = torch.softmax(q, dim=-1)  # rows sum to 1 over M
        psi_k_t = torch.softmax(k.permute(0, 1, 3, 2), dim=-1)  # rows sum to 1 over N
        assert torch.allclose(phi_q.sum(dim=-1), torch.ones(B, H, N), atol=1e-5)
        assert torch.allclose(psi_k_t.sum(dim=-1), torch.ones(B, H, M), atol=1e-5)
        global_context = psi_k_t @ v  # [B, H, M, D]
        attn_out = (phi_q @ global_context).permute(0, 2, 1, 3).contiguous().view(B, N, C)
        y = x + block.out_proj(attn_out)
        ref = y + block.mlp(block.norm2(y))

    assert torch.allclose(out, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Model plumbing: dropout / activation must reach the blocks in every mode
# ---------------------------------------------------------------------------

def test_dropout_plumbing_t3():
    model = _make_model("t3", dropout=0.25)
    for block in model.blocks:
        assert block.out_dropout.p == 0.25


def test_dropout_plumbing_physics():
    model = _make_model("physics", dropout=0.15)
    for block in model.blocks:
        assert block.dropout.p == 0.15


def test_dropout_plumbing_linear():
    model = _make_model("linear", dropout=0.3)
    for block in model.blocks:
        assert block.dropout.p == 0.3


def test_activation_plumbing_t3_and_physics():
    t3 = _make_model("t3", activation=nn.SiLU)
    for block in t3.blocks:
        # MLP sublayer of the Transolver-3 block: [Linear, act, Linear] inside MLPBlock.mlp.
        assert isinstance(block.mlp.mlp[1], nn.SiLU)

    phys = _make_model("physics", activation=nn.SiLU)
    for block in phys.blocks:
        assert isinstance(block.proj_temperature[1], nn.SiLU)


# ---------------------------------------------------------------------------
# TransolverBlockv3 fidelity to the Transolver-3 reference (arXiv:2602.04940,
# lbruand-db/distributed-transolver3: PhysicsAttentionV3 + Transolver3Block)
# ---------------------------------------------------------------------------

def _t3_manual_reference(block: TransolverBlockv3, x: torch.Tensor) -> torch.Tensor:
    """Manual re-implementation of the paper's block (eval mode, no dropout)."""
    B, N, C = x.shape
    H, M, D = block.heads, block.slice_num, block.dim_head

    xn = block.ln_1(x)
    logits = block.in_project_slice(xn).reshape(B, N, H, M).permute(0, 2, 1, 3)
    temp = torch.clamp(block.temperature, min=0.1, max=5.0)
    w = torch.softmax(logits / temp, dim=-1)  # [B, H, N, M]

    d = w.sum(dim=2)  # [B, H, M]
    s_raw = torch.einsum("bhnm,bnc->bhmc", w, xn)  # [B, H, M, C]
    s = block.slice_linear1(s_raw / (d.unsqueeze(-1) + 1e-5))

    q, k, v = block.to_q(s), block.to_k(s), block.to_v(s)
    s_out = nn.functional.scaled_dot_product_attention(q, k, v)
    s_out = block.slice_linear3(s_out)

    x_attn = torch.matmul(w, s_out).permute(0, 2, 1, 3).reshape(B, N, C)
    fx = x_attn + x
    return block.mlp(block.ln_2(fx)) + fx


def test_t3block_matches_manual_reference():
    """The full forward must equal the paper's composition (per-head slicing, SDPA, head concat)."""
    torch.manual_seed(0)
    block = TransolverBlockv3(dim=16, heads=4, slice_num=5, mlp_ratio=2).eval()
    x = torch.randn(2, 9, 16)

    with torch.no_grad():
        out = block(x)
        ref = _t3_manual_reference(block, x)

    assert out.shape == (2, 9, 16)
    assert torch.allclose(out, ref, atol=1e-5)


def test_t3block_per_head_slice_weights():
    """Each head owns an independent soft assignment of the grid points to its slices."""
    torch.manual_seed(0)
    block = TransolverBlockv3(dim=16, heads=4, slice_num=5).eval()
    x = torch.randn(2, 9, 16)

    with torch.no_grad():
        w = block._slice_weights(block.ln_1(x))

    assert w.shape == (2, 4, 9, 5)
    # Rows sum to 1 over the slice dimension.
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 9), atol=1e-5)
    # With random init the per-head assignments must differ (single-set slicing would not).
    assert not torch.allclose(w[0, 0], w[0, 1])


def test_t3block_temperature_clamped():
    """The effective temperature is clamp(param, 0.1, 5) as in the reference."""
    torch.manual_seed(0)
    block = TransolverBlockv3(dim=16, heads=2, slice_num=4).eval()
    x = torch.randn(2, 9, 16)

    with torch.no_grad():
        xn = block.ln_1(x)
        logits = block.in_project_slice(xn).reshape(2, 9, 2, 4).permute(0, 2, 1, 3)
        for extreme, clamped in ((10.0, 5.0), (0.01, 0.1)):
            block.temperature.fill_(extreme)
            w = block._slice_weights(xn)
            ref = torch.softmax(logits / clamped, dim=-1)
            assert torch.allclose(w, ref, atol=1e-5)


def test_t3block_requires_dim_equals_heads_times_dim_head():
    with pytest.raises(AssertionError):
        TransolverBlockv3(dim=16, heads=4, dim_head=5)  # 4 * 5 != 16


# ---------------------------------------------------------------------------
# Resolution behavior: no state mutation, works at multiple resolutions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["linear", "t3", "physics"])
def test_no_state_mutation_across_resolutions(mode):
    model = _make_model(mode).eval()
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}

    with torch.no_grad():
        y1 = model(torch.randn(2, 2, 16, 16))
        y2 = model(torch.randn(2, 2, 24, 20))

    assert y1.shape == (2, 1, 16, 16)
    assert y2.shape == (2, 1, 24, 20)
    for n, p in model.named_parameters():
        assert torch.equal(p, snapshot[n]), f"parameter {n} mutated by forward()"
