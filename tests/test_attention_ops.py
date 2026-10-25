"""Tests for attention classes in pynop/core/ops.py and their model integration.

Covers:
- GalerkinAttention (used by LatentGalerkin)
- LinearAttentionELU (Performer-style)
- LinearAttention (normalized factorization)
- Attention (standard SDPA, used by TransformerBlock)
- ComplexAttention (Hermitian product)
- TransformerBlock integration (with/without RoPE)
- Model compatibility: LatentNO, ITLNO, LatentGalerkin, LatentTransolver, SwinNO, UNO
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


# =============================================================================
# GalerkinAttention
# =============================================================================


class TestGalerkinAttention:
    """GalerkinAttention: linear integral operator (no softmax)."""

    def test_self_attention_shape(self):
        """Self-attention: output shape matches input."""
        B, M, d = 2, 16, 32
        att = nn.Sequential(nn.Flatten(2), nn.Identity())  # not used, just placeholder
        from pynop.core.ops import GalerkinAttention
        att = GalerkinAttention(d, heads=4)
        x = torch.randn(B, M, d)
        y = att(x)
        assert y.shape == (B, M, d)

    def test_cross_attention_shape(self):
        """Cross-attention: query length differs from context length."""
        B, M, N, d = 2, 8, 16, 32
        from pynop.core.ops import GalerkinAttention
        att = GalerkinAttention(d, heads=4)
        x = torch.randn(B, M, d)
        context = torch.randn(B, N, d)
        y = att(x, context=context)
        assert y.shape == (B, M, d)

    def test_gradient_flow(self):
        """Gradients flow to all parameters."""
        B, M, d = 2, 8, 32
        from pynop.core.ops import GalerkinAttention
        att = GalerkinAttention(d, heads=4)
        x = torch.randn(B, M, d, requires_grad=True)
        y = att(x)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        for name, p in att.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"

    def test_kv_normalization(self):
        """With kv_normalization=True, shapes and gradients still work."""
        B, M, d = 2, 8, 32
        from pynop.core.ops import GalerkinAttention
        att = GalerkinAttention(d, heads=4, kv_normalization=True)
        x = torch.randn(B, M, d) * 10  # large magnitude to test normalization
        y = att(x)
        assert y.shape == (B, M, d)
        assert torch.isfinite(y).all()


# =============================================================================
# LinearAttentionELU
# =============================================================================


class TestLinearAttentionELU:
    """LinearAttentionELU: Performer-style kernelized attention."""

    def test_shape(self):
        """Output shape matches query shape."""
        B, N_q, N_kv, d = 2, 16, 8, 32
        from pynop.core.ops import LinearAttentionELU
        att = LinearAttentionELU(d, n_heads=4)
        q = torch.randn(B, N_q, d)
        kv = torch.randn(B, N_kv, d)
        y = att(q, kv)
        assert y.shape == (B, N_q, d)

    def test_self_attention(self):
        """Self-attention (query == key/value)."""
        B, N, d = 2, 16, 32
        from pynop.core.ops import LinearAttentionELU
        att = LinearAttentionELU(d, n_heads=4)
        x = torch.randn(B, N, d)
        y = att(x, x)
        assert y.shape == (B, N, d)

    def test_gradient_flow(self):
        """Gradients flow to all parameters."""
        B, N_q, N_kv, d = 2, 8, 4, 32
        from pynop.core.ops import LinearAttentionELU
        att = LinearAttentionELU(d, n_heads=4)
        q = torch.randn(B, N_q, d, requires_grad=True)
        kv = torch.randn(B, N_kv, d, requires_grad=True)
        y = att(q, kv)
        loss = (y**2).mean()
        loss.backward()
        assert q.grad is not None
        assert kv.grad is not None
        for name, p in att.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"


# =============================================================================
# LinearAttention
# =============================================================================


class TestLinearAttention:
    """LinearAttention: normalized Q/K factorization."""

    def test_self_attention_shape(self):
        """Self-attention: context defaults to x."""
        B, N, d = 2, 16, 32
        from pynop.core.ops import LinearAttention
        att = LinearAttention(d, num_heads=4)
        x = torch.randn(B, N, d)
        y = att(x)
        assert y.shape == (B, N, d)

    def test_cross_attention_shape(self):
        """Cross-attention: context length differs from query length."""
        B, N_q, N_kv, d = 2, 16, 8, 32
        from pynop.core.ops import LinearAttention
        att = LinearAttention(d, num_heads=4)
        q = torch.randn(B, N_q, d)
        kv = torch.randn(B, N_kv, d)
        y = att(q, context=kv)
        assert y.shape == (B, N_q, d)

    def test_gradient_flow(self):
        """Gradients flow to all parameters."""
        B, N, d = 2, 8, 32
        from pynop.core.ops import LinearAttention
        att = LinearAttention(d, num_heads=4)
        x = torch.randn(B, N, d, requires_grad=True)
        y = att(x)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        for name, p in att.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


# =============================================================================
# Attention (standard SDPA — used by TransformerBlock)
# =============================================================================


class TestAttention:
    """Standard scaled dot-product attention with optional RoPE."""

    def test_self_attention_shape(self):
        """Self-attention: Q=K=V."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8)
        x = torch.randn(B, N, d)
        y = att(x, x, x)
        assert y.shape == (B, N, d)

    def test_cross_attention_shape(self):
        """Cross-attention: different Q/K/V lengths."""
        B, N_q, N_kv, d = 2, 16, 8, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8)
        q = torch.randn(B, N_q, d)
        kv = torch.randn(B, N_kv, d)
        y = att(q, kv, kv)
        assert y.shape == (B, N_q, d)

    def test_causal_mask(self):
        """Causal attention produces finite output."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8)
        x = torch.randn(B, N, d)
        y = att(x, x, x, causal_attn=True)
        assert torch.isfinite(y).all()

    def test_mask(self):
        """Additive mask works."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8)
        x = torch.randn(B, N, d)
        # Additive mask: 0 = attend, -inf = block
        mask = torch.full((1, 1, N, N), float("-inf"))
        mask = mask.tril().float()
        y = att(x, x, x, mask=mask)
        assert torch.isfinite(y).all()

    def test_rope_enabled(self):
        """RoPE enabled at init, coords passed at forward."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8, rope=True)
        assert att.rope is not None
        x = torch.randn(B, N, d)
        # coords: [B, N, 2] absolute in [-1, 1]
        coords = torch.rand(B, N, 2) * 2 - 1
        y = att(x, x, x, coords=coords)
        assert y.shape == (B, N, d)
        assert torch.isfinite(y).all()

    def test_rope_disabled(self):
        """RoPE disabled: coords are ignored."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8, rope=False)
        assert att.rope is None
        x = torch.randn(B, N, d)
        coords = torch.rand(B, N, 2) * 2 - 1
        y = att(x, x, x, coords=coords)
        assert y.shape == (B, N, d)

    def test_rope_changes_output(self):
        """RoPE should change the output vs no RoPE (with same init)."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention

        torch.manual_seed(42)
        att_no_rope = Attention(d, d, num_heads=8, rope=False)
        torch.manual_seed(42)
        att_rope = Attention(d, d, num_heads=8, rope=True)

        # Copy weights to make them identical
        att_rope.load_state_dict(att_no_rope.state_dict())

        x = torch.randn(B, N, d)
        coords = torch.rand(B, N, 2) * 2 - 1

        y_no_rope = att_no_rope(x, x, x)
        y_rope = att_rope(x, x, x, coords=coords)

        # With RoPE and non-zero coords, outputs should differ
        diff = (y_rope - y_no_rope).abs().mean()
        assert diff > 1e-6, f"RoPE had no effect (diff={diff:.2e})"

    def test_rope_head_dim_validation(self):
        """RoPE requires head_dim % 4 == 0."""
        from pynop.core.ops import Attention
        # head_dim = 28 // 4 = 7, which is NOT divisible by 4
        with pytest.raises(ValueError, match="divisible by 4"):
            Attention(28, 28, num_heads=4, rope=True)

    def test_gradient_flow(self):
        """Gradients flow to all parameters (self-attention)."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8)
        x = torch.randn(B, N, d, requires_grad=True)
        y = att(x, x, x)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        for name, p in att.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"

    def test_gradient_flow_rope(self):
        """Gradients flow through RoPE to coords and all params."""
        B, N, d = 2, 16, 64
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=8, rope=True)
        x = torch.randn(B, N, d, requires_grad=True)
        coords = (torch.rand(B, N, 2) * 2 - 1).requires_grad_(True)
        y = att(x, x, x, coords=coords)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        assert coords.grad is not None
        for name, p in att.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"

    def test_sdpa_equivalence(self):
        """Verify against manual SDPA computation (no RoPE, no dropout)."""
        B, N, d = 2, 8, 32
        n_heads = 4
        head_dim = d // n_heads
        from pynop.core.ops import Attention
        att = Attention(d, d, num_heads=n_heads)
        x = torch.randn(B, N, d)

        y = att(x, x, x)

        # Manual computation
        with torch.no_grad():
            q = att.wq(x).view(B, N, n_heads, head_dim).transpose(1, 2)
            k = att.wk(x).view(B, N, n_heads, head_dim).transpose(1, 2)
            v = att.wv(x).view(B, N, n_heads, head_dim).transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
            weights = F.softmax(scores, dim=-1)
            out = (weights @ v).transpose(1, 2).reshape(B, N, d)
            expected = att.out_proj(out)

        diff = (y - expected).abs().max()
        assert diff < 1e-5, f"SDPA mismatch: {diff}"

    def test_in_ch_differs_from_out_ch(self):
        """in_ch != out_ch (cross-attention with projection)."""
        B, N_q, N_kv = 2, 16, 8
        in_ch, out_ch = 32, 64
        from pynop.core.ops import Attention
        att = Attention(in_ch, out_ch, num_heads=8)
        q = torch.randn(B, N_q, in_ch)
        kv = torch.randn(B, N_kv, in_ch)
        y = att(q, kv, kv)
        assert y.shape == (B, N_q, out_ch)

    def test_invalid_head_dim(self):
        """out_ch not divisible by num_heads raises ValueError."""
        from pynop.core.ops import Attention
        with pytest.raises(ValueError):
            Attention(64, 64, num_heads=7)


# =============================================================================
# ComplexAttention
# =============================================================================


class TestComplexAttention:
    """Complex-valued attention with Hermitian inner product."""

    def test_shape(self):
        """Output shape matches input (complex dtype)."""
        B, N, d = 2, 16, 32
        from pynop.core.ops import ComplexAttention
        att = ComplexAttention(d, d, num_heads=8)
        x = torch.randn(B, N, d) + 1j * torch.randn(B, N, d)
        y = att(x, x, x)
        assert y.shape == (B, N, d)
        assert y.is_complex()

    def test_finite_output(self):
        """Output is finite (no NaN/Inf)."""
        B, N, d = 2, 8, 32
        from pynop.core.ops import ComplexAttention
        att = ComplexAttention(d, d, num_heads=8)
        x = (torch.randn(B, N, d) + 1j * torch.randn(B, N, d)) * 0.5
        y = att(x, x, x)
        assert torch.isfinite(y.real).all()
        assert torch.isfinite(y.imag).all()

    def test_gradient_flow(self):
        """Gradients flow to all parameters."""
        B, N, d = 2, 8, 32
        from pynop.core.ops import ComplexAttention
        att = ComplexAttention(d, d, num_heads=8)
        x = (torch.randn(B, N, d) + 1j * torch.randn(B, N, d)) * 0.5
        x.requires_grad_(True)
        y = att(x, x, x)
        loss = (torch.abs(y) ** 2).mean()
        loss.backward()
        assert x.grad is not None
        for name, p in att.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing grad for {name}"


# =============================================================================
# TransformerBlock integration
# =============================================================================


class TestTransformerBlock:
    """TransformerBlock: pre-LN residual with optional RoPE."""

    def test_forward_no_rope(self):
        """Default forward (no RoPE)."""
        B, N, d = 2, 16, 32
        from pynop.core.blocks import TransformerBlock
        block = TransformerBlock(d, n_heads=4)
        x = torch.randn(B, N, d)
        y = block(x)
        assert y.shape == (B, N, d)
        assert torch.isfinite(y).all()

    def test_forward_with_rope(self):
        """Forward with RoPE enabled and coords provided."""
        B, N, d = 2, 16, 32
        from pynop.core.blocks import TransformerBlock
        block = TransformerBlock(d, n_heads=4, rope=True)
        x = torch.randn(B, N, d)
        coords = torch.rand(B, N, 2) * 2 - 1
        y = block(x, coords=coords)
        assert y.shape == (B, N, d)
        assert torch.isfinite(y).all()

    def test_rope_property(self):
        """block.rope is not None when rope=True, None when rope=False."""
        from pynop.core.blocks import TransformerBlock
        block_rope = TransformerBlock(32, n_heads=4, rope=True)
        assert block_rope.rope is not None
        block_plain = TransformerBlock(32, n_heads=4)
        assert block_plain.rope is None

    def test_gradient_flow(self):
        """Gradients flow through the full block."""
        B, N, d = 2, 16, 32
        from pynop.core.blocks import TransformerBlock
        block = TransformerBlock(d, n_heads=4)
        x = torch.randn(B, N, d, requires_grad=True)
        y = block(x)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        for name, p in block.named_parameters():
            assert p.grad is not None, f"Missing grad for {name}"

    def test_gradient_flow_rope(self):
        """Gradients flow through RoPE when enabled."""
        B, N, d = 2, 16, 32
        from pynop.core.blocks import TransformerBlock
        block = TransformerBlock(d, n_heads=4, rope=True)
        x = torch.randn(B, N, d, requires_grad=True)
        coords = (torch.rand(B, N, 2) * 2 - 1).requires_grad_(True)
        y = block(x, coords=coords)
        loss = (y**2).mean()
        loss.backward()
        assert x.grad is not None
        assert coords.grad is not None

    def test_rope_head_dim_validation(self):
        """RoPE requires dim // n_heads % 4 == 0."""
        from pynop.core.blocks import TransformerBlock
        # head_dim = 28 // 4 = 7, which is NOT divisible by 4
        with pytest.raises(ValueError, match="divisible by 4"):
            TransformerBlock(28, n_heads=4, rope=True)

    def test_residual_structure(self):
        """Verify the residual: output != input (learned transformation)."""
        B, N, d = 1, 8, 32
        from pynop.core.blocks import TransformerBlock
        block = TransformerBlock(d, n_heads=4)
        x = torch.randn(B, N, d)
        y = block(x)
        # After initialization, the block should produce a different output
        assert not torch.allclose(x, y)


# =============================================================================
# Model compatibility tests
# =============================================================================


class TestModelCompatibility:
    """Verify attention classes work correctly within full models."""

    def _check_model(self, model, x_in, name, allowed_dead=()):
        """Helper: forward + backward + no NaN + optional dead param check."""
        model.train()
        y = model(x_in)
        assert torch.isfinite(y).all(), f"{name}: output contains NaN/Inf"
        loss = (y**2).mean()
        loss.backward()
        dead = []
        for pname, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None:
                    dead.append(pname)
                elif torch.isnan(p.grad).any():
                    raise AssertionError(f"{name}: NaN grad for {pname}")
        # Filter out known-dead params
        dead = [p for p in dead if not any(a in p for a in allowed_dead)]
        assert not dead, f"{name}: dead params: {dead}"

    def test_latentno(self):
        """LatentNO uses TransformerBlock → Attention."""
        from pynop.models.LNO import LatentNO
        model = LatentNO(in_channels=1, out_channels=1, d_model=32, modes=8, num_heads=4, num_blocks=2)
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "LatentNO")

    def test_itlno(self):
        """ITLNO uses TransformerBlock → Attention."""
        from pynop.models.LNO import ITLNO
        model = ITLNO(in_channels=1, out_channels=1, modes=(4, 4), hidden_channels=32, num_blocks=2, num_heads=4)
        x = torch.randn(1, 1, 16, 16)
        # ITLNO has expected dead params: branch_generator (linear kernel mode)
        self._check_model(model, x, "ITLNO", allowed_dead=("branch_generator", "cond_embedding"))

    def test_latentgalerkin_galerkin(self):
        """LatentGalerkin with latent_attention='galerkin' → GalerkinTransformerBlock."""
        from pynop.models.LNO import LatentGalerkin
        model = LatentGalerkin(
            in_channels=1, out_channels=1, modes=8, hidden_channels=32,
            num_blocks=2, num_heads=4, latent_attention="galerkin",
        )
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "LatentGalerkin (galerkin)")

    def test_latentgalerkin_transformer(self):
        """LatentGalerkin with latent_attention='vanilla' → TransformerBlock (standard Attention)."""
        from pynop.models.LNO import LatentGalerkin
        model = LatentGalerkin(
            in_channels=1, out_channels=1, modes=8, hidden_channels=32,
            num_blocks=2, num_heads=4, latent_attention="vanilla",
        )
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "LatentGalerkin (transformer)")

    def test_latenttransolver_transformer(self):
        """LatentTransolver with latent_attention='vanilla' (default) → TransformerBlock."""
        from pynop.models.LNO import LatentTransolver
        model = LatentTransolver(
            in_channels=1, out_channels=1, modes=8, hidden_channels=32,
            num_blocks=2, num_heads=4, latent_attention="vanilla",
        )
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "LatentTransolver (transformer)")

    def test_latenttransolver_galerkin(self):
        """LatentTransolver with latent_attention='galerkin' → GalerkinTransformerBlock."""
        from pynop.models.LNO import LatentTransolver
        model = LatentTransolver(
            in_channels=1, out_channels=1, modes=8, hidden_channels=32,
            num_blocks=2, num_heads=4, latent_attention="galerkin",
        )
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "LatentTransolver (galerkin)")

    def test_swinno(self):
        """SwinNO uses WindowAttention (blocks.py) — verify it still works."""
        from pynop.models.SWIN import SwinNO
        model = SwinNO(num_windows=4, in_channels=1, out_channels=1, latent_dim=32, num_blocks=2, num_heads=4)
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "SwinNO")

    def test_uno(self):
        """UNO uses TransformerBlock (bottleneck) + Swin blocks."""
        from pynop.models.UNO import UNO
        model = UNO(
            in_channels=1,
            out_channels=1,
            model_dim=32,
            enc_num_windows=[(4, 4)],
            dec_num_windows=[(4, 4)],
            enc_repeats=1,
            dec_repeats=1,
            bottleneck_repeats=1,
            num_heads=4,
        )
        x = torch.randn(1, 1, 16, 16)
        self._check_model(model, x, "UNO")
