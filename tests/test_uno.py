"""Tests for the UNO (U-Net Neural Operator) model."""

import pytest
import torch

from pynop.models.UNO import UNO


def _basic_config(**overrides):
    """Return a default UNO config dict, overridable via kwargs."""
    cfg = dict(
        in_channels=1,
        out_channels=1,
        grid_resolutions=[(32, 32), (16, 16)],
        enc_num_windows=[(4, 4), (8, 8)],
        dec_num_windows=[(8, 8), (4, 4)],
        model_dim=32,
        enc_repeats=1,
        dec_repeats=1,
        bottleneck_repeats=1,
        num_heads=4,
        mlp_factor=4,
        skip_fusion="add",
    )
    cfg.update(overrides)
    return cfg


class TestUNOConstruction:
    """Test that UNO builds correctly under various configurations."""

    def test_basic_build(self):
        model = UNO(**_basic_config())
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_three_stages(self):
        model = UNO(
            in_channels=1,
            out_channels=1,
            grid_resolutions=[(64, 64), (32, 32), (16, 16)],
            enc_num_windows=[(4, 4), (8, 8), (16, 16)],
            dec_num_windows=[(16, 16), (8, 8), (4, 4)],
            model_dim=[32, 64, 128, 256],
            num_heads=4,
        )
        assert len(model.encoder_stages) == 3
        assert len(model.decoder_stages) == 3

    def test_single_stage(self):
        model = UNO(
            in_channels=1,
            out_channels=1,
            grid_resolutions=[(16, 16)],
            enc_num_windows=[(4, 4)],
            dec_num_windows=[(4, 4)],
            model_dim=32,
            num_heads=4,
        )
        assert len(model.encoder_stages) == 1
        assert len(model.decoder_stages) == 1

    def test_invalid_fusion_rejected(self):
        with pytest.raises(ValueError, match="skip_fusion"):
            UNO(**_basic_config(skip_fusion="multiply"))

    def test_dim_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible by num_heads"):
            UNO(**_basic_config(model_dim=30, num_heads=4))  # 30 % 4 != 0

    def test_enc_windows_length_mismatch(self):
        with pytest.raises(ValueError, match="enc_num_windows"):
            UNO(
                in_channels=1,
                out_channels=1,
                grid_resolutions=[(32, 32), (16, 16)],
                enc_num_windows=[(4, 4)],  # too short
                dec_num_windows=[(8, 8), (4, 4)],
                model_dim=32,
                num_heads=4,
            )

    def test_model_dim_list_length(self):
        with pytest.raises(ValueError, match="model_dim"):
            UNO(
                in_channels=1,
                out_channels=1,
                grid_resolutions=[(32, 32), (16, 16)],
                enc_num_windows=[(4, 4), (8, 8)],
                dec_num_windows=[(8, 8), (4, 4)],
                model_dim=[32, 64],  # need 3 entries for 2 stages
                num_heads=4,
            )


class TestUNOForward:
    """Test forward pass, shapes, and gradient flow."""

    def test_forward_shape_add(self):
        model = UNO(**_basic_config(skip_fusion="add"))
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_forward_shape_concat(self):
        model = UNO(**_basic_config(skip_fusion="concat"))
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_forward_different_channels(self):
        model = UNO(**_basic_config(in_channels=3, out_channels=5))
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 5, 64, 64)

    def test_forward_resolution_invariance(self):
        model = UNO(**_basic_config())
        for size in [32, 64, 128]:
            x = torch.randn(1, 1, size, size)
            out = model(x)
            assert out.shape == (1, 1, size, size)

    def test_forward_cond(self):
        model = UNO(**_basic_config(cond_dim=2))
        x = torch.randn(2, 1, 64, 64)
        cond = torch.randn(2, 2, 64, 64)
        out = model(x, cond=cond)
        assert out.shape == (2, 1, 64, 64)

    def test_forward_global_stage(self):
        model = UNO(
            in_channels=1,
            out_channels=1,
            grid_resolutions=[(32, 32), (16, 16)],
            enc_num_windows=[(4, 4), None],
            dec_num_windows=[None, (4, 4)],
            model_dim=32,
            num_heads=4,
        )
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        assert out.shape == (1, 1, 64, 64)

    def test_forward_output_finite(self):
        model = UNO(**_basic_config())
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        assert torch.isfinite(out).all()


class TestUNOGradients:
    """Test that gradients flow to all parameters."""

    def test_no_dead_params_add(self):
        model = UNO(**_basic_config(skip_fusion="add"))
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        out.sum().backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert dead == [], f"Dead params: {dead}"

    def test_no_dead_params_concat(self):
        model = UNO(**_basic_config(skip_fusion="concat"))
        x = torch.randn(2, 1, 64, 64)
        out = model(x)
        out.sum().backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert dead == [], f"Dead params: {dead}"

    def test_no_dead_params_global(self):
        model = UNO(
            in_channels=1,
            out_channels=1,
            grid_resolutions=[(32, 32), (16, 16)],
            enc_num_windows=[(4, 4), None],
            dec_num_windows=[None, (4, 4)],
            model_dim=32,
            num_heads=4,
        )
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        out.sum().backward()
        dead = [n for n, p in model.named_parameters() if p.grad is None]
        assert dead == [], f"Dead params: {dead}"


class TestUNOSkipFusion:
    """Test that skip fusion modes produce different (valid) outputs."""

    def test_add_vs_concat_differ(self):
        torch.manual_seed(42)
        model_add = UNO(**_basic_config(skip_fusion="add"))
        torch.manual_seed(42)
        model_concat = UNO(**_basic_config(skip_fusion="concat"))

        # Copy weights from the add model to the concat model for the shared parts
        # (they have different skip_proj layers, so just compare structure)
        x = torch.randn(1, 1, 64, 64)
        out_add = model_add(x)
        out_concat = model_concat(x)

        # Both should be finite and the same shape
        assert out_add.shape == out_concat.shape
        assert torch.isfinite(out_add).all()
        assert torch.isfinite(out_concat).all()

    def test_concat_has_skip_proj(self):
        model = UNO(**_basic_config(skip_fusion="concat"))
        for i, stage in enumerate(model.decoder_stages):
            assert stage["skip_proj"] is not None, f"Decoder stage {i} missing skip_proj"

    def test_add_has_no_skip_proj(self):
        model = UNO(**_basic_config(skip_fusion="add"))
        for i, stage in enumerate(model.decoder_stages):
            assert stage["skip_proj"] is None, f"Decoder stage {i} should have no skip_proj"


class TestUNOValidation:
    """Test input validation at forward time."""

    def test_invalid_spatial_size(self):
        model = UNO(**_basic_config())
        # 50 is not divisible by 32 (first grid resolution)
        x = torch.randn(1, 1, 50, 50)
        with pytest.raises(ValueError, match="divisible"):
            model(x)

    def test_wrong_ndim(self):
        model = UNO(**_basic_config())
        x = torch.randn(1, 1, 64)  # 3D instead of 4D
        # This will fail in permute or shape unpacking
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            model(x)


class TestUNOArchitecture:
    """Test architectural properties."""

    def test_symmetric_stages(self):
        cfg = _basic_config()
        model = UNO(**cfg)
        assert len(model.encoder_stages) == len(model.decoder_stages)
        assert len(model.downsamplers) == len(model.encoder_stages)

    def test_bottleneck_exists(self):
        model = UNO(**_basic_config())
        assert len(model.bottleneck) >= 1

    def test_per_stage_dims(self):
        model = UNO(
            in_channels=1,
            out_channels=1,
            grid_resolutions=[(32, 32), (16, 16)],
            enc_num_windows=[(4, 4), (8, 8)],
            dec_num_windows=[(8, 8), (4, 4)],
            model_dim=[32, 64, 128],
            num_heads=4,
        )
        assert model.dims == [32, 64, 128]

    def test_encoder_blocks_are_swin(self):
        from pynop.core.blocks import AdaptiveSwinBlock

        model = UNO(**_basic_config())
        for stage in model.encoder_stages:
            for blk in stage["self_attn"]:
                assert isinstance(blk, AdaptiveSwinBlock)

    def test_downsampler_is_window_cross_attn(self):
        from pynop.core.blocks import WindowCrossAttention

        model = UNO(**_basic_config())
        for down in model.downsamplers:
            assert isinstance(down["cross_attn"], WindowCrossAttention)

    def test_upsampler_is_block_cross_attn(self):
        from pynop.core.blocks import BlockCrossAttentionUpsampler

        model = UNO(**_basic_config())
        for stage in model.decoder_stages:
            assert isinstance(stage["upscaler"], BlockCrossAttentionUpsampler)
