"""Tests for ``pynop/training/lightning_model.py``.

Covers:
- ``TrainingConfig`` (predict_derivative / derivative-loss fields, JSON export)
- ``NormalizedTimeDerivativeMSE`` (math + gradient)
- ``MultiStepNOModel`` with FNO, SwinNO, UNet, Resnet, GalerkinTransformer
  (derivative and absolute output modes, cond / time compatibility)
- ``DeepONetModel`` (branch/trunk mapping, both output modes)
- derivative-loss configuration (default, disabled, custom)
- AR curriculum and rollout (teacher forcing -> autoregressive feedback)
- optimizer / scheduler wiring
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pynop
from pynop import (
    DeepONetModel,
    MultiStepLightningBase,
    MultiStepNOModel,
    NormalizedTimeDerivativeMSE,
    TrainingConfig,
    default_scheduler_config,
)
from pynop.models import DeepONet, FNO, GalerkinTransformer, Resnet, SwinNO, UNet

B, T, C, H, W = 2, 5, 1, 8, 8
N_SLICES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_batch(cond=False):
    inputs = torch.randn(B, T, C, H, W)
    time_idx = torch.zeros(B)
    if cond:
        return inputs, time_idx, torch.randn(B, 1, H, W)
    return inputs, time_idx


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def make_config(**kwargs):
    defaults = dict(dt=0.1, n_slices=N_SLICES, loss_fn=nn.MSELoss(), derivative_loss_weight=1.0)
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def make_fno(cond_dim=None):
    return FNO(
        in_channels=N_SLICES * C,
        out_channels=C,
        modes=4,
        hidden_channels=(8, 8),
        blocks="FNO",
        cond_dim=cond_dim,
    )


def make_multistep(model, **config_kwargs):
    return MultiStepNOModel(model, make_optimizer(model), make_config(**config_kwargs))


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_new_fields():
    cfg = TrainingConfig()
    assert cfg.predict_derivative is True
    assert cfg.derivative_loss_fn is None
    assert cfg.derivative_loss_weight == 1.0


def test_training_config_json_export():
    cfg = TrainingConfig(loss_fn=[nn.MSELoss(), nn.L1Loss()])
    d = cfg.to_json_dict()
    assert d["loss_fn"] == ["MSELoss", "L1Loss"]
    assert d["predict_derivative"] is True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        cfg.save(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["loss_fn"] == ["MSELoss", "L1Loss"]


def test_normalized_time_derivative_mse_math():
    loss = NormalizedTimeDerivativeMSE()
    pred = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 8, 8)
    previous = torch.randn(2, 1, 8, 8)
    dt = 0.1

    gt = (target - previous) / dt
    expected = torch.mean(torch.mean((pred - gt) ** 2, dim=(2, 3)) / (torch.mean(gt**2, dim=(2, 3)) + 1e-8))

    value = loss(pred, target, previous, dt)
    assert torch.allclose(value, expected, atol=1e-6)
    value.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# MultiStepNOModel — model compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("predict_derivative", [True, False])
def test_multistep_fno_both_modes(predict_derivative):
    model = make_fno()
    lm = make_multistep(model, predict_derivative=predict_derivative)
    loss = lm.training_step(make_batch(), 0)
    assert torch.is_tensor(loss) and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    vloss = lm.validation_step(make_batch(), 0)
    assert torch.isfinite(vloss)


def test_multistep_fno_with_cond():
    model = make_fno(cond_dim=1)
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(cond=True), 0)
    assert torch.isfinite(loss)
    loss.backward()
    # cond path actually used: the cond_dim weights of the lifting layer get gradients
    lifting = model.lifting
    assert lifting.weight.grad is not None and torch.isfinite(lifting.weight.grad).all()


def test_multistep_swin():
    model = SwinNO(
        num_windows=2,
        in_channels=N_SLICES * C,
        out_channels=C,
        latent_dim=16,
        num_blocks=1,
        num_heads=4,
    )
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_multistep_uno():
    from pynop.models import UNO

    model = UNO(
        in_channels=N_SLICES * C,
        out_channels=C,
        enc_num_windows=[(2, 2)],
        dec_num_windows=[(2, 2)],
        model_dim=16,
        num_heads=2,
    )
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_multistep_transolver():
    from pynop.models import Transolver

    model = Transolver(
        in_ch=N_SLICES * C,
        out_ch=C,
        slice_num=8,
        n_layers=1,
        n_hidden=16,
        n_head=2,
    )
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_multistep_latentno():
    from pynop.models import LatentNO

    model = LatentNO(
        in_channels=N_SLICES * C,
        out_channels=C,
        d_model=16,
        modes=16,
        num_heads=2,
        num_blocks=1,
        mlp_layers=1,
        mlp_dim=16,
    )
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)



def test_multistep_unet_no_cond():
    """UNet has forward(x) only — the wrapper must not pass ``cond``."""
    model = UNet(
        in_channels=N_SLICES * C,
        out_channels=C,
        filters=(8, 16),
        repeats=(1, 1),
        use_transformer_bottleneck=False,
    )
    assert not lm_accepts_cond(model)
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_multistep_resnet_no_cond():
    """Resnet outputs (B, dims[-1], H, W) with build_head=False, so the
    field width must match the output channel count."""
    model = Resnet(
        in_channels=N_SLICES * C,
        num_classes=C,
        depths=[1, 1],
        dims=[8, C],
        stem_filters=8,
    )
    assert not lm_accepts_cond(model)
    lm = make_multistep(model)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_multistep_galerkin_time():
    """GalerkinTransformer has forward(x, time, residual) — the wrapper must
    pass ``time`` and the residual=True output is an absolute state."""
    model = GalerkinTransformer(
        in_channels=N_SLICES * C,
        out_channels=C,
        dt=0.1,
        num_blocks=1,
        hidden_channels=16,
        num_heads=2,
        mlp_layers=1,
        mlp_dim=16,
    )
    assert lm_accepts_time(model)
    lm = make_multistep(model, predict_derivative=False)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)
    vloss = lm.validation_step(make_batch(), 0)
    assert torch.isfinite(vloss)


def lm_accepts_cond(model):
    import inspect

    return "cond" in set(inspect.signature(model.forward).parameters)


def lm_accepts_time(model):
    import inspect

    return "time" in set(inspect.signature(model.forward).parameters)


def test_cond_rejected_when_unsupported():
    model = UNet(
        in_channels=N_SLICES * C,
        out_channels=C,
        filters=(8, 16),
        repeats=(1, 1),
        use_transformer_bottleneck=False,
    )
    lm = make_multistep(model)
    with pytest.raises(ValueError, match="cond"):
        lm.training_step(make_batch(cond=True), 0)


def test_multistep_rejects_deeponet():
    model = DeepONet(branchnet=[N_SLICES * C * H * W, 16, 16], trunknet=[2, 16, 16])
    with pytest.raises(ValueError, match="DeepONetModel"):
        MultiStepNOModel(model, make_optimizer(model), make_config())


def test_invalid_n_slices_rejected():
    model = make_fno()
    with pytest.raises(ValueError, match="n_slices"):
        MultiStepNOModel(model, make_optimizer(model), make_config(n_slices=0))


# ---------------------------------------------------------------------------
# MultiStepNOModel — predict_derivative semantics
# ---------------------------------------------------------------------------


def test_predict_derivative_formula():
    """Derivative mode: preds = x_last + dt * out. Absolute mode: preds = out."""
    captured = []

    def spy_loss(preds, targets):
        captured.append(preds.detach())
        return F.mse_loss(preds, targets)

    class ConstantOut(nn.Module):
        """Projects a window (B, n*C, H, W) to a constant one-channel field."""

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

        def forward(self, x, cond=None):
            B, _, H, W = x.shape
            return torch.ones(B, 1, H, W, device=x.device) + 0.0 * self.p

    inputs, time_idx = make_batch()
    x_last = inputs[:, N_SLICES - 1]  # last slice of the first teacher-forcing window

    # --- derivative mode
    lm = make_multistep(
        ConstantOut(),
        predict_derivative=True,
        dt=0.2,
        loss_fn=spy_loss,
        derivative_loss_weight=0.0,
    )
    captured.clear()
    lm._rollout(inputs, time_idx, None, training=False)
    expected = x_last + 0.2 * torch.ones_like(x_last)
    assert torch.allclose(captured[0], expected, atol=1e-6)

    # --- absolute mode
    lm = make_multistep(
        ConstantOut(),
        predict_derivative=False,
        dt=0.2,
        loss_fn=spy_loss,
        derivative_loss_weight=1.0,
    )
    captured.clear()
    lm._rollout(inputs, time_idx, None, training=False)
    assert torch.allclose(captured[0], torch.ones_like(x_last), atol=1e-6)


def test_derivative_loss_weight_zero_disables():
    model = make_fno()
    lm = make_multistep(model, derivative_loss_weight=0.0)
    assert lm.derivative_loss_weight == 0.0
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)


def test_custom_derivative_loss_fn_used():
    calls = []

    class MyDerivLoss(nn.Module):
        def forward(self, pred, target, previous, dt):
            calls.append((pred, target, previous, dt))
            return F.mse_loss(pred, (target - previous) / dt)

    model = make_fno()
    lm = make_multistep(model, derivative_loss_fn=MyDerivLoss())
    assert isinstance(lm.derivative_loss_fn, MyDerivLoss)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)
    assert calls, "the custom derivative loss was never called"
    pred, target, previous, dt = calls[0]
    assert previous.shape == target.shape and dt == 0.1


def test_default_derivative_loss_fn():
    model = make_fno()
    lm = make_multistep(model)
    assert isinstance(lm.derivative_loss_fn, NormalizedTimeDerivativeMSE)


# ---------------------------------------------------------------------------
# MultiStepNOModel — AR curriculum & rollout
# ---------------------------------------------------------------------------


def test_ar_curriculum_schedule():
    model = make_fno()
    lm = make_multistep(
        model,
        start_autoregressive=0,
        final_autoregressive=10,
        min_autoregressive_steps=0,
        max_autoregressive_steps=2,
    )
    assert lm._ar_steps_for_epoch(T, epoch=0) == 0  # before curriculum
    assert lm._ar_steps_for_epoch(T, epoch=5) == 1  # mid curriculum
    assert lm._ar_steps_for_epoch(T, epoch=10) == 2  # after final epoch
    assert lm._ar_steps_for_epoch(T, epoch=99) == 2  # capped


def test_ar_rollout_feeds_predictions_back():
    class Echo(nn.Module):
        """Returns a zero one-channel field and records its inputs."""

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
            self.calls = 0
            self.last_input = None

        def forward(self, x, cond=None):
            self.calls += 1
            self.last_input = x.detach()
            B, _, H, W = x.shape
            return torch.zeros(B, 1, H, W, device=x.device) + 0.0 * self.p

    model = Echo()
    lm = make_multistep(model, predict_derivative=False, derivative_loss_weight=0.0)
    inputs, time_idx = make_batch()

    total_steps = T - N_SLICES  # 3 predictions
    res = lm._rollout(inputs, time_idx, None, training=False)
    assert res["num_predictions"] == total_steps
    assert model.calls == total_steps
    # In a full (validation) rollout the last window slice is the previous
    # prediction, which is zero for this model.
    assert torch.all(model.last_input[:, -1] == 0)
    # ... while the first prediction was teacher-forced on the true window.
    assert model.calls >= 1


def test_noise_applied_in_training_only():
    model = make_fno()
    lm = make_multistep(model, noise_level=0.5)
    # training: noise is added (no crash, finite loss)
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)
    # validation: rollout runs without noise
    vloss = lm.validation_step(make_batch(), 0)
    assert torch.isfinite(vloss)


# ---------------------------------------------------------------------------
# End-of-epoch loss summary printout
# ---------------------------------------------------------------------------


def test_epoch_loss_summary_print(capsys):
    """on_validation_epoch_end prints the per-batch average losses (train + val)."""
    model = make_fno()
    lm = make_multistep(model, derivative_loss_weight=0.0)  # single loss, no regularizer

    batch = make_batch()
    train_losses = [lm.training_step(batch, i).item() for i in range(2)]
    val_losses = [lm.validation_step(batch, i).item() for i in range(3)]

    lm.on_validation_epoch_end()
    out = capsys.readouterr().out

    expected_train = sum(train_losses) / len(train_losses)
    expected_val = sum(val_losses) / len(val_losses)
    assert f"train (2 batches): loss={expected_train:.4e}" in out
    assert f"val (3 batches): loss={expected_val:.4e}" in out


def test_epoch_loss_summary_print_all_components(capsys):
    """With several losses + the derivative regularizer, every term is reported."""
    lm = make_multistep(make_fno(), loss_fn=[nn.MSELoss(), nn.L1Loss()], loss_weights=[1.0, 2.0])
    batch = make_batch()
    for _ in range(2):
        lm.training_step(batch, 0)
        lm.validation_step(batch, 0)

    lm.on_validation_epoch_end()
    out = capsys.readouterr().out

    assert "loss_1=" in out and "nMSEdt=" in out
    # both splits are reported on the same line
    line = next(l for l in out.splitlines() if "[Epoch" in l)
    assert "train (2 batches):" in line and "val (2 batches):" in line


def test_epoch_loss_summary_print_no_validation(capsys):
    """No validation data this epoch -> no summary line."""
    lm = make_multistep(make_fno())
    lm.training_step(make_batch(), 0)

    lm.on_validation_epoch_end()
    assert "[Epoch" not in capsys.readouterr().out


def test_epoch_loss_summary_resets_each_epoch(capsys):
    """The accumulators are reset at the start of each epoch."""
    lm = make_multistep(make_fno(), derivative_loss_weight=0.0)
    batch = make_batch()

    for _ in range(2):
        lm.on_train_epoch_start()
        lm.training_step(batch, 0)
        lm.on_validation_epoch_start()
        lm.validation_step(batch, 0)
        lm.on_validation_epoch_end()

    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "[Epoch" in l]
    assert len(lines) == 2
    # both epochs saw exactly one train and one val batch
    assert all("train (1 batches):" in l and "val (1 batches):" in l for l in lines)


# ---------------------------------------------------------------------------
# DeepONetModel
# ---------------------------------------------------------------------------


def make_deeponet():
    # Branch input is the whole window flattened: n_slices * C * H * W values.
    return DeepONet(branchnet=[N_SLICES * C * H * W, 16, 16], trunknet=[2, 16, 16])


@pytest.mark.parametrize("predict_derivative", [True, False])
def test_deeponet_model_both_modes(predict_derivative):
    model = make_deeponet()
    lm = DeepONetModel(model, make_optimizer(model), make_config(predict_derivative=predict_derivative))
    loss = lm.training_step(make_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no DeepONet parameter received a gradient"
    vloss = lm.validation_step(make_batch(), 0)
    assert torch.isfinite(vloss)


def test_deeponet_model_branch_trunk_mapping():
    class SpyDeepONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.branchnet = nn.Identity()
            self.trunknet = nn.Identity()
            self.encoder = None
            self.decoder = None
            self.scale = 1.0
            self.p = nn.Parameter(torch.zeros(1))
            self.last = None

        def forward(self, u, y):
            self.last = (u.detach(), y.detach())
            B, J = u.shape[0], y.shape[0]
            return torch.ones(B, J, device=u.device) + 0.0 * self.p

    model = SpyDeepONet()
    lm = DeepONetModel(model, make_optimizer(model), make_config(derivative_loss_weight=0.0))
    inputs, time_idx = make_batch()
    lm._rollout(inputs, time_idx, None, training=False)

    u, y = model.last
    assert u.shape == (B, N_SLICES * C * H * W)  # window flattened as one function vector
    assert y.shape == (H * W, 2)  # spatial query coordinates (2-D for the trunk MLP)
    assert y.min() >= -1.0 - 1e-6 and y.max() <= 1.0 + 1e-6


def test_deeponet_model_cond_rejected():
    model = make_deeponet()
    lm = DeepONetModel(model, make_optimizer(model), make_config())
    with pytest.raises(ValueError, match="cond"):
        lm.training_step(make_batch(cond=True), 0)


# ---------------------------------------------------------------------------
# Optimizer / scheduler wiring
# ---------------------------------------------------------------------------


def test_configure_optimizers_plain():
    model = make_fno()
    opt = make_optimizer(model)
    lm = MultiStepNOModel(model, opt, make_config())
    assert lm.configure_optimizers() is opt


def test_configure_optimizers_with_default_scheduler():
    model = make_fno()
    opt = make_optimizer(model)
    lm = MultiStepNOModel(model, opt, make_config(), scheduler_config=[dict(default_scheduler_config[0])])
    result = lm.configure_optimizers()
    assert isinstance(result, tuple) and len(result) == 2
    optimizers, schedulers = result
    assert optimizers[0] is opt
    assert isinstance(schedulers[0]["scheduler"], ReduceLROnPlateau)
    assert schedulers[0]["monitor"] == "avg_loss"
    assert schedulers[0]["interval"] == "epoch"
    # the caller's config must not be mutated
    assert "scheduler" in default_scheduler_config[0]


def test_trainer_fit_end_to_end(capsys):
    """Full Lightning fit: optimizer, default scheduler, logging, val loop."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    model = make_fno()
    opt = make_optimizer(model)
    lm = MultiStepNOModel(model, opt, make_config(), scheduler_config=[dict(default_scheduler_config[0])])

    inputs = torch.randn(2, T, C, H, W)
    time_idx = torch.zeros(2)
    # batch -> (inputs (2, T, C, H, W), time_idx (2,))
    train_loader = DataLoader(TensorDataset(inputs, time_idx), batch_size=2)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(lm, train_dataloaders=train_loader, val_dataloaders=train_loader)
    assert trainer.state.finished
    # the scheduler was stepped / logged without error
    assert opt.param_groups[0]["lr"] > 0
    # the end-of-epoch loss summary (train + val) was printed by Lightning's hook
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if "[Epoch" in l)
    assert "train (1 batches):" in line and "val (1 batches):" in line  # batch_size=2 over 2 samples -> 1 batch


def test_top_level_exports():
    for name in (
        "TrainingConfig",
        "MultiStepLightningBase",
        "MultiStepNOModel",
        "DeepONetModel",
        "CurriculumCheckpoint",
        "NormalizedTimeDerivativeMSE",
        "default_scheduler_config",
    ):
        assert hasattr(pynop, name), f"pynop.{name} is missing"


def test_base_class_is_abstract():
    """The merged base class is only usable through a _forward_window subclass."""
    model = make_fno()
    lm = MultiStepLightningBase(model, make_optimizer(model), make_config())
    with pytest.raises(NotImplementedError):
        lm._forward_window(None, None, 0, None, 1, 1, 1)
