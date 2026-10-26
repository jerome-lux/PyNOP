"""Tests for the command-line training driver :mod:`pynop.training.train`.

Covers config parsing / builders (pure functions) and a tiny end-to-end CPU
run on a synthetic PDEBench HDF5 file, including the checkpoint policy:

* ``last.ckpt`` is saved every epoch from the start;
* best checkpoints only appear once the rollout reaches the maximum number of
  AR steps (epoch >= ``final_AR``).
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch.nn as nn

from pynop.training.train import (
    build_losses,
    build_optimizer,
    build_scheduler_config,
    build_pdebench_dataloaders,
    load_json,
    main,
    parse_args,
    run_training,
    select_phase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SIMS = 4
T_STEPS = 30
H = W = 16
C = 2


@pytest.fixture(scope="module")
def pdebench_h5(tmp_path_factory) -> Path:
    """A tiny synthetic PDEBench HDF5 file (keys = simulation IDs, ``data`` [T,H,W,C])."""
    path = tmp_path_factory.mktemp("data") / "tiny.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for sid in range(N_SIMS):
            # PDEBench layout: one group per simulation ID holding a "data" dataset [T,H,W,C].
            grp = f.create_group(str(sid))
            grp.create_dataset("data", data=rng.standard_normal((T_STEPS, H, W, C)).astype(np.float32))
    return path


def _model_config() -> dict:
    """Tiny FNO matching input_steps=2 and 2 channels (in_channels = 4)."""
    return {
        "name": "TinyFNO",
        "model": "FNO",
        "params": {
            "in_channels": 4,
            "out_channels": C,
            "modes": [2, 2],
            "hidden_channels": [8, 8],
            "blocks": ["FNO", "FNO"],
            "spectral_layer_type": "standard",
        },
    }


def _training_config(h5_path: Path, out_root: Path, pretrain_epochs: int = 2) -> dict:
    """Tiny training config reproducing the notebook protocol at test scale."""
    return {
        "input_steps": 2,
        "dataset": {
            "type": "pdebench",
            "path": str(h5_path),
            "batch_size": 2,
            "num_workers": 0,
            "load_in_ram": True,
            "split_ratio": 0.9,
            "random_seed": 42,
        },
        "shared_params": {
            "timestep": 5e-2,
            "loss_weights": [1e3, 1e3],
            "temporal_weighting": 1.1,
            "detach_grad_steps": 4,
            "derivative_loss_weight": 1.0,
            "predict_derivative": True,
            "noise_level": 0.0,
            "time_normalization": T_STEPS,
        },
        "losses": [
            {"name": "DWMSELoss"},
            {"name": "SpectralLoss", "params": {"beta": 2}},
        ],
        "optimizer": {"name": "AdamW", "weight_decay": 5e-4, "betas": [0.9, 0.99]},
        "scheduler": {
            "name": "OneCycleLR",
            "div_factor": 10,
            "final_div_factor": 10,
            "pct_start": 0.1,
            "total_steps": "$total_steps",
            "interval": "step",
        },
        "trainer": {"gradient_clip_val": 1.0, "log_every_n_steps": 5},
        "output": {"root": str(out_root), "experiment": "TEST"},
        "loggers": ["csv"],
        # window_size = max_AR_steps(2) + input_steps(2) + 1 = 5; final_AR=1 < pretrain_epochs
        "pretraining": {
            "delta_ts": 2,
            "min_id": 0,
            "max_id": None,
            "n_samples": N_SIMS,
            "lr": 1e-3,
            "max_epochs": pretrain_epochs,
            "start_AR": 0,
            "final_AR": 1,
            "min_AR_steps": 1,
            "max_AR_steps": 2,
        },
        "fulltraining": {
            "delta_ts": 4,
            "min_id": 0,
            "max_id": None,
            "n_samples": N_SIMS,
            "lr": 5e-4,
            "max_epochs": 1,
            "start_AR": 0,
            "final_AR": 0,
            "min_AR_steps": 2,
            "max_AR_steps": 2,
        },
    }


def _write_configs(tmp_path: Path, h5_path: Path) -> tuple:
    model_cfg = tmp_path / "model.json"
    train_cfg = tmp_path / "training.json"
    with open(model_cfg, "w") as f:
        json.dump(_model_config(), f)
    with open(train_cfg, "w") as f:
        json.dump(_training_config(h5_path, tmp_path / "out"), f)
    return model_cfg, train_cfg


# ---------------------------------------------------------------------------
# CLI / phase selection
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(["m.json", "t.json"])
    assert args.model_config == Path("m.json")
    assert args.training_config == Path("t.json")
    assert args.pretrain is False
    assert args.device == "auto"
    assert args.resume is None and args.init_from is None


def test_parse_args_pretrain_flag():
    args = parse_args(["--pretrain", "--device", "cpu", "m.json", "t.json"])
    assert args.pretrain is True
    assert args.device == "cpu"


def test_select_phase(pdebench_h5, tmp_path):
    cfg = _training_config(pdebench_h5, tmp_path)
    name, phase = select_phase(cfg, pretrain=True)
    assert name == "pretraining" and phase is cfg["pretraining"]
    name, phase = select_phase(cfg, pretrain=False)
    assert name == "fulltraining" and phase is cfg["fulltraining"]

    with pytest.raises(KeyError):
        select_phase({"other": {}}, pretrain=True)


def test_load_json_errors(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_json(missing)
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        load_json(bad)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_build_losses_specs():
    losses = build_losses([{"name": "DWMSELoss"}, {"name": "SpectralLoss", "params": {"beta": 2}}, "MSELoss"])
    assert [type(l).__name__ for l in losses] == ["DWMSELoss", "SpectralLoss", "MSELoss"]
    assert losses[1].beta == 2

    default = build_losses(None)
    assert len(default) == 1 and type(default[0]).__name__ == "MSELoss"


def test_build_losses_unknown_name():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_losses([{"name": "DoesNotExist"}])


def test_activation_resolution_in_model_params(pdebench_h5):
    from pynop.training.train import _resolve_activation

    resolved = _resolve_activation({"activation": "GELU", "pe": "random_ipe", "nested": ["SiLU"]})
    assert resolved["activation"] is nn.GELU
    assert resolved["pe"] == "random_ipe"  # non-activation strings untouched
    assert resolved["nested"][0] is nn.SiLU


def test_build_optimizer(pdebench_h5):
    from pynop.training.train import build_model

    model = build_model(_model_config())
    opt = build_optimizer(model, {"name": "AdamW", "weight_decay": 1e-2, "betas": [0.9, 0.99]}, lr=3e-4)
    assert type(opt).__name__ == "AdamW"
    assert opt.param_groups[0]["lr"] == 3e-4
    assert opt.param_groups[0]["weight_decay"] == 1e-2
    assert opt.param_groups[0]["betas"] == (0.9, 0.99)

    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, {"name": "NopeOpt"}, lr=1e-3)


def test_build_scheduler_tokens():
    cfg = build_scheduler_config(
        {
            "name": "OneCycleLR",
            "div_factor": 10,
            "pct_start": 0.1,
            "total_steps": "$total_steps",
            "interval": "step",
        },
        phase_lr=2e-4,
        total_steps=123,
        steps_per_epoch=17,
    )
    assert len(cfg) == 1
    entry = cfg[0]
    assert entry["scheduler"].__name__ == "OneCycleLR"
    assert entry["total_steps"] == 123
    assert entry["max_lr"] == 2e-4  # defaults to the phase lr
    assert entry["interval"] == "step"

    with pytest.raises(ValueError, match="Unknown scheduler token"):
        build_scheduler_config({"name": "StepLR", "step_size": "$bogus"}, 1e-3, 10, 5)


def test_build_pdebench_dataloaders_split(pdebench_h5):
    ds_cfg = _training_config(pdebench_h5, Path("."))["dataset"]
    phase = {"delta_ts": 2, "min_id": 0, "max_id": None, "n_samples": N_SIMS, "max_AR_steps": 2}

    train1, val1, window = build_pdebench_dataloaders(ds_cfg, phase, input_steps=2)
    assert window == 5  # max_AR_steps + input_steps + 1
    assert len(train1.dataset.sample_ids) == int(N_SIMS * 0.9)
    assert len(val1.dataset.sample_ids) == N_SIMS - int(N_SIMS * 0.9)

    # Same seed -> same (deterministic) split; train/val are disjoint.
    train2, val2, _ = build_pdebench_dataloaders(ds_cfg, phase, input_steps=2)
    assert train1.dataset.sample_ids == train2.dataset.sample_ids
    assert not set(train1.dataset.sample_ids) & set(val1.dataset.sample_ids)

    # A batch has the expected shape (B, T_unroll, C, H, W).
    batch = next(iter(train1))
    fields, time_idx = batch[0], batch[1]
    assert fields.shape == (2, 5, C, H, W)


def test_build_pdebench_dataloaders_rejects_other_types(pdebench_h5):
    ds_cfg = _training_config(pdebench_h5, Path("."))["dataset"]
    ds_cfg["type"] = "thewell"
    with pytest.raises(ValueError, match="only 'pdebench'"):
        build_pdebench_dataloaders(ds_cfg, {"max_AR_steps": 2}, input_steps=2)


# ---------------------------------------------------------------------------
# End-to-end (tiny CPU runs)
# ---------------------------------------------------------------------------


def test_end_to_end_pretrain_checkpoint_policy(pdebench_h5, tmp_path):
    """final_AR=1 < max_epochs=2: last.ckpt from epoch 0, best checkpoints from epoch 1."""
    model_cfg, train_cfg = _write_configs(tmp_path, pdebench_h5)
    logdir = run_training(model_cfg, train_cfg, pretrain=True, device="cpu")

    assert Path(logdir).is_dir()
    # Bookkeeping files.
    for fname in ("model_config.json", "training_config.json", "run_info.json"):
        assert (logdir / fname).is_file(), f"missing {fname}"
    info = json.loads((logdir / "run_info.json").read_text())
    assert info["phase"] == "pretraining"
    assert info["window_size"] == 5
    assert info["checkpoint_gate_epoch"] == 1

    # Checkpoint policy.
    assert (logdir / "last.ckpt").is_file()
    assert (logdir / "best_RMSE.ckpt").is_file(), "gate reached at epoch 1: best_RMSE must exist"
    assert (logdir / "best_val_RMSE.ckpt").is_file()


def test_end_to_end_gate_not_reached(pdebench_h5, tmp_path):
    """final_AR=1 >= max_epochs=1: only last.ckpt is written."""
    model_cfg = tmp_path / "model.json"
    with open(model_cfg, "w") as f:
        json.dump(_model_config(), f)
    train_cfg_dict = _training_config(pdebench_h5, tmp_path / "out", pretrain_epochs=1)
    train_cfg = tmp_path / "training.json"
    with open(train_cfg, "w") as f:
        json.dump(train_cfg_dict, f)

    logdir = run_training(model_cfg, train_cfg, pretrain=True, device="cpu")

    assert (logdir / "last.ckpt").is_file(), "last.ckpt must exist even when the gate is never reached"
    best_files = [p for p in Path(logdir).glob("best_*.ckpt")]
    assert not best_files, f"no best checkpoint may be saved before max AR steps: {best_files}"


def test_end_to_end_fulltraining_and_warm_start(pdebench_h5, tmp_path):
    """fulltraining phase (final_AR=0) + warm start from a previous last.ckpt."""
    model_cfg = tmp_path / "model.json"
    with open(model_cfg, "w") as f:
        json.dump(_model_config(), f)

    # Phase 1 run to produce a last.ckpt.
    train_cfg_1 = tmp_path / "training1.json"
    with open(train_cfg_1, "w") as f:
        json.dump(_training_config(pdebench_h5, tmp_path / "out"), f)
    logdir_1 = run_training(model_cfg, train_cfg_1, pretrain=True, device="cpu")
    last_ckpt = Path(logdir_1) / "last.ckpt"
    assert last_ckpt.is_file()

    # Phase 2 run warm-started from phase 1 weights (fresh optimizer).
    train_cfg_2 = tmp_path / "training2.json"
    with open(train_cfg_2, "w") as f:
        json.dump(_training_config(pdebench_h5, tmp_path / "out"), f)
    logdir_2 = run_training(model_cfg, train_cfg_2, pretrain=False, device="cpu", init_from=last_ckpt)

    info = json.loads((Path(logdir_2) / "run_info.json").read_text())
    assert info["phase"] == "fulltraining"
    # final_AR=0 -> best checkpoints are monitored from epoch 0.
    assert (Path(logdir_2) / "best_RMSE.ckpt").is_file()


def test_main_cli_entry_point(pdebench_h5, tmp_path):
    """main() with an explicit argv runs the fulltraining phase."""
    model_cfg = tmp_path / "model.json"
    with open(model_cfg, "w") as f:
        json.dump(_model_config(), f)
    train_cfg = tmp_path / "training.json"
    cfg_dict = _training_config(pdebench_h5, tmp_path / "out")
    cfg_dict["fulltraining"]["max_epochs"] = 1
    with open(train_cfg, "w") as f:
        json.dump(cfg_dict, f)

    logdir = main([str(model_cfg), str(train_cfg), "--device", "cpu"])
    info = json.loads((Path(logdir) / "run_info.json").read_text())
    assert info["phase"] == "fulltraining"
    assert (Path(logdir) / "last.ckpt").is_file()
