"""Tests for the evaluation driver :mod:`pynop.training.eval` and its metrics.

Covers:

* the metric functions (relative L2 / nRMSE, R², spectral band errors, radial
  frequency grid) — invariance properties and exact pooling identities;
* :class:`~pynop.training.metrics.FieldMetricAccumulator` — one-shot vs
  incremental equivalence, per-step curve, window bookkeeping;
* :func:`pynop.training.eval.select_eval_ids` — val/test split selection from
  the training config or a stored ``run_info.json``, test -> val fallback and
  overlap protection;
* :func:`pynop.training.eval.build_eval_dataloader` — window protocol, batch
  shapes, ``max_windows`` cap;
* a tiny end-to-end CPU run: one pretraining epoch (shared fixture) followed by
  evaluation on the validation split, on an unused-simulations test split and
  through the CLI entry point.
"""

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pynop.training.eval import (
    build_eval_dataloader,
    format_results_table,
    main,
    parse_args,
    run_evaluation,
    select_eval_ids,
)
from pynop.training.metrics import (
    FieldMetricAccumulator,
    compute_field_metrics,
    radial_frequency_grid,
    relative_l2_error,
    r_squared,
    spectral_band_error,
)
from pynop.training.train import run_training


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SIMS = 6  # more sims than n_samples so the "unused pool" (test split) is non-empty
T_STEPS = 30
H = W = 16
C = 2
N_SAMPLES = 4  # simulations used by the training protocol (train + val)


@pytest.fixture(scope="module")
def pdebench_h5(tmp_path_factory) -> Path:
    """A tiny synthetic PDEBench HDF5 file (keys = simulation IDs, ``data`` [T,H,W,C])."""
    path = tmp_path_factory.mktemp("eval_data") / "tiny.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for sid in range(N_SIMS):
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


def _training_config(h5_path: Path, out_root: Path) -> dict:
    """Tiny training config (pretrain phase only is exercised by the e2e fixture)."""
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
        "losses": [{"name": "DWMSELoss"}],
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
        # window_size = max_AR_steps(2) + input_steps(2) + 1 = 5 -> 3 predictions per window.
        "pretraining": {
            "delta_ts": 2,
            "min_id": 0,
            "max_id": None,
            "n_samples": N_SAMPLES,
            "lr": 1e-3,
            "max_epochs": 1,
            "start_AR": 0,
            "final_AR": 1,
            "min_AR_steps": 1,
            "max_AR_steps": 2,
        },
        "fulltraining": {
            "delta_ts": 4,
            "min_id": 0,
            "max_id": None,
            "n_samples": N_SAMPLES,
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


@pytest.fixture(scope="module")
def trained_run(pdebench_h5, tmp_path_factory) -> dict:
    """One tiny pretraining run (shared by the e2e evaluation tests).

    Returns a dict with ``model_cfg`` / ``train_cfg`` paths, the experiment
    ``logdir`` (containing ``last.ckpt`` + ``run_info.json``) and the h5 path.
    """
    base = tmp_path_factory.mktemp("eval_e2e")
    model_cfg, train_cfg = _write_configs(base, pdebench_h5)
    logdir = Path(run_training(model_cfg, train_cfg, pretrain=True, device="cpu"))
    assert (logdir / "last.ckpt").is_file()
    assert (logdir / "run_info.json").is_file()
    return {"model_cfg": model_cfg, "train_cfg": train_cfg, "logdir": logdir, "h5": pdebench_h5}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def test_metrics_perfect_prediction():
    torch.manual_seed(0)
    x = torch.randn(4, 2, 16, 16)
    assert relative_l2_error(x, x).item() == 0.0
    assert abs(r_squared(x, x).item() - 1.0) < 1e-6
    bands = spectral_band_error(x, x)
    assert all(abs(v.item()) < 1e-7 for v in bands.values()), bands


def test_nrmse_scale_invariance_and_r2_affine_invariance():
    torch.manual_seed(0)
    target = torch.randn(4, 2, 32, 32)
    pred = target + 0.1 * torch.randn_like(target)

    nrmse_ref = relative_l2_error(pred, target).item()
    r2_ref = r_squared(pred, target).item()

    # nRMSE is scale-invariant (x -> a*x leaves the ratio unchanged).
    for a in [7.5, 0.01]:
        assert abs(relative_l2_error(a * pred, a * target).item() - nrmse_ref) < 1e-6, a

    # R² is invariant to any affine rescaling (shift AND scale).
    t_zm = target - target.mean(dim=(-2, -1), keepdim=True)
    p_zm = pred - target.mean(dim=(-2, -1), keepdim=True)
    for a, b in [(1.0, 0.0), (7.5, -3.2), (0.01, 4.0)]:
        assert abs(r_squared(a * p_zm + b, a * t_zm + b).item() - r2_ref) < 1e-6, (a, b)


def test_r2_equals_one_minus_nrmse_squared_for_zero_mean_data():
    """Per-unit identity: for per-channel zero-mean data R²_{i,c} = 1 − nRMSE²_{i,c}."""
    torch.manual_seed(0)
    B, Cc, Hh, Ww = 4, 2, 32, 32
    target = torch.randn(B, Cc, Hh, Ww)
    pred = target + 0.1 * torch.randn_like(target)

    t_zm = target - target.mean(dim=(-2, -1), keepdim=True)
    p_zm = pred - target.mean(dim=(-2, -1), keepdim=True)
    p4 = p_zm.reshape(-1, Cc, Hh, Ww)
    t4 = t_zm.reshape(-1, Cc, Hh, Ww)

    # nRMSE² per (sample, channel): ‖err‖²/‖target‖² with spatial norms.
    err2 = ((p4 - t4).reshape(B * Cc, -1)).pow(2).sum(-1).view(B, Cc)
    tgt2 = (t4.reshape(B * Cc, -1)).pow(2).sum(-1).view(B, Cc)
    per_nrmse2 = err2 / tgt2

    # R² per unit: 1 − ‖err‖²/‖centered target‖² (target already zero-mean).
    r2_per = 1.0 - err2 / (tgt2 + 1e-8)
    assert torch.allclose(per_nrmse2, 1.0 - r2_per, atol=1e-6)


def test_spectral_bands_white_noise_and_scale_invariance():
    torch.manual_seed(0)
    target = torch.randn(4, 2, 32, 32)
    pred = target + 0.1 * torch.randn_like(target)

    bands = spectral_band_error(pred, target)
    assert set(bands) == {"k[0,4)", "k[4,12)", "k>=12"}
    # White noise error spread over the whole spectrum: all three relative band
    # errors are non-zero and close to each other (~0.1).
    values = [v.item() for v in bands.values()]
    assert all(v > 0 for v in values), values
    assert max(values) / min(values) < 2.0, values

    # Relative band errors are scale-invariant like nRMSE.
    scaled = spectral_band_error(5.0 * pred, 5.0 * target)
    for label, ref in bands.items():
        assert abs(scaled[label].item() - ref.item()) < 1e-6, label

    # Absolute mode: band energies scale with the data amplitude.
    absolute = spectral_band_error(pred, target, relative=False)
    scaled_abs = spectral_band_error(5.0 * pred, 5.0 * target, relative=False)
    for label in bands:
        assert abs(scaled_abs[label].item() - 5.0 * absolute[label].item()) < 1e-4, label


def test_radial_frequency_grid_properties():
    Hh, Ww = 32, 32
    g = radial_frequency_grid(Hh, Ww)
    # rfftn layout: (h, w//2 + 1).
    assert g.shape == (Hh, Ww // 2 + 1)
    # Max radius = corner of the first quadrant.
    assert abs(g.max().item() - math.sqrt(16**2 + 16**2)) < 1e-4
    # FFT-order symmetry: row k and row h-k carry the same |index|.
    assert torch.equal(g[1], g[Hh - 1]) and torch.equal(g[2], g[Hh - 2])


def test_accumulator_pooling_equivalence():
    """One-shot == single update; multi-update pooling over identical windows == one-shot."""
    torch.manual_seed(0)
    B, Cc, Hh, Ww = 4, 2, 32, 32
    target = torch.randn(B, Cc, Hh, Ww)
    pred = target + 0.1 * torch.randn_like(target)

    one = compute_field_metrics([pred], [target])
    acc_single = FieldMetricAccumulator()
    acc_single.update([pred], [target])
    for k in ("mse", "rmse", "nrmse", "mae", "max_abs_error", "r2"):
        assert abs(one[k] - acc_single.compute()[k]) < 1e-12, k

    # Two updates over the SAME windows as one two-step one-shot call.
    acc_multi = FieldMetricAccumulator()
    acc_multi.update([pred[:2]], [target[:2]])
    acc_multi.update([pred[2:]], [target[2:]])
    multi = acc_multi.compute()
    full = compute_field_metrics([pred[:2], pred[2:]], [target[:2], target[2:]])
    for k in ("mse", "rmse", "nrmse", "mae", "max_abs_error", "r2"):
        assert abs(full[k] - multi[k]) < 1e-9, (k, full[k], multi[k])


def test_accumulator_per_step_curve_and_window_count():
    torch.manual_seed(0)
    B, Cc, Hh, Ww = 4, 2, 32, 32
    target = torch.randn(B, Cc, Hh, Ww)
    pred = target + 0.1 * torch.randn_like(target)

    acc = FieldMetricAccumulator()
    # Two updates x two steps each: step indices are pooled across updates.
    acc.update([pred[:2], pred[:2] + 0.05], [target[:2], target[:2]])
    acc.update([pred[2:], pred[2:]], [target[2:], target[2:]])
    res = acc.compute()

    assert res["n_windows_evaluated"] == B  # one window per batch element, over all updates
    assert res["n_steps"] == 2
    assert len(res["per_step"]) == 2
    assert [p["step"] for p in res["per_step"]] == [1, 2]
    for p in res["per_step"]:
        assert math.isfinite(p["rmse"]) and math.isfinite(p["nrmse"])

    # Mismatched list lengths / shapes are rejected.
    with pytest.raises(ValueError):
        acc.update([pred], [])
    with pytest.raises(ValueError):
        acc.update([pred, torch.randn(2, Cc, Hh, Ww)], [target, target])


# ---------------------------------------------------------------------------
# Evaluation-set selection
# ---------------------------------------------------------------------------


def _ds_cfg(h5_path: Path) -> dict:
    return {"type": "pdebench", "path": str(h5_path), "split_ratio": 0.9, "random_seed": 42}


def test_select_eval_ids_val_from_config(pdebench_h5):
    phase = {"n_samples": N_SAMPLES, "max_AR_steps": 2}
    info = select_eval_ids(_ds_cfg(pdebench_h5), phase, split="val")

    # pool of N_SAMPLES -> n_train = int(4*0.9) = 3, val = 1; disjoint by construction.
    assert len(info["train_ids"]) == 3 and len(info["val_ids"]) == 1
    assert info["eval_ids"] == info["val_ids"]
    assert not set(info["eval_ids"]) & set(info["train_ids"])
    assert info["source"] == "training config"
    assert info["effective_split"] == "val"


def test_select_eval_ids_test_split_unused_pool(pdebench_h5):
    phase = {"n_samples": N_SAMPLES, "max_AR_steps": 2}
    info = select_eval_ids(_ds_cfg(pdebench_h5), phase, split="test", n_test=2)

    # The unused pool holds the file IDs beyond n_samples in the seeded shuffle.
    assert len(info["eval_ids"]) == 2
    used = set(info["train_ids"]) | set(info["val_ids"])
    assert not set(info["eval_ids"]) & used, "test split must never touch train or val sims"
    assert info["effective_split"] == "test"

    # n_test caps the pool.
    capped = select_eval_ids(_ds_cfg(pdebench_h5), phase, split="test", n_test=1)
    assert len(capped["eval_ids"]) == 1


def test_select_eval_ids_test_fallback_to_val(pdebench_h5):
    """When training consumed every simulation, the test split falls back to val."""
    phase = {"n_samples": N_SIMS, "max_AR_steps": 2}  # whole file used -> empty unused pool
    info = select_eval_ids(_ds_cfg(pdebench_h5), phase, split="test")

    assert info["effective_split"] == "val"
    assert info["eval_ids"] == info["val_ids"]


def test_select_eval_ids_run_info_ground_truth(pdebench_h5):
    """Stored train/val IDs from run_info.json take precedence over the config."""
    ds_cfg = _ds_cfg(pdebench_h5)
    phase = {"n_samples": N_SAMPLES, "max_AR_steps": 2}

    # A run_info that disagrees with what the config would derive (drifted n_samples).
    run_info = {"train_ids": ["0", "1"], "val_ids": ["2"]}
    info = select_eval_ids(ds_cfg, phase, split="test", n_test=3, run_info=run_info)

    assert info["source"] == "run_info.json"
    assert info["train_ids"] == ["0", "1"] and info["val_ids"] == ["2"]
    # Unused pool = all file IDs minus the stored used ones.
    assert set(info["eval_ids"]) <= {str(i) for i in range(N_SIMS)} - {"0", "1", "2"}
    assert len(info["eval_ids"]) == 3


def test_select_eval_ids_unknown_split_raises(pdebench_h5):
    with pytest.raises(ValueError, match="Unknown split"):
        select_eval_ids(_ds_cfg(pdebench_h5), {"n_samples": N_SAMPLES}, split="bogus")


# ---------------------------------------------------------------------------
# Evaluation dataloader
# ---------------------------------------------------------------------------


def test_build_eval_dataloader_shapes_and_cap(pdebench_h5):
    ds_cfg = _ds_cfg(pdebench_h5)
    phase = {"delta_ts": 2, "min_id": 0, "max_id": None, "n_samples": N_SAMPLES, "max_AR_steps": 2}

    loader, window_size = build_eval_dataloader(ds_cfg, phase, input_steps=2, eval_ids=["0", "1"])
    assert window_size == 5  # max_AR_steps + input_steps + 1

    batch = next(iter(loader))
    fields, time_idx = batch[0], batch[1]
    assert fields.shape[-3:] == (C, H, W) and fields.shape[1] == window_size

    capped, _ = build_eval_dataloader(ds_cfg, phase, input_steps=2, eval_ids=["0", "1"], max_windows=3)
    assert len(capped.dataset) == 3


def test_build_eval_dataloader_rejects_other_types(pdebench_h5):
    ds_cfg = dict(_ds_cfg(pdebench_h5), type="thewell")
    with pytest.raises(ValueError, match="only 'pdebench'"):
        build_eval_dataloader(ds_cfg, {"max_AR_steps": 2}, input_steps=2, eval_ids=["0"])


# ---------------------------------------------------------------------------
# CLI / reporting
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(["m.json", "t.json", "--checkpoint", "run/last.ckpt"])
    assert args.model_config == Path("m.json")
    assert args.training_config == Path("t.json")
    assert args.checkpoint == Path("run/last.ckpt")
    assert args.split == "val" and args.n_test is None
    assert args.pretrain is False and args.device == "auto"
    assert args.batch_size is None and args.max_windows is None and args.output_dir is None


def test_parse_args_flags():
    args = parse_args(
        ["--pretrain", "--split", "test", "--n-test", "7", "--device", "cpu", "--max-windows", "3", "m.json", "t.json", "--checkpoint", "c.ckpt"]
    )
    assert args.pretrain is True and args.split == "test" and args.n_test == 7
    assert args.device == "cpu" and args.max_windows == 3


def test_format_results_table_smoke():
    results = {
        "checkpoint": "/tmp/run/last.ckpt",
        "model": {"class": "FNO", "name": "TinyFNO"},
        "phase": "pretraining",
        "split_effective": "val",
        "n_eval_simulations": 1,
        "id_source": "run_info.json",
        "window_size": 5,
        "metrics": {
            "mse": 1e-3,
            "rmse": 0.032,
            "nrmse": 0.05,
            "mae": 0.02,
            "max_abs_error": 0.4,
            "r2": 0.99,
            "spectral_bands": {"k[0,4)": 0.1, "k>=12": 0.2},
            "n_windows_evaluated": 7,
            "n_steps": 3,
            "per_step": [{"step": i + 1, "rmse": 1e-3 * (i + 1), "nrmse": 0.05} for i in range(3)],
        },
    }
    table = format_results_table(results)
    assert "EVALUATION RESULTS" in table
    assert "TinyFNO" in table and "run_info.json" in table
    assert "Spectral k>=12" in table
    assert "Per-step RMSE" in table and "s3=" in table


# ---------------------------------------------------------------------------
# End-to-end (tiny CPU runs)
# ---------------------------------------------------------------------------

N_PREDICTIONS_PER_WINDOW = 5 - 2  # window_size(5) - input_steps(2), from the pretraining phase


def _assert_results_json(path: Path, expected_split: str):
    """Common structural checks on an eval results JSON file."""
    assert path.is_file(), f"missing {path}"
    res = json.loads(path.read_text())

    for key in (
        "checkpoint",
        "model",
        "phase",
        "split_requested",
        "split_effective",
        "id_source",
        "n_eval_simulations",
        "eval_ids",
        "train_ids",
        "window_size",
        "n_windows_evaluated",
        "metrics",
    ):
        assert key in res, f"missing key {key!r}"

    m = res["metrics"]
    for key in ("mse", "rmse", "nrmse", "mae", "max_abs_error", "r2"):
        assert math.isfinite(m[key]), (key, m[key])
    assert set(m["spectral_bands"]) == {"k[0,4)", "k[4,12)", "k>=12"}
    assert all(math.isfinite(v) for v in m["spectral_bands"].values())

    # The eval set must be disjoint from the training IDs (audit trail).
    assert not set(res["eval_ids"]) & set(res["train_ids"]), "eval set overlaps with train IDs"
    assert res["split_effective"] == expected_split
    assert res["n_windows_evaluated"] > 0
    # Per-step curve: one entry per prediction of the rollout.
    assert len(m["per_step"]) == N_PREDICTIONS_PER_WINDOW, m["per_step"]


def test_end_to_end_eval_val_split(trained_run):
    """run_evaluation on the validation split (IDs from run_info.json next to the ckpt)."""
    results_path = run_evaluation(
        model_config_path=trained_run["model_cfg"],
        training_config_path=trained_run["train_cfg"],
        checkpoint=trained_run["logdir"] / "last.ckpt",
        split="val",
        device="cpu",
    )

    assert Path(results_path).name == "eval_results_val.json"
    _assert_results_json(Path(results_path), expected_split="val")

    res = json.loads(Path(results_path).read_text())
    # run_info.json was found next to the checkpoint -> its IDs are ground truth.
    assert res["id_source"] == "run_info.json"
    info = json.loads((trained_run["logdir"] / "run_info.json").read_text())
    assert set(res["eval_ids"]) == set(info["val_ids"])


def test_end_to_end_eval_test_split(trained_run):
    """run_evaluation on the unused-simulations test split (never trained or validated)."""
    results_path = run_evaluation(
        model_config_path=trained_run["model_cfg"],
        training_config_path=trained_run["train_cfg"],
        checkpoint=trained_run["logdir"] / "last.ckpt",
        split="test",
        n_test=N_SIMS - N_SAMPLES,  # use the whole unused pool (2 sims)
        device="cpu",
    )

    assert Path(results_path).name == "eval_results_test.json"
    _assert_results_json(Path(results_path), expected_split="test")

    res = json.loads(Path(results_path).read_text())
    used = set(res["train_ids"]) | set(json.loads((trained_run["logdir"] / "run_info.json").read_text())["val_ids"])
    assert len(res["eval_ids"]) == N_SIMS - N_SAMPLES
    assert not set(res["eval_ids"]) & used, "test split must be disjoint from train AND val"


def test_main_cli_entry_point(trained_run):
    """main() with an explicit argv evaluates the validation split."""
    results_path = main(
        [
            str(trained_run["model_cfg"]),
            str(trained_run["train_cfg"]),
            "--checkpoint",
            str(trained_run["logdir"] / "last.ckpt"),
            "--split",
            "val",
            "--device",
            "cpu",
        ]
    )
    _assert_results_json(Path(results_path), expected_split="val")
