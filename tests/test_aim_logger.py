"""Tests for the Aim integration (:mod:`pynop.training.aim_logger`, :mod:`pynop.training.import_logs`).

Covers:

* :class:`AimCallback` metric selection (per-batch ``*_step`` vs epoch-level)
  and parameter storage, verified by reading the data back from a temporary
  Aim repository;
* :func:`import_logs.discover_runs` on a synthetic log tree;
* CSV import (axis rule: ``_epoch`` columns at x=epoch, others at x=step),
  parameters from ``run_info.json``/``model_config.json``, and the idempotency
  sidecar;
* TensorBoard event-file import.

Reading back uses the same index mechanism as the Aim UI (``aim up`` builds it
automatically): :meth:`aim.Repo._recreate_index` merges the per-run shards into
the unified index, after which runs are readable in a fresh process.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("aim")

from pynop.training.aim_logger import AimCallback  # noqa: E402
from pynop.training.import_logs import discover_runs, import_logs  # noqa: E402
from pynop.training.train import build_aim_callback  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_run(home, run_hash):
    """Open a finished run from ``home`` for read-back (builds the UI index first)."""
    from aim import Repo, Run

    repo = Repo(str(home))
    repo._recreate_index()  # same mechanism `aim up` uses to make runs visible
    return Run(run_hash, repo=repo, read_only=True)


def _metric_frame(run, name):
    """A metric series as a list of ``(step, value)`` tuples (empty if missing)."""
    from aim.storage.context import Context

    metric = run.get_metric(name, context=Context({}))
    if metric is None:
        return []
    frame = metric.dataframe()
    return list(zip(frame["step"].tolist(), frame["value"].tolist()))


def _fake_trainer(callback_metrics=None, global_step=0, current_epoch=0, log_dir=None):
    """Minimal stand-in for ``pl.Trainer`` exposing what AimCallback reads."""
    return SimpleNamespace(
        is_global_zero=True,
        callback_metrics=dict(callback_metrics or {}),
        global_step=global_step,
        current_epoch=current_epoch,
        log_dir=log_dir,
    )


# ---------------------------------------------------------------------------
# AimCallback
# ---------------------------------------------------------------------------


def test_aim_callback_run_content(tmp_path):
    home = tmp_path / "aimdb"
    cb = AimCallback(experiment="TEST", name="phase/model/run1", home=str(home), params={"lr": 1e-3, "model_class": "FNO"})

    trainer = _fake_trainer(
        callback_metrics={
            "loss_step": 0.5,
            "RMSE_step": 0.25,
            "val_RMSE_step": 0.4,
            "loss_epoch": 9.0,
            # bare duplicates of suffixed metrics (Lightning keeps them too): must be skipped
            "loss": 5.0,
            "RMSE": 1.4,
            "avg_loss": 1.5,
            "AR_steps": 2,
        },
        global_step=42,
    )
    cb.setup(trainer, None, stage="fit")
    run_hash = cb.run_hash

    # Two training batches: per-batch metrics move with the global step;
    # val_*_step entries present in callback_metrics must be ignored (stale).
    trainer.global_step = 42
    cb.on_train_batch_end(trainer, None, outputs=None, batch=None, batch_idx=0)
    trainer.callback_metrics["loss_step"] = 0.3
    trainer.global_step = 91
    cb.on_train_batch_end(trainer, None, outputs=None, batch=None, batch_idx=1)

    # Validation batches: val_*_step at the cumulative validation-batch index.
    trainer.callback_metrics.update({"val_loss_step": 0.4, "val_RMSE_step": 0.2})
    cb.on_validation_batch_end(trainer, None, outputs=None, batch=None, batch_idx=0)
    trainer.callback_metrics["val_loss_step"] = 0.35
    cb.on_validation_batch_end(trainer, None, outputs=None, batch=None, batch_idx=1)

    # Validation end: *_epoch at x = epoch; everything else (avg_loss, AR_steps, ...)
    # at the recording position x = global_step.
    trainer.current_epoch = 3
    cb.on_validation_epoch_end(trainer, None)

    cb.teardown(trainer, None, stage="fit")

    run = _read_run(home, run_hash)
    assert run.name == "phase/model/run1"
    assert run.get("lr") == pytest.approx(1e-3)
    assert run.get("model_class") == "FNO"

    # Per-batch training series: only non-val *_step keys, at their global steps.
    assert _metric_frame(run, "loss_step") == [(42, 0.5), (91, 0.3)]
    assert [s for s, _ in _metric_frame(run, "RMSE_step")] == [42, 91]
    # Per-batch validation series: at the cumulative val-batch index.
    assert _metric_frame(run, "val_loss_step") == [(0, 0.4), (1, 0.35)]
    assert _metric_frame(run, "val_RMSE_step") == [(0, 0.2), (1, 0.2)]
    # Epoch aggregates: at x = epoch; *_step keys are excluded from the hook.
    assert _metric_frame(run, "loss_epoch") == [(3, 9.0)]
    # Non-suffixed metrics: at the recording position (global step).
    assert _metric_frame(run, "avg_loss") == [(91, 1.5)]
    assert _metric_frame(run, "AR_steps") == [(91, 2.0)]
    # Bare duplicates of suffixed metrics must not create extra series.
    assert _metric_frame(run, "loss") == []
    assert _metric_frame(run, "RMSE") == []


def test_aim_callback_disabled_without_fit_stage(tmp_path):
    home = tmp_path / "aimdb"
    cb = AimCallback(experiment="TEST", name="x", home=str(home))
    trainer = _fake_trainer()
    cb.setup(trainer, None, stage="validate")  # not fit -> no run created
    assert cb.run_hash is None
    cb.on_train_batch_end(trainer, None, outputs=None, batch=None, batch_idx=0)  # no-op
    cb.teardown(trainer, None, stage="validate")


def test_build_aim_callback_selection():
    logdir = Path("/tmp/out/PDEBENCH/pretraining/FNO/10steps_20260101-000000")
    run_info = {
        "model": {"class": "FNO", "params": 1234},
        "lr": 1e-3,
        "max_epochs": 10,
        "input_steps": 2,
        "window_size": 5,
        "seed": 42,
    }

    cb = build_aim_callback(
        logdir, ["csv", "aim"], {"experiment": "PDEBENCH"}, experiment="PDEBENCH", model_name="FNO", phase_name="pretraining", run_info=run_info
    )
    assert isinstance(cb, AimCallback)
    assert cb.experiment == "PDEBENCH"
    assert cb.name == "pretraining/FNO/10steps_20260101-000000"
    assert cb.params["model_class"] == "FNO"

    # Default experiment comes from output.experiment; home defaults to ~/.aim
    # (aim 3.x's own CWD-based default would pollute the launch directory).
    cb = build_aim_callback(logdir, ["aim"], {}, experiment="PDEBENCH", model_name="FNO", phase_name="pretraining", run_info=run_info)
    assert cb.experiment == "PDEBENCH" and Path(cb.home) == Path.home() / ".aim"

    # An explicit home is honored (and expanded).
    cb = build_aim_callback(logdir, ["aim"], {"home": "~/mydb"}, experiment="E", model_name="m", phase_name="p", run_info=run_info)
    assert Path(cb.home) == Path("~/mydb").expanduser()

    # Not requested -> None (including the csv+tb default when loggers is omitted).
    assert build_aim_callback(logdir, ["csv"], {}, experiment="E", model_name="m", phase_name="p", run_info=run_info) is None
    assert build_aim_callback(logdir, None, {}, experiment="E", model_name="m", phase_name="p", run_info=run_info) is None


# ---------------------------------------------------------------------------
# import_logs — discovery
# ---------------------------------------------------------------------------

CSV_HEADER = ["AR_steps", "RMSE_epoch", "RMSE_step", "avg_loss", "epoch", "loss_epoch", "loss_step", "lr-AdamW", "step", "val_RMSE_epoch"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_HEADER})


def _synthetic_tree(root: Path) -> dict:
    """Build a fake output tree with two runs (one csv+tb, one csv only)."""
    run_a = root / "pretraining" / "ModelA" / "10steps_20260101-000000"
    _write_csv(
        run_a / "lightning_logs" / "version_0" / "metrics.csv",
        [
            {"step": 0, "epoch": 0, "loss_step": 1.0, "RMSE_step": 0.5, "AR_steps": 2},
            {"step": 49, "epoch": 0, "loss_step": 0.8, "RMSE_step": 0.4, "AR_steps": 2, "loss_epoch": 0.9, "avg_loss": 0.95, "lr-AdamW": 1e-3},
            {"step": 50, "epoch": 1, "loss_step": 0.7, "RMSE_step": 0.35, "AR_steps": 2},
        ],
    )
    (run_a / "tb_logs" / "lightning_logs" / "version_0").mkdir(parents=True)
    (run_a / "tb_logs" / "lightning_logs" / "version_0" / "events.out.tfevents.fake").write_text("")
    (run_a / "run_info.json").write_text(
        json.dumps({"model": {"class": "ModelA", "params": 100}, "phase": "pretraining", "lr": 1e-3, "max_epochs": 2, "seed": 42})
    )

    run_b = root / "fulltraining" / "ModelB" / "10steps_20260102-000000"
    _write_csv(
        run_b / "lightning_logs" / "version_0" / "metrics.csv",
        [{"step": 7, "epoch": 0, "loss_step": 0.6, "val_RMSE_epoch": 0.3}],
    )

    # A directory with neither must not be discovered.
    (root / "pretraining" / "ModelC" / "10steps_empty").mkdir(parents=True)
    return {"run_a": run_a, "run_b": run_b}


def test_discover_runs(tmp_path):
    root = tmp_path / "PDEBENCH"
    refs = _synthetic_tree(root)

    found = {str(p) for p in discover_runs(root)}
    assert found == {str(refs["run_a"]), str(refs["run_b"])}  # csv+tb run counted once, empty dir ignored


def test_discover_runs_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_runs(tmp_path / "nope")


# ---------------------------------------------------------------------------
# import_logs — CSV
# ---------------------------------------------------------------------------


def test_import_csv_end_to_end(tmp_path):
    root = tmp_path / "PDEBENCH"
    refs = _synthetic_tree(root)
    home = tmp_path / "aimdb"

    results = import_logs(root, experiment="TEST", home=str(home))
    by_name = {r["name"]: r for r in results}
    assert set(by_name) == {"pretraining/ModelA/10steps_20260101-000000", "fulltraining/ModelB/10steps_20260102-000000"}

    entry = by_name["pretraining/ModelA/10steps_20260101-000000"]
    assert entry["status"] == "imported" and entry["source"] == "csv"
    # Axis rule: *_epoch columns at x=epoch, every other column at x=row step.
    # row1 -> 3 points (step axis); row2 -> 5 (step) + 1 loss_epoch (epoch axis); row3 -> 3.
    assert entry["n_points"] == 3 + 6 + 3

    run = _read_run(home, entry["hash"])
    assert run.name == "pretraining/ModelA/10steps_20260101-000000"
    assert _metric_frame(run, "loss_step") == [(0, 1.0), (49, 0.8), (50, 0.7)]
    assert _metric_frame(run, "RMSE_step") == [(0, 0.5), (49, 0.4), (50, 0.35)]
    assert _metric_frame(run, "AR_steps") == [(0, 2.0), (49, 2.0), (50, 2.0)]
    assert _metric_frame(run, "avg_loss") == [(49, 0.95)]
    assert _metric_frame(run, "lr-AdamW") == [(49, pytest.approx(1e-3))]
    assert _metric_frame(run, "loss_epoch") == [(0, 0.9)]
    # Parameters from run_info.json.
    assert run.get("model_class") == "ModelA"
    assert run.get("n_params") == 100
    assert run.get("phase") == "pretraining"

    # Sidecar written; a plain re-run skips the import (idempotent).
    assert (refs["run_a"] / ".aim_import.json").is_file()
    results2 = import_logs(root, experiment="TEST", home=str(home))
    assert all(r["status"].startswith("skipped") for r in results2)

    # --force re-imports (new Aim run, same name).
    results3 = import_logs(root, experiment="TEST", home=str(home), force=True)
    assert all(r["status"] == "imported" for r in results3)


def test_import_csv_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "PDEBENCH"
    refs = _synthetic_tree(root)
    home = tmp_path / "aimdb"

    results = import_logs(root, experiment="TEST", home=str(home), dry_run=True)
    assert all(r["status"] == "dry-run" for r in results)
    assert not (refs["run_a"] / ".aim_import.json").exists()
    assert not home.exists()  # no Aim run is created at all


def test_import_csv_bad_source(tmp_path):
    root = tmp_path / "PDEBENCH"
    _synthetic_tree(root)
    with pytest.raises(ValueError, match="Unknown source"):
        import_logs(root, experiment="TEST", home=str(tmp_path / "aimdb"), source="hdf5")


# ---------------------------------------------------------------------------
# import_logs — TensorBoard
# ---------------------------------------------------------------------------


def test_import_tb_end_to_end(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter  # re-exports the tensorboard writer

    root = tmp_path / "PDEBENCH"
    run_dir = root / "pretraining" / "ModelT" / "10steps_20260103-000000"
    tb_dir = run_dir / "tb_logs" / "lightning_logs" / "version_0"

    w = SummaryWriter(log_dir=str(tb_dir))
    for i in range(5):
        w.add_scalar("loss_step", 1.0 - 0.1 * i, global_step=i * 10)
        if i % 2 == 0:
            w.add_scalar("val_RMSE_epoch", 0.5 - 0.1 * i, global_step=i * 10)
    w.flush()
    w.close()

    results = import_logs(root, experiment="TEST", home=str(tmp_path / "aimdb"), source="tb")
    entry = results[0]
    assert entry["status"] == "imported" and entry["source"] == "tb"
    assert entry["n_points"] == 5 + 3

    run = _read_run(str(tmp_path / "aimdb"), entry["hash"])
    assert _metric_frame(run, "loss_step") == [(i * 10, pytest.approx(1.0 - 0.1 * i)) for i in range(5)]
    assert [s for s, _ in _metric_frame(run, "val_RMSE_epoch")] == [0, 20, 40]
