"""Aim integration for PyNOP training.

Provides :class:`AimCallback`, a Lightning callback that mirrors the metrics
logged by the training wrappers to an Aim run (https://github.com/aimhubio/aim),
so that runs of different models can be compared in the Aim UI (``aim up``).

Design notes
------------
* ``aim`` is imported lazily inside :meth:`AimCallback.setup`: the rest of the
  package works without it, and a missing/broken Aim installation degrades to
  a warning instead of crashing the training.
* Metric selection mirrors what the CSV logger records (same names, same
  x-axis convention), so live runs and runs imported from existing logs by
  :mod:`pynop.training.import_logs` are directly comparable in the UI:

  - per-batch training metrics — keys ending in ``_step`` without a ``val_``
    prefix — are tracked at every training batch with ``x = trainer.global_step``;
  - per-batch validation metrics (``val_*_step``) are tracked at every
    validation batch with ``x`` = cumulative validation batch index, exactly
    like the CSV logger's step column for validation rows;
  - ``*_epoch`` aggregates are tracked once at the end of validation with
    ``x = trainer.current_epoch``;
  - everything else currently known (``avg_loss``, ``AR_steps``, ``lr``,
    ``lr-AdamW``, per-AR-step RMSEs, ...) is tracked once at the end of
    validation at its recording position ``x = trainer.global_step``.

  The import applies the same rule to the CSV: columns ending in ``_epoch``
  use the row's ``epoch`` value as x, every other column uses the row's
  ``step`` value (which is already the cumulative val-batch index for
  validation rows).

* The callback only activates on the global rank 0 process and requires a
  validation loop (the training driver always provides one).
"""

import numbers
from typing import Any, Optional

import pytorch_lightning as pl

__all__ = ["AimCallback"]


def _is_number(value: Any) -> bool:
    """True for real/integer scalars (excluding booleans and tensors)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Real):
        return True
    # 0-d / scalar torch tensors are common in callback_metrics.
    try:
        import torch

        return isinstance(value, torch.Tensor) and value.dim() == 0
    except ImportError:  # pragma: no cover - torch is a hard dependency of pynop
        return False


def _to_float(value: Any) -> float:
    """Convert a scalar metric value (number or 0-d tensor) to ``float``."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
    except ImportError:  # pragma: no cover
        pass
    return float(value)


class AimCallback(pl.Callback):
    """Lightning callback logging training metrics to an Aim run.

    Args:
        experiment: Aim experiment name grouping comparable runs (e.g. the
            ``output.experiment`` of the training config, ``"PDEBENCH"``).
        name: Display name of the Aim run. Defaults to the trainer's log
            directory name when omitted.
        home: Path of the Aim repository ("home"). ``None`` falls back to aim's
            own CWD-based default — the training driver always passes an
            explicit path (defaulting to ``~/.aim``) instead.
        params: Optional dict of parameters/hyper-parameters stored with the
            run (shown in the Aim UI parameter table for cross-run comparison).

    Notes:
        All Aim interactions are guarded: any Aim error is reported once and
        disables the callback for the rest of the run — training never fails
        because of logging. The run is closed in :meth:`teardown`.
    """

    def __init__(
        self,
        experiment: str = "default",
        name: Optional[str] = None,
        home: Optional[str] = None,
        params: Optional[dict] = None,
    ):
        self.experiment = str(experiment)
        self.name = name
        self.home = str(home) if home is not None else None
        self.params = dict(params or {})
        self._run = None
        self._hash: Optional[str] = None
        self._disabled_reason: Optional[str] = None
        # Cumulative validation-batch counter — the same x-axis the CSV logger
        # uses for validation rows (it does not reset between epochs).
        self._n_val_batches = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def run_hash(self) -> Optional[str]:
        """Hash of the Aim run, once created (kept after close; ``None`` if disabled)."""
        return self._hash

    def _disable(self, reason: str) -> None:
        if self._disabled_reason is None:
            print(f"[AimCallback] Disabled for this run: {reason}")
            self._disabled_reason = reason

    def _track(self, values: dict, step: int, epoch: Optional[int] = None) -> None:
        """Track a batch of metric values; swallow Aim errors (warn once)."""
        if not values or self._run is None:
            return
        try:
            kwargs = {"step": int(step)}
            if epoch is not None:
                kwargs["epoch"] = int(epoch)
            self._run.track(values, **kwargs)
        except Exception as e:  # noqa: BLE001 - logging must never break training
            self._disable(f"track() failed ({type(e).__name__}: {e})")

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Create the Aim run at the start of ``fit`` (global rank 0 only)."""
        if stage != "fit" or not trainer.is_global_zero:
            return
        try:
            import aim
        except ImportError as e:
            self._disable(f"'aim' is not installed ({e}). Install it with `pip install aim`.")
            return

        run_name = self.name
        if run_name is None and trainer.log_dir is not None:
            from pathlib import Path

            run_name = str(Path(trainer.log_dir).name)

        try:
            kwargs = {"experiment": self.experiment}
            if self.home is not None:
                kwargs["repo"] = self.home
            self._run = aim.Run(**kwargs)
            self._hash = self._run.hash
            if run_name:
                self._run.name = run_name
            for key, value in self.params.items():
                self._run[key] = value
        except Exception as e:  # noqa: BLE001
            self._disable(f"could not create the Aim run ({type(e).__name__}: {e})")
            self._run = None

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Track per-batch training metrics (``*_step``, non-``val_``) at the global step.

        ``val_*_step`` entries are skipped here: during training they hold stale
        last-validation values. They are captured fresh in
        :meth:`on_validation_batch_end` instead.
        """
        if self._run is None:
            return
        values = {
            key: _to_float(value)
            for key, value in trainer.callback_metrics.items()
            if str(key).endswith("_step") and not str(key).startswith("val_") and _is_number(value)
        }
        self._track(values, step=trainer.global_step)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Track per-batch validation metrics (``val_*_step``) at the cumulative
        validation-batch index — the same x-axis the CSV logger uses for
        validation rows (it does not reset between epochs).

        Note: this assumes no sanity-check validation batches before training
        starts (the driver sets ``num_sanity_val_steps=0``); with sanity steps
        the counter would be offset by their count.
        """
        if self._run is None:
            return
        values = {
            key: _to_float(value)
            for key, value in trainer.callback_metrics.items()
            if str(key).startswith("val_") and str(key).endswith("_step") and _is_number(value)
        }
        self._track(values, step=self._n_val_batches)
        self._n_val_batches += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Track the remaining metrics once validation of the epoch has finished.

        At this point ``callback_metrics`` holds the final train and validation
        aggregates for the epoch; per-batch (``*_step``) entries are stale
        last-batch values and are skipped. The x-axis convention matches the CSV
        import (:mod:`pynop.training.import_logs`):

        - ``*_epoch`` aggregates are tracked at ``x = current_epoch``;
        - everything else (``avg_loss``, ``AR_steps``, ``lr``, ``lr-AdamW``,
          per-AR-step RMSEs, ...) is tracked at the recording position
          ``x = global_step``.
        """
        if self._run is None:
            return
        metrics = trainer.callback_metrics
        # Lightning also keeps a bare entry for every logged metric (e.g. "loss"
        # alongside "loss_step"/"loss_epoch"); the CSV has no such columns, so
        # skip any key that has a suffixed variant to stay consistent with it.
        epoch_values = {}
        step_values = {}
        for key, value in metrics.items():
            if not _is_number(value) or str(key).endswith("_step"):
                continue
            if f"{key}_step" in metrics or f"{key}_epoch" in metrics:
                continue
            (epoch_values if str(key).endswith("_epoch") else step_values)[key] = _to_float(value)
        self._track(epoch_values, step=trainer.current_epoch, epoch=trainer.current_epoch)
        self._track(step_values, step=trainer.global_step, epoch=trainer.current_epoch)

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Close the Aim run (flushes data to disk)."""
        if self._run is not None:
            try:
                self._run.close()
            except Exception as e:  # noqa: BLE001
                print(f"[AimCallback] Warning: could not close the Aim run cleanly ({type(e).__name__}: {e})")
            finally:
                self._run = None
