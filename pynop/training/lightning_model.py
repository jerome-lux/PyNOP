"""Lightning training wrappers for the PyNOP neural-operator models.

The dataloader contract shared by the multi-step wrappers is:

    (inputs, time_idx)  or  (inputs, time_idx, cond_field)

where ``inputs`` has shape ``(B, T, C, H, W)`` — ``T`` consecutive slices of the
same spatial field.

Prediction contract (``MultiStepNOModel`` / ``DeepONetModel``):
the wrapped model maps an input window ``(B, n_slices * C, H, W)`` to
``(B, C, H, W)`` — either

- the **time derivative** ``du/dt`` (``predict_derivative=True``, default), in
  which case the next state is ``u_{t} + dt * out``, or
- the **absolute** next state ``u_{t+dt}`` (``predict_derivative=False``), in
  which case the (optional) derivative regularizer is evaluated on the
  implicit derivative ``(out - u_{t}) / dt``.

Model-signature compatibility (auto-detected from ``model.forward``):

- ``model(x)``                e.g. ``UNet``, ``Resnet``, ``ConvNext``
- ``model(x, cond=...)``      e.g. ``FNO``, ``LatentNO``, ``LatentGalerkin``,
  ``ITLNO``, ``LatentTransolver``, ``SwinNO``, ``MultiscaleSWINNO``, ``UNO``,
  ``Transolver``
- ``model(x, time=...)``      e.g. ``GalerkinTransformer``

``DeepONet`` has a different paradigm — ``forward(u, y)`` with a branch net
evaluating a function and a trunk net consuming query coordinates — and is
therefore wrapped by its own class, :class:`DeepONetModel`.
"""

import inspect
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint

from ..core import add_noise
from ..core.loss import NormalizedTimeDerivativeMSE

__all__ = [
    "TrainingConfig",
    "CurriculumCheckpoint",
    "MultiStepLightningBase",
    "MultiStepNOModel",
    "DeepONetModel",
    "default_scheduler_config",
    "default_opt_config",
]


# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

default_scheduler_config = [
    {
        "scheduler": ReduceLROnPlateau,
        "mode": "min",
        "patience": 10,
        "factor": 0.5,
        "monitor": "avg_loss",
        "interval": "epoch",
        "frequency": 1,
    }
]

default_opt_config = {"optimizer": "AdamW", "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}


# ---------------------------------------------------------------------------
# Training configurations
# ---------------------------------------------------------------------------


def _config_to_json_dict(config) -> dict:
    """Serialize a config dataclass to a JSON-compatible dict.

    Loss functions, optimizers and other callables / classes / instances are
    replaced by their class name so the result can be dumped with ``json.dump``.
    """

    def serialize_value(v):
        # Handle lists or tuples recursively
        if isinstance(v, (list, tuple)):
            return [serialize_value(i) for i in v]

        # Handle classes or types
        if callable(v) or isinstance(v, type):
            return v.__name__ if hasattr(v, "__name__") else v.__class__.__name__

        # Handle custom class instances
        if hasattr(v, "__class__") and type(v).__module__ != "builtins":
            return v.__class__.__name__

        return v

    return {k: serialize_value(v) for k, v in asdict(config).items()}


@dataclass
class TrainingConfig:
    """Configuration for the multi-step prediction Lightning modules.

    Args:
        dt: Time step ``dt`` between two consecutive slices of a sample.
        start_autoregressive: Epoch at which the autoregressive (AR) curriculum
            starts increasing the number of AR steps.
        final_autoregressive: Epoch at which the number of AR steps reaches
            ``max_autoregressive_steps``.
        min_autoregressive_steps: Number of AR steps before
            ``start_autoregressive`` (and the lower bound of the curriculum).
        max_autoregressive_steps: Upper bound of the AR curriculum.
        detach_grad_steps: BPTT truncation — the gradient is detached through
            the AR rollout every ``detach_grad_steps`` steps (0 disables).
        loss_fn: A loss ``loss(preds, targets)`` or a list of losses.
        loss_weights: Scalar weight broadcast over ``loss_fn``, or a matching list.
        derivative_loss_weight: Weight of the derivative regularizer.
            ``0.0`` disables it (it is then never computed).
        derivative_loss_fn: A loss
            ``loss(pred_derivative, target, previous, dt) -> scalar``, or
            ``None`` to use the default :class:`NormalizedTimeDerivativeMSE`.
        predict_derivative: ``True`` if the network output is the time
            derivative ``du/dt`` (the next state is ``u_t + dt * out``);
            ``False`` if the output is the absolute next state ``u_{t+dt}``.
        noise_level: Teacher-forcing input noise (``0`` = no noise).
        time_normalization: Divisor applied to the time index before passing
            ``time`` to models that accept it (e.g. ``GalerkinTransformer``).
        n_slices: Number of consecutive input slices used to predict the next one.
        temporal_weighting: ``gamma`` — the loss of the ``k``-th AR step is
            weighted by ``gamma ** k`` (training only).
    """

    dt: float = 1.0
    start_autoregressive: Any = None
    final_autoregressive: Any = None
    min_autoregressive_steps: int = 0
    max_autoregressive_steps: int = 0
    detach_grad_steps: int = 4
    loss_fn: Any = torch.nn.MSELoss()
    loss_weights: Any = 1.0
    derivative_loss_weight: float = 1.0
    derivative_loss_fn: Any = None  # None -> NormalizedTimeDerivativeMSE
    predict_derivative: bool = True
    noise_level: float = 0.0  # no noise if 0
    time_normalization: float = 1.0
    n_slices: int = 2  # number of input slices used to predict the next one
    temporal_weighting: float = 1.2  # gamma: AR-step loss weight = gamma ** k

    def to_json_dict(self):
        return _config_to_json_dict(self)

    def save(self, path: Union[str, "Path"]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=4)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class CurriculumCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that only starts saving after ``start_epoch``.

    Used to defer best-model selection until the AR curriculum reaches its
    target (e.g. the maximum number of autoregressive steps), when per-epoch
    metrics become comparable across epochs.

    The gate is applied in :meth:`_should_skip_saving_checkpoint`, the single
    choke point shared by every save hook (train-batch-end, train-epoch-end —
    which is where Lightning 2.x saves by default with ``val_check_interval=1``
    — and validation-end), so no checkpoint can be written before
    ``start_epoch`` through any path. The on-exception crash-recovery save is
    gated as well.
    """

    def __init__(self, start_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _should_skip_saving_checkpoint(self, trainer) -> bool:  # type: ignore[override]
        if trainer.current_epoch < self.start_epoch:
            return True
        return super()._should_skip_saving_checkpoint(trainer)

    def on_exception(self, trainer, pl_module, exception):  # type: ignore[override]
        if trainer.current_epoch >= self.start_epoch:
            super().on_exception(trainer, pl_module, exception)


# ---------------------------------------------------------------------------
# Base multi-step Lightning module
# ---------------------------------------------------------------------------


class MultiStepLightningBase(pl.LightningModule):
    """Base Lightning module for the PyNOP multi-step prediction wrappers.

    Shared plumbing (wrapped model, optimizer, scheduler configuration,
    ``(loss, weight)`` pairs, forward-signature introspection and
    :meth:`configure_optimizers`) plus the teacher-forcing / autoregressive
    rollout, the derivative-vs-absolute output contract, the derivative
    regularizer and the AR curriculum.

    Subclasses only implement :meth:`_forward_window`, i.e. the mapping
    ``window (B, n*C, H, W) -> (B, C, H, W)``.

    At the end of every epoch (once validation has run), a single line is
    printed with the per-batch average losses of both splits — total loss,
    each weighted component and the derivative regularizer.

    The optimizer may be omitted when the module is used for inference /
    evaluation only (e.g. :mod:`pynop.training.eval`): in that case
    :meth:`configure_optimizers` must not be called through a trainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_config: Optional[TrainingConfig] = None,
        scheduler_config: Optional[list] = None,
    ):
        train_config = train_config if train_config is not None else TrainingConfig()
        super().__init__()
        self.model = model
        self.train_config = train_config
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.train_loss_avg = MeanMetric()
        # Per-epoch running sums used by the end-of-epoch loss summary printout.
        self._train_sums: dict[str, float] = {}
        self._val_sums: dict[str, float] = {}
        self._n_train_batches = 0
        self._n_val_batches = 0

        # --- losses -----------------------------------------------------
        raw_loss = train_config.loss_fn
        if raw_loss is None:
            self.loss_fn = []
        elif isinstance(raw_loss, Iterable) and not isinstance(raw_loss, torch.nn.Module):
            self.loss_fn = list(raw_loss)
        else:
            self.loss_fn = [raw_loss]

        raw_weights = train_config.loss_weights
        if isinstance(raw_weights, (list, tuple)):
            self.loss_weights = list(raw_weights)
            if len(self.loss_weights) != len(self.loss_fn):
                self.loss_weights = [self.loss_weights[0]] * len(self.loss_fn)
        else:
            self.loss_weights = [raw_weights] * len(self.loss_fn)

        # --- forward-signature introspection ------------------------------
        fwd_params = set(inspect.signature(self.model.forward).parameters)
        self._accepts_cond = "cond" in fwd_params
        self._accepts_time = "time" in fwd_params

        if self.loss_fn:
            print(f"Loss functions: {self.loss_fn}, with weights: {self.loss_weights}")

        # --- rollout contract ---------------------------------------------
        if train_config.n_slices <= 0:
            raise ValueError(f"n_slices must be > 0, got {train_config.n_slices}")
        if train_config.detach_grad_steps < 0:
            raise ValueError("detach_grad_steps must be >= 0")
        if float(train_config.dt) <= 0:
            raise ValueError(f"dt must be > 0, got {train_config.dt}")

        self.n_slices = train_config.n_slices
        self.dt = float(train_config.dt)
        self.predict_derivative = bool(train_config.predict_derivative)
        self.detach_every_k = train_config.detach_grad_steps

        # Derivative regularizer: configurable, skippable with weight 0.
        self.derivative_loss_fn = (
            train_config.derivative_loss_fn
            if train_config.derivative_loss_fn is not None
            else NormalizedTimeDerivativeMSE()
        )
        self.derivative_loss_weight = float(train_config.derivative_loss_weight)

        print(f"predict_derivative: {self.predict_derivative}")
        print(f"Derivative loss: {self.derivative_loss_fn}, with weight: {self.derivative_loss_weight}")

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def forward(self, x, **kwargs):
        """Passthrough to the wrapped model."""
        return self.model(x, **kwargs)

    @staticmethod
    def _unpack_batch(batch):
        """Return ``(inputs, time_idx, cond_field)`` from a batch tuple.

        ``cond_field`` is ``None`` for 2-element batches.
        """
        if len(batch) == 3:
            inputs, time_idx, cond_field = batch
        else:
            inputs, time_idx = batch
            cond_field = None
        return inputs, time_idx, cond_field

    def _weighted_losses(self, preds: torch.Tensor, targets: torch.Tensor, weighting: float = 1.0):
        """Compute ``(total_loss, per_loss_values)`` over ``self.loss_fn``.

        ``per_loss_values`` are Python floats (already weighted) for logging.
        """
        total = 0.0
        indiv = []
        for i, loss_fn in enumerate(self.loss_fn):
            l = weighting * self.loss_weights[i] * loss_fn(preds, targets)
            indiv.append(l.item())
            total = total + l
        return total, indiv

    # ------------------------------------------------------------------
    # Optimizers / schedulers
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Return the optimizer, plus one Lightning scheduler dict per entry of
        ``scheduler_config``.

        Each scheduler entry is a dict with a ``"scheduler"`` key (the
        scheduler class or a ready instance) and its constructor arguments;
        the Lightning-specific keys ``monitor`` / ``interval`` / ``frequency`` /
        ``strict`` are extracted into the Lightning dict.
        """
        schedulers = []
        if self.scheduler_config:
            for sch_cfg in self.scheduler_config:
                sch_cfg = dict(sch_cfg)  # copy: never mutate the caller's config
                scheduler = sch_cfg.pop("scheduler")

                # Extract the Lightning-specific keys *before* constructing the
                # scheduler (they are not constructor arguments of e.g.
                # ReduceLROnPlateau).
                lightning_keys = ("interval", "monitor", "frequency", "strict")
                lightning_params = {k: sch_cfg.pop(k) for k in lightning_keys if k in sch_cfg}

                if isinstance(scheduler, type):
                    scheduler = scheduler(self.optimizer, **sch_cfg)

                sch_dict = {
                    "scheduler": scheduler,
                    "interval": lightning_params.get("interval", "epoch"),
                    "frequency": lightning_params.get("frequency", 1),
                    "strict": lightning_params.get("strict", True),
                }
                if "monitor" in lightning_params:
                    sch_dict["monitor"] = lightning_params["monitor"]
                schedulers.append(sch_dict)

        if schedulers:
            return [self.optimizer], schedulers
        return self.optimizer

    def on_train_epoch_start(self):
        self.train_loss_avg.reset()
        self._train_sums = {}
        self._n_train_batches = 0

    def on_validation_epoch_start(self):
        self._val_sums = {}
        self._n_val_batches = 0

    # ------------------------------------------------------------------
    # To be implemented by the subclasses
    # ------------------------------------------------------------------

    def _forward_window(self, window, cond, t, time_idx, B, H, W):
        """Map one input window ``(B, n_slices * C, H, W)`` to ``(B, C, H, W)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # AR curriculum
    # ------------------------------------------------------------------

    def _ar_steps_for_epoch(self, T_unroll: int, epoch: Optional[int] = None) -> int:
        """Number of autoregressive steps for a given epoch (default: current).

        Linear curriculum between ``min_autoregressive_steps`` (before
        ``start_autoregressive``) and ``max_autoregressive_steps`` (after
        ``final_autoregressive``), capped by the number of available
        predictions ``T_unroll - n_slices - 1``.
        """
        n = self.n_slices
        available = max(T_unroll - n - 1, 0)
        max_AR_steps = int(min(available, self.train_config.max_autoregressive_steps))
        min_AR_steps = int(min(max(self.train_config.min_autoregressive_steps, 0), available))

        start_ep = self.train_config.start_autoregressive
        final_ep = self.train_config.final_autoregressive
        epoch = self.current_epoch if epoch is None else epoch

        if start_ep is not None and final_ep is not None:
            if epoch < start_ep:
                return min_AR_steps
            if epoch >= final_ep:
                return max_AR_steps
            progress = (epoch - start_ep) / (final_ep - start_ep)
            return int(min_AR_steps + progress * (max_AR_steps - min_AR_steps))
        return min_AR_steps

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _time_arg(self, time_idx, t: int, B: int):
        """Normalized time index for models whose forward accepts ``time``.

        Returned as a 2-D ``(B, 1)`` tensor, as expected by the time
        embeddings of e.g. ``GalerkinTransformer``.
        """
        time = (time_idx + t) / self.train_config.time_normalization
        if time.dim() == 0:
            time = time.expand(B)
        if time.dim() == 1:
            time = time.unsqueeze(-1)
        return time

    def _rollout(self, inputs, time_idx, cond, training: bool, store_fields: bool = False, on_step=None) -> dict:
        """Run the teacher-forcing / AR rollout over the time axis.

        Args:
            inputs: Window tensor ``(B, T, C, H, W)``.
            time_idx: Starting time index of each window (see :meth:`_time_arg`).
            cond: Optional conditioning field (or ``None``).
            training: If True apply the AR curriculum, temporal weighting and
                input noise; if False run the full rollout with ground-truth
                teacher forcing for the first prediction only.
            store_fields: If True also collect the detached per-step
                predictions and targets (lists of ``(B, C, H, W)`` in step
                order) under ``"preds"`` / ``"targets"`` — used by the
                evaluation driver to compute field-level metrics.
            on_step: Optional callback ``on_step(preds, targets)`` invoked with
                the detached ``(B, C, H, W)`` pair after every prediction step.
                Lets a caller consume each step immediately (e.g. an incremental
                metric accumulator) without keeping all fields in memory —
                preferred over ``store_fields`` for long rollouts.

        Returns:
            Dict with the accumulated (unnormalized) losses, the RMSE, the
            per-AR-step RMSEs and bookkeeping; plus ``"preds"``/``"targets"``
            when ``store_fields`` is set.
        """
        B, T, C, H, W = inputs.shape
        n = self.n_slices
        total_steps = T - n  # number of predictions in the rollout

        # Number of AR steps: curriculum for training, full rollout for validation.
        AR_count = self._ar_steps_for_epoch(T) if training else max(total_steps - 1, 0)
        gamma = self.train_config.temporal_weighting
        noise_level = self.train_config.noise_level if training else 0.0

        loss = 0.0
        indiv_losses = [0.0] * len(self.loss_fn)
        derivative_loss = 0.0
        RMSE = 0.0
        step_rmses = []
        stored_preds: list = []
        stored_targets: list = []
        preds = None
        current_input = None
        AR_preds = 0
        num_predictions = 0
        ARmode = False

        for t in range(n - 1, T - 1):
            # ------------------------------------------------------------------ input
            ARmode = (AR_count > 0) and (preds is not None) and (num_predictions >= total_steps - AR_count)
            if ARmode:  # autoregressive: feed the previous prediction back in
                current_input = torch.cat([current_input[:, 1:, ...], preds.unsqueeze(1)], dim=1)
                AR_preds += 1
                weighting = gamma**AR_preds if training else 1.0
            else:  # teacher forcing: use the ground-truth window
                current_input = inputs[:, t - (n - 1) : t + 1, ...]
                if training and noise_level > 0:
                    current_input = add_noise(current_input, noise_level, max_val=0.1, positive=False)
                weighting = 1.0

            # ------------------------------------------------------------------ prediction
            window = current_input.reshape(B, -1, H, W)
            out = self._forward_window(window, cond, t, time_idx, B, H, W)
            x_last = current_input[:, -1, ...]
            if self.predict_derivative:
                derivative = out
                preds = x_last + self.dt * out
            else:
                preds = out
                derivative = (preds - x_last) / self.dt

            targets_t = inputs[:, t + 1, ...]

            if store_fields or on_step is not None:
                p_det = preds.detach()
                t_det = targets_t.detach()
                if store_fields:
                    stored_preds.append(p_det)
                    stored_targets.append(t_det)
                if on_step is not None:
                    on_step(p_det, t_det)

            # ------------------------------------------------------------------ losses
            step_loss, step_indiv = self._weighted_losses(preds, targets_t, weighting)
            loss = loss + step_loss
            for i, v in enumerate(step_indiv):
                indiv_losses[i] += v

            if self.derivative_loss_weight > 0:
                d_loss = (
                    weighting
                    * self.derivative_loss_weight
                    * self.derivative_loss_fn(derivative, targets_t, x_last, self.dt)
                )
                derivative_loss += d_loss.item()
                loss = loss + d_loss

            with torch.no_grad():
                rmse = torch.sqrt(torch.mean((preds - targets_t) ** 2))
                RMSE = RMSE + rmse
                if training and ARmode:
                    step_rmses.append(rmse.item())

            num_predictions += 1

            # BPTT truncation every k autoregressive steps
            if AR_preds > 0 and self.detach_every_k > 0 and AR_preds % self.detach_every_k == 0:
                preds = preds.detach()

        result = {
            "loss": loss,
            "indiv_losses": indiv_losses,
            "derivative_loss": derivative_loss,
            "rmse": RMSE,
            "step_rmses": step_rmses,
            "ARmode": ARmode,
            "AR_steps": AR_count,
            "num_predictions": num_predictions,
        }
        if store_fields:
            result["preds"] = stored_preds
            result["targets"] = stored_targets
        return result

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def _finalize(self, res: dict, device) -> dict:
        """Normalize the rollout result by the number of predictions."""
        n = res["num_predictions"]
        if n > 0:
            return {
                "loss": res["loss"] / n,
                "indiv_losses": [v / n for v in res["indiv_losses"]],
                "derivative_loss": res["derivative_loss"] / n if self.derivative_loss_weight > 0 else 0.0,
                "rmse": res["rmse"] / n,
                "step_rmses": res["step_rmses"],
            }
        return {
            "loss": res["loss"],
            "indiv_losses": res["indiv_losses"],
            "derivative_loss": 0.0,
            "rmse": torch.zeros((), device=device),
            "step_rmses": [],
        }

    def _log_terms(self, terms: dict, prefix: str):
        """Log the per-loss terms, the derivative loss and the RMSE."""
        for i in range(len(self.loss_fn)):
            self.log(f"{prefix}loss_{i}", terms["indiv_losses"][i], on_step=True, on_epoch=True, prog_bar=True)
        if self.derivative_loss_weight > 0:
            self.log(f"{prefix}nMSEdt", terms["derivative_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}RMSE", terms["rmse"], on_step=True, on_epoch=True, prog_bar=True)

    def _accumulate_terms(self, sums: dict[str, float], terms: dict) -> None:
        """Add one batch's normalized ``terms`` to the per-epoch running sums."""
        loss = terms["loss"]
        sums["loss"] = sums.get("loss", 0.0) + (loss.item() if torch.is_tensor(loss) else float(loss))
        for i, v in enumerate(terms["indiv_losses"]):
            key = f"loss_{i}"
            sums[key] = sums.get(key, 0.0) + float(v)
        if self.derivative_loss_weight > 0:
            d = terms["derivative_loss"]
            sums["derivative_loss"] = sums.get("derivative_loss", 0.0) + (d.item() if torch.is_tensor(d) else float(d))

    def training_step(self, batch, batch_idx):
        inputs, time_idx, cond = self._unpack_batch(batch)
        res = self._rollout(inputs, time_idx, cond, training=True)
        terms = self._finalize(res, inputs.device)

        self.train_loss_avg.update(terms["loss"])
        self._accumulate_terms(self._train_sums, terms)
        self._n_train_batches += 1

        for i, rmse in enumerate(terms["step_rmses"]):
            self.log(f"step_{i + 1}", rmse, on_step=False, on_epoch=True, prog_bar=True)
        self._log_terms(terms, prefix="")
        self.log("AR_steps", int(res["AR_steps"]), prog_bar=True)
        self.log("loss", terms["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "avg_loss",
            self.train_loss_avg.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("lr", self.optimizer.param_groups[0]["lr"], prog_bar=True)
        return terms["loss"]

    def validation_step(self, batch, batch_idx):
        inputs, time_idx, cond = self._unpack_batch(batch)
        res = self._rollout(inputs, time_idx, cond, training=False)
        terms = self._finalize(res, inputs.device)

        self._accumulate_terms(self._val_sums, terms)
        self._n_val_batches += 1

        self._log_terms(terms, prefix="val_")
        self.log("val_loss", terms["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return terms["loss"]

    def on_validation_epoch_end(self):
        """Print the per-epoch average losses for train and validation.

        Called by Lightning once both loops of the epoch have completed, so a
        single line reports the mean (over batches) total loss, each weighted
        component and the derivative regularizer for both splits.
        """
        trainer = getattr(self, "_trainer", None)  # `self.trainer` raises when detached
        if trainer is not None and not trainer.is_global_zero:
            return  # DDP: print once only
        if not self._train_sums or not self._val_sums:
            return

        def fmt(sums: dict[str, float], n_batches: int) -> str:
            parts = [f"loss={sums['loss'] / n_batches:.4e}"]
            for i in range(len(self.loss_fn)):
                parts.append(f"loss_{i}={sums.get(f'loss_{i}', 0.0) / n_batches:.3e}")
            if self.derivative_loss_weight > 0:
                parts.append(f"nMSEdt={sums['derivative_loss'] / n_batches:.3e}")
            return " ".join(parts)

        print(
            f" | Average: "
            f"train ({self._n_train_batches} batches): {fmt(self._train_sums, self._n_train_batches)} | "
            f"val ({self._n_val_batches} batches): {fmt(self._val_sums, self._n_val_batches)}"
        )


# ---------------------------------------------------------------------------
# Multi-step NO wrapper (window models: FNO, LNO, SwinNO, UNO, Transolver, ...)
# ---------------------------------------------------------------------------


class MultiStepNOModel(MultiStepLightningBase):
    """Multi-step prediction Lightning module for window-based neural operators.

    Wraps any model whose forward maps a window ``(B, n_slices * C, H, W)`` to
    ``(B, C, H, W)``:

    - ``model(x)`` (e.g. ``UNet``, ``Resnet``, ``ConvNext``),
    - ``model(x, cond=...)`` (e.g. ``FNO``, ``LatentNO``, ``LatentGalerkin``,
      ``ITLNO``, ``LatentTransolver``, ``SwinNO``, ``MultiscaleSWINNO``, ``UNO``,
      ``Transolver``),
    - ``model(x, time=...)`` (e.g. ``GalerkinTransformer``).

    The optional keywords are passed **only if the wrapped model accepts them**
    (detected at init), so a single wrapper serves all of the above.

    The output is interpreted according to
    :attr:`TrainingConfig.predict_derivative` (derivative vs absolute value).
    See :class:`MultiStepLightningBase` for the rollout / AR-curriculum details.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train_config: Optional[TrainingConfig] = None,
        scheduler_config: Optional[list] = None,
    ):
        if _is_deeponet(model):
            raise ValueError(
                f"{type(model).__name__} has a (branch, trunk) forward signature (u, y) and cannot "
                "be wrapped by MultiStepNOModel. Use pynop.DeepONetModel instead."
            )
        super().__init__(model, optimizer, train_config, scheduler_config)

    def _forward_window(self, window, cond, t, time_idx, B, H, W):
        if cond is not None and not self._accepts_cond:
            raise ValueError(
                f"The batch provides a conditioning field, but {type(self.model).__name__}.forward "
                "does not accept a `cond` argument."
            )
        kwargs = {}
        if cond is not None:
            kwargs["cond"] = cond
        if self._accepts_time:
            kwargs["time"] = self._time_arg(time_idx, t, B)
        return self.model(window, **kwargs)


def _is_deeponet(model) -> bool:
    """Heuristic: a DeepONet exposes both a ``branchnet`` and a ``trunknet``."""
    return hasattr(model, "branchnet") and hasattr(model, "trunknet")


# ---------------------------------------------------------------------------
# DeepONet wrapper (branch / trunk paradigm)
# ---------------------------------------------------------------------------


class DeepONetModel(MultiStepLightningBase):
    r"""Multi-step prediction wrapper for :class:`~pynop.models.deeponet.DeepONet`.

    ``DeepONet.forward(u, y)`` evaluates a function: the branch net consumes the
    *function values* ``u`` and the trunk net the *query coordinates* ``y``.
    For a multi-step prediction with a window of ``n_slices`` slices
    ``(B, n*C, H, W)`` this maps to:

    - **branch input**: the window flattened as a single function vector
      ``u`` → tensor ``(B, n*C*H*W)`` (the branch MLP must be built with
      ``branchnet[0] = n*C*H*W``);
    - **trunk input**: the output-grid coordinates in ``[-1, 1]²`` →
      ``(H*W, 2)`` (``trunknet[0] = 2``);
    - **output**: ``(B, H*W)`` reshaped back to ``(B, 1, H, W)``
      (a ``(B*H*W, C)`` decoder output is also accepted).

    The ``predict_derivative`` / derivative-loss / AR-curriculum semantics are
    identical to :class:`MultiStepNOModel`.
    """

    def _forward_window(self, window, cond, t, time_idx, B, H, W):
        if cond is not None:
            raise ValueError("DeepONetModel: the wrapped DeepONet does not accept a conditioning field.")
        # The branch net receives the whole window as one flat function vector
        # (B, n*C*H*W); a per-point (B, H*W, n*C) layout would break the
        # DeepONet einsum (the branch output dim-1 must equal the trunk width).
        u = window.reshape(B, -1)  # (B, n*C*H*W)
        # The trunk net is a plain MLP over (H*W, 2) — a batch dimension would
        # be carried through its Linears and break the DeepONet einsum.
        y = self._coords(H, W, window.device).squeeze(0)  # (H*W, 2)
        out = self.model(u, y)
        return self._to_grid(out, B, H, W)

    @staticmethod
    def _coords(H: int, W: int, device) -> torch.Tensor:
        """Spatial query coordinates ``(1, H*W, 2)`` in ``[-1, 1]²``."""
        h = torch.linspace(-1, 1, H, device=device)
        w = torch.linspace(-1, 1, W, device=device)
        grid_h, grid_w = torch.meshgrid(h, w, indexing="ij")
        return torch.stack([grid_h, grid_w], dim=-1).view(1, H * W, 2)

    @staticmethod
    def _to_grid(out: torch.Tensor, B: int, H: int, W: int) -> torch.Tensor:
        """Reshape a DeepONet output back to a ``(B, C, H, W)`` field.

        Accepts ``(B, H*W)`` (scalar output), ``(B*H*W, C)`` (decoder output),
        ``(B, H*W, C)`` (per-point output) or an already-channel-first 4-D field.
        """
        if out.ndim == 2 and out.shape[0] == B and out.shape[1] == H * W:
            return out.view(B, 1, H, W)
        if out.ndim == 2 and out.shape[0] == B * H * W:
            return out.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if out.ndim == 3 and out.shape[0] == B and out.shape[1] == H * W:
            return out.permute(0, 2, 1).contiguous().view(B, out.shape[2], H, W)
        if out.ndim == 4:
            return out
        raise ValueError(
            f"DeepONetModel: cannot interpret the model output shape {tuple(out.shape)} "
            f"for a batch of {B} images of size {H}x{W}."
        )
