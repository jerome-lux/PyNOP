"""Command-line training driver for PyNOP multi-step prediction models.

This module replaces the legacy in-process training loops with a single,
config-driven entry point that reproduces the PDEBench two-phase protocol of
the ``training_*_PDEBench.ipynb`` notebooks:

* **Phase 1 — pretraining**: small ensemble (e.g. 40 simulations), restricted
  time range, short window stride, AR curriculum ramping from
  ``min_AR_steps`` to ``max_AR_steps`` between epochs ``start_AR`` and
  ``final_AR``.
* **Phase 2 — full training**: larger ensemble (e.g. 400 simulations), full
  time range, fixed AR steps at the maximum.

Usage
-----
::

    python -m pynop.training.train MODEL_CONFIG.json TRAINING_CONFIG.json [--pretrain] [options]

Running the file directly (``python pynop/training/train.py ...``) works as well,
from any working directory — the package context is restored automatically.

``--pretrain`` selects the ``"pretraining"`` section of the training config;
without it the ``"fulltraining"`` section is used.

ModelConfig.json
----------------
::

    {
        "name":   "FNOvanilla",          # optional run name (default: model class name)
        "model":  "FNO",                 # class in pynop.models, or a dotted import path
        "params": { ... }                # constructor kwargs (JSON-serializable)
    }

String values matching an activation name (``"GELU"``, ``"SiLU"``, ...) are
resolved to the corresponding ``torch.nn`` class, so e.g.
``"activation": "GELU"`` works in place of ``nn.GELU``. Tuples may be written
as lists.

TrainingConfig.json
-------------------
::

    {
        "input_steps": 10,               # n_slices: past slices given to the network
        "dataset": {                     # PDEBench dataset (only type supported so far)
            "type":         "pdebench",
            "path":         "/.../133017.hdf5",
            "batch_size":   5,
            "num_workers":  10,          # optional, default 0
            "load_in_ram":  true,        # optional, default false
            "shift":        0.0,         # optional normalization (x + shift) * scale
            "scale":        1.0,
            "split_ratio":  0.9,         # train fraction of the simulation IDs
            "random_seed":  42           # seeds ID shuffle + torch/PL RNGs
        },
        "shared_params": {               # common to both phases (notebook shared_params)
            "timestep":             5e-2,   # dt between slices
            "loss_weights":         [1e4, 1e4, 1e4],
            "temporal_weighting":   1.1,    # gamma: AR-step k loss weight = gamma**k
            "detach_grad_steps":    40,     # BPTT truncation (0 disables)
            "derivative_loss_weight": 1.0,  # 0 disables the derivative regularizer
            "predict_derivative":   true,   # output is du/dt -> u_{t+dt} = u_t + dt*out
            "noise_level":          0.0,    # teacher-forcing input noise
            "time_normalization":   101     # divisor for models accepting `time`
        },
        "losses": [                      # ordered list; weights match by position
            {"name": "DWMSELoss"},
            {"name": "SpectralLoss", "params": {"beta": 2}},
            {"name": "SobolevLoss",  "params": {"l1": 1.0, "l2": 1.0}}
        ],
        "optimizer": {                   # torch.optim class name + constructor kwargs
            "name": "AdamW",
            "weight_decay": 5e-4,
            "betas": [0.9, 0.99]
        },
        "scheduler": {                   # optional; see below for tokens
            "name": "OneCycleLR",
            "div_factor": 1e3,
            "final_div_factor": 1e2,
            "pct_start": 0.02,
            "total_steps": "$total_steps",
            "interval": "step"
        },
        "trainer": {                     # optional pl.Trainer overrides
            "gradient_clip_val": 1.0,    # 0 disables clipping (default)
            "log_every_n_steps": 50
        },
        "output": {                      # where runs are written
            "root":       "/run/media/jlux/SSD2/PDETraining",
            "experiment": "PDEBENCH"
        },
        "loggers": ["csv", "aim"],         # optional, default ["csv", "tensorboard"]
        "aim": {                           # optional; only used if "aim" is in loggers
            "experiment": "PDEBENCH",      # Aim experiment (default: output.experiment)
            "home":       null             # Aim repo path (default: ~/.aim)
        },
        "pretraining":  { ...phase params... },
        "fulltraining": { ...phase params... }
    }

Logger names: ``"csv"``, ``"tensorboard"``/``"tb"`` and ``"aim"``. The Aim entry
installs an :class:`~pynop.training.aim_logger.AimCallback` (not a Lightning
logger) that mirrors the same metrics to an Aim run — per-batch ``*_step``
metrics at ``x = global_step``, epoch-level metrics at ``x = epoch`` — so runs
can be compared in the Aim UI (``aim up``). Existing runs can be imported into
the same experiment with :mod:`pynop.training.import_logs`.

Phase parameters (both sections use the same keys):

* ``delta_ts``      — window stride of :class:`~pynop.PDEBenchDataSet`.
* ``min_id`` / ``max_id`` — time range kept for windows (``null`` = full).
* ``n_samples``     — number of simulations used (``null`` = all).
* ``split_ratio``   — optional per-phase override of the dataset split.
* ``lr``            — learning rate (OneCycleLR maximum LR, optimizer base LR).
* ``max_epochs``    — number of epochs for this phase.
* ``start_AR`` / ``final_AR`` — AR curriculum ramp interval (epochs).
* ``min_AR_steps`` / ``max_AR_steps`` — AR step bounds; the window length is
  derived as ``window_size = max_AR_steps + input_steps + 1``.
* ``scheduler``     — optional dict of overrides merged over the base
  ``"scheduler"`` section (e.g. per-phase ``pct_start``).

Scheduler tokens: string values ``"$total_steps"`` and ``"$steps_per_epoch"``
are replaced by ``max_epochs * len(train_dataloader)`` and
``len(train_dataloader)`` respectively, so the config does not need to know
the dataset size. For OneCycleLR, an omitted ``max_lr`` defaults to the phase
``lr`` (as in the notebooks).

Checkpoint policy
-----------------
All checkpoints are written to::

    <output.root>/<output.experiment>/<phase logdir>/<model name>/<input_steps>steps_<timestamp>/

* ``last.ckpt`` — saved **every epoch from the start** (overwritten), so the
  weights of the last completed epoch are always available for resuming or
  warm-starting phase 2 from phase 1.
* ``best_RMSE.ckpt`` / ``best_val_RMSE.ckpt`` (top-k = 2) — monitoring starts
  **only once the rollout reaches the maximum number of AR steps**
  (epoch >= ``final_AR``), and not before: during the curriculum ramp the
  per-epoch metrics are computed over different numbers of AR steps and are
  therefore not comparable, so selecting a "best" model earlier would be
  meaningless. If ``final_AR >= max_epochs``, no best checkpoint is ever
  written (a warning is printed at startup).

Other options: ``--resume PATH`` performs a full Lightning resume from a
checkpoint (optimizer/scheduler/epoch restored); ``--init-from PATH`` loads
only the model weights from a previous run's ``last.ckpt`` (fresh optimizer —
the phase-1 -> phase-2 warm start of the notebooks).

The module contains no visualization or evaluation code: it trains, logs and
saves checkpoints only.
"""

import argparse
import datetime
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Direct-script execution support
# ---------------------------------------------------------------------------
# The canonical entry point is ``python -m pynop.training.train ...`` (or the
# installed ``pynop-train`` script). When this file is executed directly
# instead — e.g. ``python pynop/training/train.py ...`` from ``configs/`` —
# Python gives it no package context and the relative imports below would fail
# with "attempted relative import with no known parent package". Restore the
# package context (and make sure the repository root is importable) so both
# invocation styles work from any working directory.
if __package__ in (None, ""):
    _pkg_dir = Path(__file__).resolve().parents[1]  # .../PyNOP/pynop
    _repo_root = str(_pkg_dir.parent)               # .../PyNOP
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = f"{_pkg_dir.name}.training"

from ..dataloader.h5_dataloader import PDEBenchDataSet
from .aim_logger import AimCallback
from .lightning_model import (
    CurriculumCheckpoint,
    DeepONetModel,
    MultiStepNOModel,
    TrainingConfig,
)

__all__ = [
    "parse_args",
    "main",
    "run_training",
    "build_model",
    "build_losses",
    "build_optimizer",
    "build_scheduler_config",
    "build_pdebench_dataloaders",
    "select_phase",
    "build_aim_callback",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def load_json(path: PathLike) -> dict:
    """Load a JSON file and return its content as a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON object (expected to be an object/dict).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the top-level JSON value is not an object.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: the top-level JSON value must be an object, got {type(data).__name__}.")
    return data


def _resolve_activation(value: Any) -> Any:
    """Resolve a string activation name to its ``torch.nn`` class.

    Args:
        value: A config value; strings matching a known activation name are
            replaced by the corresponding class, everything else is returned
            unchanged (recursing into lists and dicts).

    Returns:
        The resolved value.
    """
    activations = {
        name: getattr(nn, name)
        for name in ("ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "ELU", "Mish", "Softplus")
    }
    if isinstance(value, str):
        return activations.get(value, value)
    if isinstance(value, list):
        return [_resolve_activation(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_activation(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Model / loss / optimizer / scheduler builders
# ---------------------------------------------------------------------------


def build_model(model_cfg: dict) -> nn.Module:
    """Instantiate a model from a ``ModelConfig.json`` dict.

    Args:
        model_cfg: Dict with keys ``"model"`` (class name in ``pynop.models``
            or a dotted import path), optional ``"name"`` and ``"params"``
            (constructor kwargs). String activation names are resolved to
            their ``torch.nn`` classes.

    Returns:
        The instantiated model (on CPU; the caller moves it to device via the
        Lightning trainer).

    Raises:
        KeyError: If the ``"model"`` key is missing.
        ValueError: If the class cannot be resolved or instantiation fails.
    """
    if "model" not in model_cfg:
        raise KeyError('ModelConfig must contain a "model" key (class name or dotted path).')

    name = str(model_cfg["model"])
    if "." in name:  # dotted import path, e.g. "pynop.models.FNO"
        module_path, cls_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_cls = getattr(module, cls_name)
    else:
        models_pkg = importlib.import_module("pynop.models")
        if not hasattr(models_pkg, name):
            available = sorted(n for n in dir(models_pkg) if n[0].isupper())
            raise ValueError(f"Model class {name!r} not found in pynop.models. Available: {available}")
        model_cls = getattr(models_pkg, name)

    params = _resolve_activation(model_cfg.get("params", {}))
    try:
        model = model_cls(**params)
    except TypeError as e:
        raise ValueError(f"Failed to instantiate {model_cls.__name__} with params {sorted(params)}: {e}") from e

    # The notebooks silence per-forward prints before training; do the same
    # unless the config explicitly sets `verbose`.
    if "verbose" not in model_cfg.get("params", {}) and hasattr(model, "verbose"):
        model.verbose = False
    return model


def build_losses(loss_specs: Optional[Sequence[Any]]) -> list:
    """Build loss modules from JSON specs.

    Args:
        loss_specs: List of entries, each either a string (loss name) or a
            dict ``{"name": ..., "params": {...}}``. Names are resolved in
            ``pynop.core.loss`` first, then in ``torch.nn``. If ``None``, a
            single :class:`torch.nn.MSELoss` is returned.

    Returns:
        List of instantiated loss modules (order preserved — it defines the
        position of each weight in ``loss_weights``).

    Raises:
        ValueError: If a loss name cannot be resolved or instantiation fails.
    """
    if not loss_specs:
        return [nn.MSELoss()]

    from ..core.loss import (  # local import: keep module import light
        DWMSELoss,
        MaskedSpectralLoss,
        NormalizedTimeDerivativeMSE,
        SobolevLoss,
        SpectralLoss,
        StructureLoss,
        WeightedMSE,
        nDWMSELoss,
    )

    registry = {
        "DWMSELoss": DWMSELoss,
        "nDWMSELoss": nDWMSELoss,
        "WeightedMSE": WeightedMSE,
        "SpectralLoss": SpectralLoss,
        "MaskedSpectralLoss": MaskedSpectralLoss,
        "StructureLoss": StructureLoss,
        "SobolevLoss": SobolevLoss,
        "NormalizedTimeDerivativeMSE": NormalizedTimeDerivativeMSE,
        "MSELoss": nn.MSELoss,
        "L1Loss": nn.L1Loss,
        "SmoothL1Loss": nn.SmoothL1Loss,
        "HuberLoss": nn.HuberLoss,
    }

    losses = []
    for spec in loss_specs:
        if isinstance(spec, str):
            name, params = spec, {}
        elif isinstance(spec, dict) and "name" in spec:
            name, params = str(spec["name"]), spec.get("params", {})
        else:
            raise ValueError(f"Invalid loss spec {spec!r}: expected a name string or {{'name': ..., 'params': ...}}.")
        if name not in registry:
            raise ValueError(
                f"Unknown loss {name!r}. Available losses: {sorted(registry)}."
            )
        try:
            losses.append(registry[name](**_resolve_activation(params)))
        except TypeError as e:
            raise ValueError(f"Failed to instantiate loss {name} with params {sorted(params)}: {e}") from e
    return losses


def build_optimizer(model: nn.Module, opt_cfg: Optional[dict], lr: float) -> torch.optim.Optimizer:
    """Build the optimizer from a JSON config dict.

    Args:
        model: Model whose (trainable) parameters are optimized.
        opt_cfg: Dict with ``"name"`` (a ``torch.optim`` class name, default
            ``"AdamW"``) plus constructor kwargs (e.g. ``weight_decay``,
            ``betas``). The learning rate is taken from the phase config and
            passed as ``lr``.
        lr: Learning rate for this phase.

    Returns:
        The instantiated optimizer over all parameters with ``requires_grad``.

    Raises:
        ValueError: If the optimizer name cannot be resolved or instantiation fails.
    """
    opt_cfg = dict(opt_cfg or {})
    name = opt_cfg.pop("name", "AdamW")
    if not hasattr(torch.optim, name):
        raise ValueError(f"Unknown optimizer {name!r}. Use a torch.optim class name (e.g. 'AdamW', 'SGD').")
    opt_cls = getattr(torch.optim, name)

    params = _resolve_activation(opt_cfg)  # e.g. betas list -> kept as-is; no activations expected here
    if "betas" in params:
        params["betas"] = tuple(params["betas"])
    try:
        return opt_cls(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, **params)
    except TypeError as e:
        raise ValueError(f"Failed to instantiate optimizer {name} with kwargs {sorted(params)}: {e}") from e


def build_scheduler_config(
    scheduler_cfg: Optional[dict], phase_lr: float, total_steps: int, steps_per_epoch: int
) -> Optional[list]:
    """Build the Lightning ``scheduler_config`` list from a JSON dict.

    Args:
        scheduler_cfg: Dict with ``"name"`` (a ``torch.optim.lr_scheduler``
            class name or ``"cosine_schedule_with_warmup"``), constructor
            kwargs, and optional Lightning keys ``interval`` / ``monitor`` /
            ``frequency`` / ``strict``. String tokens ``"$total_steps"`` and
            ``"$steps_per_epoch"`` are replaced by the computed values. For
            OneCycleLR an omitted ``max_lr`` defaults to ``phase_lr``.
        phase_lr: Learning rate of the current phase (OneCycleLR maximum LR).
        total_steps: ``max_epochs * len(train_dataloader)``.
        steps_per_epoch: ``len(train_dataloader)``.

    Returns:
        A list with a single scheduler dict in the format expected by
        :meth:`MultiStepLightningBase.configure_optimizers`, or ``None`` if no
        scheduler is configured (the optimizer then runs at constant LR).

    Raises:
        ValueError: If the scheduler name cannot be resolved, an unknown token
            is used, or instantiation fails.
    """
    if not scheduler_cfg:
        return None

    from .schedulers import cosine_schedule_with_warmup  # local import (package-level cycle safety)

    cfg = dict(scheduler_cfg)
    name = str(cfg.pop("name"))

    registry = {n: getattr(torch.optim.lr_scheduler, n) for n in ("OneCycleLR", "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "ExponentialLR")}
    registry["cosine_schedule_with_warmup"] = cosine_schedule_with_warmup

    if name not in registry:
        raise ValueError(f"Unknown scheduler {name!r}. Available: {sorted(registry)}")

    # Lightning-specific keys are consumed by configure_optimizers, not passed to the
    # scheduler constructor — keep them out of the token substitution below.
    lightning_keys = ("interval", "monitor", "frequency", "strict")
    lightning_params = {k: cfg.pop(k) for k in list(cfg) if k in lightning_keys}

    tokens = {"$total_steps": total_steps, "$steps_per_epoch": steps_per_epoch}
    for key, value in list(cfg.items()):
        if isinstance(value, str):
            if value not in tokens:
                raise ValueError(f"Unknown scheduler token {value!r} (key {key!r}). Available: {sorted(tokens)}")
            cfg[key] = tokens[value]

    if name == "OneCycleLR" and "max_lr" not in cfg:
        cfg["max_lr"] = phase_lr  # as in the notebooks: phase lr is the OneCycle maximum

    entry = {"scheduler": registry[name], **cfg, **lightning_params}
    return [entry]


# ---------------------------------------------------------------------------
# Phase selection / dataset
# ---------------------------------------------------------------------------


def select_phase(training_cfg: dict, pretrain: bool) -> tuple:
    """Select the training-phase section of a ``TrainingConfig.json`` dict.

    Args:
        training_cfg: Parsed training config.
        pretrain: If True use the ``"pretraining"`` section, otherwise the
            ``"fulltraining"`` section (mirrors the CLI ``--pretrain`` flag).

    Returns:
        Tuple ``(phase_name, phase_dict)``.

    Raises:
        KeyError: If the requested section is missing from the config.
    """
    phase_name = "pretraining" if pretrain else "fulltraining"
    if phase_name not in training_cfg or not isinstance(training_cfg[phase_name], dict):
        raise KeyError(
            f'TrainingConfig must contain a "{phase_name}" section '
            f"(use --pretrain to select the pretraining phase). Found sections: {sorted(training_cfg)}"
        )
    return phase_name, training_cfg[phase_name]


def build_pdebench_dataloaders(ds_cfg: dict, phase_cfg: dict, input_steps: int) -> tuple:
    """Build train/val :class:`DataLoader` pairs for the PDEBench dataset.

    Reproduces the notebook protocol: all simulation IDs are shuffled with a
    seeded RNG, truncated to ``n_samples``, then split by ``split_ratio`` on
    the *simulation IDs* (so both phases share the same subset of simulations).
    The window length is derived as ``max_AR_steps + input_steps + 1``.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config. Requires
            ``type == "pdebench"``, ``path`` and ``batch_size``; optional keys
            are ``num_workers``, ``load_in_ram``, ``shift``, ``scale``,
            ``split_ratio`` (default 0.9) and ``random_seed`` (default 42).
        phase_cfg: The selected phase section (provides ``delta_ts``,
            ``min_id``, ``max_id``, ``n_samples``, ``max_AR_steps``; an
            optional per-phase ``split_ratio`` override).
        input_steps: Number of past slices given to the network.

    Returns:
        Tuple ``(train_loader, val_loader, window_size)``.

    Raises:
        ValueError: If the dataset type is not supported or a required key is missing.
    """
    ds_type = str(ds_cfg.get("type", "")).lower()
    if ds_type != "pdebench":
        raise ValueError(f"Unsupported dataset type {ds_cfg.get('type')!r}: only 'pdebench' is supported for now.")
    if "path" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "path".')
    if "batch_size" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "batch_size".')

    path = Path(ds_cfg["path"])
    window_size = int(phase_cfg["max_AR_steps"]) + input_steps + 1
    delta_ts = phase_cfg.get("delta_ts")
    min_id = int(phase_cfg.get("min_id", 0))
    max_id = phase_cfg.get("max_id")
    n_samples = phase_cfg.get("n_samples")
    split_ratio = float(phase_cfg.get("split_ratio", ds_cfg.get("split_ratio", 0.9)))
    seed = int(ds_cfg.get("random_seed", 42))

    with h5py.File(path, "r") as f:
        all_ids = list(f.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    if n_samples is not None:
        all_sample_ids = all_ids[: min(len(all_ids), int(n_samples))]
    else:
        all_sample_ids = all_ids

    n_train = int(len(all_sample_ids) * split_ratio)
    train_ids, val_ids = all_sample_ids[:n_train], all_sample_ids[n_train:]
    if not train_ids or not val_ids:
        raise ValueError(
            f"Empty split: {len(train_ids)} train / {len(val_ids)} valid simulations "
            f"(n_samples={n_samples}, split_ratio={split_ratio}). Check the dataset config."
        )

    common = dict(
        T_unroll=window_size,
        step=int(delta_ts) if delta_ts is not None else None,
        load_in_ram=bool(ds_cfg.get("load_in_ram", False)),
        min_id=min_id,
        max_id=max_id,
        shift=float(ds_cfg.get("shift", 0.0)),
        scale=float(ds_cfg.get("scale", 1.0)),
    )
    train_set = PDEBenchDataSet(path, indexes=train_ids, **common)
    val_set = PDEBenchDataSet(path, indexes=val_ids, **common)

    batch_size = int(ds_cfg["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        val_kwargs["persistent_workers"] = True
    val_loader = DataLoader(val_set, shuffle=False, **val_kwargs)

    print(
        f"Dataset '{path.name}': {len(train_ids)} train / {len(val_ids)} valid simulations, "
        f"window_size={window_size}, stride={common['step']}, t in [{min_id}, {max_id if max_id is not None else 'end'}]"
    )
    return train_loader, val_loader, window_size


# ---------------------------------------------------------------------------
# TrainingConfig / callbacks / loggers
# ---------------------------------------------------------------------------


def build_training_config(shared: dict, losses: list, input_steps: int, phase_cfg: dict) -> TrainingConfig:
    """Build the :class:`TrainingConfig` from the shared + phase sections.

    Args:
        shared: The ``"shared_params"`` section (dt, loss weights, AR detach,
            derivative regularizer, noise, temporal weighting, ...).
        losses: Loss modules built by :func:`build_losses`.
        input_steps: Number of past slices given to the network (``n_slices``).
        phase_cfg: The selected phase section (AR curriculum bounds).

    Returns:
        The populated :class:`TrainingConfig`.
    """
    return TrainingConfig(
        dt=float(shared.get("timestep", 1.0)),
        start_autoregressive=phase_cfg.get("start_AR"),
        final_autoregressive=phase_cfg.get("final_AR"),
        min_autoregressive_steps=int(phase_cfg.get("min_AR_steps", 0)),
        max_autoregressive_steps=int(phase_cfg["max_AR_steps"]),
        detach_grad_steps=int(shared.get("detach_grad_steps", 4)),
        loss_fn=losses,
        loss_weights=shared.get("loss_weights", 1.0),
        derivative_loss_weight=float(shared.get("derivative_loss_weight", 1.0)),
        predict_derivative=bool(shared.get("predict_derivative", True)),
        noise_level=float(shared.get("noise_level", 0.0)),
        time_normalization=float(shared.get("time_normalization", 1.0)),
        n_slices=input_steps,
        temporal_weighting=float(shared.get("temporal_weighting", 1.2)),
    )


class _ScientificTQDMProgressBar(TQDMProgressBar):
    """TQDM progress bar printing float metrics in scientific notation."""

    def get_metrics(self, trainer, model):  # noqa: D102 - Lightning API signature
        items = super().get_metrics(trainer, model)
        return {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in items.items()}


def make_callbacks(logdir: Path, start_monitoring: int) -> list:
    """Build the checkpoint / logging callbacks.

    Checkpoint policy (see module docstring): ``last.ckpt`` is saved every
    epoch from the start; the monitored best checkpoints only start saving
    once the rollout reaches the maximum number of AR steps
    (epoch >= ``start_monitoring``, i.e. ``final_AR``).

    Args:
        logdir: Run directory where checkpoints are written.
        start_monitoring: Epoch at which best-model monitoring starts
            (the phase's ``final_AR``).

    Returns:
        List of Lightning callbacks.

    Notes:
        ``save_on_train_epoch_end=False`` pins every save to the end of
        validation, where both monitored metrics (train ``RMSE`` and
        ``val_RMSE``) are available in ``callback_metrics`` — with the PL 2.x
        default (auto), saving would happen at train-epoch-end instead.
    """
    return [
        # Always the weights of the last completed epoch (overwritten each epoch).
        ModelCheckpoint(dirpath=logdir, filename="last", save_top_k=1, save_on_train_epoch_end=False),
        # Best models: only once max AR steps are reached (metrics comparable from then on).
        CurriculumCheckpoint(
            start_epoch=start_monitoring,
            dirpath=logdir,
            monitor="RMSE",
            filename="best_RMSE",
            mode="min",
            save_top_k=2,
            save_on_train_epoch_end=False,
        ),
        CurriculumCheckpoint(
            start_epoch=start_monitoring,
            dirpath=logdir,
            monitor="val_RMSE",
            filename="best_val_RMSE",
            mode="min",
            save_top_k=2,
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        _ScientificTQDMProgressBar(leave=True),
    ]


def make_loggers(logdir: Path, names: Optional[Sequence[str]]) -> list:
    """Build the Lightning loggers from a list of names.

    Args:
        logdir: Run directory (CSV is written there, TensorBoard under ``tb_logs``).
        names: Iterable of logger names among ``"csv"`` and ``"tensorboard"``/``"tb"``.
            If None, both are enabled (notebook default); an empty list disables logging.
            ``"aim"`` is not a Lightning logger — it is filtered out by the caller
            and handled through :func:`build_aim_callback` instead.

    Returns:
        List of Lightning loggers.

    Raises:
        ValueError: On unknown logger names.
    """
    if names is None:
        names = ["csv", "tensorboard"]
    loggers = []
    for name in names:
        key = str(name).lower()
        if key == "csv":
            loggers.append(CSVLogger(save_dir=logdir))
        elif key in ("tensorboard", "tb"):
            loggers.append(TensorBoardLogger(save_dir=Path(logdir) / "tb_logs"))
        else:
            raise ValueError(f"Unknown logger {name!r}. Available: 'csv', 'tensorboard'.")
    return loggers


def build_aim_callback(
    logdir: Path,
    logger_names: Optional[Sequence[str]],
    aim_cfg: dict,
    *,
    experiment: str,
    model_name: str,
    phase_name: str,
    run_info: dict,
) -> Optional[AimCallback]:
    """Build the :class:`AimCallback` if ``"aim"`` is among the configured loggers.

    Args:
        logdir: Run directory (used for the default Aim run name).
        logger_names: The config's ``"loggers"`` list (``None`` = csv+tensorboard
            default, which does not include Aim).
        aim_cfg: The optional config ``"aim"`` section (``experiment``, ``home``).
        experiment: The config's ``output.experiment`` — default Aim experiment.
        model_name: Run name of the model config.
        phase_name: Selected training phase (``pretraining``/``fulltraining``).
        run_info: Bookkeeping dict written to ``run_info.json`` (source of the
            parameters stored with the Aim run).

    Returns:
        The callback, or ``None`` if Aim is not requested.
    """
    names = [str(n).lower() for n in logger_names] if logger_names is not None else []
    if "aim" not in names:
        return None

    aim_cfg = dict(aim_cfg or {})
    # Aim 3.x's own default repository is CWD-based (it would create a .aim/ in
    # whatever directory the driver is launched from) — default to ~/.aim instead.
    home = str(Path(aim_cfg["home"]).expanduser()) if aim_cfg.get("home") else str(Path.home() / ".aim")
    params = {
        "model_class": run_info["model"]["class"],
        "n_params": run_info["model"]["params"],
        "phase": phase_name,
        "lr": run_info["lr"],
        "max_epochs": run_info["max_epochs"],
        "input_steps": run_info["input_steps"],
        "window_size": run_info["window_size"],
        "seed": run_info["seed"],
    }
    return AimCallback(
        experiment=str(aim_cfg.get("experiment") or experiment),
        name=f"{phase_name}/{model_name}/{Path(logdir).name}",
        home=home,
        params=params,
    )


def resolve_device(device: str) -> str:
    """Resolve the ``--device`` option to a Lightning accelerator name.

    Args:
        device: One of ``"auto"``, ``"cuda"``, ``"cpu"``.

    Returns:
        ``"gpu"`` or ``"cpu"``.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    if device == "auto":
        return "gpu" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested (--device cuda) but no GPU is available.")
        return "gpu"
    return "cpu"


def _load_weights_only(model: nn.Module, path: PathLike) -> None:
    """Load model weights (only) from a checkpoint file.

    Accepts either a full Lightning checkpoint (the ``state_dict`` entry with
    the ``model.`` prefix is extracted and stripped) or a bare state dict.

    Args:
        model: Target model to load into.
        path: Path to the checkpoint file.

    Raises:
        ValueError: If the state dict cannot be matched strictly.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    has_prefix = any(k.startswith("model.") for k in state_dict)
    cleaned = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")} if has_prefix else dict(state_dict)
    try:
        model.load_state_dict(cleaned, strict=True)
    except RuntimeError as e:
        raise ValueError(f"Could not load weights from {path} into {type(model).__name__}: {e}") from e
    print(f"Weights loaded from {path} ({len(cleaned)} tensors).")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_training(
    model_config_path: PathLike,
    training_config_path: PathLike,
    pretrain: bool = False,
    device: str = "auto",
    output_root: Optional[PathLike] = None,
    resume_from: Optional[PathLike] = None,
    init_from: Optional[PathLike] = None,
) -> Path:
    """Run a full training phase from two JSON config files.

    Args:
        model_config_path: Path to the ``ModelConfig.json`` file.
        training_config_path: Path to the ``TrainingConfig.json`` file.
        pretrain: If True use the ``"pretraining"`` section, otherwise
            ``"fulltraining"``.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        output_root: Optional override of ``output.root`` from the config.
        resume_from: Optional checkpoint path for a full Lightning resume
            (optimizer/scheduler/epoch restored).
        init_from: Optional checkpoint path to warm-start the model weights
            only (fresh optimizer — phase-1 -> phase-2 handoff).

    Returns:
        The run log directory (checkpoints, logs and saved configs are there).
    """
    model_cfg = load_json(model_config_path)
    training_cfg = load_json(training_config_path)

    # ------------------------------------------------------------------ setup
    phase_name, phase_cfg = select_phase(training_cfg, pretrain)
    ds_cfg = training_cfg.get("dataset")
    if not isinstance(ds_cfg, dict):
        raise KeyError('TrainingConfig must contain a "dataset" section.')
    shared = training_cfg.get("shared_params", {})

    input_steps = int(training_cfg["input_steps"]) if "input_steps" in training_cfg else 10
    seed = int(ds_cfg.get("random_seed", 42))
    pl.seed_everything(seed, workers=True)

    accelerator = resolve_device(device)
    torch.set_float32_matmul_precision("highest")

    # ------------------------------------------------------------------ model
    model_name = str(model_cfg.get("name") or model_cfg["model"])
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__} ({n_params / 1e6:.2f} M params), run name '{model_name}', phase '{phase_name}'")

    # ------------------------------------------------------------------ data
    train_loader, val_loader, window_size = build_pdebench_dataloaders(ds_cfg, phase_cfg, input_steps)
    max_epochs = int(phase_cfg["max_epochs"])
    total_steps = max_epochs * len(train_loader)
    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------ losses / optimizer / scheduler
    losses = build_losses(training_cfg.get("losses"))
    train_config = build_training_config(shared, losses, input_steps, phase_cfg)

    lr = float(phase_cfg["lr"])
    optimizer = build_optimizer(model, training_cfg.get("optimizer"), lr=lr)

    base_scheduler = dict(training_cfg.get("scheduler") or {})
    if "scheduler" in phase_cfg:  # per-phase overrides merged over the base section
        base_scheduler.update(phase_cfg["scheduler"])
    scheduler_config = build_scheduler_config(base_scheduler, phase_lr=lr, total_steps=total_steps, steps_per_epoch=steps_per_epoch)

    # ------------------------------------------------------------------ lightning module
    if hasattr(model, "branchnet") and hasattr(model, "trunknet"):
        lightning_model = DeepONetModel(model=model, optimizer=optimizer, train_config=train_config, scheduler_config=scheduler_config)
    else:
        lightning_model = MultiStepNOModel(model=model, optimizer=optimizer, train_config=train_config, scheduler_config=scheduler_config)

    # ------------------------------------------------------------------ run directory / bookkeeping
    output_cfg = training_cfg.get("output", {})
    root = Path(output_root if output_root is not None else output_cfg.get("root", "./PDETraining"))
    experiment = str(output_cfg.get("experiment", "PDEBENCH"))
    phase_logdir = str(phase_cfg.get("logdir", phase_name))
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = root / experiment / phase_logdir / model_name / f"{input_steps}steps_{now}"
    logdir.mkdir(parents=True, exist_ok=True)

    final_ar = int(phase_cfg.get("final_AR", 0))
    if final_ar >= max_epochs:
        print(
            f"WARNING: final_AR ({final_ar}) >= max_epochs ({max_epochs}): the rollout never reaches "
            "the maximum AR steps, so no best checkpoint will be saved (only last.ckpt)."
        )

    run_info = {
        "timestamp": now,
        "phase": phase_name,
        "pretrain": pretrain,
        "model": {"name": model_name, "class": type(model).__name__, "params": n_params},
        "input_steps": input_steps,
        "window_size": window_size,
        "dataset": {
            "path": str(ds_cfg["path"]),
            "delta_ts": phase_cfg.get("delta_ts"),
            "min_id": phase_cfg.get("min_id", 0),
            "max_id": phase_cfg.get("max_id"),
            "n_samples_used": len(train_loader.dataset.sample_ids) + len(val_loader.dataset.sample_ids),
        },
        "train_ids": list(train_loader.dataset.sample_ids),
        "val_ids": list(val_loader.dataset.sample_ids),
        "lr": lr,
        "max_epochs": max_epochs,
        "total_steps": total_steps,
        "checkpoint_gate_epoch": final_ar,  # best checkpoints start saving at this epoch
        "device": accelerator,
        "seed": seed,
    }
    with open(logdir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=4)
    with open(logdir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_cfg, f, indent=4)
    with open(logdir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4)

    print(f"Total number of steps: {total_steps} ({max_epochs} epochs x {steps_per_epoch} batches)")
    print(
        f"Checkpoints in {logdir}: last.ckpt every epoch; best_RMSE/best_val_RMSE from "
        f"epoch {final_ar} (max AR steps reached)."
    )

    # ------------------------------------------------------------------ trainer
    trainer_cfg = training_cfg.get("trainer", {})
    logger_names = training_cfg.get("loggers")
    callbacks = make_callbacks(logdir, start_monitoring=final_ar)
    aim_cb = build_aim_callback(
        logdir,
        logger_names,
        training_cfg.get("aim"),
        experiment=experiment,
        model_name=model_name,
        phase_name=phase_name,
        run_info=run_info,
    )
    if aim_cb is not None:
        callbacks.append(aim_cb)
    other_logger_names = None if logger_names is None else [n for n in logger_names if str(n).lower() != "aim"]
    loggers = make_loggers(logdir, other_logger_names)

    trainer_kwargs = dict(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
        logger=loggers if loggers else False,
        num_sanity_val_steps=0,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
    )
    gradient_clip_val = float(trainer_cfg.get("gradient_clip_val", 0.0))
    if gradient_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
        trainer_kwargs["gradient_clip_algorithm"] = str(trainer_cfg.get("gradient_clip_algorithm", "norm"))

    trainer = pl.Trainer(**trainer_kwargs)

    # ------------------------------------------------------------------ (re)start
    if init_from is not None:
        _load_weights_only(lightning_model.model, Path(init_from))
    ckpt_path = str(Path(resume_from)) if resume_from is not None else None
    if ckpt_path is not None:
        print(f"Resuming from checkpoint {ckpt_path}")

    trainer.fit(
        lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    # Record the Aim run hash in run_info.json for traceability (aim up / import_logs).
    if aim_cb is not None and aim_cb.run_hash is not None:
        try:
            info = json.loads((logdir / "run_info.json").read_text(encoding="utf-8"))
            info["aim_run"] = {"hash": aim_cb.run_hash, "experiment": aim_cb.experiment}
            (logdir / "run_info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 - bookkeeping only
            print(f"WARNING: could not record the Aim run hash in run_info.json ({type(e).__name__}: {e})")

    return logdir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="pynop.train",
        description=(
            "Train a PyNOP multi-step prediction model from JSON configs "
            "(PDEBench dataset, two-phase protocol of the training notebooks)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_config", type=Path, help="Path to ModelConfig.json (model class + constructor params).")
    parser.add_argument("training_config", type=Path, help="Path to TrainingConfig.json (dataset, losses, phases).")
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help='Use the "pretraining" section of the training config (default: "fulltraining").',
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override output.root from the training config.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Full Lightning resume from a checkpoint (e.g. <logdir>/last.ckpt): optimizer/scheduler/epoch restored.",
    )
    parser.add_argument(
        "--init-from",
        dest="init_from",
        type=Path,
        default=None,
        help="Load only the model weights from a previous run's checkpoint (fresh optimizer; phase-1 -> phase-2 warm start).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The run log directory.
    """
    args = parse_args(argv)
    return run_training(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        pretrain=args.pretrain,
        device=args.device,
        output_root=args.output_root,
        resume_from=args.resume,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
