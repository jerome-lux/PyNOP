"""Command-line evaluation driver for PyNOP multi-step prediction models.

Companion to :mod:`pynop.training.train`: it loads a trained checkpoint and runs
the **full autoregressive rollout of every trajectory** in the evaluation set,
reporting field-level metrics (MSE, RMSE, normalized MSE / relative L2, MAE,
max error, R²), per-band spectral errors in Fourier space (PDEBench-style radial
bands) and the evolution of all these errors along the trajectory.

Protocol (same as the ``## Multi-step model eval`` cells of the PDEBench
training notebooks)
-------------------------------------------------------------
For **each simulation** (trajectory = all snapshots for one initial condition):

1. load the full trajectory on the phase time range ``[min_id, max_id]``;
2. feed the first ``input_steps`` slices to the model and predict every next
   slice autoregressively until the end of the trajectory — the first
   prediction is teacher-forced on the ground-truth window, all subsequent
   ones consume the previous prediction (rolling window);
3. pool the per-step errors over **all** trajectories: global metrics plus a
   per-time-step curve (RMSE / MAE / R² / nRMSE and, by default, the per-band
   spectral error at every step).

Usage
-----
::

    python -m pynop.training.eval MODEL_CONFIG.json TRAINING_CONFIG.json --checkpoint PATH [options]

Running the file directly (``python pynop/training/eval.py ...``) works as well,
from any working directory — the package context is restored automatically.

The model / training configs are the same files used for training (the phase
section — ``--pretrain`` selects ``"pretraining"``, otherwise ``"fulltraining"``
— provides the time range: ``min_id`` / ``max_id``).

Evaluation set
--------------
* ``--split val`` (default): the validation split of the training protocol — by
  construction disjoint from the training samples. If a ``run_info.json`` is
  found next to the checkpoint, its stored ``train_ids`` / ``val_ids`` are used
  as ground truth (robust against config drift after training).
* ``--split test``: simulations drawn from the *unused pool* — IDs that were
  never part of any training phase. With the seeded protocol the pretraining
  subset is contained in the full-training one, so "beyond ``n_samples``" means
  unseen by both phases. Falls back to the validation split (with a warning)
  when no unused simulations remain — e.g. when training already consumed every
  simulation of the file.

Metric normalization
--------------------
The normalized metrics are **relative L2** errors (PDEBench's "normalized
RMSE"): per sample and channel ``‖pred - target‖₂ / ‖target‖₂``, averaged over
all samples/channels. The ratio is scale-invariant, so it stays comparable
across datasets with different amplitudes; for zero-mean-normalized fields
``(x - μ)/σ`` the denominator equals the field standard deviation and the
metric remains a meaningful relative error (this is exactly how PDEBench uses
it on their zero-mean/unit-variance data). R² is reported as well: it is
invariant to *any* affine rescaling (shift and scale) and reduces to
``1 - nRMSE²`` per sample for zero-mean data. Normalizations by
``mean(target)²`` would degenerate on zero-mean fields and are deliberately not
used (see :mod:`pynop.training.metrics` for details).

All metrics are computed on the **model-space** data, i.e. after the dataset
``(x + shift) * scale`` transform — the same space the network was trained in.

Output: a formatted table on stdout, ``eval_results_<split>.json`` (every
scalar metric, the full per-step error curves and complete bookkeeping so
results are self-contained) and two figures with English labels —
``eval_error_evolution_<split>.png`` (RMSE / nRMSE / MAE / R² vs time step)
and ``eval_spectral_evolution_<split>.png`` (per-band spectral error vs time
step) — written next to the checkpoint (or to ``--output-dir``).

Optional ``"eval"`` section of the training config (all keys optional)::

    {
        "spectral_bands": [[0, 4], [4, 12], [12, null]],  # radial bands (FFT bin-index units, PDEBench iLow/iHigh)
        "spectral_relative": true,                        # false -> absolute band energies
        "per_step_spectral": true,                        # pool the per-band spectral error at every step (for plots)
        "n_test": 40                                      # default size of the test split
    }
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Direct-script execution support
# ---------------------------------------------------------------------------
# The canonical entry point is ``python -m pynop.training.eval ...`` (or the
# installed ``pynop-eval`` script). When this file is executed directly
# instead — e.g. ``python pynop/training/eval.py ...`` from ``configs/`` —
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

from .lightning_model import (
    DeepONetModel,
    MultiStepNOModel,
    TrainingConfig,
    _is_deeponet,
)
from .metrics import FieldMetricAccumulator
from .train import (
    _load_weights_only,
    build_model,
    load_json,
    resolve_device,
    select_phase,
)

__all__ = [
    "parse_args",
    "main",
    "run_evaluation",
    "select_eval_ids",
    "scan_trajectories",
    "plot_error_evolution",
    "format_results_table",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Evaluation-set selection
# ---------------------------------------------------------------------------


def _shuffled_all_ids(ds_cfg: dict) -> list:
    """All simulation IDs of the HDF5 file, shuffled with the dataset seed.

    Uses exactly the same RNG protocol as :func:`pynop.training.train.build_pdebench_dataloaders`
    (``random.Random(seed).shuffle`` over all keys), so positions in the returned
    list are comparable to the training split.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config (requires
            ``path``; optional ``random_seed``, default 42).

    Returns:
        List of simulation IDs (HDF5 keys) in seeded shuffle order.
    """
    seed = int(ds_cfg.get("random_seed", 42))
    with h5py.File(ds_cfg["path"], "r") as f:
        all_ids = list(f.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    return all_ids


def select_eval_ids(
    ds_cfg: dict,
    phase_cfg: dict,
    split: str = "val",
    n_test: Optional[int] = None,
    run_info: Optional[dict] = None,
) -> dict:
    """Select the simulation IDs to evaluate on (never used for training).

    * ``split="val"`` — the validation split of the phase protocol.
    * ``split="test"`` — the first ``n_test`` IDs of the unused pool (beyond
      ``n_samples`` in the seeded shuffle order, i.e. never part of any
      training phase). Falls back to the validation set with a warning when the
      pool is empty.

    When ``run_info`` (the JSON written by :mod:`pynop.training.train` next to
    each checkpoint) provides ``train_ids`` / ``val_ids``, those are taken as
    ground truth and the unused pool is derived as "all file IDs minus used";
    otherwise both splits are re-derived from the config exactly like training.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config.
        phase_cfg: The selected phase section (``n_samples``, optional
            per-phase ``split_ratio``).
        split: ``"val"`` or ``"test"``.
        n_test: Number of simulations for the test split; defaults to the size
            of the validation set (so both splits are comparable in size).
        run_info: Optional parsed ``run_info.json`` content.

    Returns:
        Dict with keys ``eval_ids``, ``train_ids``, ``val_ids``, ``source``
        (``"run_info.json"`` or ``"training config"``) and ``effective_split``
        (the split actually used after a possible test -> val fallback).

    Raises:
        ValueError: On an unknown split name, or if the selected eval IDs ever
            overlap with the training IDs (should be impossible by construction).
    """
    if split not in ("val", "test"):
        raise ValueError(f"Unknown split {split!r}: expected 'val' or 'test'.")

    shuffled = _shuffled_all_ids(ds_cfg)

    if run_info is not None and "train_ids" in run_info and "val_ids" in run_info:
        train_ids = [str(i) for i in run_info["train_ids"]]
        val_ids = [str(i) for i in run_info["val_ids"]]
        used = set(train_ids) | set(val_ids)
        unused = [k for k in shuffled if k not in used]
        source = "run_info.json"
    else:
        n_samples = phase_cfg.get("n_samples")
        pool = list(shuffled[: int(n_samples)]) if n_samples is not None else list(shuffled)
        split_ratio = float(phase_cfg.get("split_ratio", ds_cfg.get("split_ratio", 0.9)))
        n_train = int(len(pool) * split_ratio)
        train_ids, val_ids = pool[:n_train], pool[n_train:]
        unused = shuffled[len(pool):]
        source = "training config"

    effective_split = split
    if split == "val":
        eval_ids = list(val_ids)
    else:
        n_test = int(n_test) if n_test is not None else max(len(val_ids), 1)
        eval_ids = unused[:n_test]
        if not eval_ids:
            print(
                f"WARNING: no unused simulations available for the test split "
                f"(all {len(shuffled)} file IDs were used in training) — falling back to the validation set."
            )
            eval_ids = list(val_ids)
            effective_split = "val"

    overlap = set(eval_ids) & set(train_ids)
    if overlap:
        raise RuntimeError(f"Evaluation set overlaps with {len(overlap)} training IDs (e.g. {sorted(overlap)[:5]}).")

    return {"eval_ids": eval_ids, "train_ids": train_ids, "val_ids": val_ids, "source": source, "effective_split": effective_split}


# ---------------------------------------------------------------------------
# Full-trajectory loading for evaluation
# ---------------------------------------------------------------------------


def scan_trajectories(h5_path: PathLike, eval_ids: Sequence[str], min_id: int = 0, max_id: Optional[int] = None) -> dict:
    """Validate the simulation IDs and report their available length.

    Args:
        h5_path: Path to the PDEBench HDF5 file (keys = simulation IDs).
        eval_ids: Simulation IDs to check.
        min_id: First frame index of the evaluation time range (inclusive).
        max_id: Last frame index of the evaluation time range (exclusive);
            ``None`` means "up to the end of the trajectory".

    Returns:
        Dict mapping each ID to its available number of frames in
        ``[min_id, max_id)``.

    Raises:
        KeyError: If an ID is not present in the file.
    """
    lengths = {}
    with h5py.File(h5_path, "r") as f:
        for sid in eval_ids:
            if sid not in f:
                raise KeyError(f"Simulation ID {sid!r} not found in {h5_path}.")
            T_total = f[sid]["data"].shape[0]
            end = min(int(max_id), T_total) if max_id is not None else T_total
            lengths[str(sid)] = int(end - min_id)
    return lengths


def _load_trajectory(f: h5py.File, sid: str, min_id: int, max_id: Optional[int], shift: float, scale: float) -> torch.Tensor:
    """Load one full trajectory from an open HDF5 handle.

    Args:
        f: Open :class:`h5py.File` (read mode).
        sid: Simulation ID (HDF5 key).
        min_id: First frame index to load (inclusive).
        max_id: Last frame index to load (exclusive); ``None`` = end of file.
        shift: Normalization shift added before scaling (model space).
        scale: Normalization scale applied after the shift.

    Returns:
        Tensor ``(T, C, H, W)`` with the normalized trajectory on CPU.
    """
    T_total = f[sid]["data"].shape[0]
    end = min(int(max_id), T_total) if max_id is not None else T_total
    raw = np.asarray(f[sid]["data"][min_id:end])  # (T, H, W, C)
    fields = torch.from_numpy(np.moveaxis(raw, -1, 1)).float()  # (T, C, H, W)
    return (fields + shift) * scale


def _group_by_length(eval_ids: Sequence[str], lengths: dict, batch_size: int) -> list:
    """Group IDs by equal trajectory length and chunk into rollout batches.

    Args:
        eval_ids: Ordered simulation IDs to evaluate.
        lengths: Mapping ID -> available frame count (from :func:`scan_trajectories`).
        batch_size: Maximum number of trajectories per rollout batch.

    Returns:
        List of ID chunks (each of length <= ``batch_size`` and uniform
        trajectory length), in first-appearance order of the groups so the
        overall evaluation order follows ``eval_ids`` as closely as possible.
    """
    by_length: dict = {}
    for sid in eval_ids:
        by_length.setdefault(lengths[sid], []).append(sid)

    chunks = []
    for ids in by_length.values():  # insertion-ordered groups
        for i in range(0, len(ids), int(batch_size)):
            chunks.append(ids[i : i + int(batch_size)])
    return chunks


# ---------------------------------------------------------------------------
# Model builder for evaluation
# ---------------------------------------------------------------------------


def build_eval_module(model_cfg: dict, training_cfg: dict, phase_name: str):
    """Build the (inference-only) Lightning wrapper for evaluation.

    The rollout contract is taken from the ``"shared_params"`` section exactly
    as during training (dt, predict_derivative, time normalization); losses and
    regularizers are disabled since no gradient is computed.

    Args:
        model_cfg: Parsed ``ModelConfig.json`` dict.
        training_cfg: Parsed ``TrainingConfig.json`` dict.
        phase_name: Phase section name (provides the time range).

    Returns:
        Tuple ``(lightning_model, model_name)`` where ``lightning_model`` is a
        :class:`MultiStepNOModel` / :class:`DeepONetModel` without optimizer.
    """
    shared = training_cfg.get("shared_params", {})
    input_steps = int(training_cfg["input_steps"]) if "input_steps" in training_cfg else 10

    model = build_model(model_cfg)
    train_config = TrainingConfig(
        dt=float(shared.get("timestep", 1.0)),
        max_autoregressive_steps=0,  # unused: the eval rollout is always full-length
        detach_grad_steps=0,
        loss_fn=None,
        derivative_loss_weight=0.0,
        predict_derivative=bool(shared.get("predict_derivative", True)),
        noise_level=0.0,
        time_normalization=float(shared.get("time_normalization", 1.0)),
        n_slices=input_steps,
    )

    if _is_deeponet(model):
        lightning_model = DeepONetModel(model=model, train_config=train_config)
    else:
        lightning_model = MultiStepNOModel(model=model, train_config=train_config)

    model_name = str(model_cfg.get("name") or model_cfg["model"])
    return lightning_model, model_name


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_evaluation(
    model_config_path: PathLike,
    training_config_path: PathLike,
    checkpoint: PathLike,
    split: str = "val",
    n_test: Optional[int] = None,
    pretrain: bool = False,
    device: str = "auto",
    batch_size: Optional[int] = None,
    max_sims: Optional[int] = None,
    output_dir: Optional[PathLike] = None,
    plots: bool = True,
) -> Path:
    """Evaluate a trained checkpoint on unseen trajectories and write the results.

    Every simulation of the evaluation split is rolled out **autoregressively
    over its full trajectory** (first ``input_steps`` slices as input, then one
    prediction per remaining slice); errors are pooled over all trajectories
    and reported globally and per time step (see the module docstring).

    Args:
        model_config_path: Path to the ``ModelConfig.json`` file (same as training).
        training_config_path: Path to the ``TrainingConfig.json`` file (same as training).
        checkpoint: Checkpoint file, or a run directory (its ``last.ckpt`` is used).
        split: ``"val"`` (default) or ``"test"`` — see :func:`select_eval_ids`.
        n_test: Number of simulations for the test split (default: val size; an
            optional ``"eval"."n_test"`` config value is used when set).
        pretrain: If True use the ``"pretraining"`` phase section, otherwise
            ``"fulltraining"``.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
        batch_size: Optional override of the dataset ``batch_size`` (number of
            trajectories rolled out together).
        max_sims: Optional cap on the number of trajectories evaluated (quick
            checks); ``None`` evaluates every trajectory of the split.
        output_dir: Where to write the results JSON and figures (default: next
            to the checkpoint).
        plots: If True render the error-evolution PNG figures (requires
            matplotlib; a warning is printed when it is unavailable).

    Returns:
        Path of the written results JSON file.
    """
    model_cfg = load_json(model_config_path)
    training_cfg = load_json(training_config_path)

    phase_name, phase_cfg = select_phase(training_cfg, pretrain)
    ds_cfg = training_cfg.get("dataset")
    if not isinstance(ds_cfg, dict):
        raise KeyError('TrainingConfig must contain a "dataset" section.')
    if str(ds_cfg.get("type", "")).lower() != "pdebench":
        raise ValueError(f"Unsupported dataset type {ds_cfg.get('type')!r}: only 'pdebench' is supported for now.")
    eval_cfg = training_cfg.get("eval", {}) or {}

    # ------------------------------------------------------------------ checkpoint
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        candidate = ckpt_path / "last.ckpt"
        if not candidate.is_file():
            raise FileNotFoundError(f"No last.ckpt found in run directory {ckpt_path}.")
        print(f"Evaluating from run directory: using {candidate.name}")
        ckpt_path = candidate
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_info = None
    info_path = ckpt_path.parent / "run_info.json"
    if info_path.is_file():
        with open(info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        print(f"Found {info_path.name} next to the checkpoint — using its stored train/val IDs.")

    # ------------------------------------------------------------------ model
    lightning_model, model_name = build_eval_module(model_cfg, training_cfg, phase_name)
    _load_weights_only(lightning_model.model, ckpt_path)
    n_params = sum(p.numel() for p in lightning_model.model.parameters())
    print(f"Model: {type(lightning_model.model).__name__} ({n_params / 1e6:.2f} M params), run name '{model_name}'")

    # ------------------------------------------------------------------ data
    if n_test is None and "n_test" in eval_cfg:
        n_test = int(eval_cfg["n_test"])
    id_info = select_eval_ids(ds_cfg, phase_cfg, split=split, n_test=n_test, run_info=run_info)
    effective_split = id_info["effective_split"]

    input_steps = int(training_cfg.get("input_steps", 10))
    min_id = int(phase_cfg.get("min_id", 0))
    max_id = phase_cfg.get("max_id")

    eval_ids = list(id_info["eval_ids"])
    if max_sims is not None and len(eval_ids) > int(max_sims):
        print(f"Evaluating on the first {int(max_sims)} of {len(eval_ids)} trajectories (max_sims cap).")
        eval_ids = eval_ids[: int(max_sims)]

    lengths = scan_trajectories(ds_cfg["path"], eval_ids, min_id=min_id, max_id=max_id)
    usable = [sid for sid in eval_ids if lengths[sid] > input_steps]
    skipped = [sid for sid in eval_ids if lengths[sid] <= input_steps]
    for sid in skipped:
        print(f"WARNING: simulation {sid!r} has only {lengths[sid]} frames in [{min_id}, {max_id}) — skipped (need > {input_steps}).")
    if not usable:
        raise ValueError("No trajectory long enough to roll out on the evaluation set.")

    if batch_size is None:
        batch_size = int(ds_cfg.get("batch_size", 1))
    chunks = _group_by_length(usable, lengths, batch_size)

    accelerator = resolve_device(device)
    dev = torch.device("cuda" if accelerator == "gpu" else "cpu")
    lightning_model.eval()
    lightning_model.to(dev)
    print(
        f"Evaluating split '{effective_split}' ({len(usable)} trajectories, source: {id_info['source']}) "
        f"on device '{accelerator}': full autoregressive rollout of each trajectory "
        f"(t in [{min_id}, {max_id}], {input_steps} input slices) in {len(chunks)} batches."
    )

    # ------------------------------------------------------------------ rollout + metrics
    accumulator = FieldMetricAccumulator(
        spectral_bands=tuple(tuple(b) for b in eval_cfg.get("spectral_bands", ((0.0, 4.0), (4.0, 12.0), (12.0, None)))),
        relative_spectral=bool(eval_cfg.get("spectral_relative", True)),
        track_per_step_spectral=bool(eval_cfg.get("per_step_spectral", True)),
    )

    shift = float(ds_cfg.get("shift", 0.0))
    scale = float(ds_cfg.get("scale", 1.0))
    n_sims_done = 0
    total_predictions = 0
    length_min, length_max = min(lengths[s] for s in usable), max(lengths[s] for s in usable)

    with torch.no_grad(), h5py.File(ds_cfg["path"], "r") as f:
        for chunk in tqdm(chunks, desc="Rolling out trajectories"):
            fields = torch.stack([_load_trajectory(f, sid, min_id, max_id, shift, scale) for sid in chunk])  # (B, T, C, H, W)
            B = fields.shape[0]
            time_idx = torch.full((B,), float(min_id))

            step_counter = {"k": 0}

            def _on_step(preds: torch.Tensor, targets: torch.Tensor):
                # Consume each prediction immediately (constant memory); the
                # explicit index keeps the per-step curve pooled at the right
                # position across batches of equal-length trajectories.
                accumulator.update_per_step(preds, targets, step_counter["k"])
                step_counter["k"] += 1

            res = lightning_model._rollout(
                fields.to(dev),
                time_idx.to(dev),
                None,
                training=False,
                on_step=_on_step,
            )
            n_sims_done += B
            # num_predictions is per trajectory (the rollout loop counts steps, not batch elements).
            total_predictions += int(res["num_predictions"]) * B

    metrics = accumulator.compute()
    # Trajectory-level bookkeeping replaces the window counter of the accumulator.
    metrics.pop("n_windows_evaluated")
    metrics["total_predictions"] = total_predictions
    # Map each pooled step index to its absolute time: the k-th prediction (1-based)
    # targets frame min_id + input_steps + (k - 1) for every trajectory.
    for entry in metrics["per_step"]:
        entry["time"] = min_id + input_steps + (entry["step"] - 1)

    # ------------------------------------------------------------------ results / output
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(output_dir) if output_dir is not None else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: list = []
    # The figures need the results dict; build it first without "plots", render, then finalize.
    results = {
        "timestamp": now,
        "checkpoint": str(ckpt_path),
        "model": {"name": model_name, "class": type(lightning_model.model).__name__, "params": n_params},
        "phase": phase_name,
        "split_requested": split,
        "split_effective": effective_split,
        "id_source": id_info["source"],
        "n_eval_simulations": n_sims_done,
        "eval_ids": list(usable),
        "train_ids": list(id_info["train_ids"]),  # audit: eval set must be disjoint from these
        "dataset": {
            "path": str(ds_cfg["path"]),
            "min_id": min_id,
            "max_id": max_id,
            "shift": shift,
            "scale": scale,
        },
        "trajectory": {
            "input_steps": input_steps,
            "length_min": length_min,
            "length_max": length_max,
        },
        "metrics": metrics,
    }

    if plots:
        plot_paths = [str(p) for p in plot_error_evolution(results, out_dir)]
    results["plots"] = plot_paths

    results_path = out_dir / f"eval_results_{effective_split}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(format_results_table(results))
    print(f"Results written to {results_path}")
    return results_path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_results_table(results: dict) -> str:
    """Format an evaluation result dict as a human-readable console table.

    Args:
        results: The results dict produced by :func:`run_evaluation`.

    Returns:
        A multi-line string with the checkpoint/split bookkeeping, all scalar
        metrics (scientific notation), the spectral band errors and a condensed
        summary of the per-step error curves (the full curves live in the JSON
        and the PNG figures).
    """
    m = results["metrics"]
    traj = results["trajectory"]
    lines = [
        "=" * 70,
        " EVALUATION RESULTS — full-trajectory autoregressive rollout",
        "=" * 70,
        f" Checkpoint : {results['checkpoint']}",
        f" Model      : {results['model']['class']} ({results['model']['name']})",
        f" Phase      : {results['phase']} | Split: {results['split_effective']} "
        f"({results['n_eval_simulations']} trajectories, source: {results['id_source']})",
        f" Trajectory : t in [{results['dataset']['min_id']}, {results['dataset']['max_id']}], length "
        f"{traj['length_min']}..{traj['length_max']}, {traj['input_steps']} input slices, "
        f"{m['n_steps']} predictions per trajectory ({m.get('total_predictions', '?')} total)",
        "-" * 70,
        f" {'Metric':<28}{'Value'}",
        "-" * 70,
        f" {'MSE':<28}{m['mse']:.3e}",
        f" {'RMSE':<28}{m['rmse']:.3e}",
        f" {'nRMSE (relative L2)':<28}{m['nrmse']:.3e}",
        f" {'MAE':<28}{m['mae']:.3e}",
        f" {'Max abs error':<28}{m['max_abs_error']:.3e}",
        f" {'R^2':<28}{m['r2']:.3e}",
    ]
    for label, value in m["spectral_bands"].items():
        lines.append(f" {'Spectral ' + label:<28}{value:.3e}")
    lines += ["-" * 70]

    per_step = m.get("per_step", [])
    if per_step:
        for key, name in (("rmse", "RMSE"), ("nrmse", "nRMSE")):
            vals = [(p["time"], p[key]) for p in per_step]
            first_t, first_v = vals[0]
            last_t, last_v = vals[-1]
            min_t, min_v = min(vals, key=lambda v: v[1])
            max_t, max_v = max(vals, key=lambda v: v[1])
            lines.append(
                f" Per-step {name:<6}: first t={first_t} {first_v:.2e} | min {min_v:.2e} @t={min_t} "
                f"| max {max_v:.2e} @t={max_t} | last t={last_t} {last_v:.2e}"
            )
        lines.append(" (full per-step curves: see the JSON file and the PNG figures)")
    for p in results.get("plots", []):
        lines.append(f" Plot       : {p}")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plots (English labels)
# ---------------------------------------------------------------------------


def _log_safe(y: np.ndarray) -> np.ndarray:
    """Replace non-positive values by NaN so log-scale plots stay well-defined."""
    y = np.asarray(y, dtype=float)
    return np.where(y > 0, y, np.nan)


def plot_error_evolution(results: dict, out_dir: PathLike) -> list:
    """Render the per-step error curves of an evaluation as PNG figures.

    Produces (all labels in English):

    * ``eval_error_evolution_<split>.png`` — 2×2 grid of RMSE, nRMSE (relative
      L2), MAE (log scale) and R² vs time step;
    * ``eval_spectral_evolution_<split>.png`` — per-band spectral error vs time
      step (one line per band, log scale); only written when the results carry
      per-step spectral bands.

    Args:
        results: The results dict produced by :func:`run_evaluation` (must
            contain ``metrics.per_step`` with a ``time`` field).
        out_dir: Directory where the PNG files are written.

    Returns:
        List of paths of the figures actually written; empty when matplotlib is
        unavailable or there is no per-step data (a warning is printed then).
    """
    per_step = results.get("metrics", {}).get("per_step", [])
    if not per_step:
        print("WARNING: no per-step data to plot — skipping figures.")
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe: the figures are only saved to files
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not available — skipping figure generation.")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = results.get("split_effective", "eval")
    model_name = results["model"]["name"]
    n_sims = results.get("n_eval_simulations", "?")
    x = [p.get("time", p["step"]) for p in per_step]

    written: list = []

    # ------------------------------------------------------------------ figure 1: field errors
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    specs = (
        (axes[0][0], "rmse", "RMSE", True),
        (axes[0][1], "nrmse", "nRMSE (relative L2)", True),
        (axes[1][0], "mae", "MAE", True),
        (axes[1][1], "r2", r"R$^2$", False),
    )
    for ax, key, label, use_log in specs:
        y = np.array([p[key] for p in per_step])
        if use_log:
            y = _log_safe(y)
            ax.set_yscale("log")
        ax.plot(x, y, color="tab:blue", lw=1.5)
        ax.set_xlabel("Time step")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{model_name} — {split} split ({n_sims} trajectories): error evolution along the trajectory")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    p1 = out_dir / f"eval_error_evolution_{split}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    written.append(p1)

    # ------------------------------------------------------------------ figure 2: spectral bands
    if "spectral_bands" in per_step[0]:
        labels = list(per_step[0]["spectral_bands"].keys())
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label in labels:
            y = _log_safe([p["spectral_bands"][label] for p in per_step])
            ax.plot(x, y, lw=1.5, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Spectral band error (relative L2)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Radial band")
        fig.suptitle(f"{model_name} — {split} split: spectral error evolution along the trajectory")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        p2 = out_dir / f"eval_spectral_evolution_{split}.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        written.append(p2)

    return written


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
        prog="pynop.eval",
        description=(
            "Evaluate a trained PyNOP checkpoint by rolling out the FULL trajectory of every "
            "simulation in an unseen split (validation by default, or unused-simulations test), "
            "reporting field-level + spectral-band metrics and their evolution along the trajectory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_config", type=Path, help="Path to ModelConfig.json (same as training).")
    parser.add_argument("training_config", type=Path, help="Path to TrainingConfig.json (same as training).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint file (.ckpt) or a run directory (its last.ckpt is used).",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="'val': validation split of the training protocol; 'test': simulations never used in any phase.",
    )
    parser.add_argument("--n-test", dest="n_test", type=int, default=None, help="Number of simulations for the test split (default: val size).")
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help='Use the "pretraining" phase section for the time range (default: "fulltraining").',
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Number of trajectories rolled out together (default: dataset batch_size).")
    parser.add_argument("--max-sims", dest="max_sims", type=int, default=None, help="Evaluate at most this many trajectories (quick checks).")
    parser.add_argument("--no-plots", action="store_true", help="Skip the PNG figure generation.")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None, help="Where to write eval_results_<split>.json and figures (default: next to the checkpoint).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The path of the written results JSON file.
    """
    args = parse_args(argv)
    return run_evaluation(
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        checkpoint=args.checkpoint,
        split=args.split,
        n_test=args.n_test,
        pretrain=args.pretrain,
        device=args.device,
        batch_size=args.batch_size,
        max_sims=args.max_sims,
        output_dir=args.output_dir,
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
