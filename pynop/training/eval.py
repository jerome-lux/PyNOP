"""Command-line evaluation driver for PyNOP multi-step prediction models.

Companion to :mod:`pynop.training.train`: it loads a trained checkpoint and runs
the **full autoregressive rollout** on samples that were *not used during
training*, reporting field-level metrics (MSE, RMSE, normalized MSE / relative
L2, MAE, max error, R²), per-band spectral errors in Fourier space
(PDEBench-style radial bands) and the per-AR-step error curve.

Usage
-----
::

    python -m pynop.training.eval MODEL_CONFIG.json TRAINING_CONFIG.json --checkpoint PATH [options]

The model / training configs are the same files used for training (the phase
section — ``--pretrain`` selects ``"pretraining"``, otherwise ``"fulltraining"``
— provides the window parameters: stride, time range and AR steps).

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
RMSE"): per sample and channel ``‖pred − target‖₂ / ‖target‖₂``, averaged over
all samples/channels. The ratio is scale-invariant, so it stays comparable
across datasets with different amplitudes; for zero-mean-normalized fields
``(x − μ)/σ`` the denominator equals the field standard deviation and the
metric remains a meaningful relative error (this is exactly how PDEBench uses
it on their zero-mean/unit-variance data). R² is reported as well: it is
invariant to *any* affine rescaling (shift and scale) and reduces to
``1 − nRMSE²`` per sample for zero-mean data. Normalizations by
``mean(target)²`` would degenerate on zero-mean fields and are deliberately not
used (see :mod:`pynop.training.metrics` for details).

All metrics are computed on the **model-space** data, i.e. after the dataset
``(x + shift) * scale`` transform — the same space the network was trained in.

Output: a formatted table on stdout and ``eval_results_<split>.json`` written
next to the checkpoint (or to ``--output-dir``). The JSON contains every scalar
metric, the per-AR-step RMSE / nRMSE curve and full bookkeeping (checkpoint,
phase, split source, evaluated IDs) so results are self-contained.

Optional ``"eval"`` section of the training config (all keys optional)::

    {
        "spectral_bands": [[0, 4], [4, 12], [12, null]],  # radial bands (FFT bin-index units, PDEBench iLow/iHigh)
        "spectral_relative": true,                        # false -> absolute band energies
        "n_test": 40                                      # default size of the test split
    }
"""

import argparse
import datetime
import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from ..dataloader.h5_dataloader import PDEBenchDataSet
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
    "build_eval_dataloader",
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
# Data / model builders for evaluation
# ---------------------------------------------------------------------------


def build_eval_dataloader(
    ds_cfg: dict, phase_cfg: dict, input_steps: int, eval_ids: Sequence[str], max_windows: Optional[int] = None
) -> tuple:
    """Build the (unshuffled) :class:`DataLoader` for the evaluation set.

    Mirrors the training window protocol of the selected phase: same stride
    (``delta_ts``), time range and ``window_size = max_AR_steps + input_steps + 1``.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config.
        phase_cfg: The selected phase section.
        input_steps: Number of past slices given to the network (``n_slices``).
        eval_ids: Simulation IDs to evaluate on.
        max_windows: Optional cap on the number of windows evaluated (the first
            ``max_windows`` in dataset order); ``None`` evaluates all.

    Returns:
        Tuple ``(loader, window_size)``.

    Raises:
        ValueError: If the dataset type is not supported or a required key is missing.
    """
    ds_type = str(ds_cfg.get("type", "")).lower()
    if ds_type != "pdebench":
        raise ValueError(f"Unsupported dataset type {ds_cfg.get('type')!r}: only 'pdebench' is supported for now.")
    if "path" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "path".')

    window_size = int(phase_cfg["max_AR_steps"]) + input_steps + 1
    delta_ts = phase_cfg.get("delta_ts")

    dataset = PDEBenchDataSet(
        ds_cfg["path"],
        indexes=list(eval_ids),
        T_unroll=window_size,
        step=int(delta_ts) if delta_ts is not None else None,
        load_in_ram=bool(ds_cfg.get("load_in_ram", False)),
        min_id=int(phase_cfg.get("min_id", 0)),
        max_id=phase_cfg.get("max_id"),
        shift=float(ds_cfg.get("shift", 0.0)),
        scale=float(ds_cfg.get("scale", 1.0)),
    )
    if not dataset.windows:
        raise ValueError(
            f"No windows available for the {len(eval_ids)} evaluation simulations "
            f"(window_size={window_size}, t in [{phase_cfg.get('min_id', 0)}, {phase_cfg.get('max_id')}])."
        )

    if max_windows is not None and len(dataset) > int(max_windows):
        dataset = Subset(dataset, range(int(max_windows)))
        print(f"Evaluating on the first {int(max_windows)} of {len(dataset.dataset)} windows (max_windows cap).")

    batch_size = int(ds_cfg.get("batch_size", 1))
    num_workers = int(ds_cfg.get("num_workers", 0))
    loader_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, pin_memory=torch.cuda.is_available(), **loader_kwargs), window_size


def build_eval_module(model_cfg: dict, training_cfg: dict, phase_name: str):
    """Build the (inference-only) Lightning wrapper for evaluation.

    The rollout contract is taken from the ``"shared_params"`` section exactly
    as during training (dt, predict_derivative, time normalization); losses and
    regularizers are disabled since no gradient is computed.

    Args:
        model_cfg: Parsed ``ModelConfig.json`` dict.
        training_cfg: Parsed ``TrainingConfig.json`` dict.
        phase_name: Phase section name (provides ``max_AR_steps``).

    Returns:
        Tuple ``(lightning_model, model_name)`` where ``lightning_model`` is a
        :class:`MultiStepNOModel` / :class:`DeepONetModel` without optimizer.
    """
    shared = training_cfg.get("shared_params", {})
    phase_cfg = training_cfg[phase_name]
    input_steps = int(training_cfg["input_steps"]) if "input_steps" in training_cfg else 10

    model = build_model(model_cfg)
    train_config = TrainingConfig(
        dt=float(shared.get("timestep", 1.0)),
        max_autoregressive_steps=int(phase_cfg["max_AR_steps"]),
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
    max_windows: Optional[int] = None,
    output_dir: Optional[PathLike] = None,
) -> Path:
    """Evaluate a trained checkpoint on unseen samples and write the results.

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
        batch_size: Optional override of the dataset ``batch_size``.
        max_windows: Optional cap on the number of windows evaluated.
        output_dir: Where to write ``eval_results_<split>.json`` (default: next
            to the checkpoint).

    Returns:
        Path of the written results JSON file.
    """
    model_cfg = load_json(model_config_path)
    training_cfg = load_json(training_config_path)

    phase_name, phase_cfg = select_phase(training_cfg, pretrain)
    ds_cfg = training_cfg.get("dataset")
    if not isinstance(ds_cfg, dict):
        raise KeyError('TrainingConfig must contain a "dataset" section.')
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

    if batch_size is not None:
        ds_cfg = dict(ds_cfg)
        ds_cfg["batch_size"] = int(batch_size)
    loader, window_size = build_eval_dataloader(ds_cfg, phase_cfg, input_steps=int(training_cfg.get("input_steps", 10)), eval_ids=id_info["eval_ids"], max_windows=max_windows)

    accelerator = resolve_device(device)
    dev = torch.device("cuda" if accelerator == "gpu" else "cpu")
    lightning_model.eval()
    lightning_model.to(dev)
    print(
        f"Evaluating split '{effective_split}' ({len(id_info['eval_ids'])} simulations, source: {id_info['source']}) "
        f"on device '{accelerator}': {len(loader.dataset)} windows x {window_size} slices."
    )

    # ------------------------------------------------------------------ rollout + metrics
    accumulator = FieldMetricAccumulator(
        spectral_bands=tuple(tuple(b) for b in eval_cfg.get("spectral_bands", ((0.0, 4.0), (4.0, 12.0), (12.0, None)))),
        relative_spectral=bool(eval_cfg.get("spectral_relative", True)),
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs, time_idx, cond = lightning_model._unpack_batch(batch)
            res = lightning_model._rollout(
                inputs.to(dev),
                time_idx.to(dev) if isinstance(time_idx, torch.Tensor) else time_idx,
                cond.to(dev) if cond is not None and isinstance(cond, torch.Tensor) else cond,
                training=False,
                store_fields=True,
            )
            accumulator.update(res["preds"], res["targets"])

    metrics = accumulator.compute()

    # ------------------------------------------------------------------ results / output
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {
        "timestamp": now,
        "checkpoint": str(ckpt_path),
        "model": {"name": model_name, "class": type(lightning_model.model).__name__, "params": n_params},
        "phase": phase_name,
        "split_requested": split,
        "split_effective": effective_split,
        "id_source": id_info["source"],
        "n_eval_simulations": len(id_info["eval_ids"]),
        "eval_ids": list(id_info["eval_ids"]),
        "train_ids": list(id_info["train_ids"]),  # audit: eval set must be disjoint from these
        "dataset": {
            "path": str(ds_cfg["path"]),
            "delta_ts": phase_cfg.get("delta_ts"),
            "min_id": phase_cfg.get("min_id", 0),
            "max_id": phase_cfg.get("max_id"),
            "shift": ds_cfg.get("shift", 0.0),
            "scale": ds_cfg.get("scale", 1.0),
        },
        "input_steps": int(training_cfg.get("input_steps", 10)),
        "window_size": window_size,
        "n_windows_evaluated": metrics["n_windows_evaluated"],
        "metrics": metrics,
    }

    out_dir = Path(output_dir) if output_dir is not None else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
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
        metrics (scientific notation), the spectral band errors and the
        per-AR-step RMSE / nRMSE curve.
    """
    m = results["metrics"]
    lines = [
        "=" * 70,
        " EVALUATION RESULTS",
        "=" * 70,
        f" Checkpoint : {results['checkpoint']}",
        f" Model      : {results['model']['class']} ({results['model']['name']})",
        f" Phase      : {results['phase']} | Split: {results['split_effective']} "
        f"({results['n_eval_simulations']} sims, source: {results['id_source']})",
        f" Windows    : {m['n_windows_evaluated']} evaluated (window_size={results['window_size']} -> {m['n_steps']} predictions each)",
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
        rmse_curve = "  ".join(f"s{p['step']}={p['rmse']:.1e}" for p in per_step)
        nrmse_curve = "  ".join(f"s{p['step']}={p['nrmse']:.1e}" for p in per_step)
        lines.append(f" Per-step RMSE : {rmse_curve}")
        lines.append(f" Per-step nRMSE: {nrmse_curve}")
    lines.append("=" * 70)
    return "\n".join(lines)


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
            "Evaluate a trained PyNOP checkpoint on samples not used during training "
            "(validation split by default, or an unused-simulations test split), "
            "reporting field-level + spectral-band metrics."
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
        help='Use the "pretraining" phase section for the window parameters (default: "fulltraining").',
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="Override the dataset batch size for evaluation.")
    parser.add_argument("--max-windows", dest="max_windows", type=int, default=None, help="Evaluate at most this many windows (quick checks).")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None, help="Where to write eval_results_<split>.json (default: next to the checkpoint).")
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
        max_windows=args.max_windows,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
