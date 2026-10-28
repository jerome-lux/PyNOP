"""Import existing PyNOP training logs (CSV / TensorBoard) into an Aim repository.

The training driver writes, per run directory::

    <root>/<experiment>/<phase>/<model>/<input_steps>steps_<timestamp>/
        lightning_logs/version_*/metrics.csv          # CSVLogger
        tb_logs/lightning_logs/version_*/events.*     # TensorBoardLogger
        run_info.json / model_config.json             # bookkeeping

This script discovers every such run directory under a root (e.g.
``/run/media/jlux/SSD2/PDETraining/PDEBENCH``) and re-logs its metrics into an
Aim repository, so that **old runs can be compared with new ones in the same
Aim UI** (``aim up``).

Usage
-----
::

    python -m pynop.training.import_logs /run/media/jlux/SSD2/PDETraining/PDEBENCH \
        [--experiment PDEBENCH] [--aim-home PATH] [--source auto|csv|tb] [--force] [--dry-run]

Conventions (identical to the live :class:`~pynop.training.aim_logger.AimCallback`):

* one Aim **run** per run directory, named ``<phase>/<model>/<run dir>``;
  all runs of a root share one Aim **experiment** (default: the root's name);
* metric names are kept as logged (``loss_step``, ``RMSE_epoch``, ...);
* x-axis: columns ending in ``_epoch`` use the row's ``epoch`` value, every
  other column uses the row's ``step`` value — exactly what the CSV records;
* parameters from ``run_info.json`` are stored with the run for cross-run
  comparison, plus the full ``model_config.json`` / ``training_config.json``
  verbatim as nested parameters (``model_config``, ``training_config``) — the
  same keys the live :class:`~pynop.training.aim_logger.AimCallback` writes, so
  imported and live runs stay directly comparable in the UI.

Re-importing a run directory creates an additional Aim run (the previous one is
kept); a sidecar file ``.aim_import.json`` records the last import and makes
plain re-runs skip already-imported directories unless ``--force`` is given —
a directory is only skipped when the recorded run still exists in the *same*
Aim repository as the current target (so importing into a different home never
hides previously imported data).

Note: aim 3.x's own default repository is "the nearest ``.aim/`` from the
current working directory" — which would pollute whatever directory you launch
from. This integration therefore defaults to ``~/.aim`` instead; view runs with
``aim up --repo ~/.aim``.
"""

import argparse
import csv
import datetime
import json
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Union

__all__ = [
    "discover_runs",
    "import_run_csv",
    "import_run_tb",
    "import_logs",
    "parse_args",
    "main",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(root: PathLike) -> list[Path]:
    """Find all run directories under ``root`` holding CSV and/or TB logs.

    A run directory is any folder containing ``lightning_logs/version_*/metrics.csv``
    (CSVLogger layout) or ``tb_logs/lightning_logs/version_*/events.out.tfevents.*``
    (TensorBoardLogger layout). Directories with both are returned once.

    Args:
        root: Root of the training output tree to scan recursively.

    Returns:
        Sorted list of run directories.

    Raises:
        FileNotFoundError: If ``root`` does not exist.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    run_dirs: set[Path] = set()
    for csv_file in root.glob("**/lightning_logs/version_*/metrics.csv"):
        # <run>/lightning_logs/version_N/metrics.csv -> parents[2] is the run dir.
        run_dirs.add(csv_file.parents[2])
    for event_file in root.glob("**/tb_logs/lightning_logs/version_*/events.out.tfevents.*"):
        # <run>/tb_logs/lightning_logs/version_N/events... -> parents[3] is the run dir.
        run_dirs.add(event_file.parents[3])

    return sorted(run_dirs)


def _csv_files(run_dir: Path) -> list[Path]:
    """All ``metrics.csv`` files of a run (one per logger version, e.g. after resumes)."""
    return sorted(run_dir.glob("lightning_logs/version_*/metrics.csv"))


def _tb_version_dirs(run_dir: Path) -> list[Path]:
    """All TensorBoard log dirs of a run containing at least one event file."""
    dirs = []
    for d in sorted(run_dir.glob("tb_logs/lightning_logs/version_*")):
        if any(d.glob("events.out.tfevents.*")):
            dirs.append(d)
    return dirs


def _run_display_name(run_dir: Path, root: Path) -> str:
    """Aim run name: path of the run directory relative to ``root`` (e.g. ``pretraining/SwinNO/10steps_...``)."""
    try:
        rel = run_dir.resolve().relative_to(root.resolve())
    except ValueError:  # pragma: no cover - defensive; discovery always yields children of root
        rel = Path(run_dir.name)
    return str(rel)


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

_AXIS_COLUMNS = {"step", "epoch"}


def _parse_float(text: Optional[str]) -> Optional[float]:
    """Parse a CSV cell to float; ``None`` for empty / non-finite values."""
    if text is None or text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def import_run_csv(run_dir: Path, aim_run) -> int:
    """Re-log every ``metrics.csv`` of a run directory into an open Aim run.

    Column-to-axis rule (mirrors the live callback): columns ending in
    ``_epoch`` are tracked at ``x = row["epoch"]``, all other data columns at
    ``x = row["step"]``. Empty cells are skipped; duplicate ``(name, x)`` pairs
    (which do not occur within a single file — only across resumed logger
    versions whose val-batch counter restarts) keep the first occurrence.

    Args:
        run_dir: Run directory containing ``lightning_logs/version_*/metrics.csv``.
        aim_run: Open Aim :class:`aim.Run` to write into.

    Returns:
        Number of tracked data points.

    Raises:
        FileNotFoundError: If the run has no CSV logs.
    """
    csv_files = _csv_files(run_dir)
    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv found under {run_dir}/lightning_logs/")

    n_points = 0
    for csv_file in csv_files:
        seen: dict[tuple, None] = {}
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_cols = [c for c in reader.fieldnames if c not in _AXIS_COLUMNS]
            rows_by_step: list[dict] = []
            rows_by_epoch: list[dict] = []

            for row in reader:
                step_x = _parse_float(row.get("step"))
                epoch_x = _parse_float(row.get("epoch"))
                point_step: dict[str, float] = {}
                point_epoch: dict[str, float] = {}
                for col in data_cols:
                    value = _parse_float(row.get(col))
                    if value is None:
                        continue
                    if col.endswith("_epoch"):
                        x = epoch_x
                        target = point_epoch
                    else:
                        x = step_x
                        target = point_step
                    if x is None or not math.isfinite(x):
                        continue
                    key = (col, float(x))
                    if key in seen:  # duplicate on the natural axis: keep first occurrence
                        continue
                    seen[key] = None
                    target[col] = value

                if point_step and step_x is not None:
                    rows_by_step.append((float(step_x), point_step))
                if point_epoch and epoch_x is not None:
                    rows_by_epoch.append((float(epoch_x), point_epoch))

        for x, values in rows_by_step:
            aim_run.track(values, step=int(x))
            n_points += len(values)
        for x, values in rows_by_epoch:
            aim_run.track(values, step=int(x), epoch=int(x))
            n_points += len(values)
    return n_points


# ---------------------------------------------------------------------------
# TensorBoard import
# ---------------------------------------------------------------------------


def import_run_tb(run_dir: Path, aim_run) -> int:
    """Re-log every scalar of a run's TensorBoard event files into an Aim run.

    Scalar tags keep their Lightning names (``loss_step``, ``val_RMSE_epoch``, ...)
    and are tracked at their original global step — the same x-axis convention as
    the CSV import for per-batch metrics.

    Args:
        run_dir: Run directory containing ``tb_logs/lightning_logs/version_*/``.
        aim_run: Open Aim :class:`aim.Run` to write into.

    Returns:
        Number of tracked data points.

    Raises:
        FileNotFoundError: If the run has no TensorBoard logs.
    """
    version_dirs = _tb_version_dirs(run_dir)
    if not version_dirs:
        raise FileNotFoundError(f"No TensorBoard event files found under {run_dir}/tb_logs/")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # local: heavy import

    n_points = 0
    for version_dir in version_dirs:
        ea = EventAccumulator(str(version_dir), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                aim_run.track(event.value, name=tag, step=int(event.step))
                n_points += 1
    return n_points


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_json_dict(path: Path) -> Optional[dict]:
    """Load a JSON file expecting an object; ``None`` if missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _load_params(run_dir: Path) -> dict:
    """Collect comparison-friendly parameters from the run's bookkeeping files.

    Mirrors :func:`pynop.training.train.build_aim_callback` for live runs: a few
    flat convenience keys (from ``run_info.json``) plus the full model and
    training configs stored verbatim as nested parameters (Aim renders them as
    an expandable tree in the run-detail "Parameters" tab).
    """
    params: dict = {}
    info_file = run_dir / "run_info.json"
    if info_file.is_file():
        info = _load_json_dict(info_file)
        if isinstance(info, dict):
            model = info.get("model") or {}
            for key in ("phase", "lr", "max_epochs", "input_steps", "window_size", "seed"):
                if key in info:
                    params[key] = info[key]
            if "class" in model:
                params["model_class"] = model["class"]
            if "params" in model:
                params["n_params"] = model["params"]
    # Full configs, verbatim (same keys as the live AimCallback).
    for key, filename in (("model_config", "model_config.json"), ("training_config", "training_config.json")):
        cfg = _load_json_dict(run_dir / filename)
        if cfg is not None:
            params[key] = cfg
    return params


def _sidecar_path(run_dir: Path) -> Path:
    return run_dir / ".aim_import.json"


def import_logs(
    root: PathLike,
    experiment: Optional[str] = None,
    home: Optional[str] = None,
    source: str = "auto",
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Import every run directory under ``root`` into an Aim repository.

    Args:
        root: Root of the training output tree (e.g. ``.../PDETraining/PDEBENCH``).
        experiment: Aim experiment name for all imported runs. Defaults to the
            root's basename so that one root = one comparable experiment.
        home: Aim repository path ("home"). ``None`` uses ``~/.aim`` (aim 3.x's
            own default is CWD-based and would pollute the launch directory).
        source: ``"auto"`` (CSV if present, else TensorBoard), or force ``"csv"``/``"tb"``.
        force: Re-import run directories that already have a ``.aim_import.json`` sidecar.
        dry_run: Only report what would be imported; write nothing.

    Returns:
        List of per-run result dicts (``run_dir``, ``name``, ``source``,
        ``n_points``, ``status``, optional ``hash``).

    Raises:
        FileNotFoundError: If ``root`` does not exist or a forced source is missing for a run.
        ValueError: On an unknown ``source`` value.
    """
    if source not in ("auto", "csv", "tb"):
        raise ValueError(f"Unknown source {source!r}: expected 'auto', 'csv' or 'tb'.")

    root = Path(root)
    run_dirs = discover_runs(root)
    if not run_dirs:
        print(f"No run directories with CSV/TB logs found under {root}")
        return []

    experiment = str(experiment or root.name)
    effective_home = str(Path(home).expanduser() if home is not None else Path.home() / ".aim")
    results: list[dict] = []

    aim = None
    target_repo = None  # lazily opened, reused for sidecar run-existence checks

    def _prev_run_visible(prev: dict) -> bool:
        """True if the previously imported run still exists in ``effective_home``."""
        nonlocal target_repo
        if str(prev.get("home")) != effective_home or not prev.get("hash"):
            return False
        try:
            if target_repo is None:
                from aim import Repo

                target_repo = Repo(effective_home)
            return bool(target_repo.run_exists(str(prev["hash"])))
        except Exception as e:  # noqa: BLE001 - unreadable repo: re-import rather than skip
            print(f"    (warning: could not check the previous import in {effective_home}: {e})")
            return False

    if not dry_run:
        try:
            import aim  # noqa: F811 - required for a real import
        except ImportError as e:
            raise SystemExit(f"'aim' is not installed ({e}). Install it with `pip install aim`.") from e

    print(f"Importing {len(run_dirs)} run(s) from {root} into Aim experiment '{experiment}' (home: {effective_home})\n")
    for run_dir in run_dirs:
        name = _run_display_name(run_dir, root)
        has_csv = bool(_csv_files(run_dir))
        has_tb = bool(_tb_version_dirs(run_dir))

        if source == "auto":
            chosen = "csv" if has_csv else ("tb" if has_tb else None)
        elif source == "csv":
            chosen = "csv" if has_csv else None
        else:  # source == "tb"
            chosen = "tb" if has_tb else None

        entry = {"run_dir": str(run_dir), "name": name, "source": chosen}

        if chosen is None:
            entry["status"] = f"skipped (no {source} logs)" if source != "auto" else "skipped (no CSV/TB logs)"
            print(f"  - {name}: {entry['status']}")
            results.append(entry)
            continue

        sidecar = _sidecar_path(run_dir)
        if not force and sidecar.is_file():
            try:
                prev = json.loads(sidecar.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                prev = None  # unreadable sidecar: fall through and re-import
            if prev is not None and _prev_run_visible(prev):
                entry.update(status=f"skipped (already imported in {effective_home}, hash {str(prev.get('hash'))[:12]}...)", **prev)
                print(f"  - {name}: {entry['status']}")
                results.append(entry)
                continue

        if dry_run:
            n_points = None
            if chosen == "csv":
                n_points = _count_csv_points(run_dir)
            entry.update(status="dry-run", n_points=n_points)
            shown = f"{n_points} points" if n_points is not None else "?"
            print(f"  - {name}: would import via {chosen} ({shown})")
            results.append(entry)
            continue

        aim_run = aim.Run(experiment=experiment, repo=effective_home)
        try:
            aim_run.name = name
            for key, value in _load_params(run_dir).items():
                aim_run[key] = value
            n_points = import_run_csv(run_dir, aim_run) if chosen == "csv" else import_run_tb(run_dir, aim_run)
            entry.update(status="imported", source=chosen, n_points=n_points, hash=aim_run.hash)
            print(f"  - {name}: imported via {chosen} ({n_points} points), run {aim_run.hash[:12]}...")
        finally:
            aim_run.close()

        sidecar.write_text(
            json.dumps(
                {
                    "hash": entry["hash"],
                    "experiment": experiment,
                    "home": effective_home,
                    "source": chosen,
                    "n_points": n_points,
                    "imported_at": datetime.datetime.now().isoformat(timespec="seconds"),
                },
                indent=4,
            ),
            encoding="utf-8",
        )
        results.append(entry)

    n_imported = sum(1 for r in results if r["status"] == "imported")
    n_skipped = sum(1 for r in results if str(r["status"]).startswith("skipped"))
    summary = f"\nDone: {n_imported} imported, {n_skipped} skipped"
    if dry_run:
        summary += f", {sum(1 for r in results if r['status'] == 'dry-run')} to import"
    print(summary + ("" if dry_run else f". View them with: aim up --repo {effective_home}"))
    return results


def _count_csv_points(run_dir: Path) -> int:
    """Number of non-empty data cells across all metrics.csv files of a run."""
    total = 0
    for csv_file in _csv_files(run_dir):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_cols = [c for c in reader.fieldnames if c not in _AXIS_COLUMNS]
            for row in reader:
                total += sum(1 for col in data_cols if (row.get(col) or "").strip() != "")
    return total


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
        prog="pynop.import_logs",
        description=(
            "Import existing PyNOP training logs (CSVLogger metrics.csv and/or TensorBoard event files) "
            "into an Aim repository, so old runs can be compared with new ones in the Aim UI."
        ),
    )
    parser.add_argument("root", type=Path, help="Root of the training output tree to scan (e.g. .../PDETraining/PDEBENCH).")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Aim experiment name for all imported runs (default: the root's basename).",
    )
    parser.add_argument(
        "--aim-home",
        dest="aim_home",
        default=None,
        help="Aim repository path to import into (default: ~/.aim — aim 3.x's own CWD-based default is avoided on purpose).",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "csv", "tb"],
        default="auto",
        help="Log source per run: 'auto' = CSV if present else TensorBoard (default), or force one of them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import runs that already have a .aim_import.json sidecar (creates additional Aim runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be imported; write nothing to the Aim repository.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> list[dict]:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        The per-run result list from :func:`import_logs`.
    """
    args = parse_args(argv)
    return import_logs(
        root=args.root,
        experiment=args.experiment,
        home=args.aim_home,
        source=args.source,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
