"""Deprecated location for the evaluation driver — re-export shim only.

The driver moved to :mod:`pynop.eval.eval` (the ``pynop/eval`` package now
holds both the CLI driver and the interactive visualisation module). This shim
re-exports its API so that existing imports keep working::

    from pynop.training.eval import run_evaluation   # deprecated, still works

New code should use the canonical location instead::

    from pynop.eval import run_evaluation            # or: python -m pynop.eval.eval ...
"""

import sys
from pathlib import Path

# Direct-script execution support (same bootstrap as the real module): when
# this file is executed directly — e.g. ``python pynop/training/eval.py ...`` —
# Python gives it no package context and the relative re-export below would
# fail; restore it so both invocation styles keep working from any directory.
if __package__ in (None, ""):
    _pkg_dir = Path(__file__).resolve().parents[1]  # .../PyNOP/pynop
    _repo_root = str(_pkg_dir.parent)               # .../PyNOP
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    __package__ = f"{_pkg_dir.name}.training"

from ..eval.eval import (  # noqa: F401  (re-exported for backward compatibility)
    _group_by_length,
    _make_reader,
    build_eval_module,
    format_results_table,
    main,
    parse_args,
    plot_error_evolution,
    run_evaluation,
    scan_trajectories,
    select_eval_ids,
)

__all__ = [
    "parse_args",
    "main",
    "run_evaluation",
    "select_eval_ids",
    "plot_error_evolution",
    "format_results_table",
    "build_eval_module",
    "scan_trajectories",
]


if __name__ == "__main__":
    main()
