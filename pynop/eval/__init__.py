"""Evaluation package: CLI driver + interactive visualisation.

* :mod:`pynop.eval.eval` — command-line evaluation driver (full autoregressive
  rollout of every trajectory in an unseen split; field-level and spectral-band
  metrics, per-step error curves, self-contained JSON results and PNG figures).
  Moved here from ``pynop/training/eval.py`` — a deprecated re-export shim
  remains at the old location for backward compatibility.
* :mod:`pynop.eval.visualisation` — interactive plotly visualisation of a
  prediction against the ground truth with per-step error maps and independent,
  configurable color scales (:func:`~pynop.eval.visualisation.plot_interactive`).

Plotly is an optional dependency (``pip install pynop[viz]``): it is imported
lazily inside the plotting functions so that importing this package never
requires it.
"""

from .eval import *  # noqa: F401,F403
from .visualisation import *  # noqa: F401,F403
