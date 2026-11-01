"""Interactive visualisation of model predictions against the ground truth.

This module is the library version of the ``plot_interactive`` helper that used
to live in ``notebooks/utils.py``: it renders an **animated plotly figure** with
three panels per time step — *Ground Truth*, *Prediction* and *Abs Error* — and
shows the per-step RMSE / MAE / nRMSE in the title, so a full autoregressive
rollout can be inspected interactively (play/pause + scrubbing).

Improvements over the notebook original
---------------------------------------
* **Independent color scales, no overlapping colorbars.** The original stacked
  all three fields into a single ``px.imshow`` and therefore shared one color
  range: since the error is usually orders of magnitude smaller than the field
  amplitude, the error panel was almost uniformly dark. Each quantity now has
  its own scale; the two data panels share a **single** colorbar (drawn on the
  Prediction panel). Because Plotly.js places the colorbars of *all* subplot
  traces at one default paper position (where they overlap each other and the
  last panel), both the panel domains and the colorbar positions are set
  explicitly in paper coordinates — final layout: *GT | Pred | data colorbar
  | Error | error colorbar*.
* **Configurable scales** — for both the data and the error:

  - ``"fixed"`` (default): one range shared by *all* steps, so changes over
    time are directly comparable. The default fixed range is the min/max of
    the displayed quantity over every step (GT+Pred for the data panels,
    absolute error for the error panel); explicit ``*_vmin`` / ``*_vmax``
    values override it.
  - ``"adaptive"``: each frame auto-scales to its own min/max — useful to see
    local structure in a step at the cost of losing temporal comparability.

* The initial-layout title bug of the original (it printed the *last* step's
  MAE for step 0) is fixed; facet titles are set through ``make_subplots``
  instead of fragile annotation-index patching.
* Inputs may be torch tensors or numpy arrays, ``(T, C, H, W)`` or
  ``(T, H, W)`` (single channel); shapes and the channel index are validated.
* The figure is **returned** rather than shown: in a notebook it renders
  automatically as the last expression of a cell; call ``fig.show()`` from a
  script.

Plotly is an optional dependency — install it with ``pip install pynop[viz]``
(or ``conda install -c conda-forge plotly``). It is imported lazily inside the
functions below, so importing :mod:`pynop` never requires it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import torch

__all__ = ["plot_interactive", "compute_step_errors"]

ArrayLike = Union[torch.Tensor, np.ndarray]
_SCALES = ("fixed", "adaptive")

# Paper-coordinate layout: GT | gap | Pred | data-colorbar zone | Error | error-colorbar zone.
# Plotly.js places the colorbars of *all* subplot traces at the same default paper position
# (near the right edge), where they overlap each other and the last panel — so both the panel
# domains and the colorbar positions are set explicitly below.
_GT_DOMAIN = (0.00, 0.25)
_PRED_DOMAIN = (0.29, 0.54)
_ERR_DOMAIN = (0.64, 0.89)
_DATA_CB_X = 0.55  # inside the zone between Prediction and Abs Error
_ERR_CB_X = 0.90   # right of the Abs Error panel; tick labels fit before x=1.0
_PANEL_DOMAINS = (_GT_DOMAIN, _PRED_DOMAIN, _ERR_DOMAIN)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_plotly() -> tuple:
    """Import the plotly pieces used by this module, raising an actionable error if missing.

    Returns:
        Tuple ``(go, make_subplots)`` — ``plotly.graph_objects`` and
        ``plotly.subplots.make_subplots``.

    Raises:
        ImportError: If plotly is not installed, with installation hints.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "plotly is required for interactive visualisation — install it with "
            "`pip install pynop[viz]` (or `conda install -c conda-forge plotly`)."
        ) from e
    return go, make_subplots


def _to_numpy(x: ArrayLike, name: str) -> np.ndarray:
    """Convert a torch tensor or array-like to a float64 numpy array.

    Args:
        x: Input data (torch tensors are detached and moved to CPU).
        name: Field name used in error messages.

    Returns:
        The same data as a ``np.float64`` ndarray.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim not in (3, 4):
        raise ValueError(f"{name} must have shape (T, C, H, W) or (T, H, W), got {arr.shape}.")
    return arr


def _check_channel(arr: np.ndarray, channel_idx: int, name: str) -> None:
    """Validate the channel index against a ``(T, C, H, W)`` array.

    Args:
        arr: Array with a channel axis at position 1.
        channel_idx: Channel to select.
        name: Field name used in error messages.

    Raises:
        IndexError: If ``channel_idx`` is out of range for the number of
            channels of ``name``.
    """
    if not 0 <= int(channel_idx) < arr.shape[1]:
        raise IndexError(f"channel_idx={channel_idx} out of range: {name} has {arr.shape[1]} channel(s).")


def _resolve_range(
    scale: str,
    vmin: Optional[float],
    vmax: Optional[float],
    data_min: float,
    data_max: float,
    what: str,
) -> tuple:
    """Resolve a panel color range from the requested mode and overrides.

    Args:
        scale: ``"fixed"`` or ``"adaptive"``.
        vmin / vmax: Explicit bounds; both must be given together (or both None).
        data_min / data_max: Min/max of the displayed quantity over all steps —
            used as the default fixed range.
        what: Quantity name used in error messages ("data", "error").

    Returns:
        Tuple ``(zmin, zmax)``; ``None`` entries mean "auto per frame"
        (adaptive mode).

    Raises:
        ValueError: On an unknown scale mode, half-specified bounds or
            ``vmin > vmax``.
    """
    if scale not in _SCALES:
        raise ValueError(f"{what}_scale must be one of {_SCALES}, got {scale!r}.")
    if (vmin is None) != (vmax is None):
        raise ValueError(f"Provide both {what}_vmin and {what}_vmax, or neither.")
    if vmin is not None:
        if float(vmin) >= float(vmax):
            raise ValueError(f"{what}_vmin ({vmin}) must be strictly smaller than {what}_vmax ({vmax}).")
        return float(vmin), float(vmax)
    if scale == "fixed":
        return float(data_min), float(data_max)
    return None, None


# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------


def compute_step_errors(pred: ArrayLike, gt: ArrayLike, channel_idx: int = 0) -> dict:
    """Compute per-step error metrics of a prediction against the ground truth.

    Both inputs are aligned step by step (step ``k`` of ``pred`` is compared to
    step ``k`` of ``gt``); only the selected channel is used. The nRMSE is the
    **relative L2** error ``‖p - g‖₂ / ‖g‖₂`` per step — the same scale-invariant
    metric as :mod:`pynop.training.metrics` (for zero-mean fields the
    denominator equals the field standard deviation).

    Args:
        pred: Predicted fields, shape ``(T, C, H, W)`` or ``(T, H, W)``.
        gt: Ground-truth fields with the same shape as ``pred``.
        channel_idx: Channel to evaluate (default 0).

    Returns:
        Dict of numpy arrays:

        * ``"rmse"`` — per-step RMSE, shape ``(T,)``;
        * ``"mae"`` — per-step mean absolute error, shape ``(T,)``;
        * ``"nrmse"`` — per-step relative L2 (PDEBench "normalized RMSE"),
          shape ``(T,)``;
        * ``"error"`` — the absolute-error field ``|gt - pred|``, shape
          ``(T, H, W)``.

    Raises:
        ValueError: If the two inputs have different step counts or shapes.
        IndexError: If ``channel_idx`` is out of range.
    """
    p = _to_numpy(pred, "pred")
    g = _to_numpy(gt, "gt")
    if p.shape != g.shape:
        raise ValueError(f"pred and gt must have the same shape, got {p.shape} vs {g.shape}.")
    if p.ndim == 3:  # (T, H, W) -> single channel
        p = p[:, None]
        g = g[:, None]
    _check_channel(p, channel_idx, "pred")

    p = p[:, int(channel_idx)]  # (T, H, W)
    g = g[:, int(channel_idx)]
    diff = p - g
    error = np.abs(diff)
    rmse = np.sqrt(np.mean(diff**2, axis=(1, 2)))
    mae = np.mean(error, axis=(1, 2))
    denom = np.linalg.norm(g.reshape(g.shape[0], -1), axis=1)
    nrmse = np.where(denom > 0.0, np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1) / np.maximum(denom, 1e-30), np.nan)
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "error": error}


# ---------------------------------------------------------------------------
# Interactive figure
# ---------------------------------------------------------------------------


def plot_interactive(
    pred: ArrayLike,
    gt: ArrayLike,
    channel_idx: int = 0,
    data_scale: str = "fixed",
    error_scale: str = "fixed",
    data_vmin: Optional[float] = None,
    data_vmax: Optional[float] = None,
    error_vmin: Optional[float] = None,
    error_vmax: Optional[float] = None,
    step_labels: Optional[Sequence[float]] = None,
    height: int = 560,
) -> Any:
    """Build an animated plotly figure comparing a prediction to the ground truth.

    One frame per time step with three panels — *Ground Truth*, *Prediction*
    (both on the **data** scale) and *Abs Error* (on its own **error** scale).
    The two data panels share a single colorbar drawn on the Prediction panel;
    the error panel has its own, so no colorbars overlap. The title of every
    frame shows the step index (or its label), RMSE, MAE and nRMSE of that
    step. A play button animates the frames; the slider lets you scrub manually.

    Args:
        pred: Predicted fields, shape ``(T, C, H, W)`` or ``(T, H, W)``, torch
            tensor or numpy array (e.g. the stacked per-step predictions of a
            full autoregressive rollout).
        gt: Ground-truth fields with the same shape as ``pred`` (the target
            snapshots aligned step by step).
        channel_idx: Channel to display and evaluate (default 0).
        data_scale: Color scale of the GT/Pred panels — ``"fixed"`` (default):
            one range for all steps, defaulting to the min/max of GT+Pred over
            every displayed step; ``"adaptive"``: each frame auto-scales.
        error_scale: Color scale of the Abs-Error panel — same semantics as
            ``data_scale``, with the default fixed range taken from the min/max
            of the absolute error over every displayed step.
        data_vmin / data_vmax: Explicit bounds for the data panels (override
            both auto and fixed defaults); must be given together.
        error_vmin / error_vmax: Explicit bounds for the error panel; same rules.
        step_labels: Optional per-step labels (e.g. absolute time indices) used
            in the frame titles instead of ``Step k``; length must equal T.
        height: Figure height in pixels.

    Returns:
        The plotly ``go.Figure`` (not shown). In a notebook, returning it as
        the last expression of a cell renders it interactively; from a script
        call ``fig.show()``.

    Raises:
        ValueError: On shape/length mismatches or invalid scale/bounds values.
        IndexError: If ``channel_idx`` is out of range.
        ImportError: If plotly is not installed (see module docstring).
    """
    go, make_subplots = _require_plotly()

    p_all = _to_numpy(pred, "pred")
    g_all = _to_numpy(gt, "gt")
    if p_all.shape != g_all.shape:
        raise ValueError(f"pred and gt must have the same shape, got {p_all.shape} vs {g_all.shape}.")
    if p_all.ndim == 3:
        p_all = p_all[:, None]
        g_all = g_all[:, None]
    _check_channel(p_all, channel_idx, "pred")

    ch = int(channel_idx)
    T = p_all.shape[0]
    p = p_all[:, ch].astype(np.float32)  # (T, H, W), float32 keeps the frame payload small
    g = g_all[:, ch].astype(np.float32)
    metrics = compute_step_errors(p, g, channel_idx=0)
    err = metrics["error"].astype(np.float32)

    if step_labels is not None and len(step_labels) != T:
        raise ValueError(f"step_labels must have {T} entries (one per step), got {len(step_labels)}.")

    # ------------------------------------------------------------------ scales
    data_zmin, data_zmax = _resolve_range(
        data_scale, data_vmin, data_vmax, min(p.min(), g.min()), max(p.max(), g.max()), "data"
    )
    err_zmin, err_zmax = _resolve_range(error_scale, error_vmin, error_vmax, float(err.min()), float(err.max()), "error")

    def _title(i: int) -> str:
        label = f"t={step_labels[i]}" if step_labels is not None else f"Step {i}"
        return (
            f"{label} | RMSE: {metrics['rmse'][i]:.3e} | MAE: {metrics['mae'][i]:.3e} "
            f"| nRMSE: {metrics['nrmse'][i]:.3e} | Channel {ch}"
        )

    def _slider_label(i: int) -> str:
        """Short per-step label shown on the scrub slider (index or time)."""
        return str(step_labels[i]) if step_labels is not None else str(i)

    # ------------------------------------------------------------------ figure
    # Note: when a frame is displayed, Plotly *replaces* the base traces with
    # the frame's versions — attributes not set on the frame trace fall back to
    # defaults, not to the base trace. The color bounds and the whole colorbar
    # spec are therefore repeated on every frame (None in adaptive mode = auto-scale).
    def _panel(
        z: np.ndarray, colorscale: str, zmin: Optional[float], zmax: Optional[float], colorbar: Optional[dict]
    ) -> Any:
        """Build one heatmap panel; ``colorbar=None`` hides its scale (shared with a neighbor)."""
        return go.Heatmap(
            z=z,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=colorbar is not None,
            colorbar=dict(thickness=12, **colorbar) if colorbar else {},
        )

    def _frame_traces(i: int) -> list:
        # Layout: GT | Pred | data colorbar | Error | error colorbar — the two
        # data panels share ONE colorbar (drawn on the Prediction panel), so it
        # never overlaps; the error panel keeps its own. Both are positioned in
        # paper coordinates inside their reserved zones (see _PANEL_DOMAINS).
        return [
            _panel(g[i], "Viridis", data_zmin, data_zmax, colorbar=None),
            _panel(p[i], "Viridis", data_zmin, data_zmax, colorbar=dict(x=_DATA_CB_X, xanchor="left")),
            _panel(err[i], "Reds", err_zmin, err_zmax, colorbar=dict(x=_ERR_CB_X, xanchor="left")),
        ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Ground Truth", "Prediction", "Abs Error"))
    # Explicit panel domains (see _PANEL_DOMAINS): the default equal columns end at
    # paper x=1.0, leaving no room for a colorbar right of the last panel.
    for ax, domain in zip((fig.layout.xaxis, fig.layout.xaxis2, fig.layout.xaxis3), _PANEL_DOMAINS):
        ax.domain = domain
    # make_subplots centered its title annotations on the *default* column domains —
    # re-center them on the explicit ones (they use xref="paper").
    for ann, domain in zip(fig.layout.annotations, _PANEL_DOMAINS):
        ann.x = (domain[0] + domain[1]) / 2

    for col, trace in enumerate(_frame_traces(0), start=1):
        fig.add_trace(trace, row=1, col=col)

    # Explicit frame names: plotly >= 7 no longer auto-names frames on assignment,
    # and the scrub slider labels come from these names.
    frames = [go.Frame(name=str(i), data=_frame_traces(i), layout=go.Layout(title=_title(i))) for i in range(T)]
    if hasattr(fig, "add_frames"):  # plotly <= 6
        fig.add_frames(frames)
    else:  # plotly >= 7 removed Figure.add_frame(s); frames are set by assignment
        fig.frames = frames

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Scrub slider: one step per frame, so the user can drag to any timestep.
    # Each step jumps immediately (duration 0) to the named frame; the frame's
    # layout then updates the title with that step's RMSE / MAE / nRMSE.
    fig.update_layout(
        title=dict(text=_title(0), x=0.5, xanchor="center"),
        height=int(height),
        margin=dict(l=10, r=10, t=60, b=50),  # bottom room for the slider
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                        label=_slider_label(i),
                    )
                    for i in range(T)
                ]
            )
        ],
        # Play/Pause buttons (frames are named "0".."T-1"); parked in the free
        # corner of the title row so they never overlap a panel.
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=["stop"]),
                ],
                x=0.98,
                y=1.0,
                xanchor="right",
                yanchor="top",
            )
        ],
    )
    return fig
