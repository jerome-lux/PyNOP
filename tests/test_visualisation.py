"""Tests for :mod:`pynop.eval.visualisation` (interactive plotly figures).

Covers:

* :func:`~pynop.eval.visualisation.compute_step_errors` — per-step RMSE / MAE /
  relative-L2 values against hand-computed references, single-channel (3-D)
  inputs;
* :func:`~pynop.eval.visualisation.plot_interactive` — figure structure (three
  panels, one frame per step), **fixed vs adaptive color scales for the data
  and error panels independently**, explicit ``vmin``/``vmax`` overrides,
  input validation errors, optional step labels, the **scrub slider** (one
  step per frame, labeled by index or time), the **single shared data
  colorbar** on the Prediction panel plus a dedicated error colorbar, and a
  regression on the notebook original's initial-title bug (the step-0 title
  must show the step-0 metrics, not the last step's).

Skipped when plotly is not installed.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("plotly")

from pynop.eval.visualisation import compute_step_errors, plot_interactive  # noqa: E402


T, C, H, W = 5, 2, 8, 8


@pytest.fixture()
def fields():
    """Deterministic (pred, gt) torch pairs with distinct per-step errors."""
    rng = np.random.default_rng(0)
    gt = (rng.standard_normal((T, C, H, W)) * 0.5 + 1.0).astype(np.float32)  # positive field
    pred = (gt + 0.1 * rng.standard_normal((T, C, H, W))).astype(np.float32)
    return torch.from_numpy(pred), torch.from_numpy(gt)


# ---------------------------------------------------------------------------
# compute_step_errors
# ---------------------------------------------------------------------------


def test_compute_step_errors_values(fields):
    pred, gt = fields
    out = compute_step_errors(pred, gt, channel_idx=1)
    p = pred[:, 1].numpy()
    g = gt[:, 1].numpy()
    diff = p - g
    assert np.allclose(out["rmse"], np.sqrt(np.mean(diff**2, axis=(1, 2))))
    assert np.allclose(out["mae"], np.mean(np.abs(diff), axis=(1, 2)))
    expected_nrmse = (
        np.linalg.norm(diff.reshape(T, -1), axis=1) / np.linalg.norm(g.reshape(T, -1), axis=1)
    )
    assert np.allclose(out["nrmse"], expected_nrmse)
    assert np.allclose(out["error"], np.abs(diff))


def test_compute_step_errors_3d_single_channel(fields):
    pred, gt = fields
    out_3d = compute_step_errors(pred[:, 0], gt[:, 0])  # (T, H, W) input
    out_4d = compute_step_errors(pred, gt, channel_idx=0)
    assert np.allclose(out_3d["rmse"], out_4d["rmse"])


def test_compute_step_errors_shape_mismatch(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="same shape"):
        compute_step_errors(pred[:-1], gt)


# ---------------------------------------------------------------------------
# plot_interactive — structure
# ---------------------------------------------------------------------------


def _heatmaps(fig) -> list:
    return [t for t in fig.data if t.type == "heatmap"]


def test_figure_structure(fields):
    pred, gt = fields
    fig = plot_interactive(pred, gt, channel_idx=1)

    assert len(_heatmaps(fig)) == 3
    annotation_texts = [a.text for a in fig.layout.annotations]
    for expected in ("Ground Truth", "Prediction", "Abs Error"):
        assert expected in annotation_texts

    # One frame per step, named "0".."T-1", each carrying the three panels.
    assert len(fig.frames) == T
    for i, frame in enumerate(fig.frames):
        assert frame.name == str(i)
        assert len(_heatmaps(frame)) == 3
        assert _heatmaps(frame)[0].z.shape == (H, W)

    # Play/Pause buttons present (attribute access: plotly >= 7 returns typed objects).
    menus = fig.layout.updatemenus
    labels = [b.label for b in menus[0].buttons]
    assert "Play" in labels and "Pause" in labels


def test_single_data_colorbar_and_error_colorbar(fields):
    """GT/Pred share one colorbar (on the Prediction panel); Error has its own.

    Layout: GT | Pred | data colorbar | Error | error colorbar — no overlap.
    The showscale flags and explicit paper positions must be carried on every
    frame, not only the base traces (Plotly replaces base traces per frame).
    """
    from pynop.eval import visualisation as viz

    pred, gt = fields
    fig = plot_interactive(pred, gt)
    # Base figure and every frame must carry the same showscale flags/positions.
    for obj in [fig, *fig.frames]:
        hm = _heatmaps(obj)
        assert [t.showscale for t in hm] == [False, True, True]
        # Data colorbar sits between the Prediction and Abs Error panels...
        assert hm[1].colorbar.xanchor == "left"
        assert viz._PRED_DOMAIN[1] < hm[1].colorbar.x < viz._ERR_DOMAIN[0]
        # ...and the error colorbar right of the Abs Error panel.
        assert hm[2].colorbar.xanchor == "left"
        assert viz._ERR_DOMAIN[1] < hm[2].colorbar.x <= 1.0


def test_scrub_slider_one_step_per_frame(fields):
    """The figure must carry a draggable slider with one step per frame."""
    pred, gt = fields
    fig = plot_interactive(pred, gt)

    sliders = fig.layout.sliders
    assert len(sliders) == 1
    steps = sliders[0].steps
    assert len(steps) == T
    for i, step in enumerate(steps):
        assert step.method == "animate"
        # args: [[frame_name], {"mode": ..., ...}] — the name must exist.
        frame_names, options = step.args
        assert list(frame_names) == [str(i)]
        assert options["mode"] == "immediate"


def test_scrub_slider_labels_default_and_step_labels(fields):
    pred, gt = fields

    fig = plot_interactive(pred, gt)
    assert [s.label for s in fig.layout.sliders[0].steps] == [str(i) for i in range(T)]

    labels = list(range(10, 10 + T))
    fig = plot_interactive(pred, gt, step_labels=labels)
    assert [s.label for s in fig.layout.sliders[0].steps] == [str(lb) for lb in labels]


def test_initial_title_shows_step_zero_metrics(fields):
    """Regression: the notebook original printed the LAST step's MAE at step 0."""
    pred, gt = fields
    out = compute_step_errors(pred, gt)
    fig = plot_interactive(pred, gt)

    title = fig.layout.title.text
    assert f"RMSE: {out['rmse'][0]:.3e}" in title
    assert f"MAE: {out['mae'][0]:.3e}" in title  # would fail if mae[-1] were shown (distinct here)


def test_step_labels_in_title(fields):
    pred, gt = fields
    labels = list(range(10, 10 + T))
    fig = plot_interactive(pred, gt, step_labels=labels)
    assert "t=10" in fig.layout.title.text
    assert f"t={10 + T - 1}" in fig.frames[-1].layout.title.text


# ---------------------------------------------------------------------------
# plot_interactive — color scales (the core requirement)
# ---------------------------------------------------------------------------


def test_fixed_data_scale_default(fields):
    """Default fixed data range = min/max of GT+Pred over every displayed step."""
    pred, gt = fields
    fig = plot_interactive(pred, gt)  # channel 0
    p = pred[:, 0].numpy()
    g = gt[:, 0].numpy()
    lo, hi = float(min(p.min(), g.min())), float(max(p.max(), g.max()))
    for trace in _heatmaps(fig)[:2]:
        assert trace.zmin == pytest.approx(lo)
        assert trace.zmax == pytest.approx(hi)


def test_fixed_error_scale_default(fields):
    """Default fixed error range = min/max of the absolute error over all steps."""
    pred, gt = fields
    fig = plot_interactive(pred, gt)
    err = compute_step_errors(pred, gt)["error"]
    lo, hi = float(err.min()), float(err.max())
    trace = _heatmaps(fig)[2]
    assert trace.zmin == pytest.approx(lo)
    assert trace.zmax == pytest.approx(hi)


def test_fixed_ranges_repeated_on_frames(fields):
    """Frames replace base traces in Plotly — bounds must be carried on each frame."""
    pred, gt = fields
    fig = plot_interactive(pred, gt)
    for frame in fig.frames:
        hm = _heatmaps(frame)
        assert hm[0].zmin == pytest.approx(_heatmaps(fig)[0].zmin)
        assert hm[2].zmax == pytest.approx(_heatmaps(fig)[2].zmax)


def test_adaptive_scales(fields):
    pred, gt = fields
    fig = plot_interactive(pred, gt, data_scale="adaptive", error_scale="adaptive")
    for trace in _heatmaps(fig):
        assert trace.zmin is None and trace.zmax is None
    for frame in fig.frames:
        for trace in _heatmaps(frame):
            assert trace.zmin is None and trace.zmax is None


def test_independent_scales(fields):
    """Data adaptive + error fixed (and vice versa) must not interfere."""
    pred, gt = fields
    fig = plot_interactive(pred, gt, data_scale="adaptive")  # error stays fixed
    hm = _heatmaps(fig)
    assert hm[0].zmin is None and hm[1].zmin is None
    assert hm[2].zmin is not None and hm[2].zmax is not None

    fig = plot_interactive(pred, gt, error_scale="adaptive")  # data stays fixed
    hm = _heatmaps(fig)
    assert hm[0].zmin is not None and hm[1].zmin is not None
    assert hm[2].zmin is None and hm[2].zmax is None


def test_explicit_bounds_override(fields):
    pred, gt = fields
    fig = plot_interactive(pred, gt, data_vmin=-1.0, data_vmax=2.0, error_vmin=0.0, error_vmax=0.5)
    hm = _heatmaps(fig)
    assert (hm[0].zmin, hm[0].zmax) == (-1.0, 2.0)
    assert (hm[1].zmin, hm[1].zmax) == (-1.0, 2.0)
    assert (hm[2].zmin, hm[2].zmax) == (0.0, 0.5)


# ---------------------------------------------------------------------------
# plot_interactive — input handling / validation
# ---------------------------------------------------------------------------


def test_numpy_and_torch_mixed_inputs(fields):
    pred, gt = fields
    fig = plot_interactive(pred.numpy(), gt)  # one side numpy, the other torch
    assert len(_heatmaps(fig)) == 3 and len(fig.frames) == T


def test_3d_single_channel_input(fields):
    pred, gt = fields
    fig = plot_interactive(pred[:, 0], gt[:, 0])
    assert len(_heatmaps(fig)) == 3


def test_shape_mismatch_raises(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="same shape"):
        plot_interactive(pred[:-1], gt)


def test_bad_channel_raises(fields):
    pred, gt = fields
    with pytest.raises(IndexError, match="channel_idx"):
        plot_interactive(pred, gt, channel_idx=C + 3)


def test_unknown_scale_raises(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="data_scale must be one of"):
        plot_interactive(pred, gt, data_scale="auto")
    with pytest.raises(ValueError, match="error_scale must be one of"):
        plot_interactive(pred, gt, error_scale="minmax")


def test_half_specified_bounds_raise(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="both data_vmin and data_vmax"):
        plot_interactive(pred, gt, data_vmin=0.0)
    with pytest.raises(ValueError, match="both error_vmin and error_vmax"):
        plot_interactive(pred, gt, error_vmax=0.5)


def test_inverted_bounds_raise(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="strictly smaller"):
        plot_interactive(pred, gt, data_vmin=2.0, data_vmax=-1.0)


def test_step_labels_length_mismatch_raises(fields):
    pred, gt = fields
    with pytest.raises(ValueError, match="one per step"):
        plot_interactive(pred, gt, step_labels=[0.0, 1.0])
