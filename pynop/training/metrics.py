"""Field-level evaluation metrics for PyNOP multi-step predictions.

These metrics are computed on **model-space** fields, i.e. after the dataset
``(x + shift) * scale`` transform — the same space the network was trained in.
All functions accept channels-first tensors of shape ``(..., C, H, W)`` (any
number of leading "sample" dimensions, e.g. ``(B, C, H, W)`` or a stacked
rollout ``(S, B, C, H, W)``).

Metric definitions
------------------
* ``MSE`` / ``RMSE`` — plain mean squared error (and its square root), pooled
  over all elements.
* ``nRMSE`` — **relative L2** error: per sample and channel
  ``‖pred − target‖₂ / ‖target‖₂``, averaged over all samples/channels. This is
  PDEBench's "normalized RMSE" (``pdebench/models/metrics.py::metric_func``).
  The ratio is **scale-invariant** (``x → a·x`` leaves it unchanged), so it
  stays comparable across datasets with different amplitudes. Under an additive
  shift the error itself is unchanged but the reference energy ``‖target‖``
  changes; for zero-mean-normalized fields ``(x − μ)/σ`` this denominator is
  exactly the field's standard deviation, so the metric remains a meaningful
  relative error — which is precisely how PDEBench uses it on their
  zero-mean/unit-variance data. Normalizations by ``mean(target)²`` would
  degenerate on zero-mean fields and are deliberately not used.
* ``MAE`` / max absolute error — L1 error and worst-case pointwise error
  (outlier / blow-up detector, as in PDEBench's ``err_Max``).
* ``R²`` — coefficient of determination per sample/channel
  ``1 − ‖err‖² / ‖target − mean(target)‖²`` (spatial mean), averaged over all
  units. Unlike nRMSE it is invariant to **any affine rescaling** (shift *and*
  scale), so it is the safest normalized metric for zero-mean datasets; for
  per-channel zero-mean data it reduces to ``1 − nRMSE²`` unit by unit, i.e.
  both carry the same information there.
* **Spectral band errors** — the field is transformed with a real FFT over the
  last two spatial dimensions and the squared error energy is grouped by radial
  frequency bands in FFT bin-index units (PDEBench-style shells, default
  ``[0,4) / [4,12) / [12,+∞)``, matching their ``iLow=4``/``iHigh=12``). By
  default each band reports the
   **relative** error ``‖F(err)_band‖₂ / ‖F(target)_band‖₂`` per sample/channel,
   averaged — the frequency-domain generalization of nRMSE and therefore also
   scale-invariant. With ``relative_spectral=False`` the absolute (unscaled)
  band energy is reported instead, closer to PDEBench's raw Fourier error.

:class:`FieldMetricAccumulator` accumulates these metrics incrementally over a
dataloader of rollout predictions (constant memory), and also tracks the
per-AR-step RMSE / nRMSE curve — the error growth along the autoregressive
unroll, which is the key diagnostic for multi-step models.
"""

from typing import Optional, Sequence, Tuple

import torch

__all__ = [
    "AverageMeter",
    "radial_frequency_grid",
    "relative_l2_error",
    "r_squared",
    "spectral_band_error",
    "FieldMetricAccumulator",
    "compute_field_metrics",
]


class AverageMeter:
    """Simple running average (kept for backward compatibility)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_samples(x: torch.Tensor) -> torch.Tensor:
    """Reshape a channels-first field ``(..., C, H, W)`` to ``(N, C, H, W)``.

    Args:
        x: Tensor with at least 4 dimensions (leading dims are pooled into N).

    Returns:
        The same tensor reshaped to 4-D.

    Raises:
        ValueError: If ``x`` has fewer than 4 dimensions.
    """
    if x.ndim < 4:
        raise ValueError(f"Expected a channels-first field (..., C, H, W), got shape {tuple(x.shape)}.")
    return x.reshape(-1, *x.shape[-3:])


def radial_frequency_grid(h: int, w: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Radial frequency grid (in FFT bin-index units) for a real FFT.

    Matches the layout of ``torch.fft.rfftn(x, dim=(-2, -1))`` output
    (shape ``(h, w // 2 + 1)``). The magnitude is computed from the non-negative
    **bin indices** — exactly PDEBench's shell index
    ``it = floor(sqrt(i² + j²))`` in ``metric_func``, where ``i``/``j`` are FFT
    bin numbers (Nyquist = ``h/2`` / ``w/2``). Note the units depend on the grid
    size, as in PDEBench: a band bound of 4 covers shells 0–3 for any resolution.

    Args:
        h: Field height.
        w: Field width.
        device: Device for the returned grid (default: CPU).

    Returns:
        Tensor of shape ``(h, w // 2 + 1)`` with ``k = sqrt(iy² + ix²)``, where
        ``iy``/``ix`` are non-negative FFT bin indices.
    """
    iy = torch.arange(h, device=device, dtype=torch.float32)
    iy = torch.minimum(iy, float(h) - iy).reshape(-1, 1)  # fft order -> |index|
    ix = torch.arange(w // 2 + 1, device=device, dtype=torch.float32).reshape(1, -1)
    return torch.sqrt(ix**2 + iy**2)


def _band_label(k_min: float, k_max: Optional[float]) -> str:
    """Human-readable label for a radial band ``[k_min, k_max)``."""
    if k_max is None:
        return f"k>={float(k_min):g}"
    return f"k[{float(k_min):g},{float(k_max):g})"


def _l2_over_spatial(x4: torch.Tensor) -> torch.Tensor:
    """L2 norm over the spatial dims of a 4-D tensor ``(N, C, H, W)``.

    Args:
        x4: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Tensor of shape ``(N, C)`` with one norm per (sample, channel).
    """
    flat = x4.reshape(x4.shape[0], x4.shape[1], -1)  # (N, C, H*W)
    return flat.pow(2).sum(dim=-1).sqrt()  # (N, C)


# ---------------------------------------------------------------------------
# Pointwise / global metrics (one-shot functions)
# ---------------------------------------------------------------------------


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative L2 error ``‖pred − target‖₂ / ‖target‖₂`` per sample/channel.

    The norm is taken over the spatial dimensions ``(H, W)``; the ratio is
    computed for every (sample, channel) pair — leading dims pooled into
    samples — and averaged over all of them. This is PDEBench's "normalized
    RMSE". It is scale-invariant (``x → a·x``); under an additive shift the
    error is unchanged while the reference energy ``‖target‖`` changes, so for
    zero-mean-normalized fields the denominator equals the field standard
    deviation and the metric remains meaningful. For a fully affine-invariant
    variant see :func:`r_squared`.

    Args:
        pred: Prediction, shape ``(..., C, H, W)``.
        target: Ground truth, same shape as ``pred``.
        eps: Numerical guard added to the target norm (fields with near-zero
            energy yield a large ratio — that is informative, not an error).

    Returns:
        Scalar tensor with the mean relative L2 error.
    """
    p = _to_samples(pred)
    t = _to_samples(target)
    num = _l2_over_spatial(p - t)  # (N, C)
    den = _l2_over_spatial(t) + eps  # (N, C)
    return (num / den).mean()


def r_squared(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient of determination per sample/channel, averaged over all units.

    ``R² = 1 − ‖err‖² / ‖target − mean(target)‖²`` with the norms taken over
    the spatial dimensions ``(H, W)`` for every (sample, channel) pair — the
    same unit as :func:`relative_l2_error`. Both numerator and denominator are
    shift- and scale-invariant, so R² is invariant to **any** affine rescaling
    of the data — the safest normalized metric for zero-mean datasets. For
    per-channel zero-mean data it equals ``1 − nRMSE²`` exactly, unit by unit.

    Args:
        pred: Prediction, shape ``(..., C, H, W)``.
        target: Ground truth, same shape as ``pred``.
        eps: Numerical guard on the total sum of squares (constant fields).

    Returns:
        Scalar tensor with the mean R² (1 = perfect prediction; negative values
        mean worse than predicting the per-channel spatial mean).
    """
    p = _to_samples(pred)  # (N, C, H, W)
    t = _to_samples(target)
    n, c = p.shape[0], p.shape[1]
    ss_res = ((p - t).reshape(n, c, -1)).pow(2).sum(dim=-1)  # (N, C)
    centered = t - t.mean(dim=(-2, -1), keepdim=True)  # per-channel spatial mean
    ss_tot = centered.reshape(n, c, -1).pow(2).sum(dim=-1) + eps  # (N, C)
    return (1.0 - ss_res / ss_tot).mean()


def spectral_band_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    bands: Sequence[Tuple[float, Optional[float]]] = ((0.0, 4.0), (4.0, 12.0), (12.0, None)),
    eps: float = 1e-8,
    relative: bool = True,
) -> dict:
    """Grouped Fourier error over radial frequency bands of the last two dims.

    The real FFT ``rfftn`` is taken over ``(H, W)`` with an orthonormal
    normalization (consistent with :class:`pynop.core.loss.SpectralLoss`). For
    each band ``[k_min, k_max)`` of the radial frequency magnitude:

    * ``relative=True`` (default): per sample/channel ratio
      ``‖F(err)_band‖₂ / ‖F(target)_band‖₂``, averaged — the spectral
      generalization of nRMSE, scale-invariant like it.
    * ``relative=False``: mean band error energy ``‖F(err)_band‖₂`` per
      sample/channel (PDEBench-style absolute Fourier error, unscaled).

    A band whose target energy is near zero yields a large relative ratio —
    the model has no signal to match there and any residual is pure noise.

    Args:
        pred: Prediction, shape ``(..., C, H, W)``.
        target: Ground truth, same shape as ``pred``.
        bands: Sequence of ``(k_min, k_max)`` radial bounds in FFT bin-index
            units (see :func:`radial_frequency_grid`); ``k_max=None`` means "up
            to Nyquist". Defaults to PDEBench's ``iLow=4`` / ``iHigh=12`` shell
            split.
        eps: Numerical guard on the target band energy.
        relative: If True report relative (scale-free) band errors, otherwise
            absolute band error energies.

    Returns:
        Dict mapping each band label (e.g. ``"k[0,4)"``, ``"k>=12"``) to a
        scalar tensor with the band error.
    """
    p = _to_samples(pred)
    t = _to_samples(target)
    h, w = p.shape[-2], p.shape[-1]

    f_pred = torch.fft.rfftn(p, dim=(-2, -1), norm="ortho")
    f_tgt = torch.fft.rfftn(t, dim=(-2, -1), norm="ortho")
    k_mag = radial_frequency_grid(h, w, device=p.device)  # (h, w//2+1)

    out = {}
    for k_min, k_max in bands:
        mask = k_mag >= float(k_min)
        if k_max is not None:
            mask = mask & (k_mag < float(k_max))
        mask = mask.to(p.device).view(1, 1, *mask.shape)

        # Band energies per (sample, channel): sum over the spatial FFT dims only.
        err_energy = torch.sqrt(((f_pred - f_tgt).abs() ** 2 * mask).sum(dim=(-2, -1)))  # (N, C)
        tgt_energy = torch.sqrt((f_tgt.abs() ** 2 * mask).sum(dim=(-2, -1)) + eps)  # (N, C)

        if relative:
            out[_band_label(k_min, k_max)] = (err_energy / tgt_energy).mean()
        else:
            out[_band_label(k_min, k_max)] = err_energy.mean()
    return out


# ---------------------------------------------------------------------------
# Incremental accumulator (dataloader-friendly)
# ---------------------------------------------------------------------------


class FieldMetricAccumulator:
    """Incrementally accumulates field-level metrics over rollout predictions.

    Designed for an evaluation loop: call :meth:`update` once per batch with the
    per-step prediction/target lists produced by
    ``MultiStepLightningBase._rollout(..., store_fields=True)``, then read the
    pooled result from :meth:`compute`. Only scalar accumulators are kept in
    memory (constant footprint regardless of dataset size).

    All metrics are **exactly pooled** over every (window, step[, channel])
    unit seen so far — not a mean of batch means — except the per-step curve,
    which pools all samples at each AR step index.

    Args:
        spectral_bands: Radial bands for :func:`spectral_band_error`
            (default PDEBench split ``[0,4) / [4,12) / [12,+∞)``).
        eps: Numerical guard for the relative metrics.
        relative_spectral: If True report relative spectral band errors
            (scale-free), otherwise absolute band energies.
    """

    def __init__(
        self,
        spectral_bands: Sequence[Tuple[float, Optional[float]]] = ((0.0, 4.0), (4.0, 12.0), (12.0, None)),
        eps: float = 1e-8,
        relative_spectral: bool = True,
    ):
        self.spectral_bands = tuple((float(a), None if b is None else float(b)) for a, b in spectral_bands)
        self.eps = float(eps)
        self.relative_spectral = bool(relative_spectral)
        self.band_labels = [_band_label(a, b) for a, b in self.spectral_bands]
        self.reset()

    def reset(self):
        """Reset all accumulators to their empty state."""
        self.n_windows = 0
        self.n_elements = 0
        self.sum_err2 = 0.0
        self.sum_abserr = 0.0
        self.max_abserr = 0.0
        self.sum_rel_l2 = 0.0
        self.n_rel_l2 = 0
        self.sum_r2 = 0.0
        self.n_r2 = 0
        self.spectral_sums = {label: 0.0 for label in self.band_labels}
        self.spectral_counts = {label: 0 for label in self.band_labels}
        # Per AR-step index: pooled squared error / element count, and the
        # relative-L2 sum / unit count (per sample-channel ratios).
        self._step_err2_sum: dict = {}
        self._step_elem_count: dict = {}
        self._step_rel_l2_sum: dict = {}
        self._step_rel_l2_count: dict = {}

    def update(self, preds: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]):
        """Accumulate one batch of rollout predictions.

        Args:
            preds: List (length S) of per-step prediction tensors
                ``(B, C, H, W)``, in AR-step order.
            targets: Matching list of ground-truth tensors.

        Raises:
            ValueError: If the two lists have different lengths or a step pair
                has mismatched shapes.
        """
        if len(preds) != len(targets):
            raise ValueError(f"preds/targets length mismatch: {len(preds)} vs {len(targets)}.")

        if preds:
            self.n_windows += preds[0].shape[0]  # one window per batch element

        for s, (p, t) in enumerate(zip(preds, targets)):
            if p.shape != t.shape:
                raise ValueError(f"Step {s}: shape mismatch {tuple(p.shape)} vs {tuple(t.shape)}.")
            err = p - t

            # --- pooled pointwise metrics -----------------------------------
            self.n_elements += err.numel()
            self.sum_err2 += float(err.pow(2).sum())
            self.sum_abserr += float(err.abs().sum())
            self.max_abserr = max(self.max_abserr, float(err.abs().max()))

            # --- relative metrics (per sample / channel units) ---------------
            p4 = _to_samples(p)  # (B, C, H, W)
            t4 = _to_samples(t)
            rel_l2 = _l2_over_spatial(p4 - t4) / (_l2_over_spatial(t4) + self.eps)  # (B, C)
            self.sum_rel_l2 += float(rel_l2.sum())
            self.n_rel_l2 += rel_l2.numel()

            # R² per (sample, channel), centered over the spatial dims only —
            # same unit as rel_l2 above.
            ss_res = ((p4 - t4).reshape(p4.shape[0], p4.shape[1], -1)).pow(2).sum(dim=-1)  # (B, C)
            centered = t4 - t4.mean(dim=(-2, -1), keepdim=True)
            r2 = 1.0 - ss_res / (centered.reshape(p4.shape[0], p4.shape[1], -1).pow(2).sum(dim=-1) + self.eps)
            self.sum_r2 += float(r2.sum())
            self.n_r2 += r2.numel()

            # --- spectral bands ----------------------------------------------
            # band_vals[label] is the mean over all (sample, channel) units of
            # this step; multiply back to keep an exactly pooled sum.
            band_vals = spectral_band_error(p, t, self.spectral_bands, eps=self.eps, relative=self.relative_spectral)
            n_units = p4.shape[0] * p4.shape[1]
            for label in self.band_labels:
                self.spectral_sums[label] += float(band_vals[label]) * n_units
                self.spectral_counts[label] += n_units

            # --- per-step curve ------------------------------------------------
            self._step_err2_sum[s] = self._step_err2_sum.get(s, 0.0) + float(err.pow(2).sum())
            self._step_elem_count[s] = self._step_elem_count.get(s, 0) + err.numel()
            self._step_rel_l2_sum[s] = self._step_rel_l2_sum.get(s, 0.0) + float(rel_l2.sum())
            self._step_rel_l2_count[s] = self._step_rel_l2_count.get(s, 0) + rel_l2.numel()

    def compute(self) -> dict:
        """Return the pooled metrics as a JSON-serializable dict.

        Returns:
            Dict with scalar entries ``mse``, ``rmse``, ``nrmse`` (relative L2),
            ``mae``, ``max_abs_error``, ``r2``, a nested ``spectral_bands`` dict,
            the bookkeeping ``n_windows_evaluated`` / ``n_steps`` and the
            per-AR-step curve ``per_step`` (list of ``{"step", "rmse", "nrmse"}``
            with 1-based step indices). Empty accumulators yield zeros.
        """
        n = self.n_elements
        mse = self.sum_err2 / n if n else 0.0
        result = {
            "mse": mse,
            "rmse": float(torch.sqrt(torch.tensor(mse))),
            "nrmse": self.sum_rel_l2 / self.n_rel_l2 if self.n_rel_l2 else 0.0,
            "mae": self.sum_abserr / n if n else 0.0,
            "max_abs_error": self.max_abserr,
            "r2": self.sum_r2 / self.n_r2 if self.n_r2 else 0.0,
            "spectral_bands": {
                label: (self.spectral_sums[label] / self.spectral_counts[label] if self.spectral_counts[label] else 0.0)
                for label in self.band_labels
            },
            "n_windows_evaluated": int(self.n_windows),
            "n_steps": len(self._step_err2_sum),
            "per_step": [
                {
                    "step": s + 1,
                    "rmse": float(torch.sqrt(torch.tensor(self._step_err2_sum[s] / self._step_elem_count[s]))),
                    "nrmse": self._step_rel_l2_sum[s] / self._step_rel_l2_count[s],
                }
                for s in sorted(self._step_err2_sum)
            ],
        }
        return result


def compute_field_metrics(
    preds: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    spectral_bands: Sequence[Tuple[float, Optional[float]]] = ((0.0, 4.0), (4.0, 12.0), (12.0, None)),
    eps: float = 1e-8,
    relative_spectral: bool = True,
) -> dict:
    """One-shot convenience wrapper around :class:`FieldMetricAccumulator`.

    Args:
        preds: List of per-step prediction tensors ``(B, C, H, W)``.
        targets: Matching list of ground-truth tensors.
        spectral_bands: Radial bands (see :func:`spectral_band_error`).
        eps: Numerical guard for the relative metrics.
        relative_spectral: Relative vs absolute spectral band errors.

    Returns:
        The pooled metrics dict (see :meth:`FieldMetricAccumulator.compute`).
    """
    acc = FieldMetricAccumulator(spectral_bands=spectral_bands, eps=eps, relative_spectral=relative_spectral)
    acc.update(preds, targets)
    return acc.compute()
