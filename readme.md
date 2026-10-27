# PyNOP — Python Neural Operators

A collection of **neural operator** architectures (and classic backbones) for learning
mappings between function spaces, together with a complete **config-driven training /
evaluation pipeline** built on PyTorch Lightning. The reference use case is multi-step
prediction of PDE trajectories (PDEBench protocol), but every model can also be used
standalone as an `nn.Module`.

> Work in progress — models are validated one by one against their original references;
> see `docs/session-notes.md` for the detailed per-model analysis log.

## Repository layout

```
pynop/
├── core/          # shared building blocks (attention, spectral convs, norms, losses, ...)
├── models/        # one file per model family (see below)
├── training/      # Lightning wrappers + CLI drivers (train.py, eval.py, metrics.py)
└── dataloader/    # HDF5 datasets: PDEBenchDataSet, NSH5Dataset, RDDataSet, TheWellDataSet, ...
configs/           # ready-to-run JSON configs (one per model + one shared training config)
tests/             # pytest suite
```

## Implemented models

| Model | File / class | Description |
|---|---|---|
| **DeepONet** | `models/deeponet.py` — `DeepONet` | Branch/trunk operator: a trunk network encodes the input function, a branch network processes the query points; output is their inner product. Requires the dedicated `DeepONetModel` Lightning wrapper (branch/trunk paradigm). |
| **FNO** | `models/FNO.py` — `FNO` | Fourier Neural Operator (Li et al. 2021): lifting → stack of spectral blocks `[norm] → spectral_conv → [activation] → pointwise(1×1) → + shortcut` → projection. Three interchangeable spectral convolutions (`standard`, `separable`, **Tucker** factorization), optional fixed/trainable positional encoding, and an optional local high-pass branch in parallel with the shortcut. Resolution-invariant (effective modes = min(configured, available)). |
| **SwinNO / MultiscaleSWINNO** | `models/SWIN.py` — `SwinNO`, `MultiscaleSWINNO` | Windowed-attention operators. `SwinNO`: full-resolution stack of adaptive Swin blocks (the number of windows is fixed, the window size is derived from the grid → resolution invariance). `MultiscaleSWINNO`: encoder/decoder with a cross-attention pooling bottleneck (`WindowCrossAttention` + `BlockCrossAttentionUpsampler`) — it keeps its bottleneck even at a single scale. |
| **UNO** | `models/UNO.py` — `UNO` | U-Net Neural Operator: symmetric Swin-style encoder/decoder with first-class skip connections (`"add"` or `"concat"` fusion) and a global transformer bottleneck; resolution-invariant. |
| **ITLNO** | `models/LNO.py` — `ITLNO` | Latent operator in the learned-integral-transform domain: forward transform with learned bases, self-attention on the transformed coefficients (real bases), inverse transform. Optional local branch (`n_local_conv`). |
| **LatentGalerkin** | `models/LNO.py` — `LatentGalerkin` | Projects spatial features onto a small set of latent tokens via Galerkin cross-attention, refines them with transformer blocks (adaptive Nyquist PE), and decodes back to the grid. Optional local branch. |
| **LatentTransolver** | `models/LNO.py` — `LatentTransolver` | Projects to a reduced latent space once via Transolver-3 slicing, applies self-attention blocks in the latent space, then deslices (the transform is not repeated at every layer). Optional local branch. |
| **LNO / TLNO** | `models/LNO.py` — `LatentNO` | Transformer-based Latent Neural Operator (reference: `L-I-M-I-T/LatentNeuralOperator`). |
| **Transolver** | `models/transolver.py` — `Transolver` | Transolver family on a 2-D grid, three selectable block modes: `"linear"` (`LinearNOBlock`, arXiv:2511.06294), `"t3"` (`TransolverBlockv3`, faithful port of Transolver-3, arXiv:2602.04940 — per-head slicing weights, clamped learnable temperature, head concatenation), `"physics"` (Gumbel-Softmax `TransolverBlock`, arXiv:2402.02366). |
| **UNet** | `models/unet.py` — `UNet` | Classic 2-D U-Net (not a neural operator): encoder/decoder with skips, optional patch embedding and an optional transformer bottleneck using positional encoding **or** RoPE (`AdaptiveRoPE2D`). |
| **Resnet** | `models/resnet.py` — `Resnet` | Unified ResNet backbone: `block_type="basic"` or `"bottleneck"`, configurable `bottleneck_ratio` and SE attention; covers resnet18/34/50 and resnext-style widths. |
| **GalerkinTransformer** | `models/Galerkin.py` — `GalerkinTransformer` | Galerkin transformer for time-dependent PDEs (Cao et al., *Choose a Transformer: Fourier or Galerkin*, arXiv:2105.14995): the field is projected onto a small set of latent tokens via cross-attention, processed by standard attention blocks with adaptive normalization and time embedding, then reconstructed on the grid. |
| **ConvNeXtV2** | `models/convnext.py` — `ConvNeXtV2` | ConvNeXt V2 backbone. |

## Training

Training is driven by two JSON files through a CLI entry point that reproduces the
PDEBench **two-phase protocol** of the original notebooks:

```bash
# Phase 1 — pretraining (small ensemble, restricted time range, AR curriculum)
python -m pynop.training.train configs/model_FNOvanilla.json configs/training_PDEBench.json --pretrain

# Phase 2 — full training (larger ensemble, full time range, fixed max AR steps)
python -m pynop.training.train configs/model_FNOvanilla.json configs/training_PDEBench.json
```

* **Model config** (`configs/model_<name>.json`): `{"name", "model": "<class name>", "params": {...}}`.
  String activation names (e.g. `"GELU"`) are resolved automatically; tuples may be written as lists.
  One ready-made file exists per model of the notebooks (FNOvanilla, tuckerFNO, ITLNO, LNO,
  LatentGalerkin, SwinNO, MSSwinNO_1stage, MSSwinNO, Transolver_linear, Transolver3).
* **Training config** (`configs/training_PDEBench.json`): dataset (PDEBench HDF5 path, batch size,
  normalization `shift`/`scale`, seeded train/val split), shared multi-step parameters
  (`timestep`, `predict_derivative`, derivative regularizer weight, temporal weighting, BPTT
  truncation, input noise), ordered loss list with weights (e.g. DWMSE + spectral + Sobolev),
  optimizer/scheduler (OneCycleLR with `$total_steps` / `$steps_per_epoch` tokens), trainer
  overrides, and the two phase sections (`pretraining`, `fulltraining`) with their own
  learning rate, epochs, time range, sample count and AR curriculum bounds.

The Lightning wrappers live in `pynop/training/lightning_model.py`:

* **`MultiStepNOModel`** — trains any multi-step operator by autoregressive rollout over a
  window of `input_steps + max_AR_steps + 1` slices; supports derivative prediction
  (`u_{t+dt} = u_t + dt·out`) or absolute output, configurable weighted losses and an optional
  derivative regularizer. Prints a per-epoch average-loss summary for train and validation.
* **`DeepONetModel`** — dedicated wrapper for the branch/trunk paradigm (refuses other models).

Checkpoints: `last.ckpt` is saved every epoch; best-val checkpoints are kept only once the
rollout reaches the maximum number of AR steps (metrics are not comparable during the
curriculum ramp). `--resume` and `--init-from` allow continuing or warm-starting a run.

## Evaluation

```bash
python -m pynop.training.eval configs/model_FNOvanilla.json configs/training_PDEBench.json \
    --checkpoint /path/to/run/last.ckpt [--split val|test] [--pretrain] [--no-plots]
```

The evaluation protocol is the **full autoregressive rollout of every trajectory** in the
evaluation set (same as the notebooks): the first `input_steps` slices are fed to the model,
then all remaining snapshots are predicted one by one — the first prediction is
teacher-forced on the ground-truth window, the rest consume the previous prediction.

* **Evaluation sets**: `--split val` (default) uses the validation split of the training
  protocol (IDs read from `run_info.json` next to the checkpoint when available);
  `--split test` draws simulations that were never used in any training phase, falling back
  to val with a warning if none remain. Overlap with training raises an error.
* **Metrics** (`pynop/training/metrics.py`): field-level MSE / RMSE / MAE / max error,
  **nRMSE = relative L2** (scale-invariant PDEBench "normalized RMSE") and R² (invariant to
  any affine rescaling; `R² = 1 − nRMSE²` per sample on zero-mean data), plus PDEBench-style
  radial spectral error bands in Fourier space. All metrics are computed in model space
  (after the dataset normalization).
* **Outputs**: a formatted table on stdout, a self-contained `eval_results_<split>.json`
  (all scalars + full per-step error curves with timestamps), and two English-labelled PNGs —
  error evolution vs time step (`RMSE / nRMSE / MAE / R²`) and spectral-band evolution.
