"""Unified time-windowed dataset loaders for PyNOP.

This module provides a single base class (:class:`TemporalWindowDataset`) that
factorizes everything the per-format loaders share — window generation,
normalization, RAM/disk storage modes and the batch contract — plus one
subclass per on-disk format:

=================  ==========================================================
``type`` key       Class / layout
=================  ==========================================================
``"pdebench"``     :class:`PDEBenchDataSet` — one top-level HDF5 group per
                   simulation ID, each holding ``data`` ``(T, H, W, C)``
                   (channels-last). No native splits: the train/val split is
                   derived by the seeded shuffle + ratio protocol.
``"custom"``       :class:`CustomDataset` — same layout as PDEBench plus a
                   time-varying ``porosity`` field concatenated on the channel
                   dimension. Samples may carry a native per-sample ``split``
                   HDF5 attribute (e.g. 400 ``train`` / 50 ``val``).
``"ayz2"``         :class:`Ayz2Dataset` — ayz2/temporal_pdes format
                   (https://huggingface.co/datasets/ayz2/temporal_pdes): one
                   HDF5 group per split (``train``/``valid``), each holding
                   ``u`` ``(N, T, H, W)`` or ``(N, T, X)``, grid datasets
                   ``x``/``t`` and per-sample scalar coefficient datasets of
                   shape ``(N,)``. Splits are usually **separate files**
                   (e.g. ``NS_1024.h5`` = train, ``NS_256.h5`` = valid).
``"thewell"``      :class:`TheWellDataSet` — TheWell layout: file-level
                   ``n_trajectories`` attribute and fields grouped by tensor
                   rank (``t0_fields`` scalars, ``t1_fields`` vectors,
                   ``t2_fields`` tensors), each dataset ``(B, T, ...)``.
                   Splits live in separate files/directories
                   (``data/train``, ``data/valid``, ``data/test``).
``"npy"``          :class:`npyPDEDataset` — single ``.npy`` file
                   ``(Nsample, Nt, Nx, Ny[, C])`` loaded fully in RAM with
                   optional time binning (legacy helper).
=================  ==========================================================

Batch contract
--------------
Every dataset returns either::

    (fields, t0)              # fields: (T_unroll, C, H, W), t0: tensor([t0])

or, when the format provides a static conditioning output (ayz2 coefficients)::

    (fields, t0, cond)        # cond: (D, H, W) — one constant spatial field
                              # per coefficient

which is exactly what :class:`pynop.training.lightning_model.MultiStepNOModel`
accepts (``cond`` is forwarded to models whose ``forward`` takes a ``cond=``
keyword). All values are already normalized in model space:
``(x + shift) * scale``.

Train/val split protocol (:func:`resolve_split_ids`)
----------------------------------------------------
Given the ``"dataset"`` section of a training config (and optionally the phase
section), sample IDs for both splits are resolved with this priority:

1. **Explicit** — ``train_ids`` / ``val_ids`` lists in the dataset config are
   used verbatim.
2. **Native splits** — when the format provides them: separate files
   (``path`` = train file, optional ``val_path``), HDF5 split groups within a
   single file, or per-sample ``split`` attributes (custom). Pools may be
   capped with ``n_train_samples`` / ``n_val_samples``, and the phase-level
   ``n_samples`` further truncates the *train* pool — so two-phase protocols
   keep their nesting property (phase-1 IDs ⊂ phase-2 IDs) because native
   pools are consumed in stable file order.
3. **Single pool** — otherwise the original PDEBench protocol applies,
   bit-for-bit: all IDs shuffled with ``random.Random(seed)``, truncated to
   ``n_samples``, then split by ``split_ratio`` on the simulation IDs.

:func:`build_dataloaders` turns a config section into ready-to-use
:class:`~torch.utils.data.DataLoader` pairs (the same function the training
driver and, for ID selection, the evaluation driver share).
"""

import random
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

__all__ = [
    "TemporalWindowDataset",
    "PDEBenchDataSet",
    "CustomDataset",
    "Ayz2Dataset",
    "TheWellDataSet",
    "npyPDEDataset",
    "DATASET_REGISTRY",
    "get_dataset_class",
    "resolve_split_ids",
    "build_dataloaders",
    "inspect_PDEBenchDataset",
    "inspect_the_well_fields",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TemporalWindowDataset(Dataset):
    """Base class for time-windowed PDE simulation datasets.

    Factorizes the machinery shared by all formats:

    * **window generation** — for each sample ID, windows ``(sid, t0)`` with
      ``t0`` stepping from ``min_id`` to ``max_id - T_unroll`` by ``step``;
      samples shorter than a window are skipped. The loop is deterministic and
      identical across formats (and bit-identical to the original
      :class:`PDEBenchDataSet` implementation, so existing runs stay
      reproducible);
    * **normalization** — every returned tensor is ``(x + shift) * scale`` in
      model space; ``shift``/``scale`` may be scalars or per-channel tensors
      (1-D inputs are reshaped to ``(C, 1, 1, 1)``);
    * **storage modes** — disk (lazy per-worker HDF5 handle) or RAM
      (preloaded at init via :meth:`_load_to_ram`);
    * **batch contract** — ``__getitem__`` returns ``(fields, t0)`` or
      ``(fields, t0, cond)`` when the subclass provides a static conditioning
      output (:meth:`_extra_output`).

    Subclasses implement the format-specific hooks:

    * :meth:`_discover_sample_ids` — all IDs in canonical file order;
    * :meth:`_sample_lengths` — mapping ID -> available number of frames;
    * :meth:`_read_range` — raw (unnormalized) float tensor ``(T, C, H, W)``
      for the absolute frame range ``[t_start, t_end)``;
    * :meth:`_load_to_ram` — optional RAM preloading (default no-op);
    * :meth:`_extra_output` — optional static conditioning tensor per sample.

    The classmethod :meth:`list_splits` exposes the format's native split
    structure to :func:`resolve_split_ids`.
    """

    #: value of the ``"dataset"."type"`` config key (subclasses override).
    dataset_type: str = ""

    def __init__(
        self,
        path: Optional[PathLike] = None,
        T_unroll: int = 10,
        step: Optional[int] = None,
        load_in_ram: bool = False,
        indexes: Optional[Sequence[Any]] = None,
        min_id: int = 0,
        max_id: Optional[int] = None,
        shift: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 0.0,
        scale: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 1.0,
    ) -> None:
        """Initialize the dataset and build its window list.

        Args:
            path: Path to the data file (``None`` for RAM-only subclasses that
                load their source before calling ``super().__init__``).
            T_unroll: Length of the time window returned to the model
                (clamped to >= 1 with a warning, as in the original loaders).
            step: Stride between window starts; defaults to ``T_unroll``
                (non-overlapping windows). Must be >= 1.
            load_in_ram: If True call :meth:`_load_to_ram` after building the
                windows (faster access, higher memory use).
            indexes: Explicit list of sample IDs for this split; if None all
                discovered IDs are used. IDs missing from the file are kept in
                ``sample_ids`` but produce no windows (a warning is printed by
                subclasses that can detect them).
            min_id: First frame index allowed for window starts (>= 0).
            max_id: Last frame index allowed (exclusive); None = end of each
                trajectory.
            shift: Value added to the data before scaling (scalar or
                per-channel array/tensor).
            scale: Value multiplied after the shift (scalar or per-channel
                array/tensor).

        Raises:
            ValueError: If ``step`` < 1 or ``min_id`` < 0.
        """
        if T_unroll < 1:
            print("Warning: T_unroll must be at least 1. Setting to 1.")
            T_unroll = 1
        self.T_unroll = int(T_unroll)
        self.step = int(step) if step is not None else self.T_unroll
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}.")
        self.min_id = int(min_id)
        if self.min_id < 0:
            raise ValueError(f"min_id must be >= 0, got {self.min_id}.")
        self.max_id = max_id

        self.path = str(path) if path is not None else None
        self.load_in_ram = bool(load_in_ram)
        self.h5file: Optional[h5py.File] = None

        self.shift = self._prepare_transform(shift)
        self.scale = self._prepare_transform(scale)

        self.sample_ids: list = list(indexes) if indexes is not None else self._discover_sample_ids()
        self.sample_lengths: dict = self._sample_lengths(self.sample_ids)
        self.windows: list = self._build_windows(self.sample_ids, self.sample_lengths)

        if self.load_in_ram:
            self._load_to_ram()

        print(f"{type(self).__name__} initialized with {len(self.sample_ids)} simulations.")
        print(f"Total unrolled windows: {len(self.windows)}")
        print(f"Status: {'RAM' if self.load_in_ram else 'Disk'} mode.")

    # ------------------------------------------------------------------ hooks

    def _discover_sample_ids(self) -> list:
        """Return all sample IDs of the file in canonical order.

        Returns:
            List of IDs (native type per format: HDF5 key strings or integer
            indices).

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """Return the number of available frames for each sample ID.

        Args:
            ids: Sample IDs to measure (usually ``self.sample_ids``).

        Returns:
            Mapping ID -> int frame count. Missing IDs map to 0 (they then
            produce no windows) instead of raising, so explicit ``indexes``
            with typos degrade gracefully like the original loaders.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read raw (unnormalized) frames ``[t_start, t_end)`` of one sample.

        Args:
            sid: Sample ID.
            t_start: First frame index (inclusive), absolute on the file axis.
            t_end: Last frame index (exclusive).

        Returns:
            Float32 tensor ``(T, C, H, W)`` with channels first.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError

    def _load_to_ram(self) -> None:
        """Preload the selected samples into memory (called when ``load_in_ram``).

        Default no-op: RAM-only datasets load their source before calling
        ``super().__init__``, and disk-based subclasses override this.
        """
        return None

    def _extra_output(self, sid: Any) -> Optional[torch.Tensor]:
        """Return the static conditioning output for a sample, if any.

        Args:
            sid: Sample ID.

        Returns:
            Raw (unnormalized) float tensor — typically ``(D, H, W)`` — or
            ``None`` when the format has no conditioning field. The base
            :meth:`__getitem__` applies the same shift/scale transform and
            appends it as a third batch element.
        """
        return None

    # ------------------------------------------------------------------ shared machinery

    @staticmethod
    def _prepare_transform(val: Any) -> Any:
        """Normalize a shift/scale value to a broadcastable form.

        Args:
            val: Scalar, list, ndarray or tensor. 1-D array-like inputs are
                reshaped to ``(C, 1, 1, 1)`` so they broadcast over
                ``(T, C, H, W)`` windows; scalars and higher-dimensional
                tensors pass through unchanged.

        Returns:
            The value ready for ``tensor + shift`` / ``* scale`` arithmetic.
        """
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            tensor_val = torch.as_tensor(val).float()
            if tensor_val.ndim == 1:
                return tensor_val.view(1, -1, 1, 1)
            return tensor_val
        return val

    def _build_windows(self, sample_ids: Sequence[Any], lengths: dict) -> list:
        """Build the ``(sid, t0)`` window list on ``[min_id, max_id]``.

        This loop is intentionally bit-identical to the original
        :class:`PDEBenchDataSet` implementation so that existing runs keep
        producing exactly the same windows (and therefore the same training
        dynamics).

        Args:
            sample_ids: Ordered sample IDs.
            lengths: Mapping ID -> available frame count.

        Returns:
            List of ``(sid, t0)`` tuples; samples with fewer than ``T_unroll``
            frames in range are skipped silently (as before).
        """
        windows = []
        for sid in sample_ids:
            T_total = int(lengths.get(sid, 0))
            current_max_id = min(self.max_id if self.max_id is not None else T_total, T_total)

            if (current_max_id - self.min_id) < self.T_unroll:
                continue

            limit_t0 = current_max_id - self.T_unroll
            for t0 in range(self.min_id, limit_t0 + 1, self.step):
                windows.append((sid, t0))
        return windows

    def _init_h5(self) -> None:
        """Open the (shared-mode) HDF5 handle lazily — once per worker process."""
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.path, "r", swmr=True)

    # ------------------------------------------------------------------ Dataset API

    def __len__(self) -> int:
        """Number of windows (not samples)."""
        return len(self.windows)

    def __getitem__(self, idx: int):
        """Return one normalized window.

        Args:
            idx: Window index into :attr:`windows`.

        Returns:
            ``(fields, t0)`` with ``fields`` of shape ``(T_unroll, C, H, W)``,
            or ``(fields, t0, cond)`` when the subclass provides a static
            conditioning output. All tensors are float32 in model space.
        """
        sid, t0 = self.windows[idx]
        fields = self._read_range(sid, t0, t0 + self.T_unroll)  # (T_unroll, C, H, W), raw
        time_idx = torch.tensor([t0], dtype=torch.float32)

        extra = self._extra_output(sid)
        if extra is None:
            return (fields + self.shift) * self.scale, time_idx
        return (fields + self.shift) * self.scale, time_idx, (extra + self.shift) * self.scale

    def load_trajectory(self, sid: Any, min_id: int = 0, max_id: Optional[int] = None) -> torch.Tensor:
        """Load one full normalized trajectory on ``[min_id, max_id)``.

        Used by the evaluation driver (full autoregressive rollouts) so that
        every format is read through a single code path.

        Args:
            sid: Sample ID.
            min_id: First frame index to load (inclusive).
            max_id: Last frame index to load (exclusive); None = end of the
                trajectory.

        Returns:
            Float32 tensor ``(T, C, H, W)`` in model space.
        """
        T_total = int(self.sample_lengths.get(sid, 0))
        end = min(int(max_id), T_total) if max_id is not None else T_total
        raw = self._read_range(sid, int(min_id), end)
        return (raw + self.shift) * self.scale

    def load_conditioning(self, sid: Any) -> Optional[torch.Tensor]:
        """Load the normalized static conditioning output of a sample.

        Companion to :meth:`load_trajectory` for the evaluation driver: models
        trained with a ``cond=`` input (e.g. on ayz2 coefficients / ``a`` field)
        must receive it at inference time as well, otherwise the rollout is
        silently evaluated without its conditioning.

        Args:
            sid: Sample ID.

        Returns:
            Float32 tensor in model space (typically ``(D, H, W)``), or None
            when the format/sample carries no conditioning output.
        """
        extra = self._extra_output(sid)
        if extra is None:
            return None
        return (extra + self.shift) * self.scale

    # ------------------------------------------------------------------ split discovery

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """Return the native split structure of a file.

        Args:
            path: Path to the data file.

        Returns:
            Ordered mapping ``split_name -> [sample IDs]``. Formats without
            native splits return a single ``{"default": [...]}`` entry; the
            unified protocol (:func:`resolve_split_ids`) then applies the
            seeded shuffle + ratio split on that pool.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PDEBench format
# ---------------------------------------------------------------------------


class PDEBenchDataSet(TemporalWindowDataset):
    """PDEBench HDF5 layout: one top-level group per simulation ID.

    Each group holds a ``data`` dataset of shape ``(T, H, W, C)`` (channels
    last; moved to channels-first on read). There is no native train/val
    structure — the split comes from :func:`resolve_split_ids` (seeded shuffle
    + ratio protocol), exactly as in the original driver.

    Batch: ``(fields [T_unroll, C, H, W], t0)``.
    """

    dataset_type = "pdebench"

    def _discover_sample_ids(self) -> list:
        """All simulation IDs (top-level HDF5 keys) in file order."""
        with h5py.File(self.path, "r") as f:
            return list(f.keys())

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """Frame count per sample; missing IDs warn and map to 0 (skipped)."""
        lengths = {}
        with h5py.File(self.path, "r") as f:
            for sid in ids:
                if sid not in f:
                    print(f"Warning: {sid} not found in HDF5 file. Skipping.")
                    continue
                lengths[sid] = int(f[sid]["data"].shape[0])
        return lengths

    def _load_to_ram(self) -> None:
        """Load every requested sample fully into ``self.data_dict``."""
        self.data_dict = {}
        with h5py.File(self.path, "r") as f:
            for sid in tqdm(self.sample_ids, desc="Loading to RAM"):
                if sid not in f:
                    continue
                raw_data = f[sid]["data"][:]  # (T, H, W, C)
                self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read frames ``[t_start, t_end)`` as ``(T, C, H, W)``."""
        if self.load_in_ram:
            return self.data_dict[sid][t_start:t_end]
        self._init_h5()
        fields = self.h5file[sid]["data"][t_start:t_end]  # (T, H, W, C)
        return torch.from_numpy(np.moveaxis(fields, -1, 1)).float()

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """Single pool of all simulation IDs (no native splits)."""
        with h5py.File(path, "r") as f:
            return {"default": list(f.keys())}


# ---------------------------------------------------------------------------
# Custom format (data + porosity, optional per-sample 'split' attribute)
# ---------------------------------------------------------------------------


class CustomDataset(TemporalWindowDataset):
    """Custom HDF5 layout: PDEBench-like groups plus a time-varying ``porosity``.

    Each top-level group holds ``data`` ``(T, H, W, C)`` and ``porosity``
    ``(T, H, W[, C])``; both must share the same temporal length (asserted at
    scan time). The two fields are concatenated on the channel dimension, so a
    sample with 2 concentration channels + 1 porosity channel yields
    ``C = 3``.

    Samples may carry a native per-sample ``split`` HDF5 attribute (e.g.
    ``400 'train' / 50 'val'``); :meth:`list_splits` then groups the IDs by
    that attribute so :func:`resolve_split_ids` can use them directly.

    Batch: ``(fields [T_unroll, C_data + C_porosity, H, W], t0)``.
    """

    dataset_type = "custom"

    def _discover_sample_ids(self) -> list:
        """All simulation IDs (top-level HDF5 keys) in file order."""
        with h5py.File(self.path, "r") as f:
            return list(f.keys())

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """Frame count per sample, asserting data/porosity temporal consistency."""
        lengths = {}
        with h5py.File(self.path, "r") as f:
            for sid in ids:
                if sid not in f:
                    print(f"Warning: {sid} not found in HDF5 file. Skipping.")
                    continue
                T_total = int(f[sid]["data"].shape[0])
                assert f[sid]["porosity"].shape[0] == T_total, f"Time mismatch in {sid}"
                lengths[sid] = T_total
        return lengths

    def _load_to_ram(self) -> None:
        """Load ``data`` and ``porosity`` of every requested sample into RAM."""
        self.data_dict = {}
        self.porosity_dict = {}
        with h5py.File(self.path, "r") as f:
            for sid in tqdm(self.sample_ids, desc="Loading to RAM"):
                if sid not in f:
                    continue

                raw_data = f[sid]["data"][:]  # (T, H, W, C)
                self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

                raw_p = f[sid]["porosity"][:]
                p_tensor = torch.from_numpy(raw_p).float()
                if p_tensor.ndim == 3:  # (T, H, W) -> (T, 1, H, W)
                    p_tensor = p_tensor.unsqueeze(1)
                elif p_tensor.ndim == 4:  # (T, H, W, C) -> (T, C, H, W)
                    p_tensor = p_tensor.permute(0, 3, 1, 2)
                self.porosity_dict[sid] = p_tensor

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read data+porosity frames ``[t_start, t_end)``, concatenated on C."""
        if self.load_in_ram:
            fields = self.data_dict[sid][t_start:t_end]
            p = self.porosity_dict[sid][t_start:t_end]
            return torch.cat([fields, p], dim=1)

        self._init_h5()
        raw_f = self.h5file[sid]["data"][t_start:t_end]  # (T, H, W, C)
        fields = torch.from_numpy(np.moveaxis(raw_f, -1, 1)).float()

        raw_p = self.h5file[sid]["porosity"][t_start:t_end]
        p = torch.from_numpy(raw_p).float()
        if p.ndim == 3:  # (T, H, W) -> (T, 1, H, W)
            p = p.unsqueeze(1)
        else:  # (T, H, W, C) -> (T, C, H, W)
            p = p.permute(0, 3, 1, 2)
        return torch.cat([fields, p], dim=1)

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """Group IDs by their native per-sample ``split`` attribute.

        Returns:
            Mapping of attribute value -> IDs in file order (e.g.
            ``{"train": [...], "val": [...]}``). Samples without the attribute
            land in a ``"default"`` pool; when no sample carries it the result
            is the single-pool form ``{"default": all_ids}``.
        """
        splits: dict = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                name = str(f[key].attrs.get("split", "default"))
                splits.setdefault(name, []).append(key)
        return splits


# ---------------------------------------------------------------------------
# ayz2 format (temporal_pdes)
# ---------------------------------------------------------------------------


class Ayz2Dataset(TemporalWindowDataset):
    """Loader for the ayz2/temporal_pdes format.

    Reference: https://huggingface.co/datasets/ayz2/temporal_pdes — each file
    holds one group per split (``train``/``valid``), typically **one file per
    split** (e.g. ``NS_1024.h5`` = train, ``NS_256.h5`` = valid). Inside a
    split group:

    * ``u`` — solution field, ``(N, T, H, W)`` for 2D PDEs or ``(N, T, X)``
      for 1D PDEs (embedded as ``(T, C=1, H=1, W=X)``);
    * ``x``, ``t`` — spatial coordinates / time steps (not returned);
    * coefficient datasets (``cx``, ``cy``, ``nu``, ...) — one per PDE
      parameter, each of shape ``(N,)``.

    Conditioning is **always** returned when present (the batch is the 3-tuple
    ``(fields [T_unroll, 1, H, W], t0, cond [D, H, W])``, compatible with models
    whose ``forward`` accepts ``cond=``), from two possible sources:

    * an optional per-sample spatial field ``a`` of shape ``(N, H, W)`` or
      ``(N, C, H, W)`` (present in the Navier-Stokes files);
    * per-sample scalar coefficient datasets of shape ``(N,)`` — one dataset
      per PDE parameter (e.g. ``cx``, ``cy``, ``nu`` in the Burgers files),
      each broadcast to a constant spatial field.

    Scalar coefficients are auto-detected; pass an explicit ``coefficients``
    list to restrict the set. If a file has neither source, the batch degrades
    to the 2-tuple ``(fields, t0)``.

    Sample IDs are integer indices within the split group.

    Batch: ``(fields, t0)`` or ``(fields, t0, cond)`` — see class docstring.
    """

    dataset_type = "ayz2"

    #: datasets inside a split group that are grid metadata, not coefficients.
    _RESERVED_NAMES = ("u", "x", "t")

    def __init__(
        self,
        h5_path: PathLike,
        split: Optional[str] = None,
        T_unroll: int = 10,
        step: Optional[int] = None,
        load_in_ram: bool = False,
        indexes: Optional[Sequence[int]] = None,
        min_id: int = 0,
        max_id: Optional[int] = None,
        shift: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 0.0,
        scale: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 1.0,
        coefficients: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the ayz2 dataset (see :class:`TemporalWindowDataset` for
        the shared parameters).

        Args:
            h5_path: Path to the HDF5 file.
            split: Name of the split group to read. If None it is auto-resolved
                when the file contains exactly one split group; with several
                groups an explicit name is required (a ``KeyError`` lists the
                available ones).
            coefficients: Optional list of coefficient dataset names to return
                as the conditioning field. If None, every per-sample scalar
                dataset found in the split group (shape ``(N,)``, excluding
                ``u``/``x``/``t``) is used.

        Raises:
            ValueError: If no split group with a ``u`` dataset exists, or if
                several exist and ``split`` is not given.
            KeyError: If the requested ``split`` (or a requested coefficient)
                is not present in the file.
        """
        with h5py.File(h5_path, "r") as f:
            groups = [g for g in f.keys() if isinstance(f[g], h5py.Group) and "u" in f[g]]
            if not groups:
                raise ValueError(f"Ayz2Dataset: no split group with a 'u' dataset found in {h5_path}.")
            if split is None:
                if len(groups) != 1:
                    raise ValueError(
                        f"Ayz2Dataset: file has multiple splits {groups}; pass split= explicitly."
                    )
                self.split = groups[0]
            elif split in groups:
                self.split = split
            else:
                raise KeyError(f"Ayz2Dataset: split {split!r} not found. Available: {groups}")

            u_shape = f[self.split]["u"].shape  # tuple: (N, T, X) or (N, T, H, W)
            if len(u_shape) not in (3, 4):
                raise ValueError(
                    f"Ayz2Dataset: unsupported 'u' shape {u_shape} in group '{self.split}' "
                    "(expected [N, T, X] or [N, T, H, W])."
                )
            self.u_N = int(u_shape[0])
            self.u_T = int(u_shape[1])
            # Spatial extent used to broadcast scalar coefficients into fields.
            if len(u_shape) == 4:
                self._cond_spatial = (int(u_shape[2]), int(u_shape[3]))
            else:
                self._cond_spatial = (1, int(u_shape[2]))

            n_samples = self.u_N
            available_coeffs = [
                k
                for k in f[self.split].keys()
                if k not in self._RESERVED_NAMES
                and isinstance(f[self.split][k], h5py.Dataset)
                and f[self.split][k].ndim == 1
                and f[self.split][k].shape[0] == n_samples
            ]

            # Optional per-sample spatial conditioning field 'a' (N, H, W) or (N, C, H, W).
            a_ds = f[self.split].get("a")
            self.has_a_field = isinstance(a_ds, h5py.Dataset) and a_ds.ndim in (3, 4) and a_ds.shape[0] == n_samples
            if self.has_a_field:
                spatial = tuple(int(s) for s in a_ds.shape[-2:])
                if spatial != self._cond_spatial:
                    raise ValueError(
                        f"Ayz2Dataset: 'a' spatial shape {spatial} does not match 'u' "
                        f"{self._cond_spatial} in group '{self.split}'."
                    )

        if coefficients is None:
            self.coefficients = list(available_coeffs)
        else:
            missing = [c for c in coefficients if c not in available_coeffs]
            if missing:
                raise KeyError(f"Ayz2Dataset: coefficient datasets {missing} not found. Available: {available_coeffs}")
            self.coefficients = list(coefficients)

        super().__init__(
            h5_path,
            T_unroll=T_unroll,
            step=step,
            load_in_ram=load_in_ram,
            indexes=indexes,
            min_id=min_id,
            max_id=max_id,
            shift=shift,
            scale=scale,
        )

    def _discover_sample_ids(self) -> list:
        """Integer indices ``0..N-1`` of the selected split group."""
        return list(range(self.u_N))

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """All samples of a split share the temporal length ``u.shape[1]``.

        Out-of-range indices warn and map to 0 (no windows) instead of failing
        later with an opaque HDF5 error at read time.
        """
        lengths = {}
        for sid in ids:
            if not isinstance(sid, (int, np.integer)) or not 0 <= int(sid) < self.u_N:
                print(f"Warning: sample index {sid!r} out of range [0, {self.u_N}) in split '{self.split}'. Skipping.")
                continue
            lengths[sid] = self.u_T
        return lengths

    def _load_to_ram(self) -> None:
        """Preload the selected ``u`` rows, ``a`` fields and coefficient vectors into RAM."""
        # Only in-range indices can be read (out-of-range ones were warned about at scan time).
        valid_ids = [sid for sid in self.sample_ids if isinstance(sid, (int, np.integer)) and 0 <= int(sid) < self.u_N]
        with h5py.File(self.path, "r") as f:
            g = f[self.split]
            u_sel = torch.from_numpy(g["u"][valid_ids]).float()  # (n, T, ...) copy
            self.u_cache = {sid: row for sid, row in zip(valid_ids, u_sel)}

            if self.has_a_field:
                a_sel = torch.from_numpy(g["a"][valid_ids]).float()  # (n, H, W) or (n, C, H, W)
                if a_sel.ndim == 3:
                    a_sel = a_sel.unsqueeze(1)  # (n, 1, H, W)
                self.a_cache = {sid: row for sid, row in zip(valid_ids, a_sel)}

            self.coeff_cache = {}
            for name in self.coefficients:
                c_sel = torch.from_numpy(g[name][valid_ids]).float()  # (n,)
                self.coeff_cache[name] = {sid: v for sid, v in zip(valid_ids, c_sel)}

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read ``u`` frames ``[t_start, t_end)`` as ``(T, 1, H, W)``.

        1D PDEs (``u`` of shape ``(N, T, X)``) are embedded with a singleton
        height: the line becomes row 0 of an ``(H=1, W=X)`` grid.
        """
        if self.load_in_ram:
            raw = self.u_cache[sid][t_start:t_end]
        else:
            self._init_h5()
            raw = torch.from_numpy(self.h5file[self.split]["u"][sid, t_start:t_end]).float()

        if raw.ndim == 2:  # (T, X) -> (T, 1, 1, X)
            return raw.unsqueeze(1).unsqueeze(2)
        return raw.unsqueeze(1)  # (T, H, W) -> (T, 1, H, W)

    def _extra_output(self, sid: Any) -> Optional[torch.Tensor]:
        """Build the conditioning field ``(D, H, W)`` for a sample.

        Channels are ordered as: the spatial ``a`` field first (when present),
        then one constant field per scalar coefficient. Returns None when the
        split group carries no conditioning at all.
        """
        if not self.coefficients and not self.has_a_field:
            return None
        h, w = self._cond_spatial

        channels = []
        if self.has_a_field:
            if self.load_in_ram:
                a = self.a_cache[sid]  # (C', H, W)
            else:
                self._init_h5()
                raw_a = torch.from_numpy(np.asarray(self.h5file[self.split]["a"][sid])).float()
                if raw_a.ndim == 2:  # (H, W) -> (1, H, W)
                    raw_a = raw_a.unsqueeze(0)
                a = raw_a
            channels.append(a)

        for name in self.coefficients:
            if self.load_in_ram:
                v = float(self.coeff_cache[name][sid])
            else:
                self._init_h5()
                # h5py returns a numpy scalar (not an ndarray) for single-element reads.
                v = float(np.asarray(self.h5file[self.split][name][sid]))
            channels.append(torch.full((1, h, w), v))

        return torch.cat(channels, dim=0)  # (D, H, W)

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """One entry per split group containing ``u``: name -> integer indices."""
        splits = {}
        with h5py.File(path, "r") as f:
            for g in f.keys():
                if isinstance(f[g], h5py.Group) and "u" in f[g]:
                    splits[g] = list(range(int(f[g]["u"].shape[0])))
        return splits


# ---------------------------------------------------------------------------
# TheWell format
# ---------------------------------------------------------------------------


class TheWellDataSet(TemporalWindowDataset):
    """TheWell HDF5 layout: multi-trajectory files with rank-grouped fields.

    File-level ``n_trajectories`` attribute; physical fields live in the
    groups ``t0_fields`` (scalars), ``t1_fields`` (vectors) and ``t2_fields``
    (tensors), each dataset shaped ``(B, T, ...)`` with a trailing channel
    axis for rank >= 1. The requested fields are concatenated on the channel
    dimension into one window.

    Splits live in **separate files** (e.g. ``data/train/``, ``data/valid/``,
    ``data/test/``) — point :func:`resolve_split_ids` at them with
    ``path`` / ``val_path`` / ``test_path``; the test set is never mixed into
    train or val.

    Batch: ``(fields [T_unroll, C_total, H, W], t0)``.
    """

    dataset_type = "thewell"

    _FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")

    def __init__(
        self,
        h5_path: PathLike,
        fields_list: Optional[Sequence[str]] = None,
        T_unroll: int = 10,
        step: Optional[int] = None,
        load_in_ram: bool = False,
        indexes: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = None,
        min_id: int = 0,
        max_id: Optional[int] = None,
        shift: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 0.0,
        scale: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 1.0,
    ) -> None:
        """Initialize the TheWell dataset (see :class:`TemporalWindowDataset`
        for the shared parameters).

        Args:
            h5_path: Path to the HDF5 file.
            fields_list: Field names to load (e.g. ``["A", "B"]``); if None,
                every field found in ``t0/t1/t2_fields`` is loaded and
                concatenated. Unknown names raise ``KeyError``.
            indexes: Explicit trajectory indices for this split; if None the
                first ``n_samples`` trajectories (or all of them) are used.
            n_samples: Maximum number of trajectories to use when ``indexes``
                is not given (legacy convenience — the unified protocol passes
                explicit ``indexes`` instead).

        Raises:
            KeyError: If a requested field does not exist in the file.
            ValueError: If no fields are available/requested.
        """
        self.n_samples = n_samples

        with h5py.File(h5_path, "r") as f:
            self.n_total = int(f.attrs["n_trajectories"])
            available_fields = {}
            for group in self._FIELD_GROUPS:
                if group in f:
                    for name in f[group].attrs.get("field_names", []):
                        available_fields[name] = group[:2]

            if fields_list is None:
                self.fields_list = list(available_fields.keys())
            else:
                self.fields_list = list(fields_list)

            missing = [n for n in self.fields_list if n not in available_fields]
            if missing:
                raise KeyError(f"TheWellDataSet: fields {missing} not found. Available: {sorted(available_fields)}")
            if not self.fields_list:
                raise ValueError(f"TheWellDataSet: no fields to load in {h5_path}.")

            self.field_types = {n: available_fields[n] for n in self.fields_list}

            # Recap table (kept from the original loader).
            file_name = Path(h5_path).name
            print(f"\n{'=' * 50}")
            print(f" Dataset: {file_name}")
            print(f"{'-' * 50}")
            print(f"{'Champ':<20} | {'Type':<5} | {'Shape (B, T, ...)'}")
            print(f"{'-' * 50}")
            for fid in self.fields_list:
                path = self._find_field_path(f, fid)
                shape = f[path].shape
                print(f"{fid:<20} | {self.field_types[fid]:<5} | {shape}")
            print(f"{'=' * 50}\n")

        super().__init__(
            h5_path,
            T_unroll=T_unroll,
            step=step,
            load_in_ram=load_in_ram,
            indexes=indexes,
            min_id=min_id,
            max_id=max_id,
            shift=shift,
            scale=scale,
        )

    @staticmethod
    def _find_field_path(h5_obj: h5py.Group, field_name: str) -> str:
        """Return the HDF5 path of a field given its name.

        Args:
            h5_obj: Open file/group to search in.
            field_name: Field name (as listed in ``field_names`` attributes).

        Returns:
            Path string like ``"t0_fields/A"``.

        Raises:
            KeyError: If the field is not found in any rank group.
        """
        for group in TheWellDataSet._FIELD_GROUPS:
            if group in h5_obj and field_name in h5_obj[group]:
                return f"{group}/{field_name}"
        raise KeyError(f"Field '{field_name}' not found.")

    def _discover_sample_ids(self) -> list:
        """Trajectory indices ``0..n-1`` (capped by the legacy ``n_samples``)."""
        num_to_load = min(self.n_total, self.n_samples) if self.n_samples else self.n_total
        return list(range(num_to_load))

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """All trajectories share the temporal length of the first field.

        Out-of-range indices warn and map to 0 (no windows) instead of failing
        later with an opaque HDF5 error at read time.
        """
        with h5py.File(self.path, "r") as f:
            T_total = int(f[self._find_field_path(f, self.fields_list[0])].shape[1])

        lengths = {}
        for sid in ids:
            if not isinstance(sid, (int, np.integer)) or not 0 <= int(sid) < self.n_total:
                print(f"Warning: trajectory index {sid!r} out of range [0, {self.n_total}). Skipping.")
                continue
            lengths[sid] = T_total
        return lengths

    def _load_to_ram(self) -> None:
        """Preload the selected trajectories on ``[min_id, max_id]`` into RAM.

        The time crop is applied at load time (as in the original loader);
        :attr:`_ram_t0` records the offset so reads stay absolute-indexed.
        """
        self._ram_t0 = self.min_id
        # Only in-range indices can be read (out-of-range ones were warned about at scan time).
        valid_ids = [sid for sid in self.sample_ids if isinstance(sid, (int, np.integer)) and 0 <= int(sid) < self.n_total]
        with h5py.File(self.path, "r") as f:
            T_total = int(f[self._find_field_path(f, self.fields_list[0])].shape[1])
            end = min(self.max_id if self.max_id is not None else T_total, T_total)

            self.data_cache = {}
            for fid in tqdm(self.fields_list, desc="Loading to RAM"):
                path = self._find_field_path(f, fid)
                sel = f[path][valid_ids, self.min_id : end, ...]  # (n, T', ...) copy
                self.data_cache[fid] = {sid: torch.from_numpy(row).float() for sid, row in zip(valid_ids, sel)}

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read all requested fields on ``[t_start, t_end)``, concatenated on C.

        Each field is normalized to channels-first: ``(T, H, W)`` gains a
        channel axis and ``(T, H, W, C)`` is permuted; other ranks raise an
        explicit error instead of silently corrupting the concatenation.
        """
        if not self.load_in_ram:
            self._init_h5()
        offset = self._ram_t0 if self.load_in_ram else 0

        chunks = []
        for fid in self.fields_list:
            if self.load_in_ram:
                chunk = self.data_cache[fid][sid][t_start - offset : t_end - offset]
            else:
                path = self._find_field_path(self.h5file, fid)
                chunk = torch.from_numpy(self.h5file[path][sid, t_start:t_end]).float()

            if chunk.ndim == 3:  # (T, H, W) -> (T, 1, H, W)
                chunk = chunk.unsqueeze(1)
            elif chunk.ndim == 4:  # (T, H, W, C) -> (T, C, H, W)
                chunk = chunk.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"TheWellDataSet: unsupported field rank {chunk.ndim} for '{fid}' — expected 3D or 4D.")
            chunks.append(chunk)

        return torch.cat(chunks, dim=1)

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """Single pool of trajectory indices (splits are separate files)."""
        with h5py.File(path, "r") as f:
            return {"default": list(range(int(f.attrs["n_trajectories"])))}


# ---------------------------------------------------------------------------
# Legacy .npy format (RAM-only)
# ---------------------------------------------------------------------------


class npyPDEDataset(TemporalWindowDataset):
    """Loads simulation data from a single ``.npy`` file directly in RAM.

    Supports ``(Nsample, Nt, Nx, Ny)`` or ``(Nsample, Nt, Nx, Ny, C)`` arrays;
    optional time binning (take 1 frame every ``time_binning`` frames) is
    applied at load and all windowing happens on the binned axis. Always in
    RAM mode — there is no disk fallback for a single array file.

    Batch: ``(fields [T_unroll, C, H, W], t0)`` with ``t0`` expressed on the
    **binned** time axis (as in the original loader).
    """

    dataset_type = "npy"

    def __init__(
        self,
        npy_path: PathLike,
        T_unroll: int = 10,
        step: Optional[int] = None,
        time_binning: int = 1,
        indexes: Optional[Sequence[int]] = None,
        min_id: int = 0,
        max_id: Optional[int] = None,
        shift: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 0.0,
        scale: Union[float, Sequence[float], np.ndarray, torch.Tensor] = 1.0,
    ) -> None:
        """Initialize the .npy dataset (see :class:`TemporalWindowDataset` for
        the shared parameters).

        Args:
            npy_path: Path to the ``.npy`` file.
            time_binning: Take 1 frame every ``time_binning`` frames, applied
                directly at loading (default 1 = no binning). Must be >= 1.

        Raises:
            ValueError: If the array is not 4D or 5D.
        """
        if time_binning < 1:
            raise ValueError(f"time_binning must be >= 1, got {time_binning}.")
        self.time_binning = int(time_binning)

        raw_data = np.load(npy_path)
        if raw_data.ndim not in (4, 5):
            raise ValueError(f"Unsupported array shape {raw_data.shape}. Expected 4D or 5D array.")
        # Binned view on the time axis (no copy); windowing uses this timeline.
        self.data = raw_data[:, :: self.time_binning]

        super().__init__(
            npy_path,
            T_unroll=T_unroll,
            step=step,
            load_in_ram=True,  # RAM-only format
            indexes=indexes,
            min_id=min_id,
            max_id=max_id,
            shift=shift,
            scale=scale,
        )

    def _discover_sample_ids(self) -> list:
        """Integer sample indices ``0..N-1``."""
        return list(range(self.data.shape[0]))

    def _sample_lengths(self, ids: Sequence[Any]) -> dict:
        """All samples share the binned temporal length."""
        T_binned = int(self.data.shape[1])
        return {sid: T_binned for sid in ids}

    def _read_range(self, sid: Any, t_start: int, t_end: int) -> torch.Tensor:
        """Read binned frames ``[t_start, t_end)`` as ``(T, C, H, W)``."""
        seq = self.data[sid, t_start:t_end]  # (T', X, Y) or (T', X, Y, C)
        fields = torch.from_numpy(seq).float()
        if fields.ndim == 3:  # (T, X, Y) -> (T, 1, X, Y)
            return fields.unsqueeze(1)
        return fields.permute(0, 3, 1, 2)  # (T, X, Y, C) -> (T, C, X, Y)

    @classmethod
    def list_splits(cls, path: PathLike) -> dict:
        """Single pool of sample indices."""
        return {"default": list(range(int(np.load(path, mmap_mode="r").shape[0])))}


# ---------------------------------------------------------------------------
# Registry + unified split protocol
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "pdebench": PDEBenchDataSet,
    "custom": CustomDataset,
    "ayz2": Ayz2Dataset,
    "thewell": TheWellDataSet,
    "npy": npyPDEDataset,
}


def get_dataset_class(ds_type: str) -> type:
    """Resolve a ``"dataset"."type"`` config value to its dataset class.

    Args:
        ds_type: Type key (case-insensitive), e.g. ``"pdebench"``,
            ``"custom"``, ``"ayz2"``, ``"thewell"``, ``"npy"``.

    Returns:
        The corresponding :class:`TemporalWindowDataset` subclass.

    Raises:
        ValueError: If the type is unknown (the error lists the available ones).
    """
    key = str(ds_type).lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset type {ds_type!r}. Available types: {sorted(DATASET_REGISTRY)}.")
    return DATASET_REGISTRY[key]


#: Preferred split names tried in order when auto-selecting a pool.
_TRAIN_POOL_NAMES = ("train", "training")
_VAL_POOL_NAMES = ("valid", "val", "validation")


def _pick_pool(split_map: dict, preferred_names: Sequence[str], label: str) -> tuple:
    """Pick one pool from a ``list_splits`` mapping.

    Args:
        split_map: Ordered mapping split name -> IDs (from :meth:`TemporalWindowDataset.list_splits`).
        preferred_names: Names tried in order (e.g. ``("train", "training")``).
        label: Human-readable role of the pool ("train"/"val") for errors.

    Returns:
        Tuple ``(name, ids)`` of the selected pool. Falls back to the single
        non-empty pool when no preferred name matches; raises otherwise so a
        multi-split file can never be consumed ambiguously.

    Raises:
        ValueError: If several pools exist and none is named by
            ``preferred_names`` (the user must set ``split``/``val_split``).
    """
    for name in preferred_names:
        if split_map.get(name):
            return name, list(split_map[name])

    nonempty = {k: v for k, v in split_map.items() if v}
    if len(nonempty) == 1:
        (name, ids), = nonempty.items()
        return name, list(ids)

    raise ValueError(
        f"Cannot auto-select the {label} split. Available splits: {sorted(split_map)}. "
        f"Set 'split' (train file) and/or 'val_split' in the dataset config."
    )


def resolve_split_ids(ds_cfg: dict, phase_cfg: Optional[dict] = None) -> dict:
    """Resolve train/val sample IDs from a ``"dataset"`` config section.

    Implements the unified protocol documented in the module docstring:
    explicit ID lists first, then native splits (separate files via
    ``path``/``val_path``, HDF5 split groups, or per-sample ``split``
    attributes) with optional ``n_train_samples`` / ``n_val_samples`` caps and
    a phase-level ``n_samples`` truncation of the train pool, and finally the
    original PDEBench protocol (seeded shuffle + ``n_samples`` +
    ``split_ratio``) on a single pool — bit-identical to the legacy driver for
    ``type == "pdebench"``.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config. Requires
            ``type`` and ``path``; optional keys: ``val_path``,
            ``train_ids``/``val_ids``, ``split``/``val_split`` (HDF5 group or
            attribute names), ``n_train_samples``/``n_val_samples``,
            ``split_ratio`` (default 0.9) and ``random_seed`` (default 42).
        phase_cfg: Optional selected phase section; its ``n_samples`` caps the
            train pool and its ``split_ratio`` overrides the dataset one.

    Returns:
        Dict with keys:

        * ``"train"`` / ``"val"`` — each ``{"path": Path, "group": str or None,
          "ids": list}`` (IDs in native type per format);
        * ``"train_pool_order"`` — the full ordered pool the train IDs were
          drawn from, before any truncation. For the protocol case this is the
          **seeded-shuffled** ID list (so downstream consumers reproduce the
          legacy ordering); for native/explicit cases it is stable file order;
        * ``"source"`` — one of ``"explicit"``, ``"native"``, ``"protocol"``.

    Raises:
        KeyError: If ``type`` or ``path`` is missing (or an explicit ID list
            references a split group that does not exist).
        ValueError: On unknown types, empty splits, or ambiguous multi-split
            files without explicit names.
    """
    ds_type = str(ds_cfg.get("type", "")).lower()
    cls = get_dataset_class(ds_type)
    if "path" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "path".')

    train_path = Path(ds_cfg["path"])
    val_path = Path(ds_cfg.get("val_path") or ds_cfg["path"])
    same_file = train_path == val_path

    seed = int(ds_cfg.get("random_seed", 42))
    phase = phase_cfg or {}
    n_samples_phase = phase.get("n_samples")
    split_ratio = float(phase.get("split_ratio", ds_cfg.get("split_ratio", 0.9)))

    # ------------------------------------------------------------------ 1) explicit IDs
    if "train_ids" in ds_cfg and "val_ids" in ds_cfg:
        train_map = cls.list_splits(train_path)
        val_map = cls.list_splits(val_path) if not same_file else train_map

        def _resolve_group(group, split_map, label):
            """Validate an explicit group name or auto-pick one from the map."""
            if group is None:
                preferred = _TRAIN_POOL_NAMES if label == "train" else _VAL_POOL_NAMES
                return _pick_pool(split_map, preferred, label)[0]
            if group not in split_map:
                raise KeyError(f"{label} split group {group!r} not found. Available: {sorted(split_map)}")
            return group

        t_group = _resolve_group(ds_cfg.get("split"), train_map, "train")
        if same_file and len([v for v in train_map.values() if v]) > 1:
            # Several pools in one file: the val IDs must come from a *different* pool.
            others = {k: v for k, v in train_map.items() if k != t_group and v}
            v_group = _resolve_group(ds_cfg.get("val_split"), others, "val")
        else:
            v_group = _resolve_group(ds_cfg.get("val_split"), val_map, "val")

        train_ids = list(ds_cfg["train_ids"])
        val_ids = list(ds_cfg["val_ids"])
        if not train_ids or not val_ids:
            raise ValueError("Explicit 'train_ids'/'val_ids' must both be non-empty.")
        return {
            "train": {"path": train_path, "group": t_group, "ids": train_ids},
            "val": {"path": val_path, "group": v_group, "ids": val_ids},
            "train_pool_order": [i for ids in train_map.values() for i in ids],
            "source": "explicit",
        }

    # ------------------------------------------------------------------ 2) native splits
    train_map = cls.list_splits(train_path)
    val_map = cls.list_splits(val_path) if not same_file else train_map

    multi_train = len([v for v in train_map.values() if v]) > 1
    multi_val = (not same_file) and len([v for v in val_map.values() if v]) > 1
    is_native = (not same_file) or multi_train or multi_val

    if is_native:
        t_name, train_pool = _pick_pool(train_map, _TRAIN_POOL_NAMES, "train")
        if same_file:
            # Single file with several pools: the val pool must be a *different* one.
            v_candidates = {k: v for k, v in train_map.items() if k != t_name and v}
            v_name, val_pool = _pick_pool(v_candidates, _VAL_POOL_NAMES, "val")
        else:
            v_name, val_pool = _pick_pool(val_map, _VAL_POOL_NAMES, "val")

        n_train_cap = ds_cfg.get("n_train_samples")
        n_val_cap = ds_cfg.get("n_val_samples")
        if n_train_cap is not None:
            train_pool = train_pool[: int(n_train_cap)]
        if n_val_cap is not None:
            val_pool = val_pool[: int(n_val_cap)]
        if n_samples_phase is not None:
            train_pool = train_pool[: int(n_samples_phase)]

        if not train_pool or not val_pool:
            raise ValueError(
                f"Empty native split: {len(train_pool)} train / {len(val_pool)} valid samples "
                f"(n_train_samples={n_train_cap}, n_val_samples={n_val_cap}, phase n_samples={n_samples_phase}). "
                f"Check the dataset config."
            )

        return {
            "train": {"path": train_path, "group": t_name, "ids": list(train_pool)},
            "val": {"path": val_path, "group": v_name, "ids": list(val_pool)},
            "train_pool_order": [i for ids in train_map.values() for i in ids],
            "source": "native",
        }

    # ------------------------------------------------------------------ 3) single pool: PDEBench protocol (bit-identical)
    all_ids = [i for ids in train_map.values() for i in ids]
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    if n_samples_phase is not None:
        pool = all_ids[: min(len(all_ids), int(n_samples_phase))]
    else:
        pool = list(all_ids)

    n_train = int(len(pool) * split_ratio)
    train_ids, val_ids = pool[:n_train], pool[n_train:]
    if not train_ids or not val_ids:
        raise ValueError(
            f"Empty split: {len(train_ids)} train / {len(val_ids)} valid simulations "
            f"(n_samples={n_samples_phase}, split_ratio={split_ratio}). Check the dataset config."
        )

    return {
        "train": {"path": train_path, "group": None, "ids": list(train_ids)},
        "val": {"path": val_path, "group": None, "ids": list(val_ids)},
        "train_pool_order": all_ids,  # full seeded-shuffled order (legacy-compatible)
        "source": "protocol",
    }


def _instantiate_dataset(cls: type, ds_type: str, split_info: dict, ds_cfg: dict, **common: Any):
    """Instantiate one dataset for a resolved split.

    Args:
        cls: Dataset class (from :func:`get_dataset_class`).
        ds_type: The normalized ``"type"`` key (drives format-specific kwargs).
        split_info: One of the ``"train"``/``"val"`` entries returned by
            :func:`resolve_split_ids`.
        ds_cfg: Full dataset config section (format extras are read from it).
        **common: Shared constructor kwargs (T_unroll, step, load_in_ram, ...).

    Returns:
        The instantiated dataset restricted to the split's IDs.
    """
    kwargs = dict(indexes=list(split_info["ids"]), **common)
    if ds_type == "ayz2":
        kwargs["split"] = split_info["group"]  # None -> auto (single-group file)
        if ds_cfg.get("coefficients") is not None:
            kwargs["coefficients"] = list(ds_cfg["coefficients"])
    elif ds_type == "thewell":
        if ds_cfg.get("fields_list"):
            kwargs["fields_list"] = list(ds_cfg["fields_list"])
    return cls(split_info["path"], **kwargs)


def build_dataloaders(ds_cfg: dict, phase_cfg: dict, input_steps: int) -> tuple:
    """Build train/val :class:`DataLoader` pairs for any supported dataset type.

    Shared by the training driver (and reused conceptually by evaluation): it
    resolves the split with :func:`resolve_split_ids`, instantiates one
    dataset per split and wraps them in standard PyTorch dataloaders (train
    shuffled, val pinned when a GPU is available). The window length is
    derived as ``max_AR_steps + input_steps + 1``.

    Args:
        ds_cfg: The ``"dataset"`` section of the training config. Requires
            ``type``, ``path`` and ``batch_size``; optional keys are
            ``num_workers``, ``load_in_ram``, ``shift``, ``scale``,
            ``val_path``, ``split_ratio``, ``random_seed`` plus the native-split
            keys documented in :func:`resolve_split_ids` and format extras
            (``fields_list`` for TheWell, ``coefficients`` for ayz2).
        phase_cfg: The selected phase section (provides ``delta_ts``,
            ``min_id``, ``max_id``, ``n_samples``, ``max_AR_steps``; an
            optional per-phase ``split_ratio`` override).
        input_steps: Number of past slices given to the network.

    Returns:
        Tuple ``(train_loader, val_loader, window_size)``.

    Raises:
        KeyError / ValueError: On missing required keys or unknown types (see
            :func:`resolve_split_ids` and :func:`get_dataset_class`).
    """
    ds_type = str(ds_cfg.get("type", "")).lower()
    cls = get_dataset_class(ds_type)
    if "path" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "path".')
    if "batch_size" not in ds_cfg:
        raise KeyError('The "dataset" section requires a "batch_size".')

    window_size = int(phase_cfg["max_AR_steps"]) + input_steps + 1
    delta_ts = phase_cfg.get("delta_ts")
    min_id = int(phase_cfg.get("min_id", 0))
    max_id = phase_cfg.get("max_id")

    info = resolve_split_ids(ds_cfg, phase_cfg)
    train_info, val_info = info["train"], info["val"]

    common = dict(
        T_unroll=window_size,
        step=int(delta_ts) if delta_ts is not None else None,
        load_in_ram=bool(ds_cfg.get("load_in_ram", False)),
        min_id=min_id,
        max_id=max_id,
        shift=float(ds_cfg.get("shift", 0.0)),
        scale=float(ds_cfg.get("scale", 1.0)),
    )
    train_set = _instantiate_dataset(cls, ds_type, train_info, ds_cfg, **common)
    val_set = _instantiate_dataset(cls, ds_type, val_info, ds_cfg, **common)

    batch_size = int(ds_cfg["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 0))
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        val_kwargs["persistent_workers"] = True
    val_loader = DataLoader(val_set, shuffle=False, **val_kwargs)

    print(
        f"Dataset '{train_info['path'].name}' [{ds_type}] (source: {info['source']}): "
        f"{len(train_info['ids'])} train / {len(val_info['ids'])} valid simulations, "
        f"window_size={window_size}, stride={common['step']}, t in [{min_id}, {max_id if max_id is not None else 'end'}]"
    )
    return train_loader, val_loader, window_size


# ---------------------------------------------------------------------------
# Inspection utilities (format-specific, unchanged from the original module)
# ---------------------------------------------------------------------------


def inspect_PDEBenchDataset(h5_path: PathLike, DT: float = 1.0) -> dict:
    """Scan a PDEBench HDF5 file to extract physical, structural, and temporal
    derivative (velocity) statistics.

    Args:
        h5_path: Path to the PDEBench HDF5 file.
        DT: Time step between frames (for the ``du/dt`` statistics).

    Returns:
        Dict with per-channel value/derivative statistics, global min/max and
        structural info; prints a formatted report on stdout. The string
        ``"Error: Empty HDF5 file."`` is returned for empty files.
    """
    stats = {
        "n_simulations": 0,
        "spatial_shape": None,
        "temporal_steps": None,
        "n_channels": None,
        "global_min": None,
        "global_max": None,
        "mean_per_channel": None,
        "std_per_channel": None,
        "mean_deriv": None,
        "std_deriv": None,
    }

    with h5py.File(h5_path, "r") as f:
        sample_ids = list(f.keys())
        stats["n_simulations"] = len(sample_ids)

        if stats["n_simulations"] == 0:
            return "Error: Empty HDF5 file."

        # Get structural info [T, X, Y, (Z), C]
        first_data = f[sample_ids[0]]["data"]
        shape = first_data.shape
        stats["temporal_steps"] = shape[0]
        stats["spatial_shape"] = shape[1:-1]
        stats["n_channels"] = shape[-1]

        n_channels = stats["n_channels"]

        # Accumulators for global stats
        sum_val = np.zeros(n_channels)
        sum_sq = np.zeros(n_channels)
        total_elements = 0

        # Accumulators for derivative stats (v = du/dt)
        sum_v = np.zeros(n_channels)
        sum_v_sq = np.zeros(n_channels)
        total_v_elements = 0

        all_mins = []
        all_maxs = []

        for sid in tqdm(sample_ids, desc="Inspecting dataset & derivatives"):
            data = f[sid]["data"][:]  # Shape: [T, X, Y, C]

            # --- Global Value Stats ---
            reduce_axes = tuple(range(data.ndim - 1))  # Reduce T, X, Y
            all_mins.append(np.min(data, axis=reduce_axes))
            all_maxs.append(np.max(data, axis=reduce_axes))

            sum_val += np.sum(data, axis=reduce_axes)
            sum_sq += np.sum(np.square(data), axis=reduce_axes)
            total_elements += np.prod(data.shape[:-1])

            # --- Derivative Stats (du/dt) ---
            # diff shape: [T-1, X, Y, C]
            diff = (data[1:] - data[:-1]) / DT
            reduce_axes_v = tuple(range(diff.ndim - 1))

            sum_v += np.sum(diff, axis=reduce_axes_v)
            sum_v_sq += np.sum(np.square(diff), axis=reduce_axes_v)
            total_v_elements += np.prod(diff.shape[:-1])

        # Finalize Value Stats
        stats["global_min"] = np.min(all_mins, axis=0)
        stats["global_max"] = np.max(all_maxs, axis=0)
        stats["mean_per_channel"] = sum_val / total_elements
        var = (sum_sq / total_elements) - np.square(stats["mean_per_channel"])
        stats["std_per_channel"] = np.sqrt(np.maximum(var, 0))

        # Finalize Derivative Stats
        stats["mean_deriv"] = sum_v / total_v_elements
        var_v = (sum_v_sq / total_v_elements) - np.square(stats["mean_deriv"])
        stats["std_deriv"] = np.sqrt(np.maximum(var_v, 0))

    # Print report
    print("\n" + "=" * 50)
    print("              PDEBENCH INSPECTION REPORT")
    print("=" * 50)
    print(f"Simulations : {stats['n_simulations']}")
    print(f"Time steps  : {stats['temporal_steps']}")
    print(f"Resolution  : {stats['spatial_shape']}")
    print(f"Channels    : {stats['n_channels']}")
    print("-" * 50)

    for c in range(stats["n_channels"]):
        print(f"CHANNEL {c}:")
        print(f"  [Values] Min: {stats['global_min'][c]:.4f} | Max: {stats['global_max'][c]:.4f}")
        print(f"  [Values] Mean: {stats['mean_per_channel'][c]:.4f} | Std: {stats['std_per_channel'][c]:.4f}")
        print(f"  [Deriv ] Mean: {stats['mean_deriv'][c]:.6f} | Std: {stats['std_deriv'][c]:.6f}")
        print("-" * 50)
    print("=" * 50)

    return stats


def inspect_the_well_fields(h5_path: PathLike) -> None:
    """Print a summary of the fields stored in a TheWell HDF5 file.

    Args:
        h5_path: Path to the TheWell HDF5 file (prints trajectory count and,
            per rank group, each field's shape and dtype).
    """
    with h5py.File(h5_path, "r") as f:
        print(f"--- Fichier : {h5_path} ---")
        print(f"Trajectoires totales (B) : {f.attrs.get('n_trajectories')}")

        for group_name in TheWellDataSet._FIELD_GROUPS:
            if group_name in f:
                group = f[group_name]
                # Récupération de la liste des noms via l'attribut @field_names
                names = group.attrs.get("field_names", [])
                print(f"\nGroup: {group_name} (Réf: {len(names)} champs)")

                for name in names:
                    ds = group[name]
                    print(f"  - {name:<15} | Shape: {ds.shape} | Dtype: {ds.dtype}")
