"""Tests for the unified dataloader module :mod:`pynop.dataloader`.

Covers (all on tiny synthetic files, no real data required):

* the base-class machinery — window generation (bit-identical to the original
  PDEBench loop), normalization (scalar and per-channel shift/scale), RAM vs
  disk equivalence, ``load_trajectory`` cropping;
* each format subclass — :class:`PDEBenchDataSet`, :class:`CustomDataset`
  (data+porosity concat, native per-sample ``split`` attribute),
  :class:`Ayz2Dataset` (separate files / split groups, scalar coefficients and
  spatial ``a`` field as conditioning, out-of-range indices) and
  :class:`TheWellDataSet` (rank-grouped fields, ``fields_list``, trajectory
  bounds);
* the unified split protocol (:func:`resolve_split_ids`) — explicit IDs, native
  splits with ``n_train_samples``/``n_val_samples`` caps and phase
  ``n_samples`` truncation, and the single-pool PDEBench protocol checked
  against a verbatim copy of the legacy algorithm (reproducibility);
* :func:`build_dataloaders` — batch shapes per format (2-tuple vs ayz2
  3-tuple with ``cond``), unknown-type rejection.
"""

import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from pynop.dataloader import (
    Ayz2Dataset,
    CustomDataset,
    PDEBenchDataSet,
    TheWellDataSet,
    build_dataloaders,
    get_dataset_class,
    npyPDEDataset,
    resolve_split_ids,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures (tiny on purpose — the tests are about structure/protocol)
# ---------------------------------------------------------------------------

N_SIMS = 8
T_STEPS = 24
H = W = 8
C = 2


@pytest.fixture(scope="module")
def pdebench_h5(tmp_path_factory) -> Path:
    """PDEBench layout: top-level groups keyed by sim ID, ``data`` [T,H,W,C]."""
    path = tmp_path_factory.mktemp("pdebench") / "tiny.hdf5"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for sid in range(N_SIMS):
            grp = f.create_group(str(sid))
            grp.create_dataset("data", data=rng.standard_normal((T_STEPS, H, W, C)).astype(np.float32))
    return path


@pytest.fixture(scope="module")
def custom_h5(tmp_path_factory) -> Path:
    """Custom layout: ``data`` [T,H,W,C] + time-varying ``porosity`` [T,H,W,1],
    with a native per-sample ``split`` attribute (4 train / 2 val)."""
    path = tmp_path_factory.mktemp("custom") / "tiny.hdf5"
    rng = np.random.default_rng(1)
    n_total = 6
    with h5py.File(path, "w") as f:
        for sid in range(n_total):
            grp = f.create_group(f"{sid:04d}")
            grp.attrs["split"] = "train" if sid < 4 else "val"
            grp.create_dataset("data", data=rng.standard_normal((T_STEPS, H, W, C)).astype(np.float32))
            grp.create_dataset("porosity", data=rng.random((T_STEPS, H, W, 1), dtype=np.float32))
    return path


def _write_ayz2_group(f: h5py.Group, name: str, n: int, T: int, rng, with_coeffs=True, with_a=False):
    """Write one ayz2 split group (u + x/t metadata + optional coeffs / 'a')."""
    g = f.create_group(name)
    g.create_dataset("u", data=rng.standard_normal((n, T, H, W)).astype(np.float32))
    g.create_dataset("x", data=np.stack([np.linspace(0, 1, H), np.linspace(0, 1, W)]).astype(np.float32))
    g.create_dataset("t", data=np.arange(T, dtype=np.float32))
    if with_coeffs:
        for cname in ("cx", "nu"):
            g.create_dataset(cname, data=rng.uniform(0.5, 1.5, size=n).astype(np.float32))
    if with_a:
        g.create_dataset("a", data=rng.random((n, H, W), dtype=np.float32))


@pytest.fixture(scope="module")
def ayz2_train_h5(tmp_path_factory) -> Path:
    """ayz2 train file: single 'train' group with scalar coefficients."""
    path = tmp_path_factory.mktemp("ayz2_train") / "train.hdf5"
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as f:
        _write_ayz2_group(f, "train", 8, 20, rng, with_coeffs=True)
    return path


@pytest.fixture(scope="module")
def ayz2_valid_h5(tmp_path_factory) -> Path:
    """ayz2 valid file: single 'valid' group with a spatial 'a' field (no coeffs)."""
    path = tmp_path_factory.mktemp("ayz2_valid") / "valid.hdf5"
    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as f:
        _write_ayz2_group(f, "valid", 4, 20, rng, with_coeffs=False, with_a=True)
    return path


@pytest.fixture(scope="module")
def ayz2_both_h5(tmp_path_factory) -> Path:
    """ayz2 file holding BOTH split groups (same-file native splits)."""
    path = tmp_path_factory.mktemp("ayz2_both") / "both.hdf5"
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as f:
        _write_ayz2_group(f, "train", 6, 16, rng, with_coeffs=True)
        _write_ayz2_group(f, "valid", 3, 16, rng, with_coeffs=False, with_a=True)
    return path


@pytest.fixture(scope="module")
def thewell_h5(tmp_path_factory) -> Path:
    """TheWell layout: n_trajectories attr + t0_fields A/B [B,T,H,W]."""
    path = tmp_path_factory.mktemp("thewell") / "tiny.hdf5"
    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as f:
        f.attrs["n_trajectories"] = 8
        grp = f.create_group("t0_fields")
        grp.attrs["field_names"] = np.array(["A", "B"], dtype=object)
        for name in ("A", "B"):
            grp.create_dataset(name, data=rng.standard_normal((8, T_STEPS, H, W)).astype(np.float32))
    return path


@pytest.fixture(scope="module")
def npy_file(tmp_path_factory) -> Path:
    """5D .npy array [Nsample, Nt, Nx, Ny, C]."""
    path = tmp_path_factory.mktemp("npy") / "tiny.npy"
    rng = np.random.default_rng(6)
    np.save(path, rng.standard_normal((4, 16, H, W, C)).astype(np.float32))
    return path


# ---------------------------------------------------------------------------
# Base-class machinery (exercised through PDEBenchDataSet)
# ---------------------------------------------------------------------------


def test_window_generation_matches_legacy_loop(pdebench_h5):
    """The shared window loop must be bit-identical to the original PDEBench one."""

    def legacy_windows(ids, T_unroll, step, min_id, max_id):
        windows = []
        for sid in ids:
            current_max_id = min(max_id if max_id is not None else T_STEPS, T_STEPS)
            if (current_max_id - min_id) < T_unroll:
                continue
            limit_t0 = current_max_id - T_unroll
            for t0 in range(min_id, limit_t0 + 1, step):
                windows.append((sid, t0))
        return windows

    ids = [str(i) for i in range(N_SIMS)]
    cases = [
        dict(T_unroll=10, step=None, min_id=0, max_id=None),
        dict(T_unroll=6, step=3, min_id=2, max_id=20),
        dict(T_unroll=4, step=7, min_id=5, max_id=None),
    ]
    for kwargs in cases:
        ds = PDEBenchDataSet(pdebench_h5, indexes=ids, **kwargs)
        expected_step = kwargs["T_unroll"] if kwargs["step"] is None else kwargs["step"]
        assert ds.windows == legacy_windows(ids, kwargs["T_unroll"], expected_step, kwargs["min_id"], kwargs["max_id"])


def test_window_generation_skips_short_samples(pdebench_h5):
    """Samples with fewer than T_unroll frames in range produce no windows."""
    # max_id=10 keeps only 10 frames: a window of 12 cannot fit.
    ds = PDEBenchDataSet(pdebench_h5, T_unroll=12, min_id=0, max_id=10)
    assert len(ds) == 0

    # But the same range fits a window of 8: every sample contributes windows.
    ds_ok = PDEBenchDataSet(pdebench_h5, T_unroll=8, step=4, min_id=0, max_id=10)
    assert len(ds_ok) == N_SIMS * ((10 - 8) // 4 + 1)


def test_missing_index_warns_but_does_not_crash(pdebench_h5):
    ds = PDEBenchDataSet(pdebench_h5, indexes=["0", "999"], T_unroll=4)
    assert "999" in ds.sample_ids  # kept for bookkeeping (legacy behavior)
    assert all(sid != "999" for sid, _ in ds.windows)


def test_normalization_scalar_and_per_channel(pdebench_h5):
    shift, scale = -1.0, 2.0
    ds = PDEBenchDataSet(pdebench_h5, indexes=["0"], T_unroll=4, step=T_STEPS, min_id=0, max_id=4, shift=shift, scale=scale)
    fields, t0 = ds[0]
    with h5py.File(str(pdebench_h5), "r") as f:
        raw = torch.from_numpy(np.moveaxis(f["0"]["data"][0:4], -1, 1)).float()
    assert torch.allclose(fields, (raw + shift) * scale)

    # Per-channel (list) shift/scale broadcast over the channel axis.
    ds_c = PDEBenchDataSet(
        pdebench_h5, indexes=["0"], T_unroll=4, step=T_STEPS, min_id=0, max_id=4, shift=[-1.0, -2.0], scale=[2.0, 3.0]
    )
    fields_c, _ = ds_c[0]
    expected = raw + torch.tensor([-1.0, -2.0]).view(1, C, 1, 1)
    assert torch.allclose(fields_c, expected * torch.tensor([2.0, 3.0]).view(1, C, 1, 1))


def test_ram_disk_equivalence(pdebench_h5):
    ds_ram = PDEBenchDataSet(pdebench_h5, indexes=["0", "1"], T_unroll=6, step=5, load_in_ram=True)
    ds_disk = PDEBenchDataSet(pdebench_h5, indexes=["0", "1"], T_unroll=6, step=5)
    assert len(ds_ram) == len(ds_disk)
    for idx in (0, 3):
        f_r, t_r = ds_ram[idx]
        f_d, t_d = ds_disk[idx]
        assert torch.allclose(f_r, f_d) and float(t_r) == float(t_d)


def test_load_trajectory_crop_and_normalization(pdebench_h5):
    shift, scale = 3.0, 2.0
    ds = PDEBenchDataSet(pdebench_h5, indexes=["0"], T_unroll=4, min_id=4, max_id=None, shift=shift, scale=scale)
    traj = ds.load_trajectory("0", min_id=4, max_id=None)
    assert traj.shape == (T_STEPS - 4, C, H, W)
    with h5py.File(str(pdebench_h5), "r") as f:
        expected = torch.from_numpy(np.moveaxis(f["0"]["data"][4:], -1, 1)).float()
    assert torch.allclose(traj, (expected + shift) * scale)


def test_invalid_params_raise(pdebench_h5):
    with pytest.raises(ValueError, match="step must be"):
        PDEBenchDataSet(pdebench_h5, T_unroll=4, step=0)
    with pytest.raises(ValueError, match="min_id must be"):
        PDEBenchDataSet(pdebench_h5, T_unroll=4, min_id=-1)


# ---------------------------------------------------------------------------
# Custom format (data + porosity, native 'split' attribute)
# ---------------------------------------------------------------------------


def test_custom_list_splits_by_attribute(custom_h5):
    splits = CustomDataset.list_splits(custom_h5)
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "val": 2}


def test_custom_batch_concatenates_porosity(custom_h5):
    ds = CustomDataset(custom_h5, indexes=["0000"], T_unroll=6, step=6, load_in_ram=True)
    fields, t0 = ds[0]
    assert fields.shape == (6, C + 1, H, W)  # data channels + porosity channel

    with h5py.File(str(custom_h5), "r") as f:
        d = torch.from_numpy(np.moveaxis(f["0000"]["data"][0:6], -1, 1)).float()
        # porosity is (T, H, W, 1) in the fixture -> channels-first permute
        p = torch.from_numpy(f["0000"]["porosity"][0:6]).float().permute(0, 3, 1, 2)
    assert torch.allclose(fields, torch.cat([d, p], dim=1))


def test_custom_ram_disk_equivalence(custom_h5):
    ds_r = CustomDataset(custom_h5, indexes=["0000", "0001"], T_unroll=6, step=5, load_in_ram=True)
    ds_d = CustomDataset(custom_h5, indexes=["0000", "0001"], T_unroll=6, step=5)
    for idx in (0, 2):
        assert torch.allclose(ds_r[idx][0], ds_d[idx][0])


def test_custom_temporal_mismatch_asserts(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("0000")
        grp.create_dataset("data", data=np.zeros((10, H, W, C), dtype=np.float32))
        grp.create_dataset("porosity", data=np.zeros((9, H, W, 1), dtype=np.float32))
    with pytest.raises(AssertionError, match="Time mismatch"):
        CustomDataset(path, T_unroll=4)


# ---------------------------------------------------------------------------
# ayz2 format (separate files / groups, coefficients and 'a' field)
# ---------------------------------------------------------------------------


def test_ayz2_list_splits(ayz2_train_h5, ayz2_both_h5):
    assert {k: len(v) for k, v in Ayz2Dataset.list_splits(ayz2_train_h5).items()} == {"train": 8}
    both = Ayz2Dataset.list_splits(ayz2_both_h5)
    assert {k: len(v) for k, v in both.items()} == {"train": 6, "valid": 3}


def test_ayz2_auto_split_and_error_cases(ayz2_train_h5, ayz2_both_h5):
    # Single-group file: split auto-resolved.
    ds = Ayz2Dataset(ayz2_train_h5, T_unroll=4, indexes=[0])
    assert ds.split == "train"

    # Multi-group file without an explicit name: ambiguous -> error listing groups.
    with pytest.raises(ValueError, match="multiple splits"):
        Ayz2Dataset(ayz2_both_h5, T_unroll=4)

    # Unknown group: KeyError listing the available ones.
    with pytest.raises(KeyError, match="not found"):
        Ayz2Dataset(ayz2_train_h5, split="bogus", T_unroll=4)


def test_ayz2_coefficients_always_returned_as_cond(ayz2_train_h5):
    ds = Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=6, step=6, indexes=[0], load_in_ram=True)
    assert ds.coefficients == ["cx", "nu"]  # auto-detected (x/t/u excluded)

    fields, t0, cond = ds[0]
    assert fields.shape == (6, 1, H, W) and cond.shape == (2, H, W)
    with h5py.File(str(ayz2_train_h5), "r") as f:
        for i, name in enumerate(ds.coefficients):
            v = float(f["train"][name][0])
            assert torch.allclose(cond[i], torch.full((H, W), v))


def test_ayz2_spatial_a_field_as_cond(ayz2_valid_h5):
    ds = Ayz2Dataset(ayz2_valid_h5, split="valid", T_unroll=6, step=6, indexes=[1], load_in_ram=True)
    assert ds.coefficients == [] and ds.has_a_field

    fields, t0, cond = ds[0]
    with h5py.File(str(ayz2_valid_h5), "r") as f:
        ref_a = torch.from_numpy(f["valid"]["a"][1]).float()
    assert cond.shape == (1, H, W) and torch.allclose(cond[0], ref_a)


def test_ayz2_no_conditioning_degrades_to_2tuple(tmp_path):
    path = tmp_path / "plain.hdf5"
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        _write_ayz2_group(f, "train", 2, 10, rng, with_coeffs=False, with_a=False)
    ds = Ayz2Dataset(path, T_unroll=4, indexes=[0])
    out = ds[0]
    assert len(out) == 2 and out[0].shape == (4, 1, H, W)


def test_ayz2_ram_disk_equivalence_and_bounds(ayz2_train_h5):
    ds_r = Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=6, step=5, indexes=[0, 1], load_in_ram=True)
    ds_d = Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=6, step=5, indexes=[0, 1])
    for idx in (0, 3):
        assert torch.allclose(ds_r[idx][0], ds_d[idx][0])
        assert torch.allclose(ds_r[idx][2], ds_d[idx][2])

    # Out-of-range index: warned, no windows — in disk AND RAM mode (no HDF5 crash).
    for ram in (False, True):
        ds_bad = Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=4, indexes=[999], load_in_ram=ram)
        assert len(ds_bad) == 0

    # Mixed valid/invalid indexes: the valid one still loads (RAM mode).
    ds_mix = Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=4, step=4, indexes=[3, 999], load_in_ram=True)
    assert len(ds_mix) > 0 and ds_mix[0][0].shape == (4, 1, H, W)


def test_ayz2_explicit_coefficients_subset_and_transform(ayz2_train_h5):
    ds = Ayz2Dataset(
        ayz2_train_h5, split="train", T_unroll=4, step=4, indexes=[3], coefficients=["cx"], shift=-1.0, scale=2.0
    )
    fields, t0, cond = ds[0]
    with h5py.File(str(ayz2_train_h5), "r") as f:
        v_cx = float(f["train"]["cx"][3])
        ref_u = torch.from_numpy(f["train"]["u"][3, 0:4]).float().unsqueeze(1)
    assert cond.shape == (1, H, W) and torch.allclose(cond[0], torch.full((H, W), (v_cx - 1.0) * 2.0))
    assert torch.allclose(fields, (ref_u - 1.0) * 2.0)

    with pytest.raises(KeyError, match="not found"):
        Ayz2Dataset(ayz2_train_h5, split="train", T_unroll=4, indexes=[0], coefficients=["nope"])


# ---------------------------------------------------------------------------
# TheWell format (rank-grouped fields)
# ---------------------------------------------------------------------------


def test_thewell_list_splits_and_batch(thewell_h5):
    assert {k: len(v) for k, v in TheWellDataSet.list_splits(thewell_h5).items()} == {"default": 8}

    ds = TheWellDataSet(thewell_h5, T_unroll=6, step=6, indexes=[0], load_in_ram=True)
    fields, t0 = ds[0]
    assert fields.shape == (6, 2, H, W)  # A + B concatenated on the channel dim

    with h5py.File(str(thewell_h5), "r") as f:
        a = torch.from_numpy(f["t0_fields/A"][0, 0:6]).float().unsqueeze(1)
        b = torch.from_numpy(f["t0_fields/B"][0, 0:6]).float().unsqueeze(1)
    assert torch.allclose(fields, torch.cat([a, b], dim=1))


def test_thewell_fields_list_and_unknown_field(thewell_h5):
    ds = TheWellDataSet(thewell_h5, fields_list=["B"], T_unroll=6, step=6, indexes=[0])
    assert ds.fields_list == ["B"] and ds[0][0].shape == (6, 1, H, W)

    with pytest.raises(KeyError, match="not found"):
        TheWellDataSet(thewell_h5, fields_list=["nope"], T_unroll=6)


def test_thewell_ram_disk_equivalence_and_bounds(thewell_h5):
    ds_r = TheWellDataSet(thewell_h5, T_unroll=6, step=5, indexes=[0, 1], load_in_ram=True)
    ds_d = TheWellDataSet(thewell_h5, T_unroll=6, step=5, indexes=[0, 1])
    for idx in (0, 2):
        assert torch.allclose(ds_r[idx][0], ds_d[idx][0])

    # RAM mode crops at [min_id, max_id] but reads stay absolute-indexed.
    ds_crop = TheWellDataSet(thewell_h5, T_unroll=6, step=6, indexes=[0], min_id=10, load_in_ram=True)
    f_c, _ = ds_crop[0]  # window at t0=10
    with h5py.File(str(thewell_h5), "r") as f:
        a = torch.from_numpy(f["t0_fields/A"][0, 10:16]).float().unsqueeze(1)
        b = torch.from_numpy(f["t0_fields/B"][0, 10:16]).float().unsqueeze(1)
    assert torch.allclose(f_c, torch.cat([a, b], dim=1))

    # Out-of-range trajectory index: warned, no windows — in disk AND RAM mode.
    for ram in (False, True):
        ds_bad = TheWellDataSet(thewell_h5, T_unroll=4, indexes=[99], load_in_ram=ram)
        assert len(ds_bad) == 0

    # Mixed valid/invalid indexes: the valid one still loads (RAM mode).
    ds_mix = TheWellDataSet(thewell_h5, T_unroll=4, step=4, indexes=[3, 99], load_in_ram=True)
    assert len(ds_mix) > 0 and ds_mix[0][0].shape == (4, 2, H, W)


# ---------------------------------------------------------------------------
# Legacy .npy format (RAM-only, time binning)
# ---------------------------------------------------------------------------


def test_npy_dataset_5d_and_binning(npy_file):
    ds = npyPDEDataset(npy_file, T_unroll=4, step=4, indexes=[0])
    fields, t0 = ds[0]
    assert fields.shape == (4, C, H, W)

    # Time binning halves the timeline: windows are built on the binned axis.
    ds_b = npyPDEDataset(npy_file, T_unroll=4, step=8, time_binning=2, indexes=[0])
    assert len(ds_b) == (16 // 2 - 4) // 8 + 1


def test_npy_dataset_4d(tmp_path):
    path = tmp_path / "tiny4d.npy"
    rng = np.random.default_rng(8)
    np.save(path, rng.standard_normal((3, 12, H, W)).astype(np.float32))
    ds = npyPDEDataset(path, T_unroll=4, step=4, indexes=[0])
    assert ds[0][0].shape == (4, 1, H, W)

    with pytest.raises(ValueError, match="Unsupported array shape"):
        np.save(tmp_path / "bad.npy", np.zeros((3, 12), dtype=np.float32))
        npyPDEDataset(tmp_path / "bad.npy", T_unroll=4)


# ---------------------------------------------------------------------------
# Unified split protocol (resolve_split_ids)
# ---------------------------------------------------------------------------


def _legacy_pdebench_protocol(path: Path, seed: int = 42, n_samples=None, split_ratio: float = 0.9):
    """Verbatim copy of the legacy driver algorithm (reproducibility reference)."""
    with h5py.File(str(path), "r") as f:
        all_ids = list(f.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    if n_samples is not None:
        all_sample_ids = all_ids[: min(len(all_ids), int(n_samples))]
    else:
        all_sample_ids = all_ids
    n_train = int(len(all_sample_ids) * split_ratio)
    return all_sample_ids[:n_train], all_sample_ids[n_train:], all_ids


def test_protocol_bit_identical_to_legacy(pdebench_h5):
    for seed, n_samples in ((42, 6), (7, None), (123, 8)):
        t_old, v_old, _ = _legacy_pdebench_protocol(pdebench_h5, seed=seed, n_samples=n_samples)
        info = resolve_split_ids(
            {"type": "pdebench", "path": str(pdebench_h5), "random_seed": seed},
            {"n_samples": n_samples} if n_samples is not None else {},
        )
        assert info["source"] == "protocol"
        assert list(info["train"]["ids"]) == t_old and list(info["val"]["ids"]) == v_old

    # The unused pool keeps the legacy seeded-shuffle ordering (eval compatibility).
    _, _, all_shuffled = _legacy_pdebench_protocol(pdebench_h5, seed=42, n_samples=6)
    info = resolve_split_ids({"type": "pdebench", "path": str(pdebench_h5), "random_seed": 42}, {"n_samples": 6})
    used = set(info["train"]["ids"]) | set(info["val"]["ids"])
    assert [k for k in info["train_pool_order"] if k not in used] == all_shuffled[6:]


def test_protocol_phase_split_ratio_override(pdebench_h5):
    """A per-phase split_ratio overrides the dataset-level one (legacy behavior)."""
    t_old, v_old, _ = _legacy_pdebench_protocol(pdebench_h5, seed=42, n_samples=8, split_ratio=0.75)
    info = resolve_split_ids(
        {"type": "pdebench", "path": str(pdebench_h5), "random_seed": 42, "split_ratio": 0.9},
        {"n_samples": 8, "split_ratio": 0.75},
    )
    assert list(info["train"]["ids"]) == t_old and list(info["val"]["ids"]) == v_old


def test_native_custom_attribute_splits(custom_h5):
    info = resolve_split_ids({"type": "custom", "path": str(custom_h5)}, {})
    assert info["source"] == "native"
    assert len(info["train"]["ids"]) == 4 and len(info["val"]["ids"]) == 2
    # Stable file order (no shuffle) — phase nesting is preserved by truncation.
    assert info["train"]["ids"] == [f"{i:04d}" for i in range(4)]

    # Caps + phase n_samples truncation of the train pool only.
    capped = resolve_split_ids({"type": "custom", "path": str(custom_h5), "n_train_samples": 2, "n_val_samples": 1}, {"n_samples": 3})
    assert len(capped["train"]["ids"]) == 2 and len(capped["val"]["ids"]) == 1


def test_native_ayz2_separate_files(ayz2_train_h5, ayz2_valid_h5):
    info = resolve_split_ids(
        {"type": "ayz2", "path": str(ayz2_train_h5), "val_path": str(ayz2_valid_h5)},
        {"n_samples": 3},
    )
    assert info["source"] == "native"
    assert info["train"]["group"] == "train" and len(info["train"]["ids"]) == 3
    assert info["val"]["group"] == "valid" and len(info["val"]["ids"]) == 4
    # Different files: distinct ID namespaces (both pools start at 0).
    assert info["train"]["path"] != info["val"]["path"]


def test_native_ayz2_same_file_groups(ayz2_both_h5):
    info = resolve_split_ids({"type": "ayz2", "path": str(ayz2_both_h5)}, {})
    assert info["source"] == "native"
    assert len(info["train"]["ids"]) == 6 and len(info["val"]["ids"]) == 3

    # Explicit group names are honored.
    named = resolve_split_ids({"type": "ayz2", "path": str(ayz2_both_h5), "split": "train", "val_split": "valid"}, {})
    assert named["train"]["group"] == "train" and named["val"]["group"] == "valid"


def test_single_group_file_falls_back_to_protocol(ayz2_train_h5):
    """One file, one group, no val_path: the PDEBench protocol applies to it."""
    all_ids = list(range(8))  # the 'train' group of ayz2_train_h5
    rng = random.Random(42)
    rng.shuffle(all_ids)
    pool = all_ids[:6]
    n_train = int(len(pool) * 0.9)

    info = resolve_split_ids({"type": "ayz2", "path": str(ayz2_train_h5), "random_seed": 42}, {"n_samples": 6})
    assert info["source"] == "protocol"
    assert list(info["train"]["ids"]) == pool[:n_train] and list(info["val"]["ids"]) == pool[n_train:]


def test_explicit_ids_take_precedence(pdebench_h5):
    info = resolve_split_ids(
        {
            "type": "pdebench",
            "path": str(pdebench_h5),
            "train_ids": ["0", "1"],
            "val_ids": ["2"],
            "random_seed": 42,
        },
        {"n_samples": 6},  # must be ignored for explicit IDs
    )
    assert info["source"] == "explicit"
    assert list(info["train"]["ids"]) == ["0", "1"] and list(info["val"]["ids"]) == ["2"]

    with pytest.raises(ValueError, match="non-empty"):
        resolve_split_ids({"type": "pdebench", "path": str(pdebench_h5), "train_ids": [], "val_ids": ["0"]}, {})


def test_unknown_type_and_missing_keys_raise():
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        get_dataset_class("bogus")
    with pytest.raises(KeyError, match='"path"'):
        resolve_split_ids({"type": "pdebench"}, {})
    with pytest.raises(KeyError, match='"batch_size"'):
        build_dataloaders({"type": "pdebench", "path": "x.hdf5"}, {"max_AR_steps": 2}, input_steps=4)


def test_empty_split_raises(pdebench_h5):
    # split_ratio so small that the val pool is empty.
    with pytest.raises(ValueError, match="Empty split"):
        resolve_split_ids({"type": "pdebench", "path": str(pdebench_h5), "split_ratio": 1.0}, {})


# ---------------------------------------------------------------------------
# build_dataloaders (end-to-end batch contract per format)
# ---------------------------------------------------------------------------

_PHASE = {"max_AR_steps": 4, "delta_ts": 6, "min_id": 0, "max_id": None, "n_samples": 8}


def test_build_dataloaders_pdebench_batch(pdebench_h5):
    tl, vl, ws = build_dataloaders(
        {"type": "pdebench", "path": str(pdebench_h5), "batch_size": 2, "num_workers": 0, "load_in_ram": True},
        _PHASE,
        input_steps=6,
    )
    assert ws == 4 + 6 + 1
    batch = next(iter(tl))
    assert len(batch) == 2 and batch[0].shape == (2, 11, C, H, W) and batch[1].shape == (2, 1)


def test_build_dataloaders_ayz2_cond_batch(ayz2_train_h5, ayz2_valid_h5):
    tl, vl, ws = build_dataloaders(
        {
            "type": "ayz2",
            "path": str(ayz2_train_h5),
            "val_path": str(ayz2_valid_h5),
            "batch_size": 2,
            "num_workers": 0,
            "load_in_ram": True,
        },
        _PHASE,
        input_steps=6,
    )
    batch = next(iter(tl))
    # 3-tuple: fields + t0 + cond (scalar coefficients broadcast to constant fields).
    assert len(batch) == 3
    assert batch[0].shape == (2, 11, 1, H, W) and batch[2].shape == (2, 2, H, W)

    vbatch = next(iter(vl))
    # The valid file carries the spatial 'a' field instead of coefficients.
    assert vbatch[2].shape == (2, 1, H, W)


def test_build_dataloaders_custom_and_thewell(custom_h5, thewell_h5):
    tl_c, _, _ = build_dataloaders(
        {"type": "custom", "path": str(custom_h5), "batch_size": 1, "num_workers": 0, "load_in_ram": True},
        {"max_AR_steps": 2, "delta_ts": 6, "min_id": 0, "max_id": None, "n_samples": 4},
        input_steps=6,
    )
    assert next(iter(tl_c))[0].shape == (1, 9, C + 1, H, W)

    tl_w, _, _ = build_dataloaders(
        {"type": "thewell", "path": str(thewell_h5), "batch_size": 1, "num_workers": 0, "load_in_ram": True},
        {"max_AR_steps": 2, "delta_ts": 6, "min_id": 0, "max_id": None, "n_samples": 4},
        input_steps=6,
    )
    assert next(iter(tl_w))[0].shape == (1, 9, 2, H, W)
