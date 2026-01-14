import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import re
import glob
from collections import defaultdict
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob


# defaultdict is no longer needed as each series is a single file
class PairedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        series_data_dir,
        static_field_dir,
        t_n,
        series_file_pattern="*.npy",
        static_file_pattern="*.npy",
        transform=None,
        expected_channels_from_user=None,
        time_first=False,
    ):
        """
        Args:
            series_data_dir (str): Directory containing time series XXXX.npy files (each [Nt, ...] or [..., Nt]).
            static_field_dir (str): Directory containing static field YYYY.npy files (each [H, W]).
                                    It's assumed XXXX and YYYY identifiers match.
            t_n (int): Number of timesteps for unrolling (sequence length).
            series_file_pattern (str): Glob pattern for series files.
            transform (callable, optional): Optional transform to be applied to the (sequence, static_field) tuple.
                                           The transform should expect a tuple and return a tuple.
            expected_channels_from_user (int, optional):
                - If raw series slices are [H,W], this specifies the number of output channels for the series.
                - If raw series slices are [C,H,W], this can be used for validation. If None, uses raw series channels.
        """
        self.series_data_dir = series_data_dir
        self.static_field_dir = static_field_dir
        self.t_n = t_n
        self.time_first = time_first
        self.t_dim = 0 if time_first else -1  # Time dimension index
        self.transform = transform
        self.expected_channels_from_user = expected_channels_from_user

        assert (
            t_n > 1
        ), "The number of time steps must be > 1: we need at least the input f(t_i) and one target f(t_{i+1})"

        self.samples_info = []  # List of tuples: (series_filepath, static_field_filepath, start_slice_index)

        # --- Stage 1: Discover series files and match with static fields ---
        series_search_path = os.path.join(self.series_data_dir, series_file_pattern)
        all_series_files_paths = glob.glob(series_search_path)
        self.series_prefix = series_file_pattern.split("*")[0]  # Extract prefix before the wildcard
        self.series_suffix = series_file_pattern.split("*")[-1]
        self.static_prefix = static_file_pattern.split("*")[0]  # Extract prefix before the wildcard
        self.static_suffix = static_file_pattern.split("*")[-1]  # Extract suffix after the wildcard

        if not all_series_files_paths:
            raise FileNotFoundError(
                f"No series files found matching pattern '{series_search_path}' in '{self.series_data_dir}'"
            )

        # --- Stage 2: Determine properties from the first valid pair ---
        self._raw_series_slice_ndim = None
        self._raw_series_slice_channels = None
        self._target_series_channels = None
        self._static_field_shape_hw = None  # To store (H, W) of static fields for consistency check

        first_valid_pair_inspected = False

        for series_filepath in all_series_files_paths:
            series_filename = os.path.basename(series_filepath)

            idx = series_filename[len(self.series_prefix) : -len(self.series_suffix)]
            static_field_filepath = os.path.join(self.static_field_dir, self.static_prefix + idx + self.static_suffix)

            if not os.path.exists(static_field_filepath):
                print(
                    f"Warning: Static field file {static_field_filepath} not found for series {series_filepath}. Skipping series."
                )
                continue

            try:
                series_data = np.load(series_filepath, mmap_mode="r")
                if series_data.ndim < 2:
                    print(
                        f"Warning: Series file {series_filepath} has insufficient dimensions (shape {series_data.shape}). Skipping."
                    )
                    continue

                Nt_series = series_data.shape[0] if self.time_first else series_data.shape[-1]
                if Nt_series < self.t_n:
                    print(
                        f"Warning: Series file {series_filepath} has only {Nt_series} timesteps, less than t_n={self.t_n}. Skipping."
                    )
                    continue

                if not first_valid_pair_inspected:
                    # Determine series properties
                    sample_series_slice = np.take(series_data, 0, axis=self.t_dim)
                    self._raw_series_slice_ndim = sample_series_slice.ndim

                    series_h, series_w = sample_series_slice.shape[-2], sample_series_slice.shape[-1]

                    if self._raw_series_slice_ndim == 2:  # Raw series slice is [H, W]
                        self._raw_series_slice_channels = 1
                        self._target_series_channels = (
                            self.expected_channels_from_user if self.expected_channels_from_user is not None else 1
                        )
                        if self._target_series_channels <= 0:
                            raise ValueError(
                                "expected_channels_from_user must be positive if raw series slices are 2D."
                            )
                    elif self._raw_series_slice_ndim == 3:  # Raw series slice is [C, H, W]
                        self._raw_series_slice_channels = sample_series_slice.shape[0]
                        if (
                            self.expected_channels_from_user is not None
                            and self._raw_series_slice_channels != self.expected_channels_from_user
                        ):
                            print(
                                f"Warning: Raw 3D series slices have {self._raw_series_slice_channels} channels, "
                                f"but expected_channels_from_user is {self.expected_channels_from_user}. "
                                f"Using raw slice's channel count ({self._raw_series_slice_channels})."
                            )
                        self._target_series_channels = self._raw_series_slice_channels
                    else:
                        raise ValueError(
                            f"Unsupported raw series slice ndim: {self._raw_series_slice_ndim} "
                            f"from file {series_filepath} (slice shape {sample_series_slice.shape}). Expected 2 or 3."
                        )

                    # Determine and check static field properties
                    static_data = np.load(static_field_filepath, mmap_mode="r")
                    if static_data.ndim != 2:  # Expecting [H, W] for static field
                        raise ValueError(
                            f"Static field file {static_field_filepath} has ndim {static_data.ndim} (shape {static_data.shape}). Expected 2 ([H,W])."
                        )
                    self._static_field_shape_hw = static_data.shape  # (H, W)

                    # Consistency check for H, W dimensions
                    if self._static_field_shape_hw[0] != series_h or self._static_field_shape_hw[1] != series_w:
                        raise ValueError(
                            f"Spatial dimension mismatch for series {series_filepath} ([...,{series_h},{series_w}]) "
                            f"and static field {static_field_filepath} ({self._static_field_shape_hw})."
                        )
                    first_valid_pair_inspected = True

                # Add all possible starting points for sequences from this series file
                for i in range(Nt_series - self.t_n + 1):
                    self.samples_info.append((series_filepath, static_field_filepath, i))

            except Exception as e:
                print(
                    f"Warning: An error occurred while processing series {series_filepath} or its static field: {e}. Skipping."
                )
                continue

        if not self.samples_info:
            raise RuntimeError(
                "No valid samples could be generated. Check directories, t_n, file contents, and matching static fields."
            )
        if not first_valid_pair_inspected:
            raise RuntimeError(
                "Could not determine data properties. No valid series/static field pairs found or all were unsuitable."
            )

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        if idx >= len(self.samples_info):
            raise IndexError(f"Index {idx} out of bounds for a dataset of size {len(self.samples_info)}.")

        series_filepath, static_field_filepath, start_slice_index = self.samples_info[idx]

        # --- Load and process time series data ---
        try:
            series_data_mmap = np.load(series_filepath, mmap_mode="r")
            if self.time_first:
                sequence_data = np.array(series_data_mmap[start_slice_index : start_slice_index + self.t_n])
            else:
                sequence_data = np.array(series_data_mmap[..., start_slice_index : start_slice_index + self.t_n])
                sequence_data = np.moveaxis(sequence_data, -1, 0)  # Move time dimension to the front
        except Exception as e:
            print(f"Error loading or slicing series data from {series_filepath} for sample index {idx}: {e}")
            raise

        if self._raw_series_slice_ndim == 2:
            final_sequence_data = np.expand_dims(sequence_data, axis=1)
            if self._target_series_channels > 1:
                final_sequence_data = np.repeat(final_sequence_data, self._target_series_channels, axis=1)
        elif self._raw_series_slice_ndim == 3:
            final_sequence_data = sequence_data
        else:
            raise RuntimeError(
                f"Unexpected _raw_series_slice_ndim ({self._raw_series_slice_ndim}) in __getitem__ for sample {idx}."
            )

        if not final_sequence_data.flags["C_CONTIGUOUS"]:
            final_sequence_data = np.ascontiguousarray(final_sequence_data)
        sequence_tensor = torch.from_numpy(final_sequence_data).float()

        # --- Load and process static field data ---
        try:
            # Static fields are likely smaller, mmap_mode might be overkill but harmless
            static_data_mmap = np.load(static_field_filepath, mmap_mode="r")
            # static_data_mmap is [H, W]
            static_field_array = np.array(static_data_mmap)
        except Exception as e:
            print(f"Error loading static field data from {static_field_filepath} for sample index {idx}: {e}")
            raise

        # Add channel dimension: [H, W] -> [1, H, W]
        static_field_array_expanded = np.expand_dims(static_field_array, axis=0)

        if not static_field_array_expanded.flags["C_CONTIGUOUS"]:
            static_field_array_expanded = np.ascontiguousarray(static_field_array_expanded)
        static_field_tensor = torch.from_numpy(static_field_array_expanded).float()

        # --- Apply transform if any ---
        # The transform should expect a tuple (sequence_tensor, static_field_tensor)
        # and return a tuple of transformed tensors.
        if self.transform:
            sequence_tensor, static_field_tensor = self.transform((sequence_tensor, static_field_tensor))

        return static_field_tensor, sequence_tensor


# ---------------------------------------------------------
# Implemented by Fatima, July 2025
# Custom dataset for unrolled time evolution
# ---------------------------------------------------------

class UnrolledTimeVaryingDataset(Dataset):
    def __init__(self, porosity_dir, concentration_dir, T_unroll=10):
        """
        PyTorch Dataset for unrolled multi-step time evolution.

        This dataset provides sequences of consecutive time steps for learning temporal dependencies in reactive transport simulations.
        Each sample contains an unrolled sequence of porosity and concentration fields over a specified time window `T_unroll`.

        Each sample consists of:
        - Input tensor: `[T-1, 3, H, W]` representing states from `t` to `t+T_unroll-2`
        - Target tensor: `[T-1, 3, H, W]` representing the next-step states from `t+1` to `t+T_unroll-1`
            where 3 channels correspond to porosity and two species fields.

        Args:
            porosity_dir (str): Directory containing porosity `.npy` files of shape `[T, H, W]`.
            concentration_dir (str): Directory containing species concentration `.npy` files of shape `[T, H, W, 2]`.
            T_unroll (int, optional): Number of time steps in each unrolled window (default: 10).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - `input_tensor`: `[T-1, 3, H, W]`
                - `target_tensor`: `[T-1, 3, H, W]`
                ready for sequence-to-sequence model training.
        """
        self.porosity_files = sorted(glob.glob(os.path.join(porosity_dir, "*.npy")))
        self.concentration_files = sorted(glob.glob(os.path.join(concentration_dir, "*.npy")))

        print(f"Found {len(self.porosity_files)} porosity files.")
        print(f"Found {len(self.concentration_files)} concentration files.")

        assert len(self.porosity_files) == len(self.concentration_files), \
            "Mismatch between number of porosity and concentration files."

        self.T_unroll = T_unroll
        self.samples = []

        for p_path, c_path in zip(self.porosity_files, self.concentration_files):
            p_data = np.load(p_path, mmap_mode="r")
            c_data = np.load(c_path, mmap_mode="r")

            if p_data.shape[0] < T_unroll:
                print(f"Skipping {p_path} due to insufficient timesteps ({p_data.shape[0]})")
                continue

            T = p_data.shape[0]
            for t in range(T - T_unroll):
                self.samples.append((p_path, c_path, t))

        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_path, c_path, t = self.samples[idx]
        p = np.load(p_path)[t:t + self.T_unroll]        
        c = np.load(c_path)[t:t + self.T_unroll]       

        p = np.expand_dims(p, -1)                       
        x = np.concatenate([p, c], axis=-1)             
        x = np.moveaxis(x, -1, 1)                        

        input_tensor = torch.from_numpy(x[:-1]).float() 
        target_tensor = torch.from_numpy(x[1:]).float()  

        return input_tensor, target_tensor

# ---------------------------------------------------------
# [Mod] Implemented by Fatima, July 2025
# Dataset for one-step evolution using porosity + concentrations
# Predicts (p_{t+1}, c_{t+1}) from (p_t, c_t)
# ---------------------------------------------------------

class FNOTimestepDataset(Dataset):
    def __init__(self, series_dir, porosity_dir, time_first=True, transform=None):
        """
        PyTorch Dataset for one-step time evolution (porosity + species).

        Loads paired `.npy` files:
        - concentration series (e.g., 2 species) shaped `[T, H, W, C]` if `time_first=True`
            or `[H, W, C, T]` if `time_first=False`;
        - porosity series shaped `[T, H, W]`.
        Args:
            series_dir (str): Directory with concentration time-series `.npy` files.
                Expected per-file shape `[T, H, W, C]` (time-first) or `[H, W, C, T]` (time-last).
            porosity_dir (str): Directory with porosity time-series `.npy` files, shape `[T, H, W]`.
            time_first (bool, optional): Whether time is the first dimension in `series` files. Default: True.
            transform (callable, optional): Optional transform applied to `(input_tensor, target_tensor)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - input_tensor:  float32 tensor `[1 + C, H, W]` at time `t`
                - target_tensor: float32 tensor `[1 + C, H, W]` at time `t+1`
        """
       
        self.series_paths = sorted(glob.glob(os.path.join(series_dir, "*.npy")))
        self.porosity_paths = sorted(glob.glob(os.path.join(porosity_dir, "*.npy")))
        self.transform = transform
        self.time_first = time_first

        assert len(self.series_paths) == len(self.porosity_paths), "Mismatch in sample counts"

        self.samples = []
        for s_path, p_path in zip(self.series_paths, self.porosity_paths):
            series = np.load(s_path, mmap_mode="r")
            porosity = np.load(p_path, mmap_mode="r")
            assert series.shape[0] == porosity.shape[0], "Time mismatch between concentration and porosity"
            T = series.shape[0] if time_first else series.shape[-1]
            for t in range(T - 1):
                self.samples.append((s_path, p_path, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_path, p_path, t = self.samples[idx]

        series = np.load(s_path)      
        porosity = np.load(p_path) 

        if not self.time_first:
            series = np.moveaxis(series, -1, 0)  

        p_t = porosity[t]     
        p_t1 = porosity[t+1]

        c_t = series[t]       
        c_t1 = series[t+1]

        p_t = p_t[np.newaxis, ...]
        p_t1 = p_t1[np.newaxis, ...]
        c_t = np.transpose(c_t, (2, 0, 1))
        c_t1 = np.transpose(c_t1, (2, 0, 1))

        input_tensor = np.concatenate([p_t, c_t], axis=0)     
        target_tensor = np.concatenate([p_t1, c_t1], axis=0) 

        input_tensor = torch.from_numpy(input_tensor).float()
        target_tensor = torch.from_numpy(target_tensor).float()

        if self.transform:
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        return input_tensor, target_tensor


# ---------------------------------------------------------
# Implemented by Fatima, July 2025
# Custom dataset for paired image-based time evolution
# Loads PNG slices of porosity and species fields to train models
# for one-step temporal prediction from images.
# ---------------------------------------------------------
class PairedImageTimeDataset(Dataset):
    def __init__(self, p_dir, c_dir, transform=None):
        """
        PyTorch Dataset for paired image-based time evolution.

        This dataset loads consecutive PNG image slices representing
        porosity and species concentration fields at two consecutive
        time steps `(t, t+1)` from simulation outputs.
        
        Args:
            p_dir (str): Directory containing porosity PNG files named as `p_evol_<sim_id>_slice_<n>.png`.
            c_dir (str): Directory containing species concentration PNG files named as 
                `c_evol_<sim_id>_slice_<n>_species_<id>.png`.
            transform (callable, optional): Transform to apply to loaded images (default: `torchvision.transforms.ToTensor()`).

        Notes:
            - Pairs slices belonging to the same simulation ID where `slice_tp1 = slice_t + 1`.
            - Filters out incomplete or non-consecutive image pairs.
        """
        self.p_dir = p_dir
        self.c_dir = c_dir
        self.transform = transform if transform else T.ToTensor()

        self.slice_keys = []
        pattern = r"p_evol_(\d+)_slice_(\d+)\.png"
        for fname in sorted(os.listdir(p_dir)):
            match = re.match(pattern, fname)
            if match:
                sim_id = match.group(1)
                slice_id = int(match.group(2))
                self.slice_keys.append((sim_id, slice_id))

        self.slice_pairs = []
        for i in range(len(self.slice_keys) - 1):
            sim_id_1, slice_1 = self.slice_keys[i]
            sim_id_2, slice_2 = self.slice_keys[i + 1]
            if sim_id_1 == sim_id_2 and slice_2 == slice_1 + 1:
                self.slice_pairs.append((self.slice_keys[i], self.slice_keys[i + 1]))

    def __len__(self):
        return len(self.slice_pairs)

    def __getitem__(self, idx):
        (sim_id_t, slice_t), (sim_id_tp1, slice_tp1) = self.slice_pairs[idx]

        def load_full(sim_id, slice_id):
            p = self.transform(Image.open(os.path.join(self.p_dir, f"p_evol_{sim_id}_slice_{slice_id}.png")).convert("L"))
            c0 = self.transform(Image.open(os.path.join(self.c_dir, f"c_evol_{sim_id}_slice_{slice_id}_species_0.png")).convert("L"))
            c1 = self.transform(Image.open(os.path.join(self.c_dir, f"c_evol_{sim_id}_slice_{slice_id}_species_1.png")).convert("L"))
            return torch.cat([p, c0, c1], dim=0)  # [3, H, W]

        input_tensor = load_full(sim_id_t, slice_t)
        target_tensor = load_full(sim_id_tp1, slice_tp1)

        return input_tensor, target_tensor
    

# ---------------------------------------------------------
# [Mod] Implemented by Fatima, August 2025
# Custom dataset for unrolled time evolution from PNG images
# ---------------------------------------------------------
class ImageUnrolledTimeVaryingDataset(Dataset):
    def __init__(self, p_dir, c_dir, T_unroll=10, transform=None):
        """
        PyTorch Dataset for unrolled multi-step time evolution from image data.

        This dataset loads sequences of PNG image slices representing porosity 
        and species concentration fields from reactive transport simulations.
        It constructs consecutive time windows of length `T_unroll` for training
        sequence-based models to predict temporal dynamics from visual data.

        Each sample consists of:
        - Input tensor: `[T_unroll-1, 3, H, W]` = sequence from time `t` to `t+T_unroll-2`
        - Target tensor: `[1, 3, H, W]` = next-step frame at time `t+T_unroll-1`

        Args:
            p_dir (str): Directory containing porosity PNG files named as 
                `p_evol_<sim_id>_slice_<n>.png`.
            c_dir (str): Directory containing species PNG files named as 
                `c_evol_<sim_id>_slice_<n>_species_<id>.png`, where `id ? {0,1}`.
            T_unroll (int, optional): Number of consecutive time steps in each sequence (default: 10).
            transform (callable, optional): Transform applied to each image 
                (default: `torchvision.transforms.ToTensor()`).
        """
        self.p_dir = p_dir
        self.c_dir = c_dir
        self.T_unroll = T_unroll
        self.transform = transform if transform else T.ToTensor()

        self.sim_dict = {}
        pattern = r"p_evol_(\d+)_slice_(\d+)\.png"

        for fname in sorted(os.listdir(p_dir)):
            match = re.match(pattern, fname)
            if match:
                sim_id, t_str = match.groups()
                t = int(t_str)
                self.sim_dict.setdefault(sim_id, []).append(t)

        self.samples = []
        for sim_id, timesteps in self.sim_dict.items():
            timesteps = sorted(timesteps)
            for i in range(len(timesteps) - T_unroll):
                self.samples.append((sim_id, timesteps[i]))

        print(f"Total unrolled samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_id, start_t = self.samples[idx]
        input_seq = []
        target_seq = []

        for dt in range(self.T_unroll):
            t = start_t + dt

            p_img = Image.open(os.path.join(self.p_dir, f"p_evol_{sim_id}_slice_{t}.png")).convert("L")
            c0_img = Image.open(os.path.join(self.c_dir, f"c_evol_{sim_id}_slice_{t}_species_0.png")).convert("L")
            c1_img = Image.open(os.path.join(self.c_dir, f"c_evol_{sim_id}_slice_{t}_species_1.png")).convert("L")

            p = self.transform(p_img)     
            c0 = self.transform(c0_img)   
            c1 = self.transform(c1_img)   
            full = torch.cat([p, c0, c1], dim=0) 

            if dt < self.T_unroll - 1:
                input_seq.append(full)
            else:
                target_seq.append(full)

        input_tensor = torch.stack(input_seq, dim=0)  
        target_tensor = torch.stack(target_seq, dim=0)

        return input_tensor, target_tensor


class UnrolledH5ConcentrationDataset(Dataset):
    def __init__(self, h5_path, T_unroll=10):
        self.h5_path = h5_path
        self.T_unroll = T_unroll
        self.windows = []

        with h5py.File(h5_path, "r") as f:
            self.sample_ids = sorted(list(f.keys()))

            for sid in self.sample_ids:
                T = f[sid]["data"].shape[0]

                if T < T_unroll:
                    print(f"Skipping sample {sid}: insufficient T={T}")
                    continue

                for t0 in range(T - T_unroll):
                    self.windows.append((sid, t0))

        print(f"Found {len(self.sample_ids)} simulation samples")
        print(f"Total unrolled windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sid, t0 = self.windows[idx]

        with h5py.File(self.h5_path, "r") as f:
            conc = f[sid]["data"][t0:t0 + self.T_unroll]

        conc = torch.from_numpy(np.moveaxis(conc, -1, 1)).float()

        return conc[:-1], conc[1:]


class FNOH5TimestepDataset(Dataset):
    def __init__(self, h5_path,max_sims=100, time_first=True, transform=None):
        self.h5_path = h5_path
        self.time_first = time_first
        self.transform = transform
        self.samples = []
        self._file = None  

        with h5py.File(h5_path, "r") as f:
            all_sids  = sorted(list(f.keys()))  
            self.sample_ids = all_sids[:max_sims]

            for sid in self.sample_ids:
                data_ds = f[sid]["data"]
                shape = data_ds.shape

                if time_first:
                  
                    T = shape[0]
                else:
                
                    T = shape[-1]

                if T < 2:
                    print(f"Skipping {sid}: T={T} < 2")
                    continue

               
                for t in range(T - 1):
                    self.samples.append((sid, t))

        print(f"Found {len(self.sample_ids)} simulations in {os.path.basename(h5_path)}")
        print(f"Using {len(self.sample_ids)} simulations (first {max_sims})")
        print(f"Total one-step samples: {len(self.samples)}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, t = self.samples[idx]

        with h5py.File(self.h5_path, "r") as f:
            data_ds = f[sid]["data"]

            if self.time_first:
       
                c_t_np  = data_ds[t]     
                c_t1_np = data_ds[t + 1] 
            else:
             
                c_t_np  = data_ds[..., t]     
                c_t1_np = data_ds[..., t + 1] 

        c_t_np  = np.transpose(c_t_np,  (2, 0, 1))
        c_t1_np = np.transpose(c_t1_np, (2, 0, 1))

        input_tensor  = torch.from_numpy(c_t_np).float()
        target_tensor = torch.from_numpy(c_t1_np).float()

        if self.transform is not None:
            input_tensor, target_tensor = self.transform((input_tensor, target_tensor))

        return input_tensor, target_tensor

class UnrolledH5ConcentrationDataset(Dataset):
    """
    Unrolled multi-step dataset for PDEBench.
        input_tensor  = seq[0 : T_unroll-1]  -> [T-1, C, H, W]
        target_tensor = seq[1 : T_unroll]    -> [T-1, C, H, W]
    """

    def __init__(self, h5_path, T_unroll=10, max_sims=None):
        super().__init__()
        self.h5_path = str(h5_path)
        self.T_unroll = T_unroll
        self.samples = []          
        self._file = None      

        with h5py.File(self.h5_path, "r") as f:
            all_sids = sorted(list(f.keys()))
            if max_sims is not None:
                all_sids = all_sids[:max_sims]

            self.sample_ids = all_sids

            for sid in self.sample_ids:
                data_ds = f[sid]["data"]        
                T = data_ds.shape[0]

                if T < T_unroll:
                    print(f"Skipping sample {sid}: T={T} < T_unroll={T_unroll}")
                    continue

                for t0 in range(T - T_unroll):
                    self.samples.append((sid, t0))

        print(f"Using {len(self.sample_ids)} simulations from {os.path.basename(self.h5_path)}")
        print(f"Total unrolled windows: {len(self.samples)}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, t0 = self.samples[idx]
        f = self._get_file()

        seq = f[sid]["data"][t0 : t0 + self.T_unroll]

        seq = np.moveaxis(seq, -1, 1)

        input_tensor  = torch.from_numpy(seq[:-1]).float()  
        target_tensor = torch.from_numpy(seq[1:]).float()  

        return input_tensor, target_tensor

class UnrolledNSDataset(Dataset):
    """
    Per-simulation files:
        velocity: (T, H, W, 2)

    Returns:
        input:  (T_unroll-1, 2, H, W)
        target: (T_unroll-1, 2, H, W)
    """

    def __init__(self, per_sim_glob, T_unroll=10, max_files=None):
        super().__init__()
        self.files = sorted(glob.glob(per_sim_glob))
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise FileNotFoundError(f"No files matched: {per_sim_glob}")

        self.T_unroll = int(T_unroll)
        self.samples = []     
        self._handles = {}   

        for file_i, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                v = f["velocity"]
                if v.ndim != 4 or v.shape[-1] != 2:
                    raise ValueError(f"{path}: expected velocity (T,H,W,2), got {v.shape}")
                T = v.shape[0]
                if T < self.T_unroll:
                    continue
                for t0 in range(T - self.T_unroll):
                    self.samples.append((file_i, t0))

        print(f"Per-sim files: {len(self.files)}")
        print(f"T_unroll = {self.T_unroll}")
        print(f"Total windows: {len(self.samples)}")

    def _get_file(self, file_i):
        if file_i not in self._handles:
            self._handles[file_i] = h5py.File(self.files[file_i], "r")
        return self._handles[file_i]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_i, t0 = self.samples[idx]
        f = self._get_file(file_i)

        seq = f["velocity"][t0 : t0 + self.T_unroll]  
        seq = np.moveaxis(seq, -1, 1)                

        x = torch.from_numpy(seq[:-1]).float()
        y = torch.from_numpy(seq[1:]).float()
        return x, y
    

class UnrolledH5DatasetWithTime(Dataset):
    """
    Unrolled multi-step dataset for PDEBench HDF5 layout:
      <sid>/data: (T, H, W, C)
      <sid>/grid/t: (T,)

    Returns sliding windows of length T_unroll:
      input_tensor  = seq[0 : T_unroll-1]  -> (T_unroll-1, C, H, W)
      target_tensor = seq[1 : T_unroll]    -> (T_unroll-1, C, H, W)
      t_in          = t[0 : T_unroll-1]    -> (T_unroll-1,)
      t_out         = t[1 : T_unroll]      -> (T_unroll-1,)
      dt            = t_out - t_in         -> (T_unroll-1,)
    """

    def __init__(self, h5_path, T_unroll=10, max_sims=None, return_dt=True):
        super().__init__()
        self.h5_path = str(h5_path)
        self.T_unroll = T_unroll
        self.return_dt = return_dt

        self.samples = []
        self.sample_ids = []
        self._file = None

        with h5py.File(self.h5_path, "r") as f:
            all_sids = sorted(list(f.keys()))
            if max_sims is not None:
                all_sids = all_sids[:max_sims]

            for sid in all_sids:
                if "data" not in f[sid] or "grid" not in f[sid] or "t" not in f[sid]["grid"]:
                    continue

                T = f[sid]["data"].shape[0]
                if T < T_unroll:
                    print(f"Skipping sample {sid}: T={T} < T_unroll={T_unroll}")
                    continue

                self.sample_ids.append(sid)
                for t0 in range(T - T_unroll):
                    self.samples.append((sid, t0))

        print(f"Using {len(self.sample_ids)} simulations from {os.path.basename(self.h5_path)}")
        print(f"Total unrolled windows: {len(self.samples)}")

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, t0 = self.samples[idx]
        f = self._get_file()

        seq = f[sid]["data"][t0 : t0 + self.T_unroll]
        t = f[sid]["grid"]["t"][t0 : t0 + self.T_unroll]

        seq = np.moveaxis(seq, -1, 1)

        x_in  = torch.from_numpy(seq[:-1]).float() 
        x_out = torch.from_numpy(seq[1:]).float()

        t_in  = torch.from_numpy(t[:-1]).float()    
        t_out = torch.from_numpy(t[1:]).float()

        if self.return_dt:
            dt = t_out - t_in
            return x_in, x_out, t_in, t_out, dt

        return x_in, x_out, t_in, t_out
