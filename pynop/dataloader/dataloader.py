import numpy as np
from PIL import Image
import os
from pathlib import Path
import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import glob
from collections import defaultdict

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
            series_data_dir (str): Directory containing time series XXXX.npy files (each [Nt, ...]).
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

        return sequence_tensor, static_field_tensor
