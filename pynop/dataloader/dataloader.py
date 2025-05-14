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


class SlicedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        data_dir,
        t_n,
        file_pattern="*.npy",
        time_first=True,
        expected_channels_from_user=None,
    ):
        """
        Args:
            data_dir (str): Directory containing the XXXX.npy files (each file [Nt, ...]).
            t_n (int): Number of timesteps for unrolling (sequence length).
            file_pattern (str): Glob pattern to find series files (e.g., "*.npy").
            transform (callable, optional): Optional transform to be applied to a sample.
            expected_channels_from_user (int, optional):
                - If raw slices are [H,W] (ndim=2), specifies the number of output channels.
                - If raw slices are [C,H,W] (ndim=3), can be used for validation.
                  If None, uses the raw channel count.
        """
        self.data_dir = data_dir
        self.t_n = t_n
        self.time_first = time_first  # True for [t_n, H, W] output
        # self.channels_first = channels_first # Implicitly True for [t_n, C, H, W] output
        self.expected_channels_from_user = expected_channels_from_user

        self.samples_info = []  # List of tuples: (filepath, start_slice_index_in_file)

        # --- Stage 1: Discover series files ---
        search_path = os.path.join(data_dir, file_pattern)
        all_series_files = glob.glob(search_path)

        if not all_series_files:
            raise FileNotFoundError(f"No files found matching pattern '{search_path}' in '{data_dir}'")

        # --- Stage 2: Determine slice properties from the first valid file ---
        # Assumes all .npy files have a consistent slice structure.
        self._raw_slice_ndim = None  # Ndim of a single time slice, e.g., 2 for [H,W], 3 for [C,H,W]
        self._raw_slice_channels = None  # Channels in the raw slice if 3D [C,H,W]
        self._target_channels = None  # Channels for the output tensor [t_n, C, H, W]
        self.t_dim = 0 if time_first else -1  # Time dimension index
        first_valid_file_inspected = False

        for series_filepath in all_series_files:
            try:
                # Use mmap_mode to avoid loading the whole file just for shape and a sample slice.
                # This is particularly important for large files.
                # np.load with mmap_mode returns an mmaparray object which is good to manage with 'with'
                # if not directly assigned to a persistent variable in the class.
                # Here, it's used temporarily for inspection.
                data = np.load(series_filepath, mmap_mode="r")
                # The entire file has a shape like [Nt, H, W] or [Nt, C, H, W]
                # We expect at least 2 dimensions for the entire file (Nt, H_or_W_dim)
                # and for a slice, at least 1 dimension (H_or_W_dim).
                # Typically, the data file will be 3D ([Nt,H,W]) or 4D ([Nt,C,H,W])
                if data.ndim < 2:  # E.g., a file with only [Nt] is not good.
                    print(
                        f"Warning: File {series_filepath} has insufficient dimensions (shape {data.shape}). Skipping."
                    )
                    continue

                Nt_series = data.shape[0] if time_first else data.shape[-1]

                if Nt_series < self.t_n:
                    print(
                        f"Warning: File {series_filepath} has only {Nt_series} timesteps, "
                        f"less than t_n={self.t_n}. Skipping."
                    )
                    continue

                if not first_valid_file_inspected:
                    # Determine properties from the first slice of the first valid file
                    sample_slice = np.take(data, 0, axis=self.t_dim)  # This is [H,W] or [C,H,W]
                    self._raw_slice_ndim = sample_slice.ndim  # ndim of the slice, not the entire file

                    if self._raw_slice_ndim == 2:  # Raw slice is [H, W]
                        self._raw_slice_channels = 1  # Conceptually, or could be None
                        self._target_channels = (
                            self.expected_channels_from_user if self.expected_channels_from_user is not None else 1
                        )
                        if self._target_channels <= 0:  # Added check
                            raise ValueError(
                                "expected_channels_from_user must be positive if raw slices are 2D and it's specified."
                            )
                    elif (
                        self._raw_slice_ndim == 3
                    ):  # Raw slice is [C, H, W] (assuming channels_first for the raw slice)
                        self._raw_slice_channels = sample_slice.shape[0]  # C from [C,H,W]
                        if (
                            self.expected_channels_from_user is not None
                            and self._raw_slice_channels != self.expected_channels_from_user
                        ):
                            print(
                                f"Warning: Raw 3D slices have {self._raw_slice_channels} channels, "
                                f"but expected_channels_from_user is {self.expected_channels_from_user}. "
                                f"Using raw slice's channel count ({self._raw_slice_channels})."
                            )
                        self._target_channels = self._raw_slice_channels  # For 3D slices, use their own channel count
                    else:
                        raise ValueError(
                            f"Unsupported raw slice ndim: {self._raw_slice_ndim} "
                            f"from file {series_filepath} (slice shape {sample_slice.shape}). Expected 2 or 3."
                        )
                    first_valid_file_inspected = True

                # Add all possible starting points for sequences from this file
                for i in range(Nt_series - self.t_n + 1):
                    self.samples_info.append((series_filepath, i))

            except FileNotFoundError:  # Should not happen with glob but good practice
                print(f"Warning: File {series_filepath} not found during processing. Skipping.")
                continue
            except ValueError as ve:  # Handles issues with np.load or unexpected shapes with mmap
                print(f"Warning: Could not process file {series_filepath} due to ValueError: {ve}. Skipping.")
                continue
            except Exception as e:  # Catch other potential errors during file inspection
                print(f"Warning: An unexpected error occurred while processing file {series_filepath}: {e}. Skipping.")
                continue

        if not self.samples_info:
            raise RuntimeError("No valid samples could be generated. Check data_dir, t_n, and file contents.")
        if not first_valid_file_inspected:  # Implies all files were skipped or empty
            raise RuntimeError("Could not determine slice properties. No valid files found or all were unsuitable.")

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, idx):
        if idx >= len(self.samples_info):  # Basic bounds check
            raise IndexError(f"Index {idx} out of bounds for a dataset of size {len(self.samples_info)}.")

        filepath, start_slice_index = self.samples_info[idx]

        try:
            # Use mmap_mode='r' for efficient read-only access.
            # Data is paged into memory as needed.
            # The file descriptor is automatically closed when the 'with' block exits.
            data_array_mmap = np.load(filepath, mmap_mode="r")
            # Slice the sequence: gives [t_n, C, H, W] or or [C, H, W, t_n,]
            # It's important to note that this is a VIEW on the mmap array.
            # To avoid issues with multiprocessing (num_workers > 0) if data
            # is modified or if the mmap is shared unsafely, an explicit copy
            # with np.array() or .copy() is safer BEFORE passing to torch.from_numpy.
            # However, if only reading and subsequent operations create new arrays (like expand_dims),
            # this might be omitted for performance.
            # For safety and clarity, especially with mmap and multiprocessing, a copy is good practice.
            if self.time_first:
                sequence_data = np.array(data_array_mmap[start_slice_index : start_slice_index + self.t_n])
            else:
                sequence_data = np.array(data_array_mmap[..., start_slice_index : start_slice_index + self.t_n])
                sequence_data = np.moveaxis(sequence_data, -1, 0)  # Move time dimension to the front

        except FileNotFoundError:  # Should not occur if __init__ was correct
            print(f"FATAL: File {filepath} not found in __getitem__. Sample index: {idx}")
            raise
        except Exception as e:
            print(f"Error loading or slicing data from {filepath} for sample index {idx}: {e}")
            raise

        # `sequence_data` shape: [t_n, H, W] (if _raw_slice_ndim=2)
        #                     or [t_n, C_raw, H, W] (if _raw_slice_ndim=3)

        if self._raw_slice_ndim == 2:  # Raw slices were [H, W]
            # Expand to [t_n, 1, H, W]
            final_sequence_data = np.expand_dims(sequence_data, axis=1)  # axis=1 for channels
            if self._target_channels > 1:
                final_sequence_data = np.repeat(final_sequence_data, self._target_channels, axis=1)
        elif self._raw_slice_ndim == 3:  # Raw slices were [C_raw, H, W]
            # Shape is already [t_n, C_raw, H, W]. Here, C_raw is self._target_channels for the 3D case.
            final_sequence_data = sequence_data
        else:
            # This case should be caught by __init__ or the check at the start of __getitem__
            raise RuntimeError(
                f"Unexpected raw slice ndim ({self._raw_slice_ndim}) in __getitem__ for sample index {idx}."
            )

        # `final_sequence_data` should now be [t_n, target_C, H, W]
        # Ensure the array is C-contiguous before converting to a tensor,
        # especially if it comes from mmap + slicing/operations.
        if not final_sequence_data.flags["C_CONTIGUOUS"]:
            final_sequence_data = np.ascontiguousarray(final_sequence_data)

        data_tensor = torch.from_numpy(final_sequence_data).float()  # Ensure float type

        return data_tensor
