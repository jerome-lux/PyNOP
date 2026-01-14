import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, data_key="data", timesteps=None):

        super().__init__()
        self.hdf5_path = hdf5_path
        self.data_key = data_key
        self.h5file = None
        self.timesteps = timesteps

        with h5py.File(self.hdf5_path, "r") as h5_object:
            self.sample_ids = list(h5_object.keys())
            self.num_samples = len(self.sample_ids)

            if self.num_samples > 0:
                first_id = self.sample_ids[0]
                self.data_shape = h5_object[first_id][self.data_key].shape
                print(f"Dataset initialisé. Nombre d'échantillons: {self.num_samples}")
                print(f"Shape par échantillon: {self.data_shape}")

            else:
                print(f"hdf5 file {hdf5_path}  is empty")
        if self.timesteps is not None:
            if self.timesteps > self.data_shape[0]:
                raise ValueError(
                    f"Number of timesteps ({self.timesteps}) greater than the full number of time steps ({self.data_shape[0]})."
                )

    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):

        if self.h5file is None:
            self.h5file = h5py.File(self.hdf5_path, "r")

        sample_id = self.sample_ids[idx]
        dataset = self.h5file[sample_id][self.data_key]
        nt = dataset.shape[0]

        # Return data[t0, t0+timesteps, ...], t0, dataset idx
        if self.timesteps is not None:
            max_start = nt - self.timesteps
            start_index = np.random.randint(0, max_start + 1)
            # 3. Définir l'indice de fin (stop)
            stop_index = start_index + self.timesteps
            return torch.from_numpy(dataset[start_index:stop_index, :, :, :]), start_index, idx
        else:
            return torch.from_numpy(dataset[:]), 0, idx


class UnrolledH5Dataset(Dataset):
    """
    Loads simulation data from an HDF5 file and generates time-unrolled windows.
    Optionally loads the entire dataset into RAM for faster training.
    when data is loaded from disk, there can be problems in torch.DataLoader when num_workers>0

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file containing the simulation data. Each group within the HDF5 file
        is expected to represent a single simulation run, containing a 'data' key.
    T_unroll : int, optional
        The length of the time window (number of timesteps) to extract from each simulation.
        Defaults to 10.
    step : int, optional
        The stride (step size) used when generating consecutive time windows from a simulation.
        If None, `step` defaults to `T_unroll`, resulting in non-overlapping windows.
    format : str, optional
        Data format specification (currently unused, default is 'pdebench').
    load_in_ram: bool, Default False
    split_type: "train", "val" or None (loads all)
    split_ratio: ratio for the training set
    seed: for reproductibility
    """

    def __init__(self, h5_path, T_unroll=10, step=None, load_in_ram=False, split_type=None, split_ratio=0.8, seed=42):

        if T_unroll < 1:
            print("Warning, T_unroll must ba at least 1. Setting it to 1 to continue")
            T_unroll = 1

        self.h5_path = h5_path
        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step
        self.load_in_ram = load_in_ram
        self.h5file = None
        self.data_dict = {}
        self.windows = []

        with h5py.File(h5_path, "r") as f:
            all_sample_ids = sorted(list(f.keys()))

            if split_type is not None:
                import random

                random.seed(seed)
                random.shuffle(all_sample_ids)

                n_train = int(len(all_sample_ids) * split_ratio)
                if split_type == "train":
                    self.sample_ids = all_sample_ids[:n_train]
                elif split_type == "val":
                    self.sample_ids = all_sample_ids[n_train:]
            else:
                self.sample_ids = all_sample_ids

            for sid in tqdm(self.sample_ids, desc=f"Scanning {split_type or 'all'}"):
                T = f[sid]["data"].shape[0]
                if T < T_unroll:
                    continue

                if self.load_in_ram:
                    raw_data = f[sid]["data"][:]
                    self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

                # Génération des fenêtres pour ce SID spécifique
                max_t0 = T - T_unroll
                for t0 in range(0, max_t0 + 1, self.step):
                    self.windows.append((sid, t0))

        print(f"Split {split_type}: {len(self.sample_ids)} simulations")
        print(f"Total unrolled windows: {len(self.windows)}")
        if self.load_in_ram:
            print("Status: All data loaded in RAM.")
        else:
            print("Status: Reading from Disk (Lazy Loading).")

    def __len__(self):
        return len(self.windows)

    def _init_h5(self):
        """Initialise le file handle pour le multi-processing (Disk mode uniquement)."""
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, idx):
        sid, t0 = self.windows[idx]

        if self.load_in_ram:
            # --- RAM ---
            fields = self.data_dict[sid][t0 : t0 + self.T_unroll]
        else:
            # --- DISK ---
            self._init_h5()
            fields = self.h5file[sid]["data"][t0 : t0 + self.T_unroll]
            # (T, X, Y, C) -> (T, C, X, Y)
            fields = torch.from_numpy(np.moveaxis(fields, -1, 1)).float()

        return fields, t0
