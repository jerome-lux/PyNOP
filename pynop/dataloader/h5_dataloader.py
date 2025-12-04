import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


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
