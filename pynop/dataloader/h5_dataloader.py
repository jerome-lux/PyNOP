import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path


class PDEBenchDataSet(Dataset):
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
    step : int, optional
        The stride (step size) used when generating consecutive time windows from a simulation.
        If None, `step` defaults to `T_unroll`, resulting in non-overlapping windows.
    format : str, optional
        Data format specification (currently unused, default is 'pdebench').
    load_in_ram: bool, Default False
    split_type: "train", "val" or None (loads all)
    split_ratio: ratio for the training set
    n_samples: maximum number of samples
    seed: for reproductibility

    To generate train/val sets with distincts samples do:
    train_set = pynop.PDEBenchDataSet(
    datapath, T_unroll=10, step=10, load_in_ram=True, split_type="train", split_ratio=0.9, n_samples=100, seed=42
    )
    val_set = pynop.PDEBenchDataSet(
        datapath, T_unroll=10, step=10, load_in_ram=True, split_type="val", split_ratio=0.9, n_samples=100, seed=42
    )

    """

    def __init__(
        self,
        h5_path,
        T_unroll=10,
        step=None,
        load_in_ram=False,
        split_type=None,
        split_ratio=0.8,
        n_samples=None,
        min_id=0,
        max_id=None,
        seed=42,
    ):

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
                if n_samples is not None:
                    n_samples = min(len(all_sample_ids), n_samples)
                    all_sample_ids = all_sample_ids[:n_samples]

                n_train = int(len(all_sample_ids) * split_ratio)
                if split_type == "train":
                    self.sample_ids = all_sample_ids[:n_train]
                elif split_type == "val":
                    self.sample_ids = all_sample_ids[n_train:]
            else:
                self.sample_ids = all_sample_ids

            for sid in tqdm(self.sample_ids, desc=f"Scanning {split_type or 'all'}"):
                T_total = f[sid]["data"].shape[0]

                current_max_id = max_id if max_id is not None else T_total
                current_max_id = min(current_max_id, T_total)

                if (current_max_id - min_id) < T_unroll:
                    continue

                if self.load_in_ram:
                    raw_data = f[sid]["data"][:]
                    self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

                # Windows in [min_id, current_max_id]
                limit_t0 = current_max_id - T_unroll
                for t0 in range(min_id, limit_t0 + 1, self.step):
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

        return fields, torch.Tensor([t0])


class TheWellDataSet(Dataset):
    """
    A PyTorch Dataset for loading simulation data from 'TheWell' HDF5 format.

    This loader handles multi-trajectory simulations stored in HDF5 files where fields
    are categorized by their tensor rank (t0_fields for scalars, t1_fields for vectors,
    and t2_fields for tensors). It supports time-unrolled window generation,
    selective field loading, and optional RAM caching for accelerated training.

    Args:
            h5_path (str or Path): Path to the HDF5 file containing simulation data.
                Accepts string or pathlib objects.
            fields_list (list, optional): List of specific field names to load
                (e.g., ['velocity', 'pressure']). If None, the loader automatically
                identifies and concatenates all available physical fields from
                t0_fields, t1_fields, and t2_fields.
            T_unroll (int): The number of timesteps to include in each window.
                Defaults to 10.
            step (int, optional): The stride (step size) between the start of
                consecutive windows. If None, it defaults to T_unroll, resulting
                in non-overlapping windows.
            load_in_ram (bool): If True, the requested trajectories and time
                range are loaded into memory for faster access. Defaults to False.
            n_samples (int, optional): Maximum number of trajectories (the 'B'
                dimension) to load from the file. If None, all available
                trajectories are used.
            min_id (int): The starting timestep index (inclusive) to define the
                temporal cropping of the dataset. Defaults to 0.
            max_id (int, optional): The ending timestep index (exclusive) for
                temporal cropping. If None, it defaults to the maximum available
                length of the simulation.
            seed (int): Random seed for reproducibility. Defaults to 42.
    """

    def __init__(
        self,
        h5_path,
        fields_list=None,
        T_unroll=10,
        step=None,
        load_in_ram=False,
        n_samples=None,
        min_id=0,
        max_id=None,
        seed=42,
    ):
        # Conversion du chemin pour compatibilité h5py et extraction du nom
        self.h5_path = str(h5_path)
        file_name = Path(h5_path).name

        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step
        self.load_in_ram = load_in_ram
        self.h5file = None
        self.data_cache = {}
        self.windows = []
        self.field_types = {}

        with h5py.File(self.h5_path, "r") as f:
            # 1. Identification de TOUS les champs disponibles
            available_fields = {}
            for group in ["t0_fields", "t1_fields", "t2_fields"]:
                if group in f:
                    names = f[group].attrs.get("field_names", [])
                    for name in names:
                        available_fields[name] = group[:2]

            if fields_list is None:
                self.fields_list = list(available_fields.keys())
            else:
                self.fields_list = fields_list

            self.field_types = {n: available_fields[n] for n in self.fields_list if n in available_fields}

            # 2. Affichage récapitulatif propre
            print(f"\n{'='*50}")
            print(f" Dataset: {file_name}")
            print(f"{'-'*50}")
            print(f"{'Champ':<20} | {'Type':<5} | {'Shape (B, T, ...)'}")
            print(f"{'-'*50}")
            for fid in self.fields_list:
                path = self._find_field_path(f, fid)
                shape = f[path].shape
                print(f"{fid:<20} | {self.field_types[fid]:<5} | {shape}")
            print(f"{'='*50}\n")

            # 3. Trajectoires et Temps
            total_trajectories = f.attrs["n_trajectories"]
            num_to_load = min(total_trajectories, n_samples) if n_samples else total_trajectories
            self.traj_ids = list(range(num_to_load))

            first_field_path = self._find_field_path(f, self.fields_list[0])
            T_total = f[first_field_path].shape[1]

            actual_max_id = min(max_id if max_id is not None else T_total, T_total)
            self.min_id = min_id

            # 4. Chargement RAM
            if self.load_in_ram:
                for fid in tqdm(self.fields_list, desc="Loading to RAM"):
                    path = self._find_field_path(f, fid)
                    # Slicing optimisé sur la plage temporelle demandée
                    data = f[path][:num_to_load, min_id:actual_max_id, ...]
                    self.data_cache[fid] = torch.from_numpy(data).float()

            # 5. Génération des fenêtres
            limit_t0 = actual_max_id - T_unroll
            for traj_idx in range(num_to_load):
                for t0 in range(min_id, limit_t0 + 1, self.step):
                    self.windows.append((traj_idx, t0))

    def _find_field_path(self, h5_obj, field_name):
        for group in ["t0_fields", "t1_fields", "t2_fields"]:
            if group in h5_obj and field_name in h5_obj[group]:
                return f"{group}/{field_name}"
        raise KeyError(f"Field '{field_name}' not found.")

    def __len__(self):
        return len(self.windows)

    def _init_h5(self):
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, idx):
        traj_idx, t0 = self.windows[idx]
        all_fields = []

        for fid in self.fields_list:
            if self.load_in_ram:
                t_offset = t0 - self.min_id
                chunk = self.data_cache[fid][traj_idx, t_offset : t_offset + self.T_unroll]
            else:
                self._init_h5()
                path = self._find_field_path(self.h5file, fid)
                chunk = torch.from_numpy(self.h5file[path][traj_idx, t0 : t0 + self.T_unroll]).float()

            # Normalisation vers (T, C, W, H)
            if chunk.ndim == 3:  # (T, W, H) -> (T, 1, W, H)
                chunk = chunk.unsqueeze(1)
            elif chunk.ndim == 4:  # (T, W, H, C) -> (T, C, W, H)
                chunk = chunk.permute(0, 3, 1, 2)

            all_fields.append(chunk)

        return torch.cat(all_fields, dim=1), torch.Tensor([t0])


def inspect_the_well_fields(h5_path):
    with h5py.File(h5_path, "r") as f:
        print(f"--- Fichier : {h5_path} ---")
        print(f"Trajectoires totales (B) : {f.attrs.get('n_trajectories')}")

        for group_name in ["t0_fields", "t1_fields", "t2_fields"]:
            if group_name in f:
                group = f[group_name]
                # Récupération de la liste des noms via l'attribut @field_names
                names = group.attrs.get("field_names", [])
                print(f"\nGroup: {group_name} (Réf: {len(names)} champs)")

                for name in names:
                    ds = group[name]
                    print(f"  - {name:<15} | Shape: {ds.shape} | Dtype: {ds.dtype}")
