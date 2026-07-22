import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path


class npyPDEDataset(Dataset):
    """Loads simulation data from a single .npy file (4D/5D) directly in RAM"""

    def __init__(
        self,
        npy_path: str,
        T_unroll: int = 10,
        step: int = None,
        time_binning: int = 1,
        indexes: list = None,
        min_id: int = 0,
        max_id: int = None,
        shift: float = 0.0,
        scale: float = 1.0,
    ):
        """
        Parameters
        ----------
        npy_path : str
            Path to the .npy file. Supports (Nsample, Nt, Nx, Ny) or (Nsample, Nt, Nx, Ny, C).
        T_unroll : int
            Length of the time window returned to the model.
        step : int
            Stride between window starts, applied AFTER time binning. Defaults to T_unroll.
        time_binning : int
            Take 1 frame every 'time_binning' frames. Applied directly at loading. Default is 1.
        indexes : list
            List of sample IDs to include in this split.
        min_id : int
            Start frame index (applied on the BINNED time axis).
        max_id : int
            End frame index (applied on the BINNED time axis).
        shift : float
            Value added to the data for normalization.
        scale : float
            Value multiplied to the data for normalization.
        """
        if T_unroll < 1:
            T_unroll = 1
        if time_binning < 1:
            time_binning = 1

        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step

        # 1. Load completely into RAM and apply time binning immediately
        raw_data = np.load(npy_path)
        self.ndims = raw_data.ndim

        if self.ndims == 4:
            # Shape: [Nsample, Nt_binned, Nx, Ny]
            self.data = raw_data[:, ::time_binning, :, :]
        elif self.ndims == 5:
            # Shape: [Nsample, Nt_binned, Nx, Ny, C]
            self.data = raw_data[:, ::time_binning, :, :, :]
        else:
            raise ValueError(f"Unsupported array shape {raw_data.shape}. Expected 4D or 5D array.")

        self.sample_ids = indexes if indexes is not None else list(range(self.data.shape[0]))

        self.windows = []
        self.shift = self._prepare_transform(shift)
        self.scale = self._prepare_transform(scale)

        # 2. Build windows on the binned timeline
        for sid in self.sample_ids:
            T_binned = self.data.shape[1]
            current_max_id = min(max_id if max_id is not None else T_binned, T_binned)

            if (current_max_id - min_id) < T_unroll:
                continue

            limit_t0 = current_max_id - T_unroll
            for t0 in range(min_id, limit_t0 + 1, self.step):
                self.windows.append((sid, t0))

        print(f"Dataset initialized with {len(self.sample_ids)} simulations (RAM Mode - Binning: 1/{time_binning}).")
        print(f"Total unrolled windows: {len(self.windows)}")

    def _prepare_transform(self, val):
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            tensor_val = torch.as_tensor(val).float()
            if tensor_val.ndim == 1:
                return tensor_val.view(1, -1, 1, 1)
            return tensor_val
        return val

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sid, t0 = self.windows[idx]

        # Extract the sequence from the already binned tensor (O(1) memory view)
        seq = self.data[sid, t0 : t0 + self.T_unroll]
        fields = torch.from_numpy(seq).float()

        if self.ndims == 4:
            # [T_unroll, Nx, Ny] -> [T_unroll, 1, Nx, Ny]
            fields = fields.unsqueeze(1)
        else:
            # [T_unroll, Nx, Ny, C] -> [T_unroll, C, Nx, Ny]
            fields = fields.permute(0, 3, 1, 2)

        return (fields + self.shift) * self.scale, torch.tensor([t0], dtype=torch.float32)


class NSH5Dataset(Dataset):
    """
    H5 dataset tailored to match PDEBenchDataSet behavior with 'u' and 'a' fields.

    HDF5 structure:
        [split]/u : (N, T, H, W)
        [split]/a : (N, H, W) ou (N, cond_dim, H, W) -> Static conditioning field

    Returns:
        u        : [T_unroll, 1, H, W]
        time_idx : [1] (t0 as torch.float32)
        a        : [cond_dim, H, W] (static conditioning field)
    """

    def __init__(
        self,
        h5_path,
        split="train",
        T_unroll=10,
        step=None,
        load_in_ram=False,
        indexes=None,
        min_id=0,
        max_id=None,
        shift=0.0,
        scale=1.0,
    ):
        """
        Initializes the dataset, synchronizes temporal filtering, and prepares transforms.

        Parameters
        ----------
        h5_path : str
            Path to the HDF5 file.
        split : str
            Dataset split, either 'train' or 'valid'.
        T_unroll : int
            Number of time steps to unroll.
        step : int, optional
            Stride between time windows.
        load_in_ram : bool
            Whether to pre-load selected samples into memory.
        indexes : list of int, optional
            List of requested global sample indices (will be sorted).
        min_id : int
            Minimum time index allowed for windows.
        max_id : int, optional
            Maximum time index allowed for windows.
        shift : float or list or np.ndarray or torch.Tensor
            Value added to the tensors for normalization.
        scale : float or list or np.ndarray or torch.Tensor
            Value multiplied to the tensors for normalization.
        """
        if T_unroll < 1:
            print("Warning: T_unroll must be at least 1. Setting to 1.")
            T_unroll = 1

        self.h5_path = h5_path
        self.split = split
        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step
        self.load_in_ram = load_in_ram

        self.h5file = None
        self.windows = []

        self.shift = shift
        self.scale = scale

        with h5py.File(h5_path, "r") as f:
            u = f[split]["u"]
            a = f[split]["a"]

            global_N, self.T, self.H, self.W = u.shape

            # D�termination de la structure de a (statique)
            # Si len(shape) == 3 -> (N, H, W), il faut ajouter l'axe cond_dim=1
            # Si len(shape) == 4 -> (N, cond_dim, H, W), cond_dim est d�j� pr�sent
            self.a_has_channel = len(a.shape) == 4

            # Alignement du filtrage temporel
            current_max_id = max_id if max_id is not None else self.T
            current_max_id = min(current_max_id, self.T)

            if indexes is not None:
                self.sample_ids = sorted([int(i) for i in indexes if i < global_N])
                self.N = len(self.sample_ids)
            else:
                self.sample_ids = list(range(global_N))
                self.N = global_N

            if load_in_ram:
                raw_u = torch.from_numpy(u[self.sample_ids]).float()  # [N, T, H, W]
                raw_a = torch.from_numpy(a[self.sample_ids]).float()  # [N, H, W] ou [N, cond_dim, H, W]

                self.u_dict = raw_u.unsqueeze(2)  # [N, T, 1, H, W]

                if self.a_has_channel:
                    self.a_dict = raw_a  # [N, cond_dim, H, W]
                else:
                    self.a_dict = raw_a.unsqueeze(1)  # [N, 1, H, W]
            else:
                self.u_dict = None
                self.a_dict = None

            # G�n�ration des fen�tres temporelles
            limit_t0 = current_max_id - T_unroll
            for local_id, global_id in enumerate(self.sample_ids):
                for t0 in range(min_id, limit_t0 + 1, self.step):
                    sample_idx_to_store = local_id if load_in_ram else global_id
                    self.windows.append((sample_idx_to_store, t0))

        print(f"NSH5Dataset [{split}] initialized with {self.N} simulations.")
        print(f"Total unrolled windows: {len(self.windows)}")
        print(f"Status: {'RAM' if load_in_ram else 'Disk'} mode.")

    def __len__(self):
        return len(self.windows)

    def _init_h5(self):
        """Initializes the HDF5 file object for thread-safe disk reading."""
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, idx):
        """
        Retrieves a window and returns u, time_idx, and a.
        """
        sample_id, t0 = self.windows[idx]

        if self.load_in_ram:
            fields_u = self.u_dict[sample_id, t0 : t0 + self.T_unroll]  # [T_unroll, 1, H, W]
            fields_a = self.a_dict[sample_id]  # [cond_dim, H, W]
        else:
            self._init_h5()

            # Lecture de la fen�tre u
            raw_u = self.h5file[self.split]["u"][sample_id, t0 : t0 + self.T_unroll]
            fields_u = torch.from_numpy(raw_u).float().unsqueeze(1)  # [T_unroll, 1, H, W]

            # Lecture du champ statique a
            raw_a = self.h5file[self.split]["a"][sample_id]
            fields_a = torch.from_numpy(raw_a).float()

            if not self.a_has_channel:
                fields_a = fields_a.unsqueeze(0)  # [1, H, W] si pas de canal natif

        # Normalisations
        u_out = (fields_u + self.shift) * self.scale
        a_out = (fields_a + self.shift) * self.scale
        time_idx = torch.tensor([t0], dtype=torch.float32)

        return u_out, time_idx, a_out


class PDEBenchDataSet(Dataset):
    """
    Loads simulation data from an HDF5 file and generates time-unrolled windows.
    """

    def __init__(
        self,
        h5_path,
        T_unroll=10,
        step=None,
        load_in_ram=False,
        indexes=None,
        min_id=0,
        max_id=None,
        shift=0.0,
        scale=1.0,
    ):
        """
        Args:
            h5_path: Path to HDF5 file.
            T_unroll: Length of time window.
            step: Stride for windows.
            load_in_ram: Load data into memory.
            indexes: List of sample IDs (keys in HDF5) to process.
                     If None, processes all available keys.
            min_id: Start index for windows.
            max_id: End index for windows.
            shift: Input shift.
            scale: Input scale.
        """

        if T_unroll < 1:
            print("Warning: T_unroll must be at least 1. Setting to 1.")
            T_unroll = 1

        self.h5_path = h5_path
        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step
        self.load_in_ram = load_in_ram
        self.h5file = None
        self.data_dict = {}
        self.windows = []
        self.max_t_idx = 0
        self.shift = self._prepare_transform(shift)
        self.scale = self._prepare_transform(scale)

        with h5py.File(h5_path, "r") as f:
            # Use provided indexes or default to all keys
            all_keys = list(f.keys())
            self.sample_ids = indexes if indexes is not None else all_keys

            for sid in tqdm(self.sample_ids, desc="Scanning samples"):
                # Safety check: ensure index exists
                if sid not in f:
                    print(f"Warning: {sid} not found in HDF5 file. Skipping.")
                    continue

                T_total = f[sid]["data"].shape[0]
                self.max_t_idx = max(self.max_t_idx, T_total)

                current_max_id = max_id if max_id is not None else T_total
                current_max_id = min(current_max_id, T_total)

                if (current_max_id - min_id) < T_unroll:
                    continue

                if self.load_in_ram:
                    raw_data = f[sid]["data"][:]
                    self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

                # Generate windows within [min_id, current_max_id]
                limit_t0 = current_max_id - T_unroll
                for t0 in range(min_id, limit_t0 + 1, self.step):
                    self.windows.append((sid, t0))

        print(f"Dataset initialized with {len(self.sample_ids)} simulations.")
        print(f"Total unrolled windows: {len(self.windows)}")
        print(f"Status: {'RAM' if self.load_in_ram else 'Disk'} mode.")

    def _prepare_transform(self, val):
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            tensor_val = torch.as_tensor(val).float()
            if tensor_val.ndim == 1:
                return tensor_val.view(1, -1, 1, 1)
            return tensor_val
        return val

    def __len__(self):
        return len(self.windows)

    def _init_h5(self):
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, idx):
        sid, t0 = self.windows[idx]

        if self.load_in_ram:
            fields = self.data_dict[sid][t0 : t0 + self.T_unroll]
        else:
            self._init_h5()
            fields = self.h5file[sid]["data"][t0 : t0 + self.T_unroll]
            fields = torch.from_numpy(np.moveaxis(fields, -1, 1)).float()

        return (fields + self.shift) * self.scale, torch.Tensor([t0])


class RDDataSet(Dataset):

    # Load the field in the data field as well as the porosity field
    def __init__(
        self,
        h5_path,
        T_unroll=10,
        step=None,
        load_in_ram=False,
        indexes=None,
        min_id=0,
        max_id=None,
        shift=0.0,
        scale=1.0,
    ):

        if T_unroll < 1:
            print("Warning: T_unroll must be at least 1. Setting to 1.")
            T_unroll = 1

        self.h5_path = h5_path
        self.T_unroll = T_unroll
        self.step = T_unroll if step is None else step
        self.load_in_ram = load_in_ram
        self.h5file = None
        self.data_dict = {}
        self.porosity_dict = {}
        self.windows = []
        self.shift = self._prepare_transform(shift)
        self.scale = self._prepare_transform(scale)

        with h5py.File(h5_path, "r") as f:
            all_keys = list(f.keys())
            self.sample_ids = indexes if indexes is not None else all_keys

            for sid in tqdm(self.sample_ids, desc="Scanning samples"):
                if sid not in f:
                    continue

                # On vérifie la cohérence temporelle entre data et porosity
                T_total = f[sid]["data"].shape[0]
                assert f[sid]["porosity"].shape[0] == T_total, f"Time mismatch in {sid}"

                current_max_id = min(max_id if max_id is not None else T_total, T_total)
                if (current_max_id - min_id) < T_unroll:
                    continue

                if self.load_in_ram:
                    # Load main field (T, C, H, W)
                    raw_data = f[sid]["data"][:]
                    self.data_dict[sid] = torch.from_numpy(np.moveaxis(raw_data, -1, 1)).float()

                    raw_p = f[sid]["porosity"][:]
                    # Add channel dim if missing: (T, H, W) -> (T, 1, H, W)
                    p_tensor = torch.from_numpy(raw_p).float()
                    if p_tensor.ndim == 3:  # (T, H, W)
                        p_tensor = p_tensor.unsqueeze(1)
                    elif p_tensor.ndim == 4:  # (T, H, W, C)
                        p_tensor = p_tensor.permute(0, 3, 1, 2)
                    self.porosity_dict[sid] = p_tensor

                limit_t0 = current_max_id - T_unroll
                for t0 in range(min_id, limit_t0 + 1, self.step):
                    self.windows.append((sid, t0))

    def _prepare_transform(self, val):
        if isinstance(val, (list, np.ndarray, torch.Tensor)):
            tensor_val = torch.as_tensor(val).float()
            if tensor_val.ndim == 1:
                return tensor_val.view(1, -1, 1, 1)
            return tensor_val
        return val

    def __len__(self):
        return len(self.windows)

    def _init_h5(self):
        if self.h5file is None and not self.load_in_ram:
            self.h5file = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, idx):
        sid, t0 = self.windows[idx]

        if self.load_in_ram:
            fields = self.data_dict[sid][t0 : t0 + self.T_unroll]
            p = self.porosity_dict[sid][t0 : t0 + self.T_unroll]
            fields = torch.cat([fields, p], dim=1)  # Concat on Channel dim
        else:
            self._init_h5()
            # Get data slice
            raw_f = self.h5file[sid]["data"][t0 : t0 + self.T_unroll]
            fields = torch.from_numpy(np.moveaxis(raw_f, -1, 1)).float()

            raw_p = self.h5file[sid]["porosity"][t0 : t0 + self.T_unroll]
            p = torch.from_numpy(raw_p).float()
            # Handle axis movement according to shape
            if p.ndim == 3:  # (T_unroll, H, W)
                p = p.unsqueeze(1)
            else:  # (T_unroll, H, W, C)
                p = p.permute(0, 3, 1, 2)
            fields = torch.cat([fields, p], dim=1)

        return (fields + self.shift) * self.scale, torch.Tensor([t0])


def inspect_PDEBenchDataset(h5_path, DT=1.0):
    """
    Scans a PDEBench HDF5 file to extract physical, structural,
    and temporal derivative (velocity) statistics.
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
