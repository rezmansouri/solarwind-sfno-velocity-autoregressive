import torch
import math
import numpy as np
from tqdm import tqdm
from pyhdf.SD import SD, SDC
from torch.utils.data import Dataset
from os.path import join as path_join
from neuralop import LpLoss
import os

FILE_NAMES = ["vr002.hdf"]

DEFAULT_INSTRUMENTS = [
    "kpo_mas_mas_std_0101",
    "mdi_mas_mas_std_0101",
    "hmi_mast_mas_std_0101",
    "hmi_mast_mas_std_0201",
    "hmi_masp_mas_std_0201",
    "mdi_mas_mas_std_0201",
]


def read_hdf(hdf_path, dataset_names):
    f = SD(hdf_path, SDC.READ)
    datasets = []
    for dataset_name in dataset_names:
        datasets.append(f.select(dataset_name).get())
    return datasets


def get_sim(sim_path):
    (v_path,) = [path_join(sim_path, file_name) for file_name in FILE_NAMES]
    v = read_hdf(v_path, ["Data-Set-2"])[0]
    v = v.transpose(2, 1, 0)

    return v


def get_sims(sim_paths):
    sims = []
    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        sims.append(get_sim(sim_path))
    sims = np.stack(sims, axis=0)
    return sims


def min_max_normalize(array, min_=None, max_=None):
    if min_ is None or max_ is None:
        min_ = np.min(array)
        max_ = np.max(array)
    array = (array - min_) / (max_ - min_)
    return array, min_, max_


def compute_climatology(data: np.ndarray) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (n, 139, 111, 128)

    Returns:
        np.ndarray: Climatology array of shape (139, 111, 128)
    """
    assert data.ndim == 4 and data.shape[1:] == (
        139,
        111,
        128,
    ), "Unexpected input shape."
    climatology = np.mean(data, axis=0)
    climatology = torch.tensor(climatology, dtype=torch.float32)
    return climatology


def get_cr_dirs(data_path):
    """Return list of CR directories (crXXXX) inside data_path."""
    cr_dirs = sorted(
        [
            d
            for d in os.listdir(data_path)
            if d.startswith("cr") and os.path.isdir(os.path.join(data_path, d))
        ]
    )
    return cr_dirs


def collect_sim_paths(data_path, cr_list, instruments=None):
    """Collect simulation paths given a list of CR directories."""
    if instruments is None:
        instruments = DEFAULT_INSTRUMENTS

    sim_paths = []
    for cr in cr_list:
        cr_path = os.path.join(data_path, cr)
        for instrument in instruments:
            instrument_path = os.path.join(cr_path, instrument)
            if os.path.exists(instrument_path):
                sim_paths.append(instrument_path)
    return sim_paths


class SphericalNODataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        v_min=None,
        v_max=None,
        instruments=None,
    ):
        super().__init__()
        sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        self.sim_paths = sim_paths
        sims = get_sims(sim_paths)
        sims, self.v_min, self.v_max = min_max_normalize(sims, v_min, v_max)
        self.sims = sims
        self.climatology = compute_climatology(sims[:, 1:, :, :])

    def __getitem__(self, index):
        cube = self.sims[index]
        path = self.sim_paths[index]
        return torch.tensor(cube, dtype=torch.float32), path

    def __len__(self):
        return len(self.sims)

    def get_min_max(self):
        return {"v_min": float(self.v_min), "v_max": float(self.v_max)}


class AreaWeightedLpLoss(LpLoss):
    """
    LpLoss with optional area weighting using sin(θ) for spherical latitude grids.
    """

    def __init__(self, d=2, p=2, measure=1.0, reduction="sum", area_weighted=True):
        super().__init__(d, p, measure, reduction)
        self.area_weighted = area_weighted

    def get_area_weights(self, x):
        """
        Returns area weights per latitude assuming equiangular grid.
        Assumes the latitude is at axis -2.
        """
        H = x.size(-2)
        theta = torch.linspace(0, np.pi, H, device=x.device).view(1, 1, H, 1)
        area_weights = torch.cos(theta) ** 2  # shape (1, 1, H, 1)
        return area_weights

    def rel(self, x, y):
        """
        Relative Lp loss with optional area weighting.
        """
        diff = x - y
        if self.area_weighted:
            area_weights = self.get_area_weights(diff)
            diff = diff * area_weights
            y = y * area_weights

        diff = torch.norm(
            torch.flatten(diff, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm
        diff = self.reduce_all(diff).squeeze()
        return diff


class L1L2Loss:
    """
    L1L2Loss combines relative L1 and L2 losses over spatial dimensions.
    Useful when you want to balance between L1 robustness and L2 smoothness.

    Parameters
    ----------
    d : int
        Number of spatial dimensions
    measure : float or list
        Domain size for quadrature weighting
    reduction : str
        'sum' or 'mean' over batch/channel dims
    alpha : float
        Weight for L1 loss (default 1.0)
    beta : float
        Weight for L2 loss (default 1.0)
    """

    def __init__(self, d, measure=1.0, reduction="sum", alpha=1.0, beta=1.0):
        self.l1 = LpLoss(d=d, p=1, measure=measure, reduction=reduction)
        self.l2 = LpLoss(d=d, p=2, measure=measure, reduction=reduction)
        self.alpha = alpha
        self.beta = beta

    def __call__(self, y_pred, y):
        l1_loss = self.l1.rel(y_pred, y)
        l2_loss = self.l2.rel(y_pred, y)
        return self.alpha * l1_loss + self.beta * l2_loss
