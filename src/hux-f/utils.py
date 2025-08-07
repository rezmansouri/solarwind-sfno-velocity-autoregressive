import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from psipy.model import MASOutput
from hux_code.hux_propagation import apply_hux_f_model
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


def get_sim(sim_path):
    model = MASOutput(sim_path)

    vr_model = model["vr"]

    p = vr_model.phi_coords
    t = vr_model.theta_coords
    r = vr_model.r_coords
    v = vr_model.data.squeeze()

    return v, r, p, t


def get_sims(sim_paths):
    vs = []
    rs = []
    ps = []
    ts = []
    for sim_path in tqdm(sim_paths, desc="Loading simulations"):
        v, r, p, t = get_sim(sim_path)
        vs.append(v)
        rs.append(r)
        ps.append(p)
        ts.append(t)

    vs = np.stack(vs, axis=0)
    rs = np.stack(rs, axis=0)
    ps = np.stack(ps, axis=0)
    ts = np.stack(ts, axis=0)
    return vs, rs, ps, ts


def compute_climatology(data: np.ndarray) -> np.ndarray:
    """
    Compute per-voxel climatology (mean field) from a dataset.

    Args:
        data (np.ndarray): Array of shape (n, 139, 111, 128)

    Returns:
        np.ndarray: Climatology array of shape (139, 111, 128)
    """
    assert data.ndim == 4 and data.shape[1:] == (
        128,
        111,
        140,
    ), "Unexpected input shape."
    data = np.transpose(data, (0, 3, 2, 1))  # Transpose to (B, 140, 111, 128)
    data = data[:, 1:, ::-1, :]
    assert data.shape[1:] == (139, 111, 128), "Unexpected shape after transpose."
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


class HUXDataset(Dataset):
    def __init__(
        self,
        data_path,
        cr_list,
        instruments=None,
    ):
        super().__init__()
        sim_paths = collect_sim_paths(data_path, cr_list, instruments)
        vs, rs, ps, ts = get_sims(sim_paths)
        self.vs = vs
        self.rs = rs
        self.ps = ps
        self.ts = ts
        self.climatology = compute_climatology(vs)

    def __getitem__(self, index):
        v = self.vs[index]
        p = self.ps[index]
        t = self.ts[index]
        r = self.rs[index]
        y = np.transpose(v, (2, 1, 0))  # Transpose to (140, 111, 128)
        y = y[1:, ::-1, :]
        return {
            "x": {
                "v": v,
                "p": p,
                "t": t,
                "r": r,
            },
            "y": y,
        }

    def __len__(self):
        return len(self.vs)


def get_hux_pred(f, r, p, t):
    r_plot = (695700) * r
    dr_vec = r_plot[1:] - r_plot[:-1]
    dp_vec = p[1:] - p[:-1]

    dr_vec = np.array(dr_vec, dtype=np.float32)
    dp_vec = np.array(dp_vec, dtype=np.float32)

    hux_f_res = np.ones((np.shape(f)[0], np.shape(f)[1], np.shape(f)[2]))
    for ii in range(len(t)):
        hux_f_res[:, ii, :] = apply_hux_f_model(f[:, ii, 0], dr_vec, dp_vec).T
    result = np.transpose(hux_f_res, [2, 1, 0])
    return result[1:, ::-1, :]
