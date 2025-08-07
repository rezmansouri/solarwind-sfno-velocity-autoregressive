import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from utils import get_cr_dirs, HUXDataset, get_hux_pred
from metrics import mse_score_per_slice, mse_score_per_slice_masked, sobel_edge_map


def main():
    data_path = sys.argv[1]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_val = cr_dirs[split_ix:]

    val_dataset = HUXDataset(data_path, cr_val)

    mses = []
    masked_mses = []

    for i in trange(len(val_dataset)):
        instance = val_dataset[i]
        v, r, p, t = (
            instance["x"]["v"],
            instance["x"]["r"],
            instance["x"]["p"],
            instance["x"]["t"],
        )
        y = torch.tensor(instance["y"].copy(), dtype=torch.float32).unsqueeze(0)
        pred = get_hux_pred(v, r, p, t)
        pred = torch.tensor(pred.copy(), dtype=torch.float32).unsqueeze(0)
        mask = sobel_edge_map(y)
        mses_batch = mse_score_per_slice(y, pred)
        mses_masked_batch = mse_score_per_slice_masked(y, pred, mask)

        mses.extend(mses_batch.detach().cpu().numpy().tolist())
        masked_mses.extend(mses_masked_batch.detach().cpu().numpy().tolist())

    mses = np.array(mses)
    mse_df_dict = {f"slice_{i+1}": mses[:, i] for i in range(mses.shape[1])}
    df = pd.DataFrame(mse_df_dict)
    df.to_csv("mse_per_slice_all.csv", index=False)
    
    masked_mses = np.array(masked_mses)
    masked_mse_df_dict = {f"slice_{i+1}": masked_mses[:, i] for i in range(masked_mses.shape[1])}
    df = pd.DataFrame(masked_mse_df_dict)
    df.to_csv("masked_mse_per_slice_all.csv", index=False)


    print("Done!")


if __name__ == "__main__":
    main()
