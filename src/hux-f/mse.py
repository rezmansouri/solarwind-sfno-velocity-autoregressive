import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from utils import get_cr_dirs, HUXDataset, get_hux_pred
from metrics import mse_score_per_sample, mse_score_per_sample_masked, sobel_edge_map


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
        mses_batch = mse_score_per_sample(y, pred)
        mses_masked_batch = mse_score_per_sample_masked(y, pred, mask)

        mses.extend(mses_batch.detach().cpu().numpy().tolist())
        masked_mses.extend(mses_masked_batch.detach().cpu().numpy().tolist())

    mse = np.sum(mses) / len(val_dataset)
    masked_mse = np.sum(masked_mses) / len(val_dataset)

    with open("hux_evaluation_mse.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mse": float(mse),
                "masked_mse": float(masked_mse),
            },
            f,
            indent=4,
        )

    df = pd.DataFrame(
        {
            "mse": mses,
            "masked_mse": masked_mses,
        }
    )
    df.to_csv("hux_evaluation_mse.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
