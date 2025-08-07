import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from utils import get_cr_dirs, HUXDataset, get_hux_pred
from metrics import uqi_per_slice


def main():
    data_path = sys.argv[1]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_val = cr_dirs[split_ix:]

    val_dataset = HUXDataset(data_path, cr_val)

    uqis = []

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

        uqis_batch = uqi_per_slice(y, pred)

        uqis.extend(uqis_batch.tolist())

    uqis = np.mean(uqis, axis=0)

    uqi = np.mean(uqis, axis=0)

    with open(
        "hux_evaluation_uqi_per_slice.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "uqi": float(uqi),
            },
            f,
            indent=4,
        )

    df = pd.DataFrame({"uqi": uqis})
    df.to_csv("hux_evaluation_uqi_per_slice.csv", index=False)

    print("Done!")


if __name__ == "__main__":
    main()
