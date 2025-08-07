import sys
import json
import torch
from tqdm import trange
from pytorch_msssim import MS_SSIM
from utils import get_cr_dirs, HUXDataset, get_hux_pred
from metrics import nnse_score, mssim_score, rmse_score, acc_score, psnr_score


MSSSIM_MODULE = MS_SSIM(
    data_range=1.0,
    size_average=True,
    channel=1,  # after unsqueeze(1)
    spatial_dims=3,  # 3D input
    win_size=7,
    win_sigma=1.0,
)


def main():
    data_path = sys.argv[1]
    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train, cr_val = cr_dirs[:split_ix], cr_dirs[split_ix:]

    train_dataset = HUXDataset(data_path, cr_train)
    val_dataset = HUXDataset(data_path, cr_val)

    climatology = train_dataset.climatology

    running_rmse = 0.0
    running_nnse = 0.0
    running_msssim = 0.0
    running_acc = 0.0
    running_psnr = 0.0

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
        running_rmse += rmse_score(y, pred)
        running_nnse += nnse_score(y, pred, climatology)
        running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
        running_acc += acc_score(y, pred, climatology)
        running_psnr += psnr_score(y, pred)

    val_rmse = running_rmse / len(val_dataset)
    val_nnse = running_nnse / len(val_dataset)
    val_msssim = running_msssim / len(val_dataset)
    val_acc = running_acc / len(val_dataset)
    val_psnr = running_psnr / len(val_dataset)

    with open("hux_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "val_rmse": val_rmse,
                "val_nnse": val_nnse,
                "val_msssim": val_msssim,
                "val_acc": val_acc,
                "val_psnr": val_psnr,
            },
            f,
            indent=4,
        )

    print("Done!")


if __name__ == "__main__":
    main()