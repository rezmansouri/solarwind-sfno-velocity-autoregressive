import sys
import torch
import json
from neuralop.losses import LpLoss
from trainer import train_cv
from utils import get_cr_dirs

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    data_path, batch_size, n_epochs, experiment_name = sys.argv[1:]
    batch_size, n_epochs = (int(batch_size), int(n_epochs))

    cr_dirs = get_cr_dirs(data_path)
    split_ix = int(len(cr_dirs) * 0.8)
    cr_train = cr_dirs[:split_ix]

    hyperparams = {
        "hidden_channels": [64, 128, 256],
        "n_layers": [4, 8],
    }

    print("search space:\n", hyperparams, "experiment name:", experiment_name)

    loss_fn = LpLoss(d=2, p=2, reduction="sum")

    results = train_cv(
        data_path,
        cr_train,
        hyperparams,
        n_splits=5,
        n_epochs=n_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        device=device,
    )

    with open(f"{experiment_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("Training completed. Results saved to results.json.")


if __name__ == "__main__":
    main()
