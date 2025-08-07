import itertools
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from neuralop.models import SFNO
import torch
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm, trange
from pytorch_msssim import MS_SSIM
import torch.optim as optim
from metrics import nnse_score, mssim_score, rmse_score, acc_score, psnr_score


MSSSIM_MODULE = MS_SSIM(
    data_range=1.0,
    size_average=True,
    channel=1,  # after unsqueeze(1)
    spatial_dims=3,  # 3D input
    win_size=7,
    win_sigma=1.0,
)


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    n_epochs: int,
    batch_size: int,
    loss_fn: nn.Module,
    buffer: int,
    device: str,
    lr: float = 8e-4,
    weight_decay: float = 0.0,
    verbose=True,
):
    """
    Train a model on train_dataset, validate on val_dataset.

    Returns
    -------
    train_losses : list of float
    val_losses : list of float
    best_epoch : int
    best_state_dict : model weights at best epoch
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    autocast_device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler(device=autocast_device_type)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None

    train_losses = []
    val_losses = []
    train_rmse = []
    val_rmse = []
    train_nnse = []
    val_nnse = []
    train_msssim = []
    val_msssim = []
    train_acc = []
    val_acc = []
    train_psnr = []
    val_psnr = []

    climatology = train_dataset.climatology.to(device)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_nnse = 0.0
        running_msssim = 0.0
        running_acc = 0.0
        running_psnr = 0.0

        for cube, paths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False):
            pred = []
            t = cube.shape[1]  # 140

            # Use starting indices for x
            x_indices = list(range(0, t - 1, buffer))  # [0, 20, 40, 60, 80, 100, 120]

            for i in trange(len(x_indices), leave=False):
                x_idx = x_indices[i]
                y_start = x_idx + 1
                y_end = min(x_idx + buffer + 1, t)  # make sure we don't go out of bounds

                x = cube[:, x_idx:x_idx+1, :, :].to(device)              # (B, 1, H, W)
                y = cube[:, y_start:y_end, :, :].to(device)              # (B, <buffer>, H, W)
                
                pred_buf = model(x)[:, :y.shape[1], :, :]                                      # should return same shape as y
                
                # print(x_idx, x_idx+1, pred_buf.shape)
                # print(y_start, y_end, y.shape)
                
                pred.append(pred_buf)

                optimizer.zero_grad()
                loss = loss_fn(pred_buf, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            # Concatenate predictions and evaluate metrics
            pred = torch.cat(pred, dim=1)  # (B, T-1, H, W)
            y = cube[:, 1:, :, :].to(device)  # ground truth full sequence
            
            # print(pred.shape, y.shape)

            running_rmse += rmse_score(y, pred)
            running_nnse += nnse_score(y, pred, climatology)
            running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
            running_acc += acc_score(y, pred, climatology)
            running_psnr += psnr_score(y, pred)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_rmse = running_rmse / len(train_loader)
        epoch_train_nnse = running_nnse / len(train_loader)
        epoch_train_msssim = running_msssim / len(train_loader)
        epoch_train_acc = running_acc / len(train_loader)
        epoch_train_psnr = running_psnr / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_rmse.append(epoch_train_rmse)
        train_nnse.append(epoch_train_nnse)
        train_msssim.append(epoch_train_msssim)
        train_acc.append(epoch_train_acc)
        train_psnr.append(epoch_train_psnr)

        scheduler.step(epoch_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        running_rmse = 0.0
        running_nnse = 0.0
        running_msssim = 0.0
        running_acc = 0.0
        running_psnr = 0.0

        with torch.no_grad():
            for cube, paths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False):
                t = cube.shape[1]  # 140
                x = cube[:, 0:1, :, :].to(device)  # initial input
                pred = []

                # Step through time autoregressively
                for i in trange(0, t - 1, buffer, leave=False):
                    y_start = i + 1
                    y_end = min(i + buffer + 1, t)

                    y = cube[:, y_start:y_end, :, :].to(device)  # target: ground truth next slices
                    
                    # print(y_start, y_end, y.shape)
                    pred_buf = model(x)[:, :y.shape[1], :, :]                          # model predicts next <buffer> steps

                    pred.append(pred_buf)
                    val_loss += loss_fn(pred_buf, y).item()

                    # next input is last predicted frame (1-step only)
                    x = pred_buf[:, -1:, :, :].detach()

                # stack predictions and evaluate full metrics
                pred = torch.cat(pred, dim=1)  # shape (B, T-1, H, W)
                y = cube[:, 1:, :, :].to(device)
                
                # print(pred.shape, y.shape)

                running_rmse += rmse_score(y, pred)
                running_nnse += nnse_score(y, pred, climatology)
                running_msssim += mssim_score(MSSSIM_MODULE, y, pred)
                running_acc += acc_score(y, pred, climatology)
                running_psnr += psnr_score(y, pred)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_rmse = running_rmse / len(val_loader)
        epoch_val_nnse = running_nnse / len(val_loader)
        epoch_val_msssim = running_msssim / len(val_loader)
        epoch_val_acc = running_acc / len(val_loader)
        epoch_val_psnr = running_psnr / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_rmse.append(epoch_val_rmse)
        val_nnse.append(epoch_val_nnse)
        val_msssim.append(epoch_val_msssim)
        val_acc.append(epoch_val_acc)
        val_psnr.append(epoch_val_psnr)

        if verbose:
            print(
                f"Epoch {epoch+1}:\n",
                f"Train Loss = {epoch_train_loss:.6f} | Val Loss = {epoch_val_loss:.6f}\n",
                f"Train RMSE = {epoch_train_rmse:.6f} | Val RMSE = {epoch_val_rmse:.6f}\n",
                f"Train NNSE = {epoch_train_nnse:.6f} | Val NNSE = {epoch_val_nnse:.6f}\n",
                f"Train MS-SSIM = {epoch_train_msssim:.6f} | Val MS-SSIM = {epoch_val_msssim:.6f}\n",
                f"Train ACC = {epoch_train_acc:.6f} | Val ACC = {epoch_val_acc:.6f}\n",
                f"Train PSNR = {epoch_train_psnr:.6f} | Val PSNR = {epoch_val_psnr:.6f}\n",
                "================================================================================================",
            )

        # Save best model
        if epoch_val_loss < best_val_loss:
            del best_state_dict
            best_state_dict = deepcopy(model.state_dict())
            best_val_loss = epoch_val_loss
            best_epoch = epoch
    if verbose:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    return (
        train_losses,
        val_losses,
        train_rmse,
        val_rmse,
        train_nnse,
        val_nnse,
        train_msssim,
        val_msssim,
        train_acc,
        val_acc,
        train_psnr,
        val_psnr,
        best_epoch,
        best_state_dict,
    )
