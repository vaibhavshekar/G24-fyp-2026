"""Training utilities for SITS forecasting."""

import os
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .config import Config
from .model import UNet3DTemporal
from .dataset import FLAIR2ForecastDataset, split_zones
from .loss import MAEPlusWeightedSSIMLoss, ssim_tensor


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_grad_scaler(amp_enabled: bool) -> torch.amp.GradScaler:
    """Create gradient scaler for mixed precision training."""
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except TypeError:
        return torch.amp.GradScaler(enabled=amp_enabled)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ssim_window: int = 11,
    ssim_sigma: float = 1.5,
) -> Dict[str, float]:
    """
    Evaluate model on a dataloader.

    Args:
        model: Model to evaluate
        loader: DataLoader for evaluation
        device: Device to use
        ssim_window: Window size for SSIM
        ssim_sigma: Sigma for SSIM

    Returns:
        Dictionary with MAE, MSE, RMSE, PSNR, SSIM metrics
    """
    model.eval()
    maes, mses, rmses, psnrs, ssims = [], [], [], [], []

    for x, y, _, _ in loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)

        yp = torch.clamp(y_pred, 0.0, 1.0)

        mae = F.l1_loss(yp, y).item()
        mse = F.mse_loss(yp, y).item()
        rmse = math.sqrt(mse)
        psnr = 99.0 if mse <= 1e-12 else (20 * math.log10(1.0) - 10 * math.log10(mse))

        yp32 = yp.float()
        y32 = y.float()
        with torch.amp.autocast(device_type=device.type, enabled=False):
            ssim = float(ssim_tensor(yp32, y32, 1.0, ssim_window, ssim_sigma).item())

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        psnrs.append(psnr)
        ssims.append(ssim)

    return {
        "MAE": float(np.mean(maes)),
        "MSE": float(np.mean(mses)),
        "RMSE": float(np.mean(rmses)),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
    }


def train_single_run(
    train_zones: List[str],
    val_zones: List[str],
    cfg: Config,
    device: Optional[torch.device] = None,
) -> Tuple[str, str, Dict[str, float]]:
    """
    Train a single model run.

    Args:
        train_zones: List of zone names for training
        val_zones: List of zone names for validation
        cfg: Configuration object
        device: Device to use (defaults to CUDA if available)

    Returns:
        Tuple of (checkpoint_path, loss_curve_path, final_metrics)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Create datasets
    train_ds = FLAIR2ForecastDataset(
        cfg.root,
        cfg.input_len,
        cfg.pred_len,
        cfg.use_normalized,
        zones_filter=train_zones,
        mode="train",
        patch_size=cfg.patch_size,
        use_patches=cfg.use_patches,
        val_deterministic_patch=cfg.val_deterministic_patch,
        clamp_lo=cfg.clamp_lo,
        clamp_hi=cfg.clamp_hi,
        in_channels=cfg.in_channels,
        augment=cfg.augment,
        input_noise_std=cfg.input_noise_std,
        temporal_dropout_p=cfg.temporal_dropout_p,
        channel_dropout_p=cfg.channel_dropout_p,
        seed=cfg.seed,
    )
    val_ds = FLAIR2ForecastDataset(
        cfg.root,
        cfg.input_len,
        cfg.pred_len,
        cfg.use_normalized,
        zones_filter=val_zones,
        mode="val",
        patch_size=cfg.patch_size,
        use_patches=cfg.use_patches,
        val_deterministic_patch=cfg.val_deterministic_patch,
        clamp_lo=cfg.clamp_lo,
        clamp_hi=cfg.clamp_hi,
        in_channels=cfg.in_channels,
        augment=False,
        seed=cfg.seed,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Train/Val dataset has 0 samples.")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_mem,
    )

    model = UNet3DTemporal(
        cfg.in_channels, cfg.in_channels, cfg.base_channels, pred_len=cfg.pred_len
    ).to(device)

    criterion = MAEPlusWeightedSSIMLoss(cfg.ssim_weight, cfg.ssim_window, cfg.ssim_sigma)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.sched_patience, min_lr=cfg.min_lr
    )

    amp_enabled = device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.checkpoint_dir, f"RUN_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("TRAINING SINGLE RUN")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")
    print(
        f"Batch: {cfg.batch_size} | Accum: {cfg.accum_steps} | EffBatch: {cfg.batch_size * cfg.accum_steps}"
    )
    print(f"Patch: {cfg.use_patches} size={cfg.patch_size} | Augment(train)={cfg.augment}")
    print(f"VAL deterministic patch: {cfg.val_deterministic_patch}")
    print(f"Loss: MAE + {cfg.ssim_weight}*(1-SSIM)")
    print("-" * 70)

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        steps = 0
        exploded = False

        for bi, (x, y, _, _) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                y_pred = model(x)
                loss = criterion(y_pred, y)
                if not torch.isfinite(loss):
                    exploded = True
                loss = loss / cfg.accum_steps

            if exploded:
                print("Non-finite loss detected (NaN/Inf). Stopping early.")
                break

            scaler.scale(loss).backward()

            if (bi % cfg.accum_steps == 0) or (bi == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * cfg.accum_steps
            steps += 1

        if exploded:
            break

        train_loss = running / max(1, steps)

        # Validation
        model.eval()
        vloss = 0.0
        vsteps = 0
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                if not torch.isfinite(loss):
                    exploded = True
                    break
                vloss += float(loss.item())
                vsteps += 1

        if exploded:
            print("Non-finite VAL loss detected (NaN/Inf). Stopping early.")
            break

        val_loss = vloss / max(1, vsteps)
        scheduler.step(val_loss)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        lr_now = optimizer.param_groups[0]["lr"]

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "val_loss": float(best_val),
                "loss_name": f"mae+{cfg.ssim_weight}*(1-ssim)",
            }
            print(
                f"Epoch {epoch:03d} | LR {lr_now:.2e} | Train {train_loss:.6f} | Val {val_loss:.6f} <-- best"
            )
        else:
            no_improve += 1
            print(
                f"Epoch {epoch:03d} | LR {lr_now:.2e} | Train {train_loss:.6f} | Val {val_loss:.6f} (no improve {no_improve})"
            )

        if no_improve >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save loss curve
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (MAE + w*(1-SSIM))")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(run_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve: {loss_curve_path}")

    if best_state is None:
        raise RuntimeError("No best checkpoint saved (run exploded too early).")

    ckpt_path = os.path.join(run_dir, "unet3d_best_mae_w_ssim.pth")
    torch.save(best_state, ckpt_path)
    print(f"Saved BEST checkpoint: {ckpt_path}")

    # Final metrics on best
    model.load_state_dict(best_state["model_state_dict"])
    metrics = evaluate_model(model, val_loader, device, cfg.ssim_window, cfg.ssim_sigma)

    print("\n=== FINAL METRICS (best checkpoint) ===")
    print(f"Best Val Loss : {best_val:.6f}")
    print(f"MAE : {metrics['MAE']:.4f}")
    print(f"MSE : {metrics['MSE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"PSNR: {metrics['PSNR']:.2f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")

    return ckpt_path, loss_curve_path, metrics


def train_model(cfg: Config) -> Tuple[str, str, Dict[str, float]]:
    """
    Main training function.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (checkpoint_path, loss_curve_path, final_metrics)
    """
    set_seed(cfg.seed)
    train_zones, val_zones = split_zones(
        cfg.root, cfg.use_normalized, cfg.val_zone_frac, cfg.seed
    )

    print("\n==================== ZONE SPLIT ====================")
    print(f"Train zones: {len(train_zones)}")
    print(f"Val zones  : {len(val_zones)}")
    print(
        "Val zones:",
        ", ".join(val_zones[:10]) + (" ..." if len(val_zones) > 10 else ""),
    )

    return train_single_run(train_zones, val_zones, cfg)
