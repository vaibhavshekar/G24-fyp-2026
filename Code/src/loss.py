"""Loss functions for SITS forecasting."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Cache for SSIM Gaussian windows
_SSIM_WIN_CACHE: Dict[Tuple, torch.Tensor] = {}


def _gaussian_window(
    window_size: int, sigma: float, device: torch.device, dtype: torch.dtype, channels: int
) -> torch.Tensor:
    """Create Gaussian window for SSIM computation."""
    key = (channels, window_size, float(sigma), str(device), str(dtype))
    win = _SSIM_WIN_CACHE.get(key, None)
    if win is not None:
        return win

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / (g.sum() + 1e-12)
    window_1d = g.view(1, 1, window_size)
    window_2d = window_1d.transpose(2, 1) @ window_1d
    win = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    _SSIM_WIN_CACHE[key] = win
    return win


def _safe_ssim_window_size(h: int, w: int, requested_ws: int) -> int:
    """Ensure window size is valid for the given image dimensions."""
    m = min(h, w)
    ws = min(requested_ws, m)
    if ws < 3:
        ws = 3
    if ws % 2 == 0:
        ws -= 1
    if ws < 3:
        ws = 3
    return ws


def ssim_tensor(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute SSIM between predicted and target tensors.

    Args:
        pred: Predicted tensor [B, T, C, H, W]
        target: Target tensor [B, T, C, H, W]
        data_range: Data range of input (default 1.0 for normalized images)
        window_size: Size of Gaussian window
        sigma: Standard deviation of Gaussian window
        eps: Small epsilon for numerical stability

    Returns:
        Mean SSIM value (differentiable)
    """
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    B, T, C, H, W = pred.shape
    ws = _safe_ssim_window_size(H, W, window_size)

    x = pred.reshape(B * T, C, H, W)
    y = target.reshape(B * T, C, H, W)

    win = _gaussian_window(ws, sigma, device=x.device, dtype=x.dtype, channels=C)

    mu_x = F.conv2d(x, win, padding=ws // 2, groups=C)
    mu_y = F.conv2d(y, win, padding=ws // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, win, padding=ws // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=ws // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=ws // 2, groups=C) - mu_xy

    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + eps)

    ssim_map = torch.nan_to_num(ssim_map, nan=0.0, posinf=0.0, neginf=0.0)
    return ssim_map.mean()


class SSIMLoss(nn.Module):
    """SSIM-based loss function (1 - SSIM)."""

    def __init__(self, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute SSIM in FP32 with autocast OFF (prevents NaNs)
        pred32 = pred.float()
        tgt32 = target.float()
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            ssim_val = ssim_tensor(
                pred32,
                tgt32,
                data_range=self.data_range,
                window_size=self.window_size,
                sigma=self.sigma,
            )
        return 1.0 - ssim_val


class MAEPlusWeightedSSIMLoss(nn.Module):
    """
    Combined loss: MAE + w * (1 - SSIM)

    Args:
        ssim_weight: Weight for SSIM loss component
        ssim_window: Window size for SSIM computation
        ssim_sigma: Sigma for SSIM Gaussian window
    """

    def __init__(
        self,
        ssim_weight: float = 0.2,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
    ):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim_w = float(ssim_weight)
        self.ssim = SSIMLoss(1.0, ssim_window, ssim_sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae = self.l1(pred, target)
        ssim_loss = self.ssim(pred, target)  # Already (1 - SSIM) in safe FP32
        return mae + self.ssim_w * ssim_loss


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_window: int = 11,
    ssim_sigma: float = 1.5,
) -> Dict[str, float]:
    """
    Compute evaluation metrics between prediction and target.

    Args:
        pred: Predicted tensor [B, T, C, H, W]
        target: Target tensor [B, T, C, H, W]
        ssim_window: Window size for SSIM
        ssim_sigma: Sigma for SSIM

    Returns:
        Dictionary with MAE, MSE, RMSE, PSNR, SSIM
    """
    import math

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    mae = F.l1_loss(pred, target).item()
    mse = F.mse_loss(pred, target).item()
    rmse = math.sqrt(mse)
    psnr = 99.0 if mse <= 1e-12 else (20 * math.log10(1.0) - 10 * math.log10(mse))

    pred32 = pred.float()
    target32 = target.float()
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        ssim = float(ssim_tensor(pred32, target32, 1.0, ssim_window, ssim_sigma).item())

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "PSNR": psnr,
        "SSIM": ssim,
    }
