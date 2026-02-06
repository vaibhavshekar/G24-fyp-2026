"""XAI attribution methods for SITS forecasting."""

from typing import Tuple

import torch
import torch.nn as nn

from .config import XAIConfig


def scalar_target(pred: torch.Tensor, gt: torch.Tensor, cfg: XAIConfig) -> torch.Tensor:
    """
    Compute scalar target for attribution.

    Args:
        pred: Predictions [B, T_out, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        cfg: XAI configuration

    Returns:
        Scalar target tensor
    """
    B, T, C, H, W = pred.shape
    t = min(cfg.t_out, T - 1)

    p = pred[:, t]  # [B, C, H, W]

    if cfg.target_type == "prediction":
        base = p
    else:
        g = gt[:, t]
        if cfg.target_type == "error_abs":
            base = (p - g).abs()
        elif cfg.target_type == "error_mse":
            base = (p - g) ** 2
        else:
            raise ValueError(f"Unknown target_type: {cfg.target_type}")

    if cfg.target_mode == "full_mean":
        return base.mean()

    base2d = base.mean(dim=1)  # [B, H, W]
    y, x = cfg.yx
    y = max(0, min(H - 1, y))
    x = max(0, min(W - 1, x))

    if cfg.target_mode == "pixel":
        return base2d[:, y, x].mean()

    if cfg.target_mode == "patch":
        r = cfg.patch_radius
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        return base2d[:, y0:y1, x0:x1].mean()

    raise ValueError(f"Unknown target_mode: {cfg.target_mode}")


def saliency_xai(
    model: nn.Module,
    x: torch.Tensor,
    gt: torch.Tensor,
    cfg: XAIConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute saliency map (gradient-based attribution).

    Args:
        model: Model to explain
        x: Input tensor [B, T_in, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        cfg: XAI configuration

    Returns:
        Tuple of (predictions, attributions)
    """
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    pred = model(x)
    target = scalar_target(pred, gt, cfg)
    model.zero_grad(set_to_none=True)
    target.backward()
    attr = x.grad.detach()
    return pred.detach(), attr


def integrated_gradients_xai(
    model: nn.Module,
    x: torch.Tensor,
    gt: torch.Tensor,
    cfg: XAIConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Integrated Gradients attribution.

    Args:
        model: Model to explain
        x: Input tensor [B, T_in, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        cfg: XAI configuration

    Returns:
        Tuple of (predictions, attributions)
    """
    model.eval()
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)

    for i in range(1, cfg.ig_steps + 1):
        alpha = i / cfg.ig_steps
        xi = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        pred = model(xi)
        target = scalar_target(pred, gt, cfg)
        model.zero_grad(set_to_none=True)
        target.backward()
        total_grad += xi.grad.detach()

    avg_grad = total_grad / cfg.ig_steps
    attr = (x - baseline) * avg_grad
    pred = model(x).detach()
    return pred, attr


@torch.no_grad()
def occlusion_xai(
    model: nn.Module,
    x: torch.Tensor,
    gt: torch.Tensor,
    cfg: XAIConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Occlusion Sensitivity attribution.

    Args:
        model: Model to explain
        x: Input tensor [B, T_in, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        cfg: XAI configuration

    Returns:
        Tuple of (predictions, pixel_heat, per_month, per_band)
    """
    model.eval()
    B, T, C, H, W = x.shape

    pred0 = model(x)
    base = scalar_target(pred0, gt, cfg).item()

    if cfg.occ_value is None:
        fill_map = x.mean(dim=(-1, -2), keepdim=True)
    else:
        fill_map = torch.full((B, T, C, 1, 1), float(cfg.occ_value), device=x.device, dtype=x.dtype)

    heat = torch.zeros((H, W), device=x.device)

    for y0 in range(0, H, cfg.occ_stride):
        for x0 in range(0, W, cfg.occ_stride):
            y1 = min(H, y0 + cfg.occ_patch)
            x1 = min(W, x0 + cfg.occ_patch)

            xo = x.clone()
            xo[:, :, :, y0:y1, x0:x1] = fill_map.expand(-1, -1, -1, y1 - y0, x1 - x0)

            pred = model(xo)
            val = scalar_target(pred, gt, cfg).item()
            heat[y0:y1, x0:x1] += abs(base - val)

    heat = heat / (heat.max() + 1e-8)

    # Per-month importance
    per_month = torch.zeros((T,), device=x.device)
    for t in range(T):
        xo = x.clone()
        xo[:, t : t + 1] = fill_map[:, t : t + 1].expand(-1, 1, -1, H, W)
        pred = model(xo)
        val = scalar_target(pred, gt, cfg).item()
        per_month[t] = abs(base - val)

    # Per-band importance
    per_band = torch.zeros((C,), device=x.device)
    for c in range(C):
        xo = x.clone()
        xo[:, :, c : c + 1] = fill_map[:, :, c : c + 1].expand(-1, -1, 1, H, W)
        pred = model(xo)
        val = scalar_target(pred, gt, cfg).item()
        per_band[c] = abs(base - val)

    return pred0.detach(), heat.cpu(), per_month.cpu(), per_band.cpu()


def aggregate_attribution(
    attr: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate attribution tensor into pixel map, per-month, and per-band importance.

    Args:
        attr: Attribution tensor [B, T, C, H, W]

    Returns:
        Tuple of (pixel_map [H, W], per_month [T], per_band [C])
    """
    a = attr.abs().mean(dim=0)  # [T, C, H, W]
    pix_map = a.sum(dim=(0, 1))  # [H, W]
    per_month = a.sum(dim=(1, 2, 3))  # [T]
    per_band = a.sum(dim=(0, 2, 3))  # [C]
    return pix_map, per_month, per_band


def run_xai_method(
    model: nn.Module,
    x: torch.Tensor,
    gt: torch.Tensor,
    method: str,
    cfg: XAIConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run a specified XAI method.

    Args:
        model: Model to explain
        x: Input tensor [B, T_in, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        method: Method name ("saliency", "ig", "occlusion")
        cfg: XAI configuration

    Returns:
        Tuple of (predictions, pixel_attr, per_month, per_band)
    """
    if method == "saliency":
        pred, attr = saliency_xai(model, x, gt, cfg)
        pix, per_m, per_b = aggregate_attribution(attr)
    elif method == "ig":
        pred, attr = integrated_gradients_xai(model, x, gt, cfg)
        pix, per_m, per_b = aggregate_attribution(attr)
    elif method == "occlusion":
        pred, pix, per_m, per_b = occlusion_xai(model, x, gt, cfg)
    else:
        raise ValueError(f"Unknown XAI method: {method}")

    return pred, pix, per_m, per_b
