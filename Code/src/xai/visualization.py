"""Visualization utilities for XAI analysis."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt

from .config import XAIConfig


def _to_np(x: Optional[Union[torch.Tensor, np.ndarray]]) -> Optional[np.ndarray]:
    """Convert tensor or array to numpy."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def norm01_np(a: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    a = a.astype(np.float32)
    a = a - a.min()
    return a / (a.max() + 1e-8)


def importance_mask(attr01_hw: np.ndarray, cfg: XAIConfig) -> np.ndarray:
    """Create binary mask for important regions based on attribution."""
    if cfg.roi_threshold_mode == "quantile":
        thr = np.quantile(attr01_hw, cfg.roi_quantile)
    else:
        thr = float(cfg.roi_value)
    return (attr01_hw >= thr).astype(np.float32)


def rgb_from_chw(chw: torch.Tensor, cfg: XAIConfig) -> np.ndarray:
    """
    Extract RGB image from multispectral tensor.

    Args:
        chw: Tensor [C, H, W]
        cfg: XAI configuration with RGB band indices

    Returns:
        RGB image [H, W, 3]
    """
    arr = chw.detach().cpu().float().numpy()
    C, H, W = arr.shape
    b0, b1, b2 = cfg.rgb_bands
    b0 = int(np.clip(b0, 0, C - 1))
    b1 = int(np.clip(b1, 0, C - 1))
    b2 = int(np.clip(b2, 0, C - 1))

    rgb = np.stack([arr[b0], arr[b1], arr[b2]], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    lo, hi = cfg.rgb_stretch
    for k in range(3):
        ch = rgb[..., k]
        p1, p2 = np.percentile(ch, [lo, hi])
        ch = (ch - p1) / (p2 - p1 + 1e-8)
        rgb[..., k] = np.clip(ch, 0.0, 1.0)

    if cfg.rgb_gamma is not None and abs(cfg.rgb_gamma - 1.0) > 1e-6:
        rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / float(cfg.rgb_gamma))

    return np.clip(rgb, 0.0, 1.0)


def stretch_rgb(rgb: np.ndarray, lo: int = 2, hi: int = 98, gamma: float = 1.0) -> np.ndarray:
    """Apply percentile stretch and gamma correction to RGB image."""
    out = rgb.copy()
    for k in range(3):
        ch = out[..., k]
        p1, p2 = np.percentile(ch, [lo, hi])
        ch = (ch - p1) / (p2 - p1 + 1e-8)
        out[..., k] = np.clip(ch, 0.0, 1.0)
    if gamma is not None and abs(gamma - 1.0) > 1e-6:
        out = np.clip(out, 0.0, 1.0) ** (1.0 / float(gamma))
    return np.clip(out, 0.0, 1.0)


def visualize_xai_rgb(
    x: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    pix_attr: Union[torch.Tensor, np.ndarray],
    cfg: XAIConfig,
    save_path: Optional[str] = None,
    zone_name: str = "",
    show_inline: bool = False,
    apply_stretch: bool = False,
    figsize: Tuple[int, int] = (24, 5),
    return_fig: bool = False,
) -> Tuple[float, Optional[plt.Figure]]:
    """
    Visualize XAI results with RGB images.

    Args:
        x: Input tensor [B, T_in, C, H, W]
        pred: Predictions [B, T_out, C, H, W]
        gt: Ground truth [B, T_out, C, H, W]
        pix_attr: Pixel attribution map [H, W]
        cfg: XAI configuration
        save_path: Path to save visualization
        zone_name: Zone identifier for title
        show_inline: Whether to show inline (for Jupyter)
        apply_stretch: Whether to apply RGB stretch
        figsize: Figure size

    Returns:
        Tuple of (MAE, figure)
    """

    def rgb_simple(chw: torch.Tensor) -> np.ndarray:
        arr = chw.detach().cpu().float().numpy()  # [C, H, W]
        C, H, W = arr.shape
        b0, b1, b2 = cfg.rgb_bands
        b0 = int(np.clip(b0, 0, C - 1))
        b1 = int(np.clip(b1, 0, C - 1))
        b2 = int(np.clip(b2, 0, C - 1))
        rgb = np.stack([arr[b0], arr[b1], arr[b2]], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    # Move to CPU
    x = x.detach().cpu()
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()

    t = min(cfg.t_out, pred.shape[1] - 1)

    last_in = x[0, -1]  # [C, H, W]
    p = pred[0, t]  # [C, H, W]
    g = gt[0, t]  # [C, H, W]

    in_rgb = rgb_simple(last_in)
    p_rgb = rgb_simple(p)
    g_rgb = rgb_simple(g)

    if apply_stretch:
        lo, hi = getattr(cfg, "rgb_stretch", (2, 98))
        gamma = getattr(cfg, "rgb_gamma", 1.0)
        in_rgb = stretch_rgb(in_rgb, lo, hi, gamma)
        p_rgb = stretch_rgb(p_rgb, lo, hi, gamma)
        g_rgb = stretch_rgb(g_rgb, lo, hi, gamma)

    # Difference
    diff_rgb = np.abs(g_rgb - p_rgb)
    diff_map = diff_rgb.mean(axis=-1)
    mae = float(diff_rgb.mean())

    # XAI map
    if torch.is_tensor(pix_attr):
        attr = pix_attr.detach().cpu().float().numpy()
    else:
        attr = np.asarray(pix_attr, dtype=np.float32)
    attr = attr - attr.min()
    attr01 = attr / (attr.max() + 1e-8)

    # ROI mask
    if cfg.roi_threshold_mode == "quantile":
        thr = np.quantile(attr01, cfg.roi_quantile)
    else:
        thr = float(cfg.roi_value)
    mask = (attr01 >= thr).astype(np.float32)
    masked_in = in_rgb * mask[..., None]

    # Plot (no ground-truth panel)
    fig, axes = plt.subplots(1, 5, figsize=figsize)

    axes[0].imshow(in_rgb)
    axes[0].set_title("Last Input (RGB)")

    axes[1].imshow(p_rgb)
    axes[1].set_title("Predicted Next (RGB)")

    im2 = axes[2].imshow(diff_map, cmap=getattr(cfg, "diff_cmap", "magma"))
    axes[2].set_title(f"Difference\nMAE: {mae:.4f}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(p_rgb)
    axes[3].imshow(attr01, alpha=getattr(cfg, "overlay_alpha", 0.5), cmap=getattr(cfg, "attr_cmap", "magma"))
    axes[3].set_title("XAI Overlay")

    axes[4].imshow(masked_in)
    axes[4].set_title("Important Region Only")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"{zone_name} | MAE: {mae:.4f}", fontsize=12)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plt.savefig(str(save_path), dpi=200)
        print(f"Saved XAI visualization: {save_path}")

    if show_inline:
        plt.show()

    if not return_fig:
        plt.close(fig)
        fig = None

    return mae, fig


def xai_summary_dict(
    zone: str,
    method: str,
    cfg: XAIConfig,
    mae: float,
    per_m: Optional[Union[torch.Tensor, np.ndarray]],
    per_b: Optional[Union[torch.Tensor, np.ndarray]],
    pix_attr: Union[torch.Tensor, np.ndarray],
    png_path: str,
    topk: int = 20,
) -> Dict[str, Any]:
    """
    Create XAI summary dictionary for reporting.

    Args:
        zone: Zone identifier
        method: XAI method name
        cfg: XAI configuration
        mae: Mean Absolute Error
        per_m: Per-month importance
        per_b: Per-band importance
        pix_attr: Pixel attribution map
        png_path: Path to saved visualization
        topk: Number of top hotspots to include

    Returns:
        Summary dictionary
    """
    pix = _to_np(pix_attr)
    pix = pix - pix.min()
    pix01 = pix / (pix.max() + 1e-8)

    H, W = pix01.shape
    flat = pix01.reshape(-1)

    k = min(topk, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    hotspots = [{"y": int(i // W), "x": int(i % W), "score": float(flat[i])} for i in idx]

    # Entropy for concentration measure
    p = flat / (flat.sum() + 1e-8)
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size + 1e-12))

    per_m = _to_np(per_m)
    per_b = _to_np(per_b)
    if per_m is not None and per_m.sum() > 0:
        per_m = (per_m / per_m.sum()).tolist()
    if per_b is not None and per_b.sum() > 0:
        per_b = (per_b / per_b.sum()).tolist()

    return {
        "zone": str(zone),
        "method": str(method),
        "target": {
            "type": cfg.target_type,
            "mode": cfg.target_mode,
            "t_out": int(cfg.t_out),
        },
        "metrics": {
            "mae": float(mae),
            "importance_entropy": entropy,
        },
        "per_month_importance": per_m,
        "per_band_importance": per_b,
        "hotspots_topk": hotspots,
        "rgb_bands": list(cfg.rgb_bands),
        "artifact_paths": {"xai_png": str(png_path)},
    }


def save_json(d: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")


def plot_month_importance(
    per_month: Union[List[float], np.ndarray],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot temporal importance bar chart."""
    per_m = np.asarray(per_month, dtype=np.float32)
    if per_m.sum() > 0:
        per_m = per_m / per_m.sum()

    labels = [f"M{i+1}" for i in range(len(per_m))]

    fig, ax = plt.subplots()
    ax.bar(labels, per_m)
    ax.set_ylabel("Importance share")
    ax.set_ylim(0, max(1e-6, float(per_m.max()) * 1.2))
    ax.set_title(title or "Temporal importance (per month)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return fig


def plot_band_importance(
    per_band: Union[List[float], np.ndarray],
    title: Optional[str] = None,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot spectral importance bar chart."""
    per_b = np.asarray(per_band, dtype=np.float32)
    if per_b.sum() > 0:
        per_b = per_b / per_b.sum()

    labels = [f"B{i}" for i in range(len(per_b))]
    if top_k is not None and top_k < len(per_b):
        idx = np.argsort(-per_b)[:top_k]
        per_b = per_b[idx]
        labels = [labels[i] for i in idx]

    fig, ax = plt.subplots()
    ax.bar(labels, per_b)
    ax.set_ylabel("Importance share")
    ax.set_ylim(0, max(1e-6, float(per_b.max()) * 1.2))
    ax.set_title(title or f"Spectral importance (top {len(per_b)})")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    return fig
