"""Inference utilities for SITS forecasting."""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import torch
import numpy as np
from torch.utils.data import DataLoader

from .model import UNet3DTemporal, load_model, get_model_info
from .dataset import SingleZoneFolderDataset
from .loss import compute_metrics
from .xai import XAIConfig
from .xai.methods import run_xai_method
from .xai.visualization import visualize_xai_rgb, xai_summary_dict, save_json


def predict_folder(
    checkpoint_path: str,
    data_folder: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Run prediction on a folder of satellite images.

    Args:
        checkpoint_path: Path to model checkpoint
        data_folder: Path to normalized folder with .tif files
        output_dir: Optional directory to save outputs
        device: Device to use ("cuda" or "cpu")

    Returns:
        Dictionary with predictions and metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model info
    info = get_model_info(checkpoint_path, device)
    config = info.get("config", {})

    # Get parameters
    input_len = config.get("input_len", config.get("INPUT_LEN", 6))
    pred_len = config.get("pred_len", config.get("PRED_LEN", 3))
    in_channels = config.get("in_channels", config.get("IN_CHANNELS", 10))
    base_channels = config.get("base_channels", config.get("BASE_CHANNELS", 12))
    patch_size = config.get("patch_size", config.get("PATCH_SIZE", 128))
    clamp_lo = config.get("clamp_lo", config.get("CLAMP_LO", -2.0))
    clamp_hi = config.get("clamp_hi", config.get("CLAMP_HI", 6.0))

    # Load model
    model = load_model(
        checkpoint_path,
        in_channels=in_channels,
        base_channels=base_channels,
        pred_len=pred_len,
        device=device,
    )

    # Load dataset
    dataset = SingleZoneFolderDataset(
        folder=data_folder,
        input_len=input_len,
        pred_len=pred_len,
        in_channels=in_channels,
        patch_size=patch_size,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
        deterministic_patch=True,
    )

    # Get data
    x, y, _, zone = dataset[0]

    # Run prediction
    model.eval()
    with torch.no_grad():
        x_device = x.unsqueeze(0).to(device)
        pred = model(x_device)
        pred = torch.clamp(pred, 0.0, 1.0)

    pred = pred.cpu()

    # Compute metrics if ground truth available
    metrics = None
    if y.sum() > 0:
        metrics = compute_metrics(pred, y.unsqueeze(0))

    result = {
        "zone": zone,
        "input": x,
        "prediction": pred.squeeze(0),
        "ground_truth": y if y.sum() > 0 else None,
        "metrics": metrics,
        "dates": dataset.get_date_info(),
    }

    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save prediction as numpy
        np.save(output_path / f"{zone}_prediction.npy", pred.squeeze(0).numpy())

        if metrics:
            save_json(metrics, output_path / f"{zone}_metrics.json")

    return result


def run_xai_analysis(
    checkpoint_path: str,
    data_folder: str,
    method: str = "saliency",
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    xai_config: Optional[XAIConfig] = None,
) -> Dict:
    """
    Run XAI analysis on predictions.

    Args:
        checkpoint_path: Path to model checkpoint
        data_folder: Path to normalized folder with .tif files
        method: XAI method ("saliency", "ig", or "occlusion")
        output_dir: Optional directory to save outputs
        device: Device to use
        xai_config: Optional XAI configuration

    Returns:
        Dictionary with XAI results
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if xai_config is None:
        xai_config = XAIConfig()

    # Load model info
    info = get_model_info(checkpoint_path, device)
    config = info.get("config", {})

    # Get parameters
    input_len = config.get("input_len", config.get("INPUT_LEN", 6))
    pred_len = config.get("pred_len", config.get("PRED_LEN", 3))
    in_channels = config.get("in_channels", config.get("IN_CHANNELS", 10))
    base_channels = config.get("base_channels", config.get("BASE_CHANNELS", 12))
    patch_size = config.get("patch_size", config.get("PATCH_SIZE", 128))
    clamp_lo = config.get("clamp_lo", config.get("CLAMP_LO", -2.0))
    clamp_hi = config.get("clamp_hi", config.get("CLAMP_HI", 6.0))

    # Load model
    model = load_model(
        checkpoint_path,
        in_channels=in_channels,
        base_channels=base_channels,
        pred_len=pred_len,
        device=device,
    )

    # Load dataset
    dataset = SingleZoneFolderDataset(
        folder=data_folder,
        input_len=input_len,
        pred_len=pred_len,
        in_channels=in_channels,
        patch_size=patch_size,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
        deterministic_patch=True,
    )

    # Get data
    x, y, _, zone = dataset[0]

    # Move to device
    x_device = x.unsqueeze(0).to(device)
    y_device = y.unsqueeze(0).to(device)

    # Run XAI
    pred, pix_attr, per_month, per_band = run_xai_method(
        model, x_device, y_device, method, xai_config
    )

    result = {
        "zone": zone,
        "method": method,
        "prediction": pred.cpu(),
        "pix_attr": pix_attr.cpu() if torch.is_tensor(pix_attr) else pix_attr,
        "per_month": per_month.cpu() if torch.is_tensor(per_month) else per_month,
        "per_band": per_band.cpu() if torch.is_tensor(per_band) else per_band,
        "input": x,
        "ground_truth": y if y.sum() > 0 else None,
    }

    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save visualization
        png_path = output_path / f"xai_{method}_{zone}.png"
        mae, _ = visualize_xai_rgb(
            x.unsqueeze(0),
            pred.cpu(),
            y.unsqueeze(0),
            pix_attr,
            xai_config,
            save_path=str(png_path),
        )

        # Save summary JSON
        summary = xai_summary_dict(
            zone=zone,
            method=method,
            cfg=xai_config,
            mae=mae,
            per_m=per_month,
            per_b=per_band,
            pix_attr=pix_attr,
            png_path=str(png_path),
        )
        save_json(summary, output_path / f"xai_{method}_{zone}.json")

        result["output_files"] = {
            "visualization": str(png_path),
            "summary": str(output_path / f"xai_{method}_{zone}.json"),
        }

    return result
