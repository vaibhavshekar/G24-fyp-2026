"""Configuration for XAI methods."""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class XAIConfig:
    """Configuration for XAI analysis."""

    # Target output time step
    t_out: int = 0

    # Target type: "prediction", "error_abs", "error_mse"
    target_type: str = "error_abs"

    # Target mode: "pixel", "patch", "full_mean"
    target_mode: str = "full_mean"

    # Target pixel location (for pixel/patch modes)
    yx: Tuple[int, int] = (64, 64)

    # Patch radius (for patch mode)
    patch_radius: int = 4

    # Integrated Gradients steps
    ig_steps: int = 32

    # Occlusion parameters
    occ_patch: int = 8
    occ_stride: int = 4
    occ_value: Optional[float] = None

    # Visualization parameters
    overlay_alpha: float = 0.5

    # ROI (Region of Interest) parameters
    roi_threshold_mode: str = "quantile"  # "quantile" or "value"
    roi_quantile: float = 0.90
    roi_value: float = 0.7

    # RGB bands for visualization (Sentinel-2: 0=B, 1=G, 2=R)
    rgb_bands: Tuple[int, int, int] = (2, 1, 0)
    rgb_stretch: Tuple[int, int] = (2, 98)
    rgb_gamma: float = 1.0

    # Colormaps
    diff_cmap: str = "magma"
    attr_cmap: str = "magma"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "t_out": self.t_out,
            "target_type": self.target_type,
            "target_mode": self.target_mode,
            "yx": self.yx,
            "patch_radius": self.patch_radius,
            "ig_steps": self.ig_steps,
            "occ_patch": self.occ_patch,
            "occ_stride": self.occ_stride,
            "occ_value": self.occ_value,
            "overlay_alpha": self.overlay_alpha,
            "roi_threshold_mode": self.roi_threshold_mode,
            "roi_quantile": self.roi_quantile,
            "roi_value": self.roi_value,
            "rgb_bands": self.rgb_bands,
            "rgb_stretch": self.rgb_stretch,
            "rgb_gamma": self.rgb_gamma,
            "diff_cmap": self.diff_cmap,
            "attr_cmap": self.attr_cmap,
        }
