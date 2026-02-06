"""XAI (Explainable AI) module for SITS forecasting."""

from .config import XAIConfig
from .methods import saliency_xai, integrated_gradients_xai, occlusion_xai, aggregate_attribution
from .visualization import visualize_xai_rgb, rgb_from_chw, xai_summary_dict

__all__ = [
    "XAIConfig",
    "saliency_xai",
    "integrated_gradients_xai",
    "occlusion_xai",
    "aggregate_attribution",
    "visualize_xai_rgb",
    "rgb_from_chw",
    "xai_summary_dict",
]
