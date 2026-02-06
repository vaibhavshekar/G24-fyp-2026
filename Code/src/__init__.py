"""SITS Forecasting - Satellite Image Time Series Forecasting with XAI"""

from .config import Config
from .model import UNet3DTemporal, load_model, get_model_info
from .dataset import FLAIR2ForecastDataset, SingleZoneFolderDataset
from .loss import SSIMLoss, MAEPlusWeightedSSIMLoss, compute_metrics
from .training import train_single_run, evaluate_model, set_seed
from .inference import predict_folder, run_xai_analysis
from .xai import XAIConfig

__all__ = [
    "Config",
    "UNet3DTemporal",
    "load_model",
    "get_model_info",
    "FLAIR2ForecastDataset",
    "SingleZoneFolderDataset",
    "SSIMLoss",
    "MAEPlusWeightedSSIMLoss",
    "compute_metrics",
    "train_single_run",
    "evaluate_model",
    "set_seed",
    "predict_folder",
    "run_xai_analysis",
    "XAIConfig",
]
