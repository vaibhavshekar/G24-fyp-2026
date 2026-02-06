"""Configuration classes for SITS forecasting model."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Main configuration for training and inference."""

    # Data paths
    root: str = ""
    use_normalized: bool = True

    # Sequence parameters
    input_len: int = 6
    pred_len: int = 3
    val_zone_frac: float = 0.2

    # Patch training
    use_patches: bool = True
    patch_size: int = 128
    val_deterministic_patch: bool = True

    # Normalization
    clamp_lo: float = -2.0
    clamp_hi: float = 6.0

    # Model architecture
    in_channels: int = 10
    base_channels: int = 12

    # Training hyperparameters
    batch_size: int = 1
    accum_steps: int = 4
    num_epochs: int = 100
    lr: float = 5e-5
    weight_decay: float = 2e-4
    sched_patience: int = 3
    min_lr: float = 1e-6
    early_stop_patience: int = 10

    # Augmentation
    augment: bool = True
    input_noise_std: float = 0.01
    temporal_dropout_p: float = 0.15
    channel_dropout_p: float = 0.10

    # SSIM parameters
    ssim_window: int = 11
    ssim_sigma: float = 1.5
    ssim_weight: float = 0.20

    # Misc
    seed: int = 42
    num_workers: int = 0
    checkpoint_dir: str = "./checkpoints"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "root": self.root,
            "use_normalized": self.use_normalized,
            "input_len": self.input_len,
            "pred_len": self.pred_len,
            "val_zone_frac": self.val_zone_frac,
            "use_patches": self.use_patches,
            "patch_size": self.patch_size,
            "val_deterministic_patch": self.val_deterministic_patch,
            "clamp_lo": self.clamp_lo,
            "clamp_hi": self.clamp_hi,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "sched_patience": self.sched_patience,
            "min_lr": self.min_lr,
            "early_stop_patience": self.early_stop_patience,
            "augment": self.augment,
            "input_noise_std": self.input_noise_std,
            "temporal_dropout_p": self.temporal_dropout_p,
            "channel_dropout_p": self.channel_dropout_p,
            "ssim_window": self.ssim_window,
            "ssim_sigma": self.ssim_sigma,
            "ssim_weight": self.ssim_weight,
            "seed": self.seed,
            "num_workers": self.num_workers,
            "checkpoint_dir": self.checkpoint_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
