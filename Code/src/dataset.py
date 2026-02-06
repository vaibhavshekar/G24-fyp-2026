"""Dataset classes for SITS forecasting."""

import re
import zlib
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import torch
from torch.utils.data import Dataset

# Regex pattern for parsing date from filenames
DATE_RE = re.compile(r".*?(20\d{2})[_-]?(0[1-9]|1[0-2]).*?\.tif$")


def list_zones(root: str, use_normalized: bool) -> List[str]:
    """List available zones in the dataset root directory."""
    rootp = Path(root)
    sub = "normalized" if use_normalized else "composites"
    zones = []
    for z in sorted(rootp.iterdir()):
        if z.is_dir() and (z / sub).exists():
            zones.append(z.name)
    return zones


def split_zones(
    root: str, use_normalized: bool, val_frac: float, seed: int
) -> Tuple[List[str], List[str]]:
    """Split zones into train and validation sets."""
    zones = list_zones(root, use_normalized)
    rng = random.Random(seed)
    rng.shuffle(zones)
    n_val = max(1, int(len(zones) * val_frac))
    val_zones = zones[:n_val]
    train_zones = zones[n_val:]
    return train_zones, val_zones


class FLAIR2ForecastDataset(Dataset):
    """
    Dataset for FLAIR-2 satellite image forecasting.

    Returns:
        x: [T_in, C, P, P] - Input sequence
        y: [T_out, C, P, P] - Target sequence
        mask_y: [T_out, 1, P, P] - Mask for valid pixels
        zone: str - Zone identifier
    """

    def __init__(
        self,
        root: str,
        input_len: int,
        pred_len: int,
        use_normalized: bool = True,
        zones_filter: Optional[List[str]] = None,
        mode: str = "train",
        patch_size: int = 128,
        use_patches: bool = True,
        val_deterministic_patch: bool = True,
        clamp_lo: float = -2.0,
        clamp_hi: float = 6.0,
        in_channels: int = 10,
        augment: bool = True,
        input_noise_std: float = 0.01,
        temporal_dropout_p: float = 0.15,
        channel_dropout_p: float = 0.10,
        seed: int = 42,
    ):
        super().__init__()
        assert mode in ("train", "val")
        self.mode = mode

        self.root = Path(root)
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.use_normalized = use_normalized
        self.zones_filter: Optional[Set[str]] = set(zones_filter) if zones_filter else None

        self.patch_size = patch_size
        self.use_patches = use_patches
        self.val_deterministic_patch = val_deterministic_patch
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.in_channels = in_channels
        self.augment = augment
        self.input_noise_std = input_noise_std
        self.temporal_dropout_p = temporal_dropout_p
        self.channel_dropout_p = channel_dropout_p
        self.seed = seed

        self.samples: List[Tuple[str, int]] = []
        self.series_cache: Dict[str, List[Path]] = {}
        self.zone_hw: Dict[str, Tuple[int, int]] = {}

        self._build()

    def _parse_month(self, fname: str) -> Optional[Tuple[int, int]]:
        m = DATE_RE.match(fname)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, self.clamp_lo, self.clamp_hi)
        arr = (arr - self.clamp_lo) / (self.clamp_hi - self.clamp_lo + 1e-8)
        return arr.astype(np.float32)

    def _build(self):
        sub = "normalized" if self.use_normalized else "composites"
        for zone_dir in sorted(self.root.iterdir()):
            if not zone_dir.is_dir():
                continue
            zone = zone_dir.name
            if self.zones_filter is not None and zone not in self.zones_filter:
                continue

            comp_dir = zone_dir / sub
            if not comp_dir.exists():
                continue

            items = []
            for tif in comp_dir.glob("*.tif"):
                mk = self._parse_month(tif.name)
                if mk is None:
                    continue
                items.append((mk, tif))
            if not items:
                continue

            items.sort(key=lambda t: t[0])
            paths = [p for _, p in items]
            self.series_cache[zone] = paths

            # Compute per-zone min H/W
            min_h, min_w = None, None
            for p in paths:
                try:
                    with rasterio.open(p) as src:
                        h, w = src.height, src.width
                    if min_h is None:
                        min_h, min_w = h, w
                    else:
                        min_h = min(min_h, h)
                        min_w = min(min_w, w)
                except RasterioIOError:
                    continue

            if min_h is None:
                continue
            self.zone_hw[zone] = (min_h, min_w)

            T = len(paths)
            if T < self.total_len:
                continue
            for s in range(0, T - self.total_len + 1):
                self.samples.append((zone, s))

        if not self.samples:
            print("WARNING: No samples found! Check ROOT structure / filenames.")
        else:
            print(f"Dataset built: {len(self.samples)} windows from {len(self.series_cache)} zones.")

    def __len__(self) -> int:
        return len(self.samples)

    def _deterministic_top_left(
        self, zone: str, start: int, base_h: int, base_w: int, P: int
    ) -> Tuple[int, int]:
        zone_crc = zlib.crc32(zone.encode("utf-8")) & 0xFFFFFFFF
        seed = (self.seed * 1315423911 + zone_crc + start * 2654435761) & 0xFFFFFFFF
        rng = random.Random(seed)
        top = rng.randint(0, base_h - P) if base_h > P else 0
        left = rng.randint(0, base_w - P) if base_w > P else 0
        return top, left

    def _augment_train(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random flip horizontal
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        # Random flip vertical
        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])
        # Random rotation
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k=k, dims=[-2, -1])
            y = torch.rot90(y, k=k, dims=[-2, -1])

        # Channel dropout
        if self.channel_dropout_p > 0:
            C = x.shape[1]
            for c in range(C):
                if random.random() < self.channel_dropout_p:
                    x[:, c, :, :] = 0.0

        # Temporal dropout
        if self.temporal_dropout_p > 0:
            T = x.shape[0]
            for t in range(T):
                if random.random() < self.temporal_dropout_p:
                    x[t] = 0.0

        # Input noise
        if self.input_noise_std > 0:
            x = x + torch.randn_like(x) * self.input_noise_std

        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        zone, start = self.samples[idx]
        seq = self.series_cache[zone]
        paths = seq[start : start + self.total_len]

        base_h, base_w = self.zone_hw.get(zone, (self.patch_size, self.patch_size))
        if self.use_patches:
            P = min(self.patch_size, base_h, base_w)
        else:
            P = min(base_h, base_w)

        if self.mode == "val" and self.val_deterministic_patch:
            top, left = self._deterministic_top_left(zone, start, base_h, base_w, P)
        else:
            top = random.randint(0, base_h - P) if base_h > P else 0
            left = random.randint(0, base_w - P) if base_w > P else 0

        imgs = []
        for p in paths:
            try:
                with rasterio.open(p) as src:
                    arr = src.read().astype(np.float32)
            except RasterioIOError:
                arr = np.zeros((self.in_channels, base_h, base_w), dtype=np.float32)

            # Adjust channels
            if arr.shape[0] > self.in_channels:
                arr = arr[: self.in_channels]
            elif arr.shape[0] < self.in_channels:
                pad_c = self.in_channels - arr.shape[0]
                arr = np.concatenate(
                    [arr, np.zeros((pad_c, arr.shape[1], arr.shape[2]), dtype=np.float32)],
                    axis=0,
                )

            arr = self._normalize(arr)
            arr = arr[:, :base_h, :base_w]
            arr = arr[:, top : top + P, left : left + P]
            imgs.append(arr)

        stack = torch.from_numpy(np.stack(imgs, axis=0))  # [T, C, P, P]
        x = stack[: self.input_len]
        y = stack[self.input_len :]

        if self.mode == "train" and self.augment and self.use_patches:
            x, y = self._augment_train(x, y)

        x = torch.clamp(x, 0.0, 1.0)

        mask_y = torch.ones((self.pred_len, 1, P, P), dtype=torch.float32)
        return x, y, mask_y, zone


class SingleZoneFolderDataset(Dataset):
    """
    Dataset for a single zone folder containing monthly .tif files.

    Returns:
        x: [T_in, C, P, P] - Input sequence
        y: [T_out, C, P, P] - Target sequence (zeros if not enough future frames)
        mask_y: [T_out, 1, P, P] - Mask for valid pixels
        zone: str - Zone identifier
    """

    def __init__(
        self,
        folder: str,
        input_len: int,
        pred_len: int,
        in_channels: int,
        patch_size: int = 128,
        clamp_lo: float = -2.0,
        clamp_hi: float = 6.0,
        seed: int = 42,
        deterministic_patch: bool = True,
    ):
        self.folder = Path(folder)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.total_len = self.input_len + self.pred_len
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.clamp_lo = float(clamp_lo)
        self.clamp_hi = float(clamp_hi)
        self.seed = int(seed)
        self.deterministic_patch = bool(deterministic_patch)

        # Collect .tif files sorted by date
        items = []
        for tif in self.folder.glob("*.tif"):
            m = DATE_RE.match(tif.name)
            if m:
                items.append(((int(m.group(1)), int(m.group(2))), tif))
            else:
                items.append(((9999, 99), tif))  # Push unknowns to end
        items.sort(key=lambda t: (t[0][0], t[0][1], t[1].name))
        self.paths = [p for _, p in items]

        if len(self.paths) < self.input_len:
            raise ValueError(
                f"Need at least INPUT_LEN={self.input_len} tif files, found {len(self.paths)}"
            )

        self.use_future = len(self.paths) >= self.total_len

        # Find common base H/W
        hs, ws = [], []
        for p in self.paths[: min(len(self.paths), self.total_len)]:
            with rasterio.open(p) as src:
                hs.append(src.height)
                ws.append(src.width)
        self.base_h = min(hs)
        self.base_w = min(ws)

        self.P = min(self.patch_size, self.base_h, self.base_w)
        self.zone = (
            self.folder.parent.name
            if self.folder.name.lower() == "normalized"
            else self.folder.name
        )

    def __len__(self) -> int:
        return 1

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, self.clamp_lo, self.clamp_hi)
        arr = (arr - self.clamp_lo) / (self.clamp_hi - self.clamp_lo + 1e-8)
        return arr.astype(np.float32)

    def _pick_patch(self, start_idx: int) -> Tuple[int, int]:
        if self.base_h <= self.P or self.base_w <= self.P:
            return 0, 0
        if not self.deterministic_patch:
            return random.randint(0, self.base_h - self.P), random.randint(
                0, self.base_w - self.P
            )

        zone_crc = zlib.crc32(self.zone.encode("utf-8")) & 0xFFFFFFFF
        seed = (self.seed * 1315423911 + zone_crc + start_idx * 2654435761) & 0xFFFFFFFF
        rng = random.Random(seed)
        top = rng.randint(0, self.base_h - self.P)
        left = rng.randint(0, self.base_w - self.P)
        return top, left

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        start = 0
        top, left = self._pick_patch(start)

        paths = self.paths[: (self.total_len if self.use_future else self.input_len)]

        frames = []
        for p in paths:
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)  # [C, H, W]

            # Channel pad/crop
            if arr.shape[0] > self.in_channels:
                arr = arr[: self.in_channels]
            elif arr.shape[0] < self.in_channels:
                pad = self.in_channels - arr.shape[0]
                arr = np.concatenate(
                    [arr, np.zeros((pad, arr.shape[1], arr.shape[2]), np.float32)],
                    axis=0,
                )

            arr = self._normalize(arr)
            arr = arr[:, : self.base_h, : self.base_w]
            arr = arr[:, top : top + self.P, left : left + self.P]
            frames.append(arr)

        stack = torch.from_numpy(np.stack(frames, axis=0))  # [T, C, P, P]
        x = stack[: self.input_len]  # [T_in, C, P, P]

        if self.use_future:
            y = stack[self.input_len : self.total_len]  # [T_out, C, P, P]
        else:
            y = torch.zeros(
                (self.pred_len, self.in_channels, self.P, self.P), dtype=x.dtype
            )

        x = torch.clamp(x, 0.0, 1.0)
        y = torch.clamp(y, 0.0, 1.0)

        dummy_mask = torch.ones((self.pred_len, 1, self.P, self.P), dtype=torch.float32)
        return x, y, dummy_mask, self.zone

    def get_full_image(self, time_idx: int = 0) -> np.ndarray:
        """Get full image at a given time index without patching."""
        if time_idx >= len(self.paths):
            raise ValueError(f"time_idx {time_idx} out of range")

        with rasterio.open(self.paths[time_idx]) as src:
            arr = src.read().astype(np.float32)

        if arr.shape[0] > self.in_channels:
            arr = arr[: self.in_channels]
        elif arr.shape[0] < self.in_channels:
            pad = self.in_channels - arr.shape[0]
            arr = np.concatenate(
                [arr, np.zeros((pad, arr.shape[1], arr.shape[2]), np.float32)],
                axis=0,
            )

        return self._normalize(arr)

    def get_date_info(self) -> List[str]:
        """Get date information for each file in the sequence."""
        dates = []
        for p in self.paths:
            m = DATE_RE.match(p.name)
            if m:
                dates.append(f"{m.group(1)}-{m.group(2)}")
            else:
                dates.append(p.stem)
        return dates
