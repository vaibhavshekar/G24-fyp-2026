"""3D U-Net model for satellite image time series forecasting."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(num_channels: int) -> nn.Module:
    """Create adaptive normalization layer based on channel count."""
    for g in [8, 4, 2, 1]:
        if num_channels % g == 0:
            if g == 1:
                return nn.InstanceNorm3d(num_channels, affine=True)
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.InstanceNorm3d(num_channels, affine=True)


class DoubleConv3D(nn.Module):
    """Double 3D convolution block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = make_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = make_norm(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        return self.act(out + identity)


class Down3D(nn.Module):
    """Downsampling block with max pooling and double convolution."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.conv = DoubleConv3D(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upsampling block with skip connection and double convolution."""

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv = DoubleConv3D(in_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffT = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffT // 2,
                diffT - diffT // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3DTemporal(nn.Module):
    """
    3D U-Net for spatiotemporal satellite image forecasting.

    Input:  [B, T_in, C, H, W]
    Output: [B, T_out, C, H, W]

    Uses residual prediction: output = last_input + learned_residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_c: int = 12,
        pred_len: int = 3,
    ):
        super().__init__()
        self.pred_len = pred_len

        # Dropout rates
        d1, d2 = 0.10, 0.20
        db = 0.40

        # Encoder
        self.inc = DoubleConv3D(in_channels, base_c, dropout_p=d1)
        self.down1 = Down3D(base_c, base_c * 2, dropout_p=d1)
        self.down2 = Down3D(base_c * 2, base_c * 4, dropout_p=d2)
        self.down3 = Down3D(base_c * 4, base_c * 8, dropout_p=d2)

        # Bottleneck
        self.bottleneck = nn.Sequential(DoubleConv3D(base_c * 8, base_c * 8, dropout_p=db))

        # Decoder
        self.up1 = Up3D(base_c * 8 + base_c * 4, base_c * 4, dropout_p=d2)
        self.up2 = Up3D(base_c * 4 + base_c * 2, base_c * 2, dropout_p=d1)
        self.up3 = Up3D(base_c * 2 + base_c, base_c, dropout_p=d1)

        # Output
        self.outc = nn.Conv3d(base_c, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_in, C, H, W]
        last_in = x[:, -1:, :, :, :]  # [B, 1, C, H, W]
        x5d = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        # Encoder
        x1 = self.inc(x5d)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.bottleneck(x4)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Output
        out_full = self.outc(x)  # [B, C, T_in, H, W]
        res = out_full[:, :, -self.pred_len :, :, :]  # [B, C, T_out, H, W]
        res = res.permute(0, 2, 1, 3, 4)  # [B, T_out, C, H, W]

        # Residual connection
        last_rep = last_in.repeat(1, self.pred_len, 1, 1, 1)
        out = last_rep + res
        return out


def load_model(
    checkpoint_path: str,
    in_channels: int = 10,
    base_channels: int = 12,
    pred_len: int = 3,
    device: str = "cpu",
) -> UNet3DTemporal:
    """Load model from checkpoint."""
    model = UNet3DTemporal(
        in_channels=in_channels,
        out_channels=in_channels,
        base_c=base_channels,
        pred_len=pred_len,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def get_model_info(checkpoint_path: str, device: str = "cpu") -> dict:
    """Get information from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    info = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "val_loss": checkpoint.get("val_loss", "N/A"),
        "loss_name": checkpoint.get("loss_name", "N/A"),
    }
    if "config" in checkpoint:
        info["config"] = checkpoint["config"]
    return info
