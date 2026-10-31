"""
Deep Learning Architectures and Loss Functions for Image Deblurring
Includes:
1. DeblurUNet: Multi-Scale Residual Channel-Attention U-Net with Large-Kernel Depthwise Deconvolution Blocks.
2. DeblurResNet: Deep Residual Convolutional Network (Nah et al. style).
3. DeblurAutoencoder: Classic Encoder-Decoder Convolutional Autoencoder.
4. DeblurLoss: Composite Charbonnier + Spatial Edge Gradient (Sobel) + Frequency (FFT) Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention Block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResidualBlock(nn.Module):
    """
    NAFNet-inspired Large-Kernel Residual Deconvolution Block:
    Uses a 7x7 depthwise convolution to capture wide motion/defocus blur trajectories
    followed by 1x1 channel mixing, GELU activation, 3x3 spatial refinement, and Channel Attention.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True)
        self.pw_conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.ca = ChannelAttention(channels)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw_conv(x)
        out = self.pw_conv2(self.act(self.pw_conv1(out)))
        out = self.ca(self.spatial_conv(out))
        return x + self.scale * out


class DeblurUNet(nn.Module):
    """
    Residual Channel-Attention U-Net for Image Deblurring.
    Learns the high-frequency residual detail `R = Sharp - Blurred` across 3 spatial scales,
    plus a high-frequency spatial deconvolution branch.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # High-frequency feature extractor with 3x3 and 7x7 receptive fields
        self.intro = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # Encoder Level 1
        self.enc1 = nn.Sequential(
            ResidualBlock(c1),
            ResidualBlock(c1),
        )
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2)

        # Encoder Level 2
        self.enc2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2),
        )
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c3),
            ResidualBlock(c3),
            ResidualBlock(c3),
        )

        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, kernel_size=1)
        self.dec2 = nn.Sequential(
            ResidualBlock(c2),
            ResidualBlock(c2),
        )

        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, kernel_size=1)
        self.dec1 = nn.Sequential(
            ResidualBlock(c1),
            ResidualBlock(c1),
            ResidualBlock(c1),
        )

        # Output residual projection (zero-initialized so initial output equals input)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad if input dimensions are not divisible by 4
        _, _, h, w = x.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        f0 = self.intro(x_in)
        e1 = self.enc1(f0)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(self.fuse2(torch.cat([d2, e2], dim=1)))

        d1 = self.up1(d2)
        d1 = self.dec1(self.fuse1(torch.cat([d1, e1], dim=1)) + f0)

        residual = self.out_conv(d1)
        out = torch.clamp(x_in + residual, 0.0, 1.0)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
        return out


class DeblurResNet(nn.Module):
    """Deep Residual CNN without downsampling for high-detail local restoration."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 48, num_blocks: int = 8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body(feat)
        residual = self.tail(feat)
        return torch.clamp(x + residual, 0.0, 1.0)


class DeblurAutoencoder(nn.Module):
    """Convolutional Autoencoder with Residual Prediction."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(c3),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.decoder(self.encoder(x))
        return torch.clamp(x + residual, 0.0, 1.0)


def build_model(arch: str = "unet", base_channels: int = 32) -> nn.Module:
    """Factory helper to instantiate the chosen deblurring architecture."""
    arch = arch.lower().strip()
    if arch == "unet":
        return DeblurUNet(base_channels=base_channels)
    elif arch == "resnet":
        return DeblurResNet(base_channels=base_channels)
    elif arch == "autoencoder":
        return DeblurAutoencoder(base_channels=base_channels)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from ['unet', 'resnet', 'autoencoder'].")


class DeblurLoss(nn.Module):
    """
    Hybrid Loss Function for Image Deblurring:
    L_total = L_charbonnier + lambda_edge * L_edge + lambda_fft * L_fft
    - Charbonnier Loss: Robust L1 loss that avoids over-smoothing.
    - Sobel Edge Gradient Loss: Explicitly penalizes blurry boundaries.
    - FFT Frequency Loss: Encourages restoration of high-frequency spectral components.
    """

    def __init__(self, eps: float = 1e-3, edge_weight: float = 0.25, fft_weight: float = 0.05):
        super().__init__()
        self.eps = eps
        self.edge_weight = edge_weight
        self.fft_weight = fft_weight

        # Register fixed Sobel filters for horizontal and vertical gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))

    def charbonnier(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_p = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        gy_p = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        gx_t = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        gy_t = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        return self.charbonnier(gx_p, gx_t) + self.charbonnier(gy_p, gy_t)

    def fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft_p = torch.fft.rfft2(pred, norm="ortho")
        fft_t = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(torch.abs(fft_p), torch.abs(fft_t))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_char = self.charbonnier(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        return l_char + self.edge_weight * l_edge + self.fft_weight * l_fft
