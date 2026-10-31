"""
Evaluation Metrics for Image Deblurring
Includes PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index),
RMSE, and Laplacian Variance (no-reference sharpness score).
"""

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: First image array (float in [0, 1] or uint8 in [0, 255]).
        img2: Second image array of the same shape.
        max_val: Maximum pixel value (1.0 for float, 255.0 for uint8).
    Returns:
        PSNR value in decibels (dB).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images (Wang et al., 2004).
    Supports both grayscale (H, W) and multi-channel RGB (H, W, C) arrays.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3 and img1.shape[2] in (3, 4):
        ssim_channels = [
            _ssim_single_channel(img1[..., c], img2[..., c], max_val)
            for c in range(img1.shape[2])
        ]
        return float(np.mean(ssim_channels))
    return _ssim_single_channel(img1, img2, max_val)


def _ssim_single_channel(ch1: np.ndarray, ch2: np.ndarray, max_val: float = 1.0) -> float:
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = cv2.GaussianBlur(ch1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(ch2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(ch1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(ch2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ch1 * ch2, (11, 11), 1.5) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def calculate_sharpness(img: np.ndarray) -> float:
    """
    Calculate no-reference image sharpness using the Variance of the Laplacian.
    Higher values indicate sharper edges and less blur.
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img

    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_u8

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


@torch.no_grad()
def batch_psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """
    Compute average PSNR and SSIM for a batch of PyTorch tensors in [B, C, H, W] with values in [0, 1].
    """
    pred_np = pred.detach().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy()

    psnr_list = []
    ssim_list = []
    for p, t in zip(pred_np, target_np):
        psnr_list.append(calculate_psnr(p, t, max_val=1.0))
        ssim_list.append(calculate_ssim(p, t, max_val=1.0))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))
