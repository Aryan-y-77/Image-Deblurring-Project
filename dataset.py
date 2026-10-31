"""
Dataset Pipeline and Blur Synthesis Engine for Image Deblurring
Supports:
1. Loading paired datasets (e.g., GoPro, DIV2K, RealBlur) from `blurred/` and `sharp/` folders.
2. Generating synthetic Motion Blur, Defocus (Disk) Blur, and Gaussian Blur from sharp images.
3. Generating a high-detail benchmark dataset automatically when no external dataset is present.
"""

import random
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def create_motion_blur_kernel(kernel_size: int = 9, angle: float = 35.0) -> np.ndarray:
    """
    Generate a 2D directional motion blur point-spread function (PSF) kernel.
    """
    kernel_size = max(3, int(kernel_size) | 1)  # ensure odd size
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)

    for i in range(-center, center + 1):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0

    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[center, center] = 1.0
    return kernel


def create_defocus_disk_kernel(radius: int = 3) -> np.ndarray:
    """
    Generate a circular disk PSF kernel representing out-of-focus (defocus) lens blur.
    """
    radius = max(1, int(radius))
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1, lineType=cv2.LINE_AA)
    kernel /= np.sum(kernel)
    return kernel


def apply_synthetic_blur(
    sharp_img: np.ndarray,
    blur_type: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, str]:
    """
    Apply realistic optical/motion blur to a sharp RGB float32 image in [0, 1].
    Args:
        sharp_img: Sharp RGB image of shape (H, W, 3) in [0, 1].
        blur_type: One of 'motion', 'defocus', 'gaussian', or 'random'.
        rng: Optional NumPy random generator for reproducibility.
    Returns:
        Tuple of (blurred_img in [0, 1], applied_blur_type).
    """
    if rng is None:
        rng = np.random.default_rng()

    if blur_type == "random":
        blur_type = str(rng.choice(["motion", "defocus", "gaussian"]))

    if blur_type == "motion":
        ksize = int(rng.choice([7, 9]))
        angle = float(rng.choice([25.0, 45.0, 135.0]))
        kernel = create_motion_blur_kernel(ksize, angle)
        blurred = cv2.filter2D(sharp_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    elif blur_type == "defocus":
        radius = int(rng.choice([3, 4]))
        kernel = create_defocus_disk_kernel(radius)
        blurred = cv2.filter2D(sharp_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    elif blur_type == "gaussian":
        ksize = int(rng.choice([7, 9]))
        sigma = float(rng.uniform(1.6, 2.2))
        blurred = cv2.GaussianBlur(sharp_img, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT)
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}")

    # Subtle sensor readout noise
    noise = rng.normal(0.0, 0.002, size=blurred.shape).astype(np.float32)
    blurred = np.clip(blurred + noise, 0.0, 1.0)
    return blurred.astype(np.float32), blur_type


def _generate_rich_sharp_scene(index: int, size: int = 96) -> np.ndarray:
    """
    Synthesize a rich, multi-scale sharp photograph containing natural gradients,
    architectural edges, geometric details, and high-frequency typography
    for training and benchmarking image deblurring models.
    """
    rng = np.random.default_rng(seed=1000 + index)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Smooth sky/background gradient
    c_top = rng.integers(40, 210, size=3)
    c_bot = rng.integers(40, 210, size=3)
    for y in range(size):
        alpha = y / max(1, size - 1)
        img[y, :] = ((1 - alpha) * c_top + alpha * c_bot).astype(np.uint8)

    # 2. Gentle natural wave shading
    yy, xx = np.meshgrid(np.linspace(0, 3 * np.pi, size), np.linspace(0, 3 * np.pi, size), indexing="ij")
    freq_x = rng.uniform(1.0, 2.5)
    freq_y = rng.uniform(1.0, 2.5)
    pattern = 14.0 * np.sin(freq_x * xx + freq_y * yy)
    img = np.clip(img.astype(np.float32) + pattern[..., None], 0, 255).astype(np.uint8)

    # 3. Architectural window / panel grid with structured contrast
    gx = int(rng.integers(8, size // 3))
    gy = int(rng.integers(8, size // 3))
    gw = int(rng.integers(size // 3, size // 2))
    gh = int(rng.integers(size // 3, size // 2))
    cell = 12
    c1 = tuple(int(v) for v in rng.integers(30, 95, size=3))
    c2 = tuple(int(v) for v in rng.integers(175, 245, size=3))
    for r in range(gy, min(size, gy + gh), cell):
        for c in range(gx, min(size, gx + gw), cell):
            color = c1 if ((r // cell) + (c // cell)) % 2 == 0 else c2
            cv2.rectangle(img, (c, r), (min(size - 1, c + cell), min(size - 1, r + cell)), color, -1)

    # 4. Crisp geometric objects (rectangles, circles, structural contours)
    for _ in range(4):
        shape_type = int(rng.integers(0, 3))
        color = tuple(int(v) for v in rng.integers(20, 240, size=3))
        border_color = tuple(int(v) for v in rng.integers(10, 250, size=3))
        if shape_type == 0:
            pt1 = (int(rng.integers(6, size - 28)), int(rng.integers(6, size - 28)))
            pt2 = (pt1[0] + int(rng.integers(18, 38)), pt1[1] + int(rng.integers(18, 38)))
            cv2.rectangle(img, pt1, pt2, color, -1)
            cv2.rectangle(img, pt1, pt2, border_color, 2, lineType=cv2.LINE_AA)
        elif shape_type == 1:
            center = (int(rng.integers(16, size - 16)), int(rng.integers(16, size - 16)))
            radius = int(rng.integers(10, 22))
            cv2.circle(img, center, radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, center, radius, border_color, 2, lineType=cv2.LINE_AA)
        else:
            pt1 = (int(rng.integers(4, size - 4)), int(rng.integers(4, size - 4)))
            pt2 = (int(rng.integers(4, size - 4)), int(rng.integers(4, size - 4)))
            cv2.line(img, pt1, pt2, color, 2, lineType=cv2.LINE_AA)

    # 5. Bold signage / text for visual sharpness verification
    words = ["AI-ML", "SHARP", "2026", "HD-4K", "OPTIC", "VISION", "DEBLUR", "UNET"]
    word = words[index % len(words)]
    tx = int(rng.integers(6, max(10, size - 68)))
    ty = int(rng.integers(22, max(28, size - 8)))
    cv2.putText(img, word, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(img, word, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)

    return img


def prepare_synthetic_dataset(
    root_dir: str | Path,
    num_train: int = 120,
    num_val: int = 24,
    num_test: int = 18,
    img_size: int = 96,
    force_regenerate: bool = False,
) -> Path:
    """
    Create a structured paired dataset (`train`, `val`, `test` splits with `sharp/` and `blurred/` folders)
    if it does not already exist.
    """
    root_path = Path(root_dir)
    splits = [("train", num_train, 0), ("val", num_val, 1000), ("test", num_test, 2000)]

    for split_name, count, offset in splits:
        sharp_dir = root_path / split_name / "sharp"
        blur_dir = root_path / split_name / "blurred"
        sharp_dir.mkdir(parents=True, exist_ok=True)
        blur_dir.mkdir(parents=True, exist_ok=True)

        existing = list(sharp_dir.glob("*.png"))
        if len(existing) >= count and not force_regenerate:
            continue

        for old_f in sharp_dir.glob("*.png"):
            old_f.unlink()
        for old_f in blur_dir.glob("*.png"):
            old_f.unlink()

        blur_modes = ["motion", "defocus", "gaussian"]
        for i in range(count):
            idx = offset + i
            rng = np.random.default_rng(seed=5000 + idx)
            sharp_rgb = _generate_rich_sharp_scene(idx, size=img_size)
            sharp_f32 = sharp_rgb.astype(np.float32) / 255.0

            b_mode = blur_modes[i % len(blur_modes)]
            blurred_f32, _ = apply_synthetic_blur(sharp_f32, blur_type=b_mode, rng=rng)
            blurred_rgb = np.clip(blurred_f32 * 255.0, 0, 255).astype(np.uint8)

            fname = f"sample_{i:04d}_{b_mode}.png"
            cv2.imwrite(str(sharp_dir / fname), cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(blur_dir / fname), cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR))

    return root_path


class DeblurDataset(Dataset):
    """
    PyTorch Dataset for Paired Image Deblurring.
    Reads paired images from `<split_dir>/blurred` and `<split_dir>/sharp`, or synthesizes
    blur on-the-fly if only sharp images are provided.
    """

    def __init__(
        self,
        split_dir: str | Path,
        patch_size: Optional[int] = 96,
        augment: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.patch_size = patch_size
        self.augment = augment

        sharp_dir = self.split_dir / "sharp"
        blur_dir = self.split_dir / "blurred"

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        self.sharp_paths: List[Path] = []
        for ext in exts:
            self.sharp_paths.extend(sorted(sharp_dir.glob(ext)))

        if not self.sharp_paths:
            raise FileNotFoundError(f"No sharp images found in {sharp_dir}")

        self.blur_paths: List[Optional[Path]] = []
        for sp in self.sharp_paths:
            bp = blur_dir / sp.name
            self.blur_paths.append(bp if bp.exists() else None)

    def __len__(self) -> int:
        return len(self.sharp_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sharp_path = self.sharp_paths[idx]
        blur_path = self.blur_paths[idx]

        sharp_bgr = cv2.imread(str(sharp_path), cv2.IMREAD_COLOR)
        sharp_rgb = cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if blur_path is not None and blur_path.exists():
            blur_bgr = cv2.imread(str(blur_path), cv2.IMREAD_COLOR)
            blur_rgb = cv2.cvtColor(blur_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            blur_rgb, _ = apply_synthetic_blur(sharp_rgb, blur_type="random")

        # Optional random crop / resize to patch_size
        if self.patch_size is not None:
            h, w, _ = sharp_rgb.shape
            ps = self.patch_size
            if h > ps and w > ps:
                if self.augment:
                    top = random.randint(0, h - ps)
                    left = random.randint(0, w - ps)
                else:
                    top = (h - ps) // 2
                    left = (w - ps) // 2
                sharp_rgb = sharp_rgb[top : top + ps, left : left + ps]
                blur_rgb = blur_rgb[top : top + ps, left : left + ps]
            elif h != ps or w != ps:
                sharp_rgb = cv2.resize(sharp_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)
                blur_rgb = cv2.resize(blur_rgb, (ps, ps), interpolation=cv2.INTER_CUBIC)

        # Geometric data augmentation (flips)
        if self.augment:
            if random.random() < 0.5:
                sharp_rgb = np.fliplr(sharp_rgb).copy()
                blur_rgb = np.fliplr(blur_rgb).copy()
            if random.random() < 0.5:
                sharp_rgb = np.flipud(sharp_rgb).copy()
                blur_rgb = np.flipud(blur_rgb).copy()

        blur_tensor = torch.from_numpy(blur_rgb.transpose(2, 0, 1)).float()
        sharp_tensor = torch.from_numpy(sharp_rgb.transpose(2, 0, 1)).float()
        return blur_tensor, sharp_tensor, sharp_path.name
