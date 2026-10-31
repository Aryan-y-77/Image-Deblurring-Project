"""
CLI Inference Tool for Image Deblurring
Restores blurred photographs using the trained Deep Learning model.
Supports:
- Restoring any real-world blurred image (`--input my_blurred_photo.jpg`).
- Optionally applying synthetic motion/defocus blur to a sharp image first (`--add-blur motion`)
  to demonstrate restoration with exact PSNR & SSIM metrics.
- Generating a side-by-side comparison figure (`--compare`).
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import apply_synthetic_blur
from metrics import calculate_psnr, calculate_sharpness, calculate_ssim
from models import build_model


def restore_single_image(
    model: torch.nn.Module,
    img_rgb_f32: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run deep learning inference on an RGB float32 image in [0, 1]."""
    model.eval()
    with torch.no_grad():
        tensor_in = torch.from_numpy(img_rgb_f32.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        restored_t = model(tensor_in).squeeze(0).cpu().clamp(0.0, 1.0)
        restored_np = restored_t.permute(1, 2, 0).numpy()
    return restored_np


def run_inference(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project_dir / ckpt_path

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run `python train.py` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "unet")
    base_channels = ckpt.get("base_channels", 24)

    model = build_model(arch=arch, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    input_path = Path(args.input)
    if not input_path.is_absolute() and not input_path.exists():
        input_path = project_dir / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image file: {input_path}")
    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    gt_rgb = None
    if args.add_blur != "none":
        gt_rgb = orig_rgb.copy()
        rng = np.random.default_rng(seed=42)
        blurred_rgb, applied_mode = apply_synthetic_blur(orig_rgb, blur_type=args.add_blur, rng=rng)
        print(f"[Info] Applied synthetic '{applied_mode}' blur to input sharp image for demonstration.")
    else:
        blurred_rgb = orig_rgb
        if args.ground_truth:
            gt_path = Path(args.ground_truth)
            if not gt_path.is_absolute() and not gt_path.exists():
                gt_path = project_dir / gt_path
            if gt_path.exists():
                gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
                gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    restored_rgb = restore_single_image(model, blurred_rgb, device)

    # Save restored image
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    restored_u8 = np.clip(restored_rgb * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(restored_u8, cv2.COLOR_RGB2BGR))

    blur_sharpness = calculate_sharpness(blurred_rgb)
    rest_sharpness = calculate_sharpness(restored_rgb)

    print("=" * 68)
    print("DEEP LEARNING IMAGE DEBLURRING — INFERENCE SUMMARY")
    print("=" * 68)
    print(f"Input Image          : {input_path}")
    print(f"Restored Output      : {output_path}")
    print(f"Laplacian Sharpness  : {blur_sharpness:.1f} (Blurred) -> {rest_sharpness:.1f} (Restored)")

    if gt_rgb is not None:
        b_psnr = calculate_psnr(blurred_rgb, gt_rgb)
        b_ssim = calculate_ssim(blurred_rgb, gt_rgb)
        r_psnr = calculate_psnr(restored_rgb, gt_rgb)
        r_ssim = calculate_ssim(restored_rgb, gt_rgb)
        print(f"PSNR (vs. Reference) : {b_psnr:.2f} dB (Blurred) -> {r_psnr:.2f} dB (Restored) [+{r_psnr - b_psnr:.2f} dB]")
        print(f"SSIM (vs. Reference) : {b_ssim:.4f} (Blurred) -> {r_ssim:.4f} (Restored) [+{r_ssim - b_ssim:.4f}]")

    if args.compare:
        comp_path = output_path.with_name(output_path.stem + "_comparison.png")
        ncols = 3 if gt_rgb is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.8), dpi=160)

        if gt_rgb is not None:
            axes[0].imshow(np.clip(blurred_rgb, 0, 1))
            axes[0].set_title(f"Blurred Input\nPSNR: {b_psnr:.2f} dB | SSIM: {b_ssim:.3f}", fontsize=11)
            axes[1].imshow(np.clip(restored_rgb, 0, 1))
            axes[1].set_title(f"Deep Learning Restored\nPSNR: {r_psnr:.2f} dB | SSIM: {r_ssim:.3f}", fontsize=11, fontweight="bold")
            axes[2].imshow(np.clip(gt_rgb, 0, 1))
            axes[2].set_title("Ground Truth Sharp Reference", fontsize=11)
        else:
            axes[0].imshow(np.clip(blurred_rgb, 0, 1))
            axes[0].set_title(f"Blurred Input\nSharpness Score: {blur_sharpness:.1f}", fontsize=11)
            axes[1].imshow(np.clip(restored_rgb, 0, 1))
            axes[1].set_title(f"Deep Learning Restored\nSharpness Score: {rest_sharpness:.1f}", fontsize=11, fontweight="bold")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(comp_path, bbox_inches="tight", dpi=160)
        plt.close(fig)
        print(f"Comparison Figure    : {comp_path}")
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore a blurred photograph using the trained Deep Learning model")
    parser.add_argument("--input", type=str, required=True, help="Path to input blurred (or sharp) image")
    parser.add_argument("--output", type=str, default="results/restored_sample.png", help="Path to save restored image")
    parser.add_argument("--ground-truth", type=str, default="", help="Optional path to sharp reference image for PSNR/SSIM")
    parser.add_argument(
        "--add-blur",
        type=str,
        default="none",
        choices=["none", "motion", "defocus", "gaussian", "random"],
        help="Optionally apply synthetic blur to a sharp input photo before restoring",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deblur_model.pth", help="Model checkpoint path")
    parser.add_argument("--compare", action="store_true", default=True, help="Save a side-by-side comparison plot")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
