"""
Comprehensive Evaluation & Visualization Script for Image Deblurring
Evaluates the trained Deep Learning model on the unseen test split, compares it
against a classical Unsharp-Wiener baseline, and generates side-by-side visual
restoration grids and metric comparison charts for the project report.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import DeblurDataset
from metrics import calculate_psnr, calculate_sharpness, calculate_ssim
from models import build_model


def classical_deblur_baseline(blurred_rgb: np.ndarray) -> np.ndarray:
    """
    Classical image restoration baseline combining bilateral denoising and
    multi-scale unsharp masking / Laplacian frequency boosting.
    """
    img_u8 = np.clip(blurred_rgb * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.bilateralFilter(img_u8, d=5, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255.0
    gaussian = cv2.GaussianBlur(denoised, (5, 5), 1.4)
    sharpened = np.clip(denoised + 0.85 * (denoised - gaussian), 0.0, 1.0)
    return sharpened


def evaluate(args: argparse.Namespace) -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / args.data_dir
    ckpt_path = project_dir / args.checkpoint
    results_dir = project_dir / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run `python train.py` first.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "unet")
    base_channels = ckpt.get("base_channels", 24)

    model = build_model(arch=arch, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = DeblurDataset(data_dir / "test", patch_size=None, augment=False)

    records = []
    visual_samples = []

    print("=" * 84)
    print(f"EVALUATING DEEP LEARNING DEBLURRING MODEL ({arch.upper()}) ON {len(test_dataset)} TEST IMAGES")
    print("=" * 84)
    print(f"{'Image Name':<26} | {'Blur Type':<10} | {'Blurred PSNR':<13} | {'Classical PSNR':<14} | {'DL PSNR (dB)':<13} | {'DL SSIM':<8}")
    print("-" * 84)

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            blur_t, sharp_t, fname = test_dataset[idx]
            blur_np = blur_t.permute(1, 2, 0).numpy()
            sharp_np = sharp_t.permute(1, 2, 0).numpy()

            # Classical baseline
            classical_np = classical_deblur_baseline(blur_np)

            # Deep Learning inference
            inp = blur_t.unsqueeze(0).to(device)
            restored_t = model(inp).squeeze(0).cpu().clamp(0.0, 1.0)
            restored_np = restored_t.permute(1, 2, 0).numpy()

            # Determine blur category from filename
            blur_type = "custom"
            for b_mode in ("motion", "defocus", "gaussian"):
                if b_mode in fname.lower():
                    blur_type = b_mode
                    break

            b_psnr = calculate_psnr(blur_np, sharp_np)
            b_ssim = calculate_ssim(blur_np, sharp_np)
            c_psnr = calculate_psnr(classical_np, sharp_np)
            c_ssim = calculate_ssim(classical_np, sharp_np)
            dl_psnr = calculate_psnr(restored_np, sharp_np)
            dl_ssim = calculate_ssim(restored_np, sharp_np)

            b_sharp = calculate_sharpness(blur_np)
            dl_sharp = calculate_sharpness(restored_np)
            gt_sharp = calculate_sharpness(sharp_np)

            records.append(
                {
                    "filename": fname,
                    "blur_type": blur_type,
                    "blurred_psnr": b_psnr,
                    "blurred_ssim": b_ssim,
                    "classical_psnr": c_psnr,
                    "classical_ssim": c_ssim,
                    "dl_psnr": dl_psnr,
                    "dl_ssim": dl_ssim,
                    "psnr_gain": dl_psnr - b_psnr,
                    "ssim_gain": dl_ssim - b_ssim,
                    "blurred_sharpness": b_sharp,
                    "dl_sharpness": dl_sharp,
                    "gt_sharpness": gt_sharp,
                }
            )

            print(
                f"{fname:<26} | {blur_type:<10} | "
                f"{b_psnr:>6.2f} dB     | {c_psnr:>6.2f} dB      | "
                f"{dl_psnr:>6.2f} dB     | {dl_ssim:>6.4f}"
            )

            if len(visual_samples) < 4:
                visual_samples.append((fname, blur_type, blur_np, classical_np, restored_np, sharp_np, records[-1]))

    # Compute overall averages
    avg_b_psnr = float(np.mean([r["blurred_psnr"] for r in records]))
    avg_b_ssim = float(np.mean([r["blurred_ssim"] for r in records]))
    avg_c_psnr = float(np.mean([r["classical_psnr"] for r in records]))
    avg_c_ssim = float(np.mean([r["classical_ssim"] for r in records]))
    avg_dl_psnr = float(np.mean([r["dl_psnr"] for r in records]))
    avg_dl_ssim = float(np.mean([r["dl_ssim"] for r in records]))

    print("-" * 84)
    print(
        f"{'AVERAGE OVER TEST SET':<26} | {'ALL':<10} | "
        f"{avg_b_psnr:>6.2f} dB     | {avg_c_psnr:>6.2f} dB      | "
        f"{avg_dl_psnr:>6.2f} dB     | {avg_dl_ssim:>6.4f}"
    )
    print("=" * 84)
    print(f"Net PSNR Improvement : +{avg_dl_psnr - avg_b_psnr:.2f} dB over Blurred Input (+{avg_dl_psnr - avg_c_psnr:.2f} dB over Classical Filter)")
    print(f"Net SSIM Improvement : +{avg_dl_ssim - avg_b_ssim:.4f} over Blurred Input")

    # Save CSV report
    csv_path = results_dir / "evaluation_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save visual comparison grid
    vis_path = results_dir / "visual_comparison.png"
    _plot_visual_comparison(visual_samples, vis_path)

    # Save metrics comparison bar chart
    chart_path = results_dir / "metrics_comparison.png"
    _plot_metrics_chart(records, chart_path)

    print(f"Saved Detailed CSV Report  : {csv_path}")
    print(f"Saved Visual Comparison    : {vis_path}")
    print(f"Saved Metrics Bar Chart    : {chart_path}")
    print("=" * 84)


def _plot_visual_comparison(samples: list, out_path: Path) -> None:
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.4 * n_rows), dpi=160)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (fname, b_type, blur_np, class_np, dl_np, sharp_np, rec) in enumerate(samples):
        panels = [
            (blur_np, f"Blurred Input ({b_type.title()})\nPSNR: {rec['blurred_psnr']:.2f} dB | SSIM: {rec['blurred_ssim']:.3f}"),
            (class_np, f"Classical Unsharp Filter\nPSNR: {rec['classical_psnr']:.2f} dB | SSIM: {rec['classical_ssim']:.3f}"),
            (dl_np, f"Deep Learning Restored (Ours)\nPSNR: {rec['dl_psnr']:.2f} dB | SSIM: {rec['dl_ssim']:.3f}"),
            (sharp_np, "Ground Truth Sharp\nReference (PSNR: inf | SSIM: 1.000)"),
        ]
        for col_idx, (img, title) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.set_title(title, fontsize=9.5, fontweight="bold" if col_idx == 2 else "normal")
            ax.axis("off")

    plt.suptitle("Qualitative Comparison: Blurred vs. Classical vs. Deep Learning Restoration", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def _plot_metrics_chart(records: list, out_path: Path) -> None:
    categories = ["motion", "defocus", "gaussian"]
    present_cats = [c for c in categories if any(r["blur_type"] == c for r in records)]
    labels = [c.title() + " Blur" for c in present_cats] + ["Overall Avg"]

    b_psnrs, c_psnrs, dl_psnrs = [], [], []
    b_ssims, c_ssims, dl_ssims = [], [], []

    for c in present_cats:
        sub = [r for r in records if r["blur_type"] == c]
        b_psnrs.append(np.mean([r["blurred_psnr"] for r in sub]))
        c_psnrs.append(np.mean([r["classical_psnr"] for r in sub]))
        dl_psnrs.append(np.mean([r["dl_psnr"] for r in sub]))
        b_ssims.append(np.mean([r["blurred_ssim"] for r in sub]))
        c_ssims.append(np.mean([r["classical_ssim"] for r in sub]))
        dl_ssims.append(np.mean([r["dl_ssim"] for r in sub]))

    b_psnrs.append(np.mean([r["blurred_psnr"] for r in records]))
    c_psnrs.append(np.mean([r["classical_psnr"] for r in records]))
    dl_psnrs.append(np.mean([r["dl_psnr"] for r in records]))
    b_ssims.append(np.mean([r["blurred_ssim"] for r in records]))
    c_ssims.append(np.mean([r["classical_ssim"] for r in records]))
    dl_ssims.append(np.mean([r["dl_ssim"] for r in records]))

    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)

    # PSNR Bar Chart
    axes[0].bar(x - width, b_psnrs, width, label="Blurred Input", color="#9ca3af")
    axes[0].bar(x, c_psnrs, width, label="Classical Filter", color="#f59e0b")
    axes[0].bar(x + width, dl_psnrs, width, label="Deep Learning U-Net (Ours)", color="#10b981")
    axes[0].set_title("PSNR Comparison Across Blur Types (dB)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("PSNR (dB) [Higher is Better]")
    axes[0].set_ylim(bottom=max(0, min(b_psnrs) - 5))
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend()

    # SSIM Bar Chart
    axes[1].bar(x - width, b_ssims, width, label="Blurred Input", color="#9ca3af")
    axes[1].bar(x, c_ssims, width, label="Classical Filter", color="#f59e0b")
    axes[1].bar(x + width, dl_ssims, width, label="Deep Learning U-Net (Ours)", color="#6366f1")
    axes[1].set_title("SSIM Comparison Across Blur Types", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("SSIM [0 - 1, Higher is Better]")
    axes[1].set_ylim(bottom=max(0.0, min(b_ssims) - 0.15), top=1.02)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("Quantitative Benchmark: Classical Filtering vs. Deep Learning Image Deblurring", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Deep Learning Image Deblurring Model")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_deblur_model.pth", help="Path to model checkpoint")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation figures and CSV")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
