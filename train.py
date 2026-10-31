"""
Training Pipeline for Image Deblurring using Deep Learning
Trains a custom Residual Attention U-Net / ResNet / Autoencoder from scratch,
logs epoch-wise Loss, PSNR, and SSIM, saves model checkpoints, and generates
publication-ready training curves for your project report.
"""

import argparse
import csv
import os
import time
from pathlib import Path

# Configure writable Matplotlib directory before importing pyplot
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DeblurDataset, prepare_synthetic_dataset
from metrics import batch_psnr_ssim
from models import DeblurLoss, build_model


def plot_training_curves(history: dict, output_path: Path) -> None:
    """
    Generate a 3-panel figure showing Training & Validation Loss, PSNR (dB), and SSIM over epochs.
    """
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Loss Curve
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2563eb", linewidth=2, markersize=4)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#dc2626", linewidth=2, markersize=4)
    axes[0].set_title("Hybrid Deblur Loss (Charbonnier + Edge + FFT)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # 2. Validation PSNR Curve
    axes[1].plot(epochs, history["train_psnr"], "o-", label="Train PSNR", color="#059669", linewidth=2, markersize=4)
    axes[1].plot(epochs, history["val_psnr"], "s--", label="Val PSNR", color="#d97706", linewidth=2, markersize=4)
    axes[1].axhline(y=history["baseline_psnr"][0], color="#6b7280", linestyle=":", linewidth=1.8, label="Blurred Input Baseline")
    axes[1].set_title("Peak Signal-to-Noise Ratio (PSNR in dB)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB) [Higher is Better]")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # 3. Validation SSIM Curve
    axes[2].plot(epochs, history["train_ssim"], "o-", label="Train SSIM", color="#7c3aed", linewidth=2, markersize=4)
    axes[2].plot(epochs, history["val_ssim"], "s--", label="Val SSIM", color="#db2777", linewidth=2, markersize=4)
    axes[2].axhline(y=history["baseline_ssim"][0], color="#6b7280", linestyle=":", linewidth=1.8, label="Blurred Input Baseline")
    axes[2].set_title("Structural Similarity Index (SSIM)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM [0 - 1, Higher is Better]")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.suptitle("Deep Learning Image Deblurring — Training & Validation Performance", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / args.data_dir
    ckpt_dir = project_dir / args.checkpoint_dir
    results_dir = project_dir / args.results_dir

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure dataset exists (auto-generates synthetic dataset if not already present)
    prepare_synthetic_dataset(
        data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        img_size=args.patch_size,
        force_regenerate=args.force_regen,
    )

    train_dataset = DeblurDataset(data_dir / "train", patch_size=args.patch_size, augment=True)
    val_dataset = DeblurDataset(data_dir / "val", patch_size=args.patch_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch=args.arch, base_channels=args.base_channels).to(device)
    criterion = DeblurLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 72)
    print("IMAGE DEBLURRING USING DEEP LEARNING -- TRAINING PIPELINE")
    print("=" * 72)
    print(f"Architecture     : {args.arch.upper()} (Base Channels: {args.base_channels})")
    print(f"Trainable Params : {num_params:,}")
    print(f"Compute Device   : {device}")
    print(f"Dataset Samples  : {len(train_dataset)} Train | {len(val_dataset)} Val (Patch: {args.patch_size}x{args.patch_size})")
    print("=" * 72)

    # Measure initial blurred input baseline PSNR/SSIM on validation set
    base_psnrs, base_ssims = [], []
    for blur_b, sharp_b, _ in val_loader:
        p, s = batch_psnr_ssim(blur_b, sharp_b)
        base_psnrs.append(p)
        base_ssims.append(s)
    baseline_psnr = float(np.mean(base_psnrs))
    baseline_ssim = float(np.mean(base_ssims))
    print(f"Blurred Input Baseline -> Val PSNR: {baseline_psnr:.2f} dB | Val SSIM: {baseline_ssim:.4f}")
    print("-" * 72)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_psnr": [],
        "val_psnr": [],
        "train_ssim": [],
        "val_ssim": [],
        "baseline_psnr": [],
        "baseline_ssim": [],
    }

    best_val_psnr = -1.0
    best_ckpt_path = ckpt_dir / f"best_{args.arch}_deblur.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_losses, tr_psnrs, tr_ssims = [], [], []

        for blur_imgs, sharp_imgs, _ in train_loader:
            blur_imgs = blur_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)

            optimizer.zero_grad()
            restored = model(blur_imgs)
            loss = criterion(restored, sharp_imgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_losses.append(loss.item())
            p, s = batch_psnr_ssim(restored, sharp_imgs)
            tr_psnrs.append(p)
            tr_ssims.append(s)

        scheduler.step()

        # Validation phase
        model.eval()
        val_losses, val_psnrs, val_ssims = [], [], []
        with torch.no_grad():
            for blur_imgs, sharp_imgs, _ in val_loader:
                blur_imgs = blur_imgs.to(device)
                sharp_imgs = sharp_imgs.to(device)
                restored = model(blur_imgs)
                v_loss = criterion(restored, sharp_imgs)
                val_losses.append(v_loss.item())
                vp, vs = batch_psnr_ssim(restored, sharp_imgs)
                val_psnrs.append(vp)
                val_ssims.append(vs)

        ep_tr_loss = float(np.mean(tr_losses))
        ep_val_loss = float(np.mean(val_losses))
        ep_tr_psnr = float(np.mean(tr_psnrs))
        ep_val_psnr = float(np.mean(val_psnrs))
        ep_tr_ssim = float(np.mean(tr_ssims))
        ep_val_ssim = float(np.mean(val_ssims))

        history["epoch"].append(epoch)
        history["train_loss"].append(ep_tr_loss)
        history["val_loss"].append(ep_val_loss)
        history["train_psnr"].append(ep_tr_psnr)
        history["val_psnr"].append(ep_val_psnr)
        history["train_ssim"].append(ep_tr_ssim)
        history["val_ssim"].append(ep_val_ssim)
        history["baseline_psnr"].append(baseline_psnr)
        history["baseline_ssim"].append(baseline_ssim)

        is_best = ep_val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = ep_val_psnr
            checkpoint = {
                "epoch": epoch,
                "arch": args.arch,
                "base_channels": args.base_channels,
                "model_state_dict": model.state_dict(),
                "val_psnr": ep_val_psnr,
                "val_ssim": ep_val_ssim,
                "baseline_psnr": baseline_psnr,
                "baseline_ssim": baseline_ssim,
            }
            torch.save(checkpoint, best_ckpt_path)
            # Also save as default best_deblur_model.pth for easy inference
            torch.save(checkpoint, ckpt_dir / "best_deblur_model.pth")

        elapsed = time.time() - t0
        marker = " *BEST*" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Loss: {ep_tr_loss:.4f} / {ep_val_loss:.4f} | "
            f"Val PSNR: {ep_val_psnr:.2f} dB (+{ep_val_psnr - baseline_psnr:.2f}) | "
            f"Val SSIM: {ep_val_ssim:.4f}{marker}"
        )

    # Save training history CSV
    csv_path = results_dir / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(history.keys()))
        for i in range(len(history["epoch"])):
            writer.writerow([history[k][i] for k in history.keys()])

    # Plot training curves
    curves_path = results_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    print("=" * 72)
    print(f"Training Complete! Best Validation PSNR: {best_val_psnr:.2f} dB (Gain: +{best_val_psnr - baseline_psnr:.2f} dB)")
    print(f"Saved Best Checkpoint : {best_ckpt_path}")
    print(f"Saved Training History: {csv_path}")
    print(f"Saved Training Curves : {curves_path}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Deep Learning Image Deblurring Model")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "resnet", "autoencoder"], help="Model architecture")
    parser.add_argument("--base-channels", type=int, default=24, help="Base feature channels")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Initial learning rate")
    parser.add_argument("--patch-size", type=int, default=96, help="Training image patch size")
    parser.add_argument("--num-train", type=int, default=120, help="Number of synthetic training pairs if auto-generated")
    parser.add_argument("--num-val", type=int, default=24, help="Number of validation pairs if auto-generated")
    parser.add_argument("--num-test", type=int, default=18, help="Number of test pairs if auto-generated")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root folder")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint output folder")
    parser.add_argument("--results-dir", type=str, default="results", help="Results output folder")
    parser.add_argument("--force-regen", action="store_true", help="Force regenerate synthetic dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
