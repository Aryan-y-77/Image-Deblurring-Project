# Image Deblurring using Deep Learning — Restore Blurred Photographs

A complete, modular **AI & Machine Learning (AIML) Final Year Project** that restores blurred photographs (Motion Blur, Defocus/Out-of-Focus Blur, and Gaussian Blur) using a custom **Residual Channel-Attention U-Net (`DeblurUNet`)**, **Deep Residual CNN (`DeblurResNet`)**, and **Convolutional Autoencoder (`DeblurAutoencoder`)** trained with a hybrid **Charbonnier + Sobel Edge Gradient + Fast Fourier Transform (FFT) Spectral Loss**.

---

## Project Architecture & Highlights

1. **Mathematical Formulation of Blur Degradation**:
   $$I_{\text{blur}} = K * I_{\text{sharp}} + \eta$$
   where $K$ is the unknown Point Spread Function (PSF) kernel (directional camera shake / motion trajectory or circular lens defocus disk), $*$ denotes 2D spatial convolution, and $\eta$ represents sensor readout noise.

2. **Global Residual Learning (`DeblurUNet`)**:
   Instead of reconstructing the entire RGB photograph from scratch, our network learns the high-frequency residual detail $\hat{R} = f_{\theta}(I_{\text{blur}})$:
   $$\hat{I}_{\text{sharp}} = \text{clamp}\big(I_{\text{blur}} + f_{\theta}(I_{\text{blur}}),\; 0,\; 1\big)$$
   Combined with **Squeeze-and-Excitation Channel Attention** and multi-scale encoder-decoder skip connections, the network restores crisp edges, textures, and text without color shift.

3. **Hybrid Edge- & Frequency-Aware Loss Function (`DeblurLoss`)**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + \lambda_{\text{edge}}\,\mathcal{L}_{\text{Sobel}} + \lambda_{\text{fft}}\,\mathcal{L}_{\text{FFT}}$$
   - **Charbonnier Loss**: $\sqrt{\|\hat{I} - I\|^2 + \epsilon^2}$ prevents the blurry averaging effect of standard $L_2$/MSE loss.
   - **Sobel Gradient Loss**: Penalizes blurry horizontal and vertical edges.
   - **FFT Spectral Loss**: Enforces recovery of high-frequency Fourier amplitude components.

---

## Directory Structure

```text
image_deblurring_project/
├── dataset.py                           # Paired Dataset Loader & Synthetic Motion/Defocus PSF Generator
├── models.py                            # DeblurUNet, DeblurResNet, DeblurAutoencoder & Hybrid DeblurLoss
├── metrics.py                           # PSNR (dB), SSIM, and Laplacian Sharpness metrics
├── train.py                             # Training script with checkpointing & loss/PSNR/SSIM plots
├── evaluate.py                          # Test-set evaluation vs. Classical Filter + visual & bar charts
├── infer.py                             # CLI tool to deblur any custom photograph
├── Image_Deblurring_Deep_Learning.ipynb # Interactive Jupyter Notebook for presentation & experiments
├── requirements.txt                     # Python dependencies
├── checkpoints/                         # Saved PyTorch model weights (.pth)
├── data/                                # Paired train/, val/, and test/ splits (blurred/ & sharp/)
└── results/                             # Generated plots, CSV reports, and restored images
```

---

## Quick Start Commands

### 1. Train the Custom Model from Scratch
```bash
python image_deblurring_project/train.py --arch unet --epochs 12 --batch-size 8 --patch-size 96
```
*Note: If `data/` is empty, `train.py` automatically synthesizes a benchmark dataset of sharp and blurred image pairs (Motion Blur, Defocus Blur, Gaussian Blur). To train on GoPro, DIV2K, or RealBlur, place paired images inside `data/train/blurred` & `data/train/sharp` and `data/val/blurred` & `data/val/sharp`.*

### 2. Evaluate Model & Generate Report Plots
```bash
python image_deblurring_project/evaluate.py
```
Generates:
- `results/training_curves.png` — Epoch-wise Loss, PSNR, and SSIM curves.
- `results/visual_comparison.png` — Side-by-side grid comparing Blurred Input, Classical Filter, Deep Learning U-Net, and Ground Truth.
- `results/metrics_comparison.png` — Quantitative PSNR & SSIM bar charts across blur types.
- `results/evaluation_metrics.csv` — Per-image quantitative evaluation table.

### 3. Restore Any Single Photograph (Inference CLI)
```bash
# Restore a blurred image and save a side-by-side comparison plot
python image_deblurring_project/infer.py --input image_deblurring_project/data/test/blurred/sample_0000_motion.png --ground-truth image_deblurring_project/data/test/sharp/sample_0000_motion.png --output image_deblurring_project/results/single_restored.png --compare
```
