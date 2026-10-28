# LAMP: Latent Asynchronous Masking Pipeline for Generative Image Inpainting

<p align="center">
  <img src="Outputs/fpga_output_pic.jpeg" alt="LAMP inpainting outputs on a Xilinx ZCU104 FPGA" width="900">
</p>

<p align="center">
  <b>LAMP inpainting results running on a Xilinx ZCU104 FPGA (INT8)</b>
</p>

![Status](https://img.shields.io/badge/Paper-In%20Preparation%20(ICCI)-orange)
![Research](https://img.shields.io/badge/IIT%20Bhilai-Research%20Internship-blue)
![Hardware](https://img.shields.io/badge/FPGA-Xilinx%20ZCU104-red)

> Research from my internship at **IIT Bhilai** (May–Aug 2026), supervised by **Dr. Hathiram Nenavath**. A paper describing this work is being prepared for submission to **ICCI**.

---

## Overview

LAMP is a latent diffusion framework for **generative image inpainting**: filling large missing regions of an image with content that is anatomically plausible and blends seamlessly into the surrounding image.

Earlier diffusion inpainting methods that schedule noise per pixel (such as RAD) work in full 256×256 pixel space, which needs a lot of memory and limits how much context each step can see. LAMP moves this **asynchronous (region-wise) noise scheduling into the compressed latent space** of a frozen, pretrained VQ-VAE:

- **16× less spatial work:** images are encoded to a 64×64 latent instead of 256×256 pixels.
- **Noise only where it's needed:** masked tokens are noised and denoised, while background tokens stay exactly clean throughout.
- **Spatial Noise Embedding (SNE):** a 1×1 convolution replaces the usual scalar timestep, so the U-Net knows *which* tokens are noisy and by how much, and its cross-attention can borrow color, lighting and structure from the clean background.
- **Runs on embedded hardware:** the model was quantized to INT8 and deployed on a Xilinx ZCU104 FPGA.

LAMP is evaluated on two criteria:
- **Semantic Plausibility:** generated structures are anatomically valid for the subject's pose and geometry.
- **Contextual Harmony:** generated content blends in with no visible seams at the mask boundary.

---

## Architecture

<p align="center">
  <img src="docs/architecture.png" width="900" alt="LAMP architecture">
</p>

*Figure: LAMP pipeline. The image is encoded by a frozen VQ-VAE, noise is applied only to masked latent tokens, and a Latent U-Net conditioned by the Spatial Noise Embedding denoises them. The result is decoded back to a 256×256 image.*

### Core Components

1. **Frozen VQ-VAE encoder and decoder** (from `CompVis/ldm-celebahq-256`): compresses 256×256×3 images to 64×64×3 latents. The binary mask is downsampled with bilinear interpolation, so boundary tokens get a smooth, partial noise level instead of a hard edge.
2. **Two-phase asynchronous forward process:**
   - **Phase 1:** only masked tokens receive noise; background tokens stay bit-exact copies of the encoder output.
   - **Phase 2:** the roles flip. Background is noised and the mask is held clean, which stops the network from over-specializing on masked tokens.
3. **Spatial Noise Embedding (SNE):** a 1×1 convolution over the per-token noise levels √ᾱ and √(1−ᾱ). It is injected into every residual block through spatial adaptive group normalization, replacing the scalar timestep embedding.
4. **Latent U-Net (~300M parameters):** fully fine-tuned, with cross-attention that routes context from clean background tokens to noisy masked tokens.
5. **Reverse process:** 500 DDIM-style steps that update masked tokens only. Background tokens are provably unchanged, then the result is decoded by the VQ-VAE.

### Training

| Setting | Value |
|---|---|
| Base model | `CompVis/ldm-celebahq-256` |
| Fine-tuning data | FFHQ (10,000 train / 1,000 test) |
| Steps | 15,000 (Phase 1: 1–7,500, Phase 2: 7,501–15,000) |
| Optimizer | AdamW, learning rate 5×10⁻⁵, batch size 4 |
| Diffusion steps | T = 2,000 (training), N = 500 (inference) |
| Resolution | 256×256 image → 64×64×3 latent |

---

## Results

### Performance across datasets (1,000-image test sets, box masking)

| Dataset | FID ↓ | LPIPS ↓ | PSNR ↑ (dB) | SSIM ↑ |
|---|---|---|---|---|
| FFHQ | **18.60** | 0.0552 | 24–25 | 0.92 |
| CelebA-HQ | **16.55** | 0.0697 | 23–24 | 0.89 |
| Places365 (out-of-domain) | 45.57 | 0.1968 | 20–22 | 0.72 |

### Comparison with baselines (512×512, box masks)

**FFHQ, light box mask (15–25%)**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| LBP-SA | 21.041 | 0.9122 | 0.0712 |
| LaMa | 24.429 | 0.9480 | 0.0583 |
| SCAT | 26.723 | 0.9424 | 0.0652 |
| CoordFill | 28.418 | 0.9544 | 0.0348 |
| **LAMP (ours)** | **28.903** | **0.9567** | **0.0320** |

**CelebA-HQ, light box mask (15–25%)**

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| LaMa | 36.987 | 0.9881 | 0.0135 |
| SCAT | 38.333 | 0.9819 | 0.0318 |
| CoordFill | **39.531** | 0.9875 | 0.0136 |
| **LAMP (ours)** | 38.523 | **0.9889** | **0.0128** |

Full results, including medium masks (25–45%) and the LOHC, LGNet and RePaint rows, are in the paper. LGNet and RePaint were evaluated on a single image only, so they are excluded from the rankings.

### Ablation study

| Variant | FFHQ PSNR ↑ | FFHQ LPIPS ↓ | CelebA-HQ PSNR ↑ | CelebA-HQ LPIPS ↓ |
|---|---|---|---|---|
| Baseline (standard LDM) | 13.28 | 0.2744 | 13.41 | 0.2773 |
| Sync-Noise (no asynchronous schedule) | 13.02 | 0.2857 | 13.20 | 0.2828 |
| No-SE (no Spatial Noise Embedding) | 20.32 | 0.1428 | 21.25 | 0.1114 |
| **Full LAMP** | **24.82** | **0.0889** | **25.96** | **0.0637** |

Both mechanisms are needed. Without asynchronous scheduling, the model produces incoherent noise in the masked region; without the SNE, completions are blurry and lack detail.

---

## FPGA Deployment (Xilinx ZCU104)

The fine-tuned Latent U-Net was compiled for the ZCU104's DPU accelerator with **Vitis AI**. The original model would not compile, so I made four targeted changes:

1. **Activation substitution:** SiLU → ReLU, since the DPU has no SiLU support.
2. **Linear → 1×1 convolution:** time-embedding projections reshaped into convolutions.
3. **CPU/DPU partitioning:** the trigonometric positional encoding runs on the ARM CPU, and all convolutions and normalization run on the DPU.
4. **Anti-fusion identity convolution:** a depthwise 1×1 identity convolution after every residual addition stops the XIR compiler from silently folding away residual connections (`XIR_REMOVE_OP_FAIL`).

The model was then quantized to **INT8** (post-training, calibrated on 486 latent samples) and compiled with `vai_c_xir`. 84 of 205 subgraphs run on the DPU.

| Metric | Host CPU (Intel i5-6600) | ZCU104 FPGA |
|---|---|---|
| Precision | FP32 | INT8 |
| Power | 65 W | 15 W |
| Accuracy vs. floating point | 100.00% | 99.98% |
| Latency (10 images) | 6.148 s | 0.550 s |
| Throughput | 1.63 FPS | **18.18 FPS** |
| Energy efficiency | 0.025 FPS/W | **1.212 FPS/W** |

**11.18× throughput gain** and **48.44× energy-efficiency gain**, with under 0.15% numerical deviation from the floating-point model.

> **Note:** the hardware numbers measure a single denoising forward pass at a fixed noise level (ᾱ = 0.64), not the full 500-step generation process. End-to-end on-device generation has not yet been measured.

---

## Limitations and Future Work

- Fine-tuned on faces (FFHQ), so quality drops on natural scenes (Places365).
- Completions are plausible but not identity-preserving, since LAMP is a generative model.
- The LGNet and RePaint comparisons use one image; a full 1,000-image evaluation is planned.
- Next steps: on-device measurement of full generation, fewer sampling steps (DDIM or distillation), and multi-domain fine-tuning.

---

## Tech Stack

Python · PyTorch · Hugging Face Diffusers · VQ-VAE / Latent Diffusion · Vitis AI (INT8 quantization, `vai_c_xir`) · Xilinx ZCU104 · NumPy · OpenCV · Matplotlib

---

## Datasets

- **FFHQ:** https://www.kaggle.com/datasets/gibi13/flickr-faces-hq-dataset-ffhq
- **CelebA-HQ**
- **Places365**

**Abhishek kumar Singh**

B.Tech, Electrical and Electronics Engineering  
National Institute of Technology Raipur
Datasets are not included in this repository because of their size.

---

## Author

**Rishik Thammisetty**
B.Tech, Electrical and Electronics Engineering (Minor in AI & Robotics), National Institute of Technology Andhra Pradesh
Research Intern, IIT Bhilai

Supervisor: **Dr. Hathiram Nenavath**, Department of Electronics and Communication Engineering, IIT Bhilai

---

## Acknowledgements

- IIT Bhilai Research Internship Program
- CompVis for the pretrained latent diffusion model
- FFHQ, CelebA-HQ and Places365 dataset contributors
- The PyTorch, Hugging Face and AMD/Xilinx Vitis AI communities
