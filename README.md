# QuickDraw MobileViT — Phone‑ready INT8 Transformer with Modern Quantization

This repo will train a **MobileViT/ViT** on the **Quick, Draw!** bitmap dataset, then:
- run a **modern, LLM‑style weight‑only quant analysis** on desktop (AWQ/GPTQ track), and
- produce a **phone‑deployable INT8 build** (ExecuTorch track).

> We're working step‑by‑step. This commit is **Step 2: QuickDraw data pipeline**.

## Planned Steps (we'll do these one by one)
1. ✅ Initialize repo, environment, and skeleton
2. ✅ Data pipeline: QuickDraw (bitmap) loader with configurable class subset
3. Baseline training: MobileViT‑S / ViT‑Tiny fine‑tune
4. Baseline evaluation & confusion matrix
5. Modern quant (desktop): AWQ/GPTQ weight‑only analysis on the ViT blocks
6. Device quant (ExecuTorch PT2E): INT8 (weights+activations) with calibration
7. Export `.pte` and minimal Android demo
8. Results tables/plots + README polish
9. Optional ONNX Runtime Mobile export
10. GitHub Actions (lint/tests/export) + Release assets

## Quantization Testing Framework

This project implements a comprehensive quantization testing suite to evaluate different compression techniques for mobile deployment. Our approach tests multiple quantization schemes to find the optimal balance between model size, accuracy, and inference speed.

### Quantization Methods Tested

**A) Weight-only 4-bit Quantization (AWQ/GPTQ)**
- Quantizes weights in Linear layers (Attention Q/K/V/Out, MLP) to 4-bit
- Keeps activations in FP16/FP32 for better accuracy
- Targets transformer blocks while preserving LayerNorms, Softmax, GELU
- Uses calibration data for optimal quantization parameters

**B) Full INT8 Quantization (ExecuTorch PT2E)**
- Quantizes both weights and activations to INT8
- Production-ready for mobile deployment via ExecuTorch
- Exports to `.pte` format for Android integration
- Optimized for CPU/NPU execution with XNNPACK backend

**C) Activation Smoothing (SmoothQuant-style)**
- Pre-scales activations per-channel to reduce quantization error
- Transfers techniques from large language model quantization
- Compares INT8 performance with and without smoothing

**D) Quantization-Aware Training (QAT)**
- Fine-tunes model with fake quantization for better accuracy
- Trains the network to be robust to quantization errors
- Provides best accuracy for aggressive quantization schemes

### Evaluation Metrics

All quantization methods are compared using consistent metrics:
- **Accuracy**: Top-1 validation accuracy on same data split
- **Model Size**: On-disk size in MB for each quantization
- **Latency**: Single-image inference time (desktop and Android)
- **Memory**: Peak memory usage during inference

### Fair Comparison Design
- Same calibration dataset across all methods (2-4k images)
- Identical validation split for accuracy measurement
- Fixed random seeds for reproducible results
- Balanced per-class sampling for calibration

## Getting Started (local)
```bash
# 1) Create and activate a virtual env 
/usr/mobileye/pkgs/python/3.10/bin/virtualenv .venv  # Or: python -m venv .venv
# For fish shell: set -gx VIRTUAL_ENV (pwd)/.venv && set -gx PATH $VIRTUAL_ENV/bin $PATH
# For bash/zsh: source .venv/bin/activate

# 2) Install dependencies
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 3) Sanity check
.venv/bin/python -c "import torch, transformers, datasets; print('✅ Dependencies OK')"

# 4) Download QuickDraw data (Step 2)
.venv/bin/python scripts/download_quickdraw.py --num-classes 10 --samples-per-class 1000

# 5) Test QuickDraw data loading
.venv/bin/python -c "
import sys; sys.path.append('src')
from data import create_dataloaders
train_loader, val_loader, meta = create_dataloaders(num_classes=3, train_samples_per_class=50)
print(f'✅ Data pipeline OK: {meta[\"num_classes\"]} classes, {meta[\"train_samples\"]} train samples')
"
```

## Repo Layout
```
scripts/
  download_quickdraw.py   # ✅ Download & convert QuickDraw to Parquet
  view_sketches.py        # ✅ Interactive sketch viewer for dataset exploration
src/
  data.py                 # ✅ QuickDraw Parquet loader (28x28→224x224, single-channel)
  models.py               # MobileViT/ViT factory with 1-channel support
  train.py                # Training entrypoint with checkpoint saving
  eval.py                 # Evaluation & confusion matrix generation
  quant_awq.py            # ✅ AWQ/GPTQ 4-bit weight quantization
  quant_executorch.py     # ✅ PT2E INT8 quantization with SmoothQuant
  export_executorch.py    # ✅ Export to .pte for Android deployment
  export_onnx.py          # (optional) ORT mobile export path
  tests/                  # (dev only) test scripts for development
data/                     # (created by download script) Parquet data files
android_demo/             # (later) Android app with ExecuTorch
results/                  # (later) metrics & plots
```

## Step 2 Completed: QuickDraw Data Pipeline (Parquet-based)

**Features implemented:**
- **Parquet format conversion** - fast, efficient, future-proof data loading
- **Single-channel grayscale** processing (mobile-optimized, 3x smaller than RGB)
- **Download script** - converts HuggingFace data to local Parquet files
- **Configurable class selection** - choose specific classes or auto-sample N classes
- **Stratified train/val splits** with balanced samples per class
- **Smart augmentation** preserving doodle characteristics (mild rotation, translation)
- **Efficient data loading** with PyTorch DataLoader integration
- **NEAREST interpolation** upsampling to maintain crisp doodle edges

**Why Parquet format?**
- **Fast loading** - columnar storage, no deprecated HuggingFace scripts
- **Reproducible** - fixed dataset snapshots, no internet dependency
- **Safe** - no arbitrary code execution warnings
- **Efficient** - compressed storage, faster than raw data
- **Future-proof** - standard data format, widely supported

**Why single-channel?**
- Matches mobile inference reality (finger drawings ≈ 28x28 grayscale)

## Viewing QuickDraw Sketches

After downloading the dataset, you can visualize the sketches using the interactive viewer:

```bash
# Interactive mode with menu
python scripts/view_sketches.py --interactive

# View specific classes  
python scripts/view_sketches.py --classes cat dog apple --num-samples 16

# Save visualization to file (useful for headless environments)
python scripts/view_sketches.py --random-classes 5 --save sketches.png

# List all available classes
python scripts/view_sketches.py --list-classes
```

**Benefits of single-channel:**
- 3x smaller model size and memory usage  
- Faster training and inference
- Authentic representation of doodle data

**Download options:**
```bash
# Small test dataset
.venv/bin/python scripts/download_quickdraw.py --num-classes 5 --samples-per-class 100

# Medium dataset for training
.venv/bin/python scripts/download_quickdraw.py --num-classes 20 --samples-per-class 1000

# Specific classes
.venv/bin/python scripts/download_quickdraw.py --classes cat dog apple car house --samples-per-class 500

# List all 345 available classes
.venv/bin/python scripts/download_quickdraw.py --list-classes
```

## License
MIT — see [LICENSE](LICENSE).
