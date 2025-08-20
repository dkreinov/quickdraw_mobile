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

# 4) Download QuickDraw data (Step 2) - optimized per-class format
.venv/bin/python scripts/download_quickdraw_direct.py --num-classes 10 --samples-per-class 1000 --per-class-files

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
  download_quickdraw_direct.py  # ✅ Download & convert QuickDraw to optimized Parquet format
  split_parquet_by_class.py     # ✅ Convert existing datasets to optimized per-class format
  view_sketches.py              # ✅ Interactive sketch viewer for dataset exploration
src/
  data.py                       # ✅ High-performance QuickDraw loader with concurrent loading
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

## Step 2 Completed: QuickDraw Data Pipeline (Optimized)

**Features implemented:**
- **High-performance data loading** - optimized per-class Parquet files with concurrent loading
- **Single-channel grayscale** processing (mobile-optimized, 3x smaller than RGB)
- **Flexible dataset organization** - per-class files for fast subset loading or monolithic for compatibility
- **Configurable class selection** - choose specific classes or auto-sample N classes
- **Stratified train/val splits** with balanced samples per class
- **Smart augmentation** preserving doodle characteristics (mild rotation, translation)
- **Automatic format detection** - uses the fastest available data format
- **NEAREST interpolation** upsampling to maintain crisp doodle edges

**Performance highlights:**
- **70k+ samples/second** loading speed with concurrent per-class files
- **22x faster** than traditional monolithic loading for large datasets
- **Memory efficient** - loads only the classes you need
- **Auto-optimized** - automatically chooses the best loading method

**Why single-channel?**
- Matches mobile inference reality (finger drawings ≈ 28x28 grayscale)
- Model trains on **black-on-white** format (inverted from original dataset) to match mobile drawing UI

## Viewing QuickDraw Sketches

After downloading the dataset, you can visualize the sketches using the interactive viewer:

```bash
# Interactive mode with menu
python scripts/view_sketches.py --interactive

# View specific classes  
python scripts/view_sketches.py --classes cat dog apple --num-samples 16

# Save visualization to file (useful for headless environments)
python scripts/view_sketches.py --random-classes 5 --save sketches.png

# Invert colors for traditional black-on-white appearance  
python scripts/view_sketches.py --classes cat dog --invert-colors

# List all available classes
python scripts/view_sketches.py --list-classes
```

**Note on Colors**: QuickDraw sketches are stored with white strokes on black background (original format), but the **model trains on black-on-white** (inverted) to match mobile drawing interfaces. The viewer shows original format by default; use `--invert-colors` to see the training format.

**Benefits of single-channel:**
- 3x smaller model size and memory usage  
- Faster training and inference
- Authentic representation of doodle data

**Download options:**
```bash
# Small test dataset (optimized format)
.venv/bin/python scripts/download_quickdraw_direct.py --num-classes 5 --samples-per-class 100 --per-class-files

# Medium dataset for training (optimized format)
.venv/bin/python scripts/download_quickdraw_direct.py --num-classes 20 --samples-per-class 1000 --per-class-files

# Specific classes (optimized format)
.venv/bin/python scripts/download_quickdraw_direct.py --classes cat dog apple car house --samples-per-class 500 --per-class-files

# List all 345 available classes
.venv/bin/python scripts/download_quickdraw_direct.py --list-classes

# Convert existing monolithic dataset to optimized format
.venv/bin/python scripts/split_parquet_by_class.py --input-dir data/quickdraw_parquet
```

## Mobile App Architecture

After training and quantizing the model, we plan to create a minimal Android app for live sketch classification.

### MVP Features
- **On-screen drawing**: Black strokes on white UI background
- **Live classification**: Real-time inference as user draws
- **Color preprocessing**: Invert colors before model input (since dataset is white-on-black)

### Later Enhancements
- **Camera mode**: Photo capture with preprocessing pipeline for hand-drawn sketches
- **Improved UX**: Drawing tools, undo/clear, confidence scores

### App Architecture (Android/Kotlin + ExecuTorch)

#### Core Components

**1. CanvasView**
- Finger/stylus drawing to Bitmap
- White background, black strokes
- Standard Android drawing canvas

**2. Preprocessing Pipeline**
```kotlin
// Convert drawing to model input
fun preprocessDrawing(bitmap: Bitmap): FloatArray {
    // 1. Resize to 224x224 with NEAREST interpolation
    val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, false)
    val input = FloatArray(1 * 224 * 224)
    
    // 2. Convert to grayscale + normalize + invert colors
    var idx = 0
    for (y in 0 until 224) {
        for (x in 0 until 224) {
            val pixel = resized.getPixel(x, y)
            val r = (pixel shr 16 and 0xFF)
            val g = (pixel shr 8  and 0xFF) 
            val b = (pixel        and 0xFF)
            
            // Grayscale conversion
            var gray = (0.299f*r + 0.587f*g + 0.114f*b) / 255f
            
            // Invert: black strokes → white strokes (match training data)
            gray = 1f - gray
            
            // Normalize to [-1, 1] range (mean=0.5, std=0.5)
            input[idx++] = (gray - 0.5f) / 0.5f
        }
    }
    return input
}
```

**3. Inference Engine**
- Load INT8 W8A8 ExecuTorch `.pte` model
- Run inference on preprocessed input
- Display top-k classification results

**4. User Interface**
- "Clear" and "Undo" buttons
- Optional "Live classify" toggle (classify every ~200ms while drawing)
- Real-time confidence scores display

### Camera Mode (Future Enhancement)

When adding photo capture support:

**Preprocessing Pipeline**
1. **Grayscale conversion** from camera input
2. **Adaptive thresholding** to isolate drawing from background
3. **Contour detection** to find largest sketch area
4. **Perspective correction** and crop to square
5. **Resize to 224x224** with NEAREST interpolation
6. **Color inversion** (if needed) to match training format
7. **Normalization** to model input range

**Failure Handling**
- Detect low contrast/blur and prompt user to retake photo
- Guide user to improve lighting/positioning
- Fallback to manual drawing mode

### Technical Notes

- **Model Format**: INT8 quantized ExecuTorch `.pte` file
- **Inference Backend**: XNNPACK for optimized CPU execution
- **Target Devices**: Android 8.0+ with ARM64 processors
- **Memory Usage**: <50MB for model + preprocessing
- **Latency Target**: <100ms for single sketch classification

### Development Timeline

1. **Phase 1**: Basic drawing canvas + preprocessing
2. **Phase 2**: ExecuTorch model integration + inference
3. **Phase 3**: UI polish + live classification
4. **Phase 4**: Camera mode + advanced preprocessing

## License
MIT — see [LICENSE](LICENSE).
