# QuickDraw MobileViT — Phone‑ready INT8 Transformer with Modern Quantization

This repo trains a **ViT-Tiny** on the **Quick, Draw!** bitmap dataset, achieving **76.42% accuracy on all 344 classes**, then:
- runs **modern, LLM‑style weight‑only quant analysis** on desktop (AWQ/GPTQ track), and
- produces a **phone‑deployable INT8 build** (ExecuTorch track).

## 🏆 **Best Results: 76.42% on 344 Classes**

**Current State-of-the-Art Configuration:**

```bash
python scripts/train_quickdraw.py \
    --classes 344 --per-class-train 4000 --per-class-val 500 \
    --epochs 75 --batch-size 1024 \
    --lr 0.0011 --optimizer adamw \
    --label-smoothing 0.1584 --warmup-epochs 1 \
    --weight-decay 0.04 --no-amp \
    --early-stopping 15 \
    --experiment-name full344_4k500_regaug
```

**Key Results:**
- **Validation Accuracy**: 76.42% (stopped at epoch 52/75)
- **Training Time**: 12.3 hours
- **Model Size**: 5.5M parameters (ViT-Tiny)
- **Dataset**: 4,000 train + 500 val samples per class (1.55M total samples)
- **Best Performing Class**: triangle (94.0%)
- **Most Challenging Class**: cooler (30.0%)

## 🧠 **Key Training Insights**

**💡 Dataset Size Impact:**
- **4k/500 samples per class**: 76.42% accuracy ✅ **BEST**
- **1k/200 samples per class**: ~73% accuracy
- **Insight**: Larger datasets significantly improve generalization (+3.4% absolute)

**💡 Augmentation Strategy:**
- **Light augmentation**: Works best for doodle data
- **Strong augmentation from scratch**: Hurts performance (-0.2%)
- **Strong augmentation fine-tuning**: Also counterproductive (-0.4%)
- **Insight**: Doodles require gentler augmentation than natural images

**💡 Early Stopping:**
- **Patience 15 epochs**: Optimal for large datasets
- **Patience 5-10 epochs**: Good for smaller datasets
- **Insight**: More data requires more patience for convergence

## 📊 **Performance Analysis**

<details>
<summary><strong>📊 Comprehensive Performance Visualizations</strong></summary>

### Per-Class Accuracy Distribution
![Per-Class Accuracy Bars](results/best_model_76p42_visuals/per_class_accuracy_bars.png)

Individual accuracy performance for all 344 classes, sorted from worst to best. Color coding clearly distinguishes performance tiers: **Red** (Poor <50%), **Orange** (Fair 50-70%), **Yellow** (Good 70-85%), and **Green** (Excellent ≥85%). The visualization highlights that while most classes achieve good performance, challenging classes like "cooler" (30.0%) and "aircraft carrier" (33.5%) still present significant difficulties for the model.

### Accuracy Distribution Analysis  
![Accuracy Distribution](results/best_model_76p42_visuals/accuracy_distribution.png)

Histogram showing the distribution of per-class accuracies reveals a right-skewed distribution with most classes performing well. The analysis shows **162 classes (47.1%) achieve Good performance** (70-85%), while only **24 classes (7.0%) fall into the Poor category** (<50%). The mean accuracy line at 76.4% demonstrates consistent model performance across the dataset.

### Model Performance Summary
![Summary Statistics](results/best_model_76p42_visuals/summary_statistics.png)

Comprehensive performance dashboard featuring: (1) **Performance tier breakdown** with class counts and percentages, (2) **Key metrics comparison** showing overall vs. per-class accuracy with error bars, (3) **Cumulative performance curve** with quartile markers for progression analysis, and (4) **Detailed summary table** with professional formatting showing model statistics and performance gaps.

### Precision-Recall Analysis
![Precision Recall Analysis](results/best_model_76p42_visuals/precision_recall_analysis.png)

Scatter plot examining the relationship between precision and recall across all classes, colored by accuracy. The diagonal reference line shows perfect balance, while the colorbar reveals that higher-accuracy classes (green/yellow) tend to achieve better precision-recall trade-offs. Most classes cluster in the high-precision, high-recall region, indicating robust model performance.

### F1-Score Correlation Analysis
![F1 Accuracy Correlation](results/best_model_76p42_visuals/f1_accuracy_correlation.png)

Strong linear correlation (r=0.98) between F1-scores and accuracy demonstrates consistent performance across different metrics. The tight clustering around the correlation line indicates that classes performing well on accuracy also achieve high F1-scores, validating the model's balanced precision-recall characteristics across all performance levels.

### Top vs Bottom Performers Comparison
![Top Bottom Performers](results/best_model_76p42_visuals/top_bottom_performers.png)

Direct comparison highlighting the **performance gap** between best and worst performing classes. Top performers include simple geometric shapes (triangle, envelope) and common objects (headphones, star), while bottom performers consist of abstract concepts (camouflage, cooler) and complex objects (aircraft carrier, marker). This analysis reveals systematic challenges with ambiguous drawings and specialized domains.

### Confusion Matrix for Challenging Classes
![Confusion Matrix](results/best_model_76p42_visuals/confusion_worst_20_classes.png)

Professional confusion matrix focusing on the 20 most challenging classes, using a clean white-to-blue gradient. Diagonal values represent true class accuracy (e.g., dragon correctly identified 45.5% of the time), while off-diagonal values show systematic confusion patterns. The matrix reveals that struggling classes often get confused with visually similar objects, highlighting the model's reliance on shape-based features.

</details>

### Quick Results Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **76.42%** |
| Classes Evaluated | 344 |
| Total Test Samples | 172,000 (500 × 344) |
| Best Performing Class | triangle (94.0%) |
| Most Challenging Class | cooler (30.0%) |
| Performance Standard Deviation | 13.1% |
| Classes Above 70% Accuracy | 65% |

## \ud83d\ude80 **Quick Start**

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Download data (small test)
python scripts/download_quickdraw_direct.py --num-classes 10 --samples-per-class 1000 --per-class-files

# 3. Train a baseline model
python scripts/train_quickdraw.py --classes 10 --epochs 30 --batch-size 256

# 4. Reproduce best 344-class result (requires GPUs + time)
python scripts/train_quickdraw.py \
    --classes 344 --per-class-train 4000 --per-class-val 500 \
    --epochs 75 --batch-size 1024 --lr 0.0011 --optimizer adamw \
    --label-smoothing 0.1584 --warmup-epochs 1 --weight-decay 0.04 \
    --no-amp --early-stopping 15
```

---

## 📱 **Mobile Deployment Pipeline**

Complete mobile deployment pipeline available in `mobile_deployment/` folder:

### 1. Model Export
- **ExecuTorch**: Export to `.pte` format for Android/iOS
- **Labels**: Generate class names file for apps  
- **Metadata**: Model info and preprocessing details

### 2. Platform Integration
- **Android**: Step-by-step ExecuTorch guide (`ANDROID_EXECUTORCH_FP32.md`)
- **iOS**: Coming soon (Core ML or TensorFlow Lite path)

### 3. Performance Features
- Real-time drawing canvas and classification
- Benchmarking tools (FP32 vs INT8 comparison)
- Latency and throughput measurement

---

## 🔬 **Experimental Ablations & Historical Results**

<details>
<summary><strong>Click to expand detailed experimental findings</strong></summary>

### Dataset Size Impact (344 Classes)
| Train/Val per class | Accuracy | Training Time | Notes |
|---------------------|----------|---------------|-------|
| **4000/500** | **76.42%** | 12.3h | **Best result** |
| 1000/200 | ~73.0% | ~4h | Progressive enhancement peak |
| 1000/200 | 72.95% | ~3h | Baseline (50 epochs) |
| 1000/200 | 72.83% | ~6h | Resume training (90 epochs) |

### Augmentation Strategy Experiments
| Strategy | Dataset | Accuracy | Insight |
|----------|---------|----------|---------|
| Light augmentation | 4k/500 | **76.42%** | **Optimal for doodles** |
| Light augmentation | 1k/200 | 73.02% | Progressive enhancement |
| Strong augmentation (scratch) | 1k/200 | 72.82% | Hurts initial learning |
| Strong augmentation (finetune) | 4k/500 | 75.98% | Counterproductive from good checkpoint |

### 50-Class Subset Results (Historical)
| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| AdamW optimized | **87.41%** | Multi-GPU, optimized hyperparams |
| LAMB optimized | 87.01% | Large batch specialist |
| AdamW default | ~60-68% | Suboptimal hyperparameters |
| Single-GPU | 68.58% | Batch size 64 |

### Key Training Lessons
1. **Larger datasets** significantly improve generalization (+3.4% absolute)
2. **Light augmentation** works better for doodle data vs natural images  
3. **Early stopping** with patience 15 optimal for large datasets
4. **Enhanced augmentation** helps from scratch but hurts fine-tuning
5. **Progressive enhancement** (baseline → resume with more aug) can work but is surpassed by larger datasets

</details>

---

## 📁 **Repository Structure**
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
  export_executorch.py    # (moved to mobile_deployment/scripts/)
  export_onnx.py          # (optional) ORT mobile export path
  tests/                  # (dev only) test scripts for development
data/                     # (created by download script) Parquet data files
mobile_deployment/        # 📱 Mobile deployment pipeline
  scripts/                # Export scripts (ExecuTorch, labels)
  android/                # Android app projects (future)
  ios/                    # iOS app projects (future)
  exports/                # Generated .pte files and assets
results/                  # Training metrics & plots
```

## 🗂️ **Data Pipeline Features**

- **High-performance loading** (70k+ samples/sec with per-class Parquet files)
- **Single-channel grayscale** processing (3x smaller than RGB, mobile-optimized)
- **Flexible class selection** (specific classes or auto-sample N classes)
- **Smart augmentation** preserving doodle characteristics
- **NEAREST interpolation** maintains crisp doodle edges

### 🖼️ **Data Download Options**

```bash
# Small test dataset
python scripts/download_quickdraw_direct.py --num-classes 10 --samples-per-class 1000 --per-class-files

# Full dataset for best results
python scripts/download_quickdraw_direct.py --num-classes 344 --samples-per-class 4500 --per-class-files

# View sketches interactively
python scripts/view_sketches.py --interactive

# List all 345 available classes
python scripts/download_quickdraw_direct.py --list-classes
```

---

## 📄 **License**

MIT — see [LICENSE](LICENSE).
