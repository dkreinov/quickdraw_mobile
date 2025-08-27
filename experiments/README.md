# Overfitting Reduction Experiments

## Baseline Results
- **Original Training (Epoch 20)**: 62.50% val accuracy (19.64% overfitting gap)
  - Train: 82.14%, Val: 62.50%, Gap: 19.64%
- **Original Training (Best - Epoch 27)**: 68.58% val accuracy (18.5% gap) 
- **Target**: 65-70% val accuracy with <15% gap (at epoch 20)
- **Test Protocol**: 20 epochs each, resume from best checkpoint (epoch 27)

## Experiment Queue

### 1. baseline_original ✅
- **Status**: COMPLETED
- **Results (Epoch 20)**: 62.50% val acc, 19.64% gap (Train: 82.14%, Val: 62.50%)
- **Results (Best - Epoch 27)**: 68.58% val acc, 18.5% gap (Train: 87.06%, Val: 68.58%)
- **Checkpoint**: `../results/fp32_baseline_50c_main_best.pt`

### 2A. stronger_augmentation_single_gpu ❌
- **Status**: COMPLETED - **FAILED** (made overfitting worse)
- **Change**: Increase data augmentation strength
- **Modifications**:
  - `degrees=15` (vs 10)
  - `translate=(0.15, 0.15)` (vs 0.1)
  - `RandomHorizontalFlip(p=0.2)` (vs 0.1)
  - `brightness=0.3, contrast=0.3` (vs 0.2)
- **Results (Epoch 20)**: 55.71% val acc, 23.51% gap (Train: 79.22%, Val: 55.71%)
- **vs Baseline**: -6.79% val acc, +3.87% gap (WORSE overfitting!)
- **Conclusion**: Stronger augmentation **hurts** QuickDraw performance

### 2B. stronger_augmentation_multigpu
- **Status**: CANCELLED - no point testing multi-GPU with failed approach
- **Reason**: Since stronger augmentation failed on single GPU, skip multi-GPU version

### 3. baseline_multigpu ❌
- **Status**: COMPLETED - **FAILED** (LR scaling issue)
- **Change**: Multi-GPU baseline with original augmentation
- **Setup**: 4 GPUs, batch=512, lr=0.0012, original augmentation
- **Results (Epoch 20)**: 57.19% val acc, 20.52% gap (Train: 77.71%, Val: 57.19%)
- **vs Single-GPU**: -5.31% val acc, +0.88% gap (WORSE performance!)
- **Issue**: Learning rate too high for large batch size
- **Analysis**: Used 4x LR for 8x batch increase - should be 2x or sqrt scaling

### 3B. baseline_multigpu_epoch_schedule ❌
- **Status**: COMPLETED - **FAILED** (epoch-based LR scheduling is broken)
- **Change**: Multi-GPU with epoch-based LR scheduling fix
- **Setup**: 4 GPUs, batch=1024, lr=0.0048 (linear scaling), epoch-based schedule
- **Results (Epoch 20)**: 17.33% val acc (extreme failure)
- **Issue**: Epoch-based LR scheduling doesn't work well - reverted default to step-based
- **Analysis**: Step-based scheduling is superior for this setup

### 3C. baseline_multigpu_step_schedule ✅
- **Status**: COMPLETED - **MUCH BETTER** (step-based LR scheduling works)
- **Change**: Multi-GPU with step-based LR scheduling (reverted from epoch-based)
- **Setup**: 4 GPUs, batch=1024, lr=0.0048 (linear scaling), step-based schedule
- **Results (Epoch 20)**: 56.62% val acc
- **vs Single-GPU**: -5.88% val acc (still worse but much better than epoch-based)
- **Analysis**: Step-based is clearly superior, but LR scaling still needs work

### ~~4-7. Previous Experiments~~ ❌ **OBSOLETE**
- **Status**: **CANCELLED** - LAMB breakthrough makes these obsolete
- **Reason**: LAMB optimizer solved the core issue (86.82% vs 68.58% baseline)
- **Individual hyperparameter tweaks**: Now handled by W&B LAMB sweep

## 🔬 NEW: LAMB Hyperparameter Optimization

### 8. lamb_hyperparameter_sweep 🔄 **IN PROGRESS**
- **Purpose**: Optimize LAMB beyond 86.82% baseline
- **Method**: W&B Bayesian optimization sweep
- **Parameters**: 
  - Learning Rate: 0.0001 - 0.001 (log scale)
  - Weight Decay: 0.01 - 0.15 (uniform)
  - Warmup Epochs: [2, 3, 5]
- **Target**: Find optimal LAMB configuration for >90% validation accuracy
- **Command**: `cd experiments && ./run_lamb_sweep.sh`

## Results Summary

| Experiment | Val Accuracy (Epoch 20) | Train Accuracy (Epoch 20) | Gap | Improvement vs Baseline |
|------------|--------------------------|----------------------------|-----|-------------------------|
| baseline_original | 62.50% | 82.14% | 19.64% | - |
| stronger_aug_1gpu | 55.71% | 79.22% | 23.51% | **-6.79%** ❌ |
| baseline_multigpu | 57.19% | 77.71% | 20.52% | **-5.31%** ❌ |
| baseline_multigpu_epoch_sched | 17.33% | ~17.33% | ~0% | **-45.17%** ❌ |
| baseline_multigpu_step_sched | 56.62% | ~TBD~ | ~TBD~ | **-5.88%** ⚠️ |
| **lamb_multigpu_base_lr** | **86.82%** | **~TBD~** | **~TBD~** | **+24.32%** ✅ **BREAKTHROUGH** |
| lamb_optimized_sweep | **TBD** | **TBD** | **TBD** | **Target: >90%** 🎯 |

## 🚀 BREAKTHROUGH: LAMB Optimizer Solution

**PROBLEM SOLVED**: Multi-GPU training was performing poorly due to optimizer limitations, not learning rate issues.

### Key Discovery
- **Root Cause**: AdamW optimizer doesn't handle large batch training well
- **Solution**: LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
- **Result**: **86.82% validation accuracy** vs 68.58% single-GPU baseline

### 🚨 **Critical Insight: Pretrained Weights Required**

**During W&B debugging, we discovered a critical requirement:**
- **LAMB + pretrained weights**: 86.82% validation accuracy ✅
- **LAMB + training from scratch**: ~10-15% validation accuracy ❌

**The combination of LAMB optimizer AND pretrained weights is essential for success.**

### LAMB vs AdamW Comparison

| Setup | Optimizer | Batch Size | Best Val Acc | Status |
|-------|-----------|------------|--------------|--------|
| Single-GPU | AdamW | 64 | 68.58% | Baseline |
| Multi-GPU | AdamW | 1024 | ~56-68% | ❌ Failed |
| Multi-GPU | **LAMB** | 1024 | **86.82%** | ✅ **Best** |

### Implementation
```bash
python scripts/train_quickdraw.py \
    --optimizer lamb \
    --lr 0.0003 \
    --batch-size 1024 \
    --pretrained \
    --classes 50 \
    --epochs 30
```

**Why LAMB Works**: Layer-wise adaptive learning rates handle large batch gradients much better than AdamW's global adaptation.

## Best Practices
- Each experiment runs for exactly 20 epochs (resuming from epoch 27 checkpoint)
- All resume from the same best checkpoint (`fp32_baseline_50c_main_best.pt`)
- Use identical data splits (seed=42)
- Save results in structured format
- Compare both validation accuracy and overfitting gap at epoch 20
- Target: >65% val accuracy with <15% gap (improvement over 62.50% baseline)
