# Experiment Execution Plan

## Current Status
- [x] Baseline completed: 68.58% val acc (19.7% gap)
- [ ] Run systematic experiments (20 epochs each)
- [ ] Analyze results and select best approaches
- [ ] Create final optimized model

## Ready-to-Run Commands

### 2. Stronger Augmentation
**Status**: Ready to run  
**Requires**: Modify `src/data.py` augmentation parameters first

```bash
# Step 1: Modify augmentation in src/data.py (see commands below)
# Step 2: Run training
python scripts/train_quickdraw.py \
  --classes 50 \
  --epochs 20 \
  --per-class-train 1000 \
  --per-class-val 200 \
  --batch-size 64 \
  --no-pretrained \
  --arch vit_tiny_patch16_224 \
  --lr 0.0003 \
  --warmup-epochs 3 \
  --seed 42 \
  --experiment-name exp2_stronger_augmentation \
  --resume results/fp32_baseline_50c_main_best.pt
```

### 3. Higher Weight Decay
**Status**: Ready to run

```bash
python scripts/train_quickdraw.py \
  --classes 50 \
  --epochs 20 \
  --per-class-train 1000 \
  --per-class-val 200 \
  --batch-size 64 \
  --no-pretrained \
  --arch vit_tiny_patch16_224 \
  --lr 0.0003 \
  --warmup-epochs 3 \
  --weight-decay 0.1 \
  --seed 42 \
  --experiment-name exp3_higher_weight_decay \
  --resume results/fp32_baseline_50c_main_best.pt
```

### 4. Higher Label Smoothing
**Status**: Ready to run

```bash
python scripts/train_quickdraw.py \
  --classes 50 \
  --epochs 20 \
  --per-class-train 1000 \
  --per-class-val 200 \
  --batch-size 64 \
  --no-pretrained \
  --arch vit_tiny_patch16_224 \
  --lr 0.0003 \
  --warmup-epochs 3 \
  --label-smoothing 0.15 \
  --seed 42 \
  --experiment-name exp4_higher_label_smoothing \
  --resume results/fp32_baseline_50c_main_best.pt
```

### 5. Lower Learning Rate
**Status**: Ready to run

```bash
python scripts/train_quickdraw.py \
  --classes 50 \
  --epochs 20 \
  --per-class-train 1000 \
  --per-class-val 200 \
  --batch-size 64 \
  --no-pretrained \
  --arch vit_tiny_patch16_224 \
  --lr 0.0001 \
  --warmup-epochs 3 \
  --seed 42 \
  --experiment-name exp5_lower_learning_rate \
  --resume results/fp32_baseline_50c_main_best.pt
```

### 6. Dropout Regularization
**Status**: Needs code modification

```bash
# Step 1: Add dropout to model (needs implementation)
# Step 2: Run training
python scripts/train_quickdraw.py \
  --classes 50 \
  --epochs 20 \
  --per-class-train 1000 \
  --per-class-val 200 \
  --batch-size 64 \
  --no-pretrained \
  --arch vit_tiny_patch16_224 \
  --lr 0.0003 \
  --warmup-epochs 3 \
  --seed 42 \
  --experiment-name exp6_dropout_regularization \
  --resume results/fp32_baseline_50c_main_best.pt
```

## Code Modifications Needed

### For Experiment 2 (Stronger Augmentation)
Modify `src/data.py` lines 515-526:

```python
# Current (mild):
transforms.RandomAffine(
    degrees=10,           # Change to: 15
    translate=(0.1, 0.1), # Change to: (0.15, 0.15)
    scale=(0.9, 1.1),     # Keep same
    fill=0
),
transforms.RandomHorizontalFlip(p=0.1),  # Change to: p=0.2

# Contrast/brightness adjustments  
transforms.RandomApply([
    transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Change to: 0.3, 0.3
], p=0.3),
```

### For Experiment 6 (Dropout)
Add dropout parameter to model creation in `src/models.py`:

```python
# Add dropout parameter to build_model function
# Modify ViT model to include dropout layers
```

## Manual Execution Order (One at a Time)
1. **Pick one experiment** from the ready-to-run list (3, 4, 5)
2. **Run it and analyze results** before proceeding
3. **Update README.md** with findings
4. **Decide next experiment** based on results
5. **Implement code changes** only when needed (experiments 2, 6)
6. **Create combined approach** after finding what works

## Analysis Protocol (After Each Experiment)
1. Record epoch 20 validation accuracy
2. Record train-val gap at epoch 20
3. Compare against 62.50% baseline
4. Note any training stability issues
5. Update results table in README.md
6. **Decide if approach is promising** before trying next
7. **Stop and think** - don't just run all experiments blindly
