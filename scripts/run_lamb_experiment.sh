#!/bin/bash
"""
Test LAMB optimizer with multi-GPU training.

This script demonstrates how to use LAMB optimizer for better large batch training
compared to the previous AdamW experiments.
"""

set -e  # Exit on any error

echo "🐑 LAMB Optimizer Multi-GPU Experiment"
echo "======================================"

# Check GPU availability
nvidia-smi | head -n 20

echo ""
echo "📝 Testing LAMB optimizer first..."
python scripts/test_lamb_optimizer.py

echo ""
echo "🚀 Starting LAMB training experiment..."

# Run with LAMB optimizer and automatic LR scaling
python scripts/train_quickdraw.py \
    --classes 50 \
    --epochs 30 \
    --batch-size 1024 \
    --lr 0.0003 \
    --optimizer lamb \
    --auto-scale-lr \
    --base-batch-size 64 \
    --weight-decay 0.05 \
    --warmup-epochs 3 \
    --schedule-time-unit step \
    --label-smoothing 0.1 \
    --grad-clip 1.0 \
    --experiment-name "lamb_multigpu_1024_auto_scaled" \
    --per-class-train 1000 \
    --per-class-val 200 \
    --seed 42

echo ""
echo "✅ LAMB experiment completed!"
echo ""
echo "🔍 To compare with your previous results:"
echo "   python scripts/compare_training_curves.py --all-experiments results/"
echo ""
echo "📊 Expected improvements with LAMB:"
echo "   - Better convergence with large batches"
echo "   - More stable training across GPUs"
echo "   - Potentially higher validation accuracy"
echo "   - Layer-wise adaptive learning rates"
