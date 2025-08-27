#!/bin/bash
"""
Launch LAMB Optimizer Hyperparameter Sweep

This script starts a W&B sweep to optimize LAMB hyperparameters
after the breakthrough discovery that LAMB achieves 86.82% vs 68.58% baseline.

Target: Find optimal LR + Weight Decay combination for LAMB.
"""

set -e

echo "🐑 LAMB Optimizer Hyperparameter Sweep"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "wandb_lamb_sweep_config.yaml" ]; then
    echo "❌ Error: wandb_lamb_sweep_config.yaml not found"
    echo "Please run this script from the experiments/ directory"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU setup..."
nvidia-smi | head -n 10

echo ""
echo "📋 Sweep Configuration:"
cat wandb_lamb_sweep_config.yaml

echo ""
echo "🚀 Starting W&B sweep..."

# Create the sweep and get the command
echo "🔸 Creating LAMB sweep in your project..."
wandb sweep wandb_lamb_sweep_config.yaml

echo ""
echo "✅ Sweep created!"
echo ""
echo "🏃 To start the sweep agent, run the command shown above:"
echo "   Example: wandb agent kredennis-mobileye/quickdraw-lr-sweep/SWEEP_ID"
echo ""
echo "🎯 This will optimize LAMB beyond your 86.82% baseline"

echo ""
echo "✅ LAMB sweep completed!"
echo ""
echo "📊 View results at: https://wandb.ai/your-username/your-project"
echo ""
echo "🎯 Expected improvements beyond 86.82% baseline with optimal hyperparameters"
