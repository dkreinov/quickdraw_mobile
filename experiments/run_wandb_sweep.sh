#!/bin/bash
# Launch W&B Sweep for LR Finding
#
# This script creates and runs a W&B sweep to find the optimal learning rate
# for multi-GPU training with batch=1024.

# Activate environment
source ../.venv/bin/activate
module load nvidia/cuda/12.1

# Change to experiments directory
cd /localdrive/users/dkreinov/quickdraw-mobilevit-quant/experiments

# Set W&B project and entity
export WANDB_PROJECT="quickdraw-lamb-optimization"
export WANDB_ENTITY="kredennis-mobileye"

echo "=== Creating LAMB Optimization Sweep ==="
echo "Project: $WANDB_PROJECT"
echo "Entity: $WANDB_ENTITY"
echo "Optimizer: LAMB (Fixed LR=0.0003)"
echo "Search: Weight Decay, Batch Size, Warmup, Label Smoothing"
echo "Target: Optimize beyond 86.82% baseline"
echo ""

# Create the sweep
echo "Creating LAMB Fixed-LR sweep..."
SWEEP_ID=$(wandb sweep wandb_lamb_fixed_lr_config.yaml 2>&1 | grep "Created sweep" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create sweep. Creating manually..."
    wandb sweep wandb_lamb_fixed_lr_config.yaml
    echo ""
    echo "Copy the sweep ID from above and run:"
    echo "  wandb agent <SWEEP_ID>"
else
    echo "Sweep created: $SWEEP_ID"
    echo ""
    echo "Starting sweep agent..."
    wandb agent $SWEEP_ID
fi

echo ""
echo "=== Sweep Complete ==="
echo "Check results at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
