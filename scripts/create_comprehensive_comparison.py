#!/usr/bin/env python3
"""
Create comprehensive comparison plot between Single-GPU AdamW and LAMB Multi-GPU.
Shows training/validation accuracy and loss curves.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

def create_comprehensive_comparison():
    """Create 4-panel comparison plot."""
    
    # Load the two experiments
    with open('results/fp32_baseline_50c_main_history.json') as f:
        single_gpu = json.load(f)
        
    with open('results/lamb_multigpu_base_lr_history.json') as f:
        lamb_gpu = json.load(f)

    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Single-GPU AdamW vs LAMB Multi-GPU Comparison', fontsize=16, fontweight='bold')

    # Extract metrics
    single_epochs = [m['epoch'] for m in single_gpu['metrics']]
    single_train_acc = [m['train_top1'] for m in single_gpu['metrics']]
    single_val_acc = [m['val_top1'] for m in single_gpu['metrics']]
    single_train_loss = [m['train_loss'] for m in single_gpu['metrics']]
    single_val_loss = [m['val_loss'] for m in single_gpu['metrics']]

    lamb_epochs = [m['epoch'] for m in lamb_gpu['metrics']]
    lamb_train_acc = [m['train_top1'] for m in lamb_gpu['metrics']]
    lamb_val_acc = [m['val_top1'] for m in lamb_gpu['metrics']]
    lamb_train_loss = [m['train_loss'] for m in lamb_gpu['metrics']]
    lamb_val_loss = [m['val_loss'] for m in lamb_gpu['metrics']]

    # Plot 1: Training Accuracy
    axes[0,0].plot(single_epochs, single_train_acc, 'b-', label='Single-GPU AdamW', linewidth=2)
    axes[0,0].plot(lamb_epochs, lamb_train_acc, 'r-', label='LAMB Multi-GPU', linewidth=2)
    axes[0,0].set_title('Training Accuracy', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Training Top-1 Accuracy (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy (KEY PLOT)
    axes[0,1].plot(single_epochs, single_val_acc, 'b-', 
                   label=f'Single-GPU AdamW (Best: {single_gpu["best_val_acc"]:.2f}%)', linewidth=2)
    axes[0,1].plot(lamb_epochs, lamb_val_acc, 'r-', 
                   label=f'LAMB Multi-GPU (Best: {lamb_gpu["best_val_acc"]:.2f}%)', linewidth=2)
    axes[0,1].set_title('Validation Accuracy (KEY METRIC)', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Validation Top-1 Accuracy (%)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Plot 3: Training Loss
    axes[1,0].plot(single_epochs, single_train_loss, 'b-', label='Single-GPU AdamW', linewidth=2)
    axes[1,0].plot(lamb_epochs, lamb_train_loss, 'r-', label='LAMB Multi-GPU', linewidth=2)
    axes[1,0].set_title('Training Loss', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Training Loss')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Plot 4: Validation Loss
    axes[1,1].plot(single_epochs, single_val_loss, 'b-', label='Single-GPU AdamW', linewidth=2)
    axes[1,1].plot(lamb_epochs, lamb_val_loss, 'r-', label='LAMB Multi-GPU', linewidth=2)
    axes[1,1].set_title('Validation Loss', fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Validation Loss')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot
    output_path = 'results/comparison_plots/comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Comprehensive comparison plot saved to: {output_path}')
    
    # Print summary
    improvement = lamb_gpu["best_val_acc"] - single_gpu["best_val_acc"]
    print(f'\n🎯 KEY RESULTS:')
    print(f'Single-GPU AdamW: {single_gpu["best_val_acc"]:.2f}% validation accuracy')
    print(f'LAMB Multi-GPU: {lamb_gpu["best_val_acc"]:.2f}% validation accuracy')
    print(f'IMPROVEMENT: +{improvement:.2f} percentage points ({improvement/single_gpu["best_val_acc"]*100:.1f}% relative)')

if __name__ == "__main__":
    create_comprehensive_comparison()
