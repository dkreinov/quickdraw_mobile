#!/usr/bin/env python3
"""
Plot training curves for all experiments to compare performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def load_experiment_data(results_dir="results"):
    """Load all experiment history files."""
    experiments = {}
    
    # Find all history files
    history_files = glob.glob(f"{results_dir}/*_history.json")
    
    for file_path in history_files:
        # Extract experiment name from filename
        file_name = Path(file_path).stem
        exp_name = file_name.replace("_history", "")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            experiments[exp_name] = data
            print(f"✅ Loaded: {exp_name}")
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
    
    return experiments

def plot_training_curves(experiments, save_dir="experiments"):
    """Plot validation accuracy curves and overfitting gap for all experiments."""
    
    # Create figure with 2 main plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Experiment Comparison: Validation Accuracy & Overfitting Gap', fontsize=16, fontweight='bold')
    
    # Color scheme for different experiments
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Filter to relevant experiments (50-class ones)
    relevant_experiments = {}
    for exp_name, data in experiments.items():
        # Focus on 50-class experiments
        if '50c' in exp_name or 'stronger_augmentation' in exp_name:
            relevant_experiments[exp_name] = data
    
    for i, (exp_name, data) in enumerate(relevant_experiments.items()):
        color = colors[i % len(colors)]
        
        # Extract epochs and metrics
        epochs = [m['epoch'] for m in data['metrics']]
        train_acc = [m['train_top1'] for m in data['metrics']]
        val_acc = [m['val_top1'] for m in data['metrics']]
        gap = [t - v for t, v in zip(train_acc, val_acc)]
        
        # Clean up experiment names for legend
        display_name = exp_name.replace('fp32_baseline_50c_main', 'Baseline (Original Aug)')
        display_name = display_name.replace('exp2_stronger_augmentation', 'Stronger Augmentation')
        
        # Plot validation accuracy (MAIN METRIC)
        ax1.plot(epochs, val_acc, color=color, linestyle='-', linewidth=3, 
                marker='o', markersize=4, label=display_name, alpha=0.8)
        
        # Plot overfitting gap 
        ax2.plot(epochs, gap, color=color, linestyle='-', linewidth=3,
                marker='s', markersize=4, label=display_name, alpha=0.8)
    
    # Configure Validation Accuracy plot
    ax1.set_title('Validation Accuracy Evolution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add baseline reference line
    ax1.axhline(y=62.5, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Baseline @ Epoch 20 (62.5%)')
    ax1.axhline(y=65.0, color='green', linestyle='--', alpha=0.5, linewidth=2, 
               label='Target (65%)')
    
    # Configure Overfitting Gap plot
    ax2.set_title('Overfitting Gap (Train - Val)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gap (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Add reference lines
    ax2.axhline(y=19.64, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='Baseline @ Epoch 20 (19.64%)')
    ax2.axhline(y=15.0, color='green', linestyle='--', alpha=0.5, linewidth=2,
               label='Target (<15%)')
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "validation_curves_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved validation curves: {save_path}")
    
    # Also save individual experiment summaries
    save_summary(experiments, save_dir)
    
    plt.show()

def save_summary(experiments, save_dir):
    """Save experiment summary table."""
    summary_file = Path(save_dir) / "experiment_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("EXPERIMENT COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"{'Experiment':<25} {'Best Val':<10} {'Epoch':<6} {'Final Val':<10} {'Final Gap':<10}\n")
        f.write("-"*60 + "\n")
        
        for exp_name, data in experiments.items():
            best_val = data['best_val_acc']
            best_epoch = data['best_epoch']
            final_metrics = data['metrics'][-1]
            final_val = final_metrics['val_top1']
            final_gap = final_metrics['train_top1'] - final_metrics['val_top1']
            
            f.write(f"{exp_name:<25} {best_val:<10.2f} {best_epoch:<6} {final_val:<10.2f} {final_gap:<10.2f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("BASELINE COMPARISON:\n")
        f.write("- Original baseline (epoch 20): 62.50% val, 19.64% gap\n")
        f.write("- Target: >65% val accuracy, <15% gap\n")
    
    print(f"📋 Saved summary: {summary_file}")

def plot_epoch_20_comparison(experiments, save_dir="experiments"):
    """Create bar chart comparing epoch 20 performance."""
    
    exp_names = []
    val_accs = []
    gaps = []
    
    baseline_val = 62.50
    baseline_gap = 19.64
    
    for exp_name, data in experiments.items():
        # Find epoch 20 data
        epoch_20 = None
        for metric in data['metrics']:
            if metric['epoch'] == 20:
                epoch_20 = metric
                break
        
        if epoch_20:
            exp_names.append(exp_name.replace('_', '\n'))  # Line break for readability
            val_accs.append(epoch_20['val_top1'])
            gaps.append(epoch_20['train_top1'] - epoch_20['val_top1'])
    
    # Add baseline
    exp_names.insert(0, 'baseline\n(original)')
    val_accs.insert(0, baseline_val)
    gaps.insert(0, baseline_gap)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Epoch 20 Performance Comparison', fontsize=14, fontweight='bold')
    
    # Validation accuracy
    bars1 = ax1.bar(exp_names, val_accs, color=['gray'] + ['skyblue'] * (len(val_accs)-1))
    ax1.set_title('Validation Accuracy at Epoch 20')
    ax1.set_ylabel('Accuracy (%)')
    ax1.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_val}%)')
    ax1.legend()
    
    # Add value labels on bars
    for bar, val in zip(bars1, val_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom')
    
    # Overfitting gap
    bars2 = ax2.bar(exp_names, gaps, color=['gray'] + ['lightcoral'] * (len(gaps)-1))
    ax2.set_title('Overfitting Gap at Epoch 20')
    ax2.set_ylabel('Gap (%)')
    ax2.axhline(y=baseline_gap, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_gap}%)')
    ax2.legend()
    
    # Add value labels on bars
    for bar, gap in zip(bars2, gaps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{gap:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_dir) / "epoch_20_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved comparison: {save_path}")
    
    plt.show()

def main():
    print("📊 Generating validation accuracy curve visualizations...")
    
    # Load all experiment data
    experiments = load_experiment_data()
    
    if not experiments:
        print("❌ No experiment data found!")
        return
    
    print(f"\n📈 Found {len(experiments)} experiments:")
    for name in experiments.keys():
        print(f"  - {name}")
    
    # Generate main plots - validation curves and overfitting gap
    plot_training_curves(experiments)
    
    print("\n✅ Validation curve visualization complete!")
    print("📊 Check experiments/validation_curves_comparison.png to see:")
    print("   - Validation accuracy evolution for each experiment")
    print("   - Overfitting gap trends")
    print("   - Momentum and convergence patterns")

if __name__ == "__main__":
    main()
