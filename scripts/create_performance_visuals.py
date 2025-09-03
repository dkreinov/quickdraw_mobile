#!/usr/bin/env python3
"""
Create comprehensive performance visualizations for GitHub documentation.
Generates publication-quality plots from evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def create_per_class_performance_plot(metrics_df, save_dir):
    """Create comprehensive per-class performance visualization."""
    
    # Sort by accuracy for better visualization
    metrics_sorted = metrics_df.sort_values('accuracy', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top plot: Per-class accuracy bar chart
    colors = ['#d62728' if acc < 0.5 else '#ff7f0e' if acc < 0.7 else '#2ca02c' 
              for acc in metrics_sorted['accuracy']]
    
    bars = ax1.barh(range(len(metrics_sorted)), metrics_sorted['accuracy'] * 100, 
                    color=colors, alpha=0.8)
    
    # Add performance zones
    ax1.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Poor (<50%)')
    ax1.axvline(x=70, color='orange', linestyle='--', alpha=0.7, label='Good (70%)')
    ax1.axvline(x=85, color='green', linestyle='--', alpha=0.7, label='Excellent (85%)')
    
    # Highlight worst and best classes
    worst_idx = 0
    best_idx = len(metrics_sorted) - 1
    ax1.barh(worst_idx, metrics_sorted.iloc[worst_idx]['accuracy'] * 100, 
             color='red', alpha=1.0, label=f"Worst: {metrics_sorted.iloc[worst_idx]['class_name']}")
    ax1.barh(best_idx, metrics_sorted.iloc[best_idx]['accuracy'] * 100, 
             color='darkgreen', alpha=1.0, label=f"Best: {metrics_sorted.iloc[best_idx]['class_name']}")
    
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classes (sorted by performance)', fontsize=12, fontweight='bold')
    ax1.set_title('Per-Class Performance: QuickDraw 344 Classes (73.02% Overall)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_acc = metrics_sorted['accuracy'].mean() * 100
    std_acc = metrics_sorted['accuracy'].std() * 100
    stats_text = f'Mean: {mean_acc:.1f}%\nStd: {std_acc:.1f}%\nClasses: {len(metrics_sorted)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom plot: Accuracy distribution histogram
    ax2.hist(metrics_sorted['accuracy'] * 100, bins=30, alpha=0.7, color='skyblue', 
             edgecolor='black', linewidth=0.5)
    ax2.axvline(x=mean_acc, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_acc:.1f}%')
    ax2.axvline(x=mean_acc - std_acc, color='red', linestyle='--', alpha=0.7, label=f'±1 Std')
    ax2.axvline(x=mean_acc + std_acc, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Classes', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Per-Class Accuracies', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "performance_overview.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Performance overview saved: {plot_path}")
    return plot_path

def create_confusion_heatmap(confusion_df, metrics_df, save_dir, top_n=20):
    """Create focused confusion matrix for worst performing classes."""
    
    # Get worst performing classes
    worst_classes = metrics_df.nsmallest(top_n, 'accuracy')['class_name'].tolist()
    
    # Extract subset of confusion matrix
    confusion_subset = confusion_df.loc[worst_classes, worst_classes]
    
    # Normalize for better visualization
    confusion_normalized = confusion_subset.div(confusion_subset.sum(axis=1), axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create heatmap
    im = ax.imshow(confusion_normalized.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(worst_classes)))
    ax.set_yticks(range(len(worst_classes)))
    ax.set_xticklabels(worst_classes, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(worst_classes, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Confusion Rate', fontsize=12, fontweight='bold')
    
    # Add title and labels
    ax.set_title(f'Confusion Matrix: {top_n} Worst Performing Classes\n(Normalized by True Class)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    # Add diagonal line for perfect classification
    ax.plot([0, len(worst_classes)-1], [0, len(worst_classes)-1], 'w-', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / f"confusion_worst_{top_n}_classes.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Confusion heatmap saved: {plot_path}")
    return plot_path

def create_performance_comparison(metrics_df, save_dir):
    """Create comparison between different performance metrics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision vs Recall scatter
    ax1.scatter(metrics_df['recall'] * 100, metrics_df['precision'] * 100, 
                alpha=0.6, s=50, c=metrics_df['accuracy'], cmap='viridis')
    ax1.set_xlabel('Recall (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Precision vs Recall by Class', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal line for balanced performance
    ax1.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect Balance')
    ax1.legend()
    
    # 2. F1-Score vs Accuracy
    ax2.scatter(metrics_df['accuracy'] * 100, metrics_df['f1_score'] * 100, 
                alpha=0.6, s=50, color='orange')
    ax2.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontsize=11, fontweight='bold')
    ax2.set_title('F1-Score vs Accuracy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(metrics_df['accuracy'] * 100, metrics_df['f1_score'] * 100, 1)
    p = np.poly1d(z)
    ax2.plot(metrics_df['accuracy'] * 100, p(metrics_df['accuracy'] * 100), 
             "r--", alpha=0.8, label=f'Correlation: {np.corrcoef(metrics_df["accuracy"], metrics_df["f1_score"])[0,1]:.3f}')
    ax2.legend()
    
    # 3. Support (sample count) vs Performance
    ax3.scatter(metrics_df['support'], metrics_df['accuracy'] * 100, 
                alpha=0.6, s=50, color='green')
    ax3.set_xlabel('Number of Test Samples', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Test Sample Count vs Accuracy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top and Bottom performers
    top_10 = metrics_df.nlargest(10, 'accuracy')[['class_name', 'accuracy']]
    bottom_10 = metrics_df.nsmallest(10, 'accuracy')[['class_name', 'accuracy']]
    
    # Create comparison bar plot
    y_pos_top = np.arange(len(top_10))
    y_pos_bottom = np.arange(len(bottom_10)) - len(bottom_10) - 1
    
    ax4.barh(y_pos_top, top_10['accuracy'] * 100, color='green', alpha=0.7, label='Top 10')
    ax4.barh(y_pos_bottom, bottom_10['accuracy'] * 100, color='red', alpha=0.7, label='Bottom 10')
    
    # Set labels for top performers
    ax4.set_yticks(y_pos_top)
    ax4.set_yticklabels([name[:15] + '...' if len(name) > 15 else name for name in top_10['class_name']], 
                        fontsize=9)
    
    # Set labels for bottom performers
    bottom_ticks = y_pos_bottom
    ax4.set_yticks(list(y_pos_top) + list(bottom_ticks))
    all_labels = ([name[:15] + '...' if len(name) > 15 else name for name in top_10['class_name']] + 
                  [name[:15] + '...' if len(name) > 15 else name for name in bottom_10['class_name']])
    ax4.set_yticklabels(all_labels, fontsize=9)
    
    ax4.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Best vs Worst Performing Classes', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "performance_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Performance analysis saved: {plot_path}")
    return plot_path

def create_summary_statistics(metrics_df, summary_data, save_dir):
    """Create a summary statistics visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Performance distribution pie chart
    poor = len(metrics_df[metrics_df['accuracy'] < 0.5])
    fair = len(metrics_df[(metrics_df['accuracy'] >= 0.5) & (metrics_df['accuracy'] < 0.7)])
    good = len(metrics_df[(metrics_df['accuracy'] >= 0.7) & (metrics_df['accuracy'] < 0.85)])
    excellent = len(metrics_df[metrics_df['accuracy'] >= 0.85])
    
    sizes = [poor, fair, good, excellent]
    labels = [f'Poor (<50%)\n{poor} classes', f'Fair (50-70%)\n{fair} classes', 
              f'Good (70-85%)\n{good} classes', f'Excellent (≥85%)\n{excellent} classes']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                                       startangle=90)
    ax1.set_title('Class Performance Distribution', fontsize=14, fontweight='bold')
    
    # 2. Key metrics comparison
    metrics_names = ['Overall Accuracy', 'Mean Per-Class', 'Worst Class', 'Best Class', 'Std Dev']
    metrics_values = [
        summary_data['overall_accuracy'] * 100,
        summary_data['mean_per_class_accuracy'] * 100,
        summary_data['worst_class']['accuracy'] * 100,
        summary_data['best_class']['accuracy'] * 100,
        summary_data['std_per_class_accuracy'] * 100
    ]
    
    bars = ax2.bar(metrics_names, metrics_values, 
                   color=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd'])
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Cumulative performance curve
    sorted_acc = metrics_df['accuracy'].sort_values(ascending=False) * 100
    cumulative_acc = np.cumsum(sorted_acc) / np.arange(1, len(sorted_acc) + 1)
    
    ax3.plot(range(1, len(cumulative_acc) + 1), cumulative_acc, 'b-', linewidth=2)
    ax3.set_xlabel('Number of Classes (sorted by performance)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Average Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Performance Curve', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add horizontal line at overall accuracy
    ax3.axhline(y=summary_data['overall_accuracy'] * 100, color='red', linestyle='--', 
                label=f'Overall: {summary_data["overall_accuracy"]*100:.1f}%')
    ax3.legend()
    
    # 4. Model information table
    ax4.axis('tight')
    ax4.axis('off')
    
    model_info = [
        ['Metric', 'Value'],
        ['Total Classes', f"{summary_data['num_classes']}"],
        ['Total Samples', f"{summary_data['total_samples']:,}"],
        ['Overall Accuracy', f"{summary_data['overall_accuracy']*100:.2f}%"],
        ['Mean Per-Class Accuracy', f"{summary_data['mean_per_class_accuracy']*100:.2f}%"],
        ['Standard Deviation', f"{summary_data['std_per_class_accuracy']*100:.2f}%"],
        ['Best Performing Class', f"{summary_data['best_class']['name']} ({summary_data['best_class']['accuracy']*100:.1f}%)"],
        ['Worst Performing Class', f"{summary_data['worst_class']['name']} ({summary_data['worst_class']['accuracy']*100:.1f}%)"],
        ['Performance Gap', f"{(summary_data['best_class']['accuracy'] - summary_data['worst_class']['accuracy'])*100:.1f} pp"]
    ]
    
    table = ax4.table(cellText=model_info[1:], colLabels=model_info[0], 
                      cellLoc='center', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(model_info)):
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 1)].set_facecolor('#f0f0f0')
        if i == 0:  # Header
            table[(i, 0)].set_text_props(weight='bold')
            table[(i, 1)].set_text_props(weight='bold')
            table[(i, 0)].set_facecolor('#d0d0d0')
            table[(i, 1)].set_facecolor('#d0d0d0')
    
    ax4.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_dir / "summary_statistics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Summary statistics saved: {plot_path}")
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Create performance visualizations')
    parser.add_argument('--eval-dir', type=str, 
                       default='results/best_model_enhanced_aug_evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, 
                       default='results/performance_visuals',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating performance visualizations...")
    print(f"📁 Input: {eval_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Load data
    metrics_df = pd.read_csv(eval_dir / "per_class_metrics.csv")
    confusion_df = pd.read_csv(eval_dir / "confusion_matrix_normalized.csv", index_col=0)
    
    with open(eval_dir / "evaluation_summary.json", 'r') as f:
        summary_data = json.load(f)
    
    print(f"📈 Loaded data for {len(metrics_df)} classes")
    
    # Create visualizations
    plots_created = []
    
    plots_created.append(create_per_class_performance_plot(metrics_df, output_dir))
    plots_created.append(create_confusion_heatmap(confusion_df, metrics_df, output_dir))
    plots_created.append(create_performance_comparison(metrics_df, output_dir))
    plots_created.append(create_summary_statistics(metrics_df, summary_data, output_dir))
    
    print(f"\n🎨 Created {len(plots_created)} visualizations:")
    for plot in plots_created:
        print(f"   📊 {plot.name}")
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"🚀 Ready for GitHub upload!")

if __name__ == "__main__":
    main()
