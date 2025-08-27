#!/usr/bin/env python3
"""
Compare results across multiple experiments.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --format table
    python scripts/compare_experiments.py --save-csv results/comparison_table.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

def find_experiments(experiments_dir: Path) -> List[Dict]:
    """Find all experiments and load their metadata."""
    experiments = []
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and exp_dir.name != 'comparison':
            # Load experiment metadata
            meta_file = exp_dir / "experiment_meta.json"
            eval_file = exp_dir / "evaluation" / "summary_table.json"
            
            if meta_file.exists() and eval_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    with open(eval_file) as f:
                        results = json.load(f)
                    
                    experiment = {
                        "name": exp_dir.name,
                        "path": str(exp_dir),
                        "meta": meta,
                        "results": results
                    }
                    experiments.append(experiment)
                    
                except Exception as e:
                    print(f"Warning: Could not load {exp_dir.name}: {e}")
    
    return experiments

def create_comparison_table(experiments: List[Dict]) -> pd.DataFrame:
    """Create a comparison table from experiments."""
    rows = []
    
    for exp in experiments:
        results = exp["results"]
        meta = exp["meta"]
        
        # Extract key metrics
        row = {
            "Experiment": exp["name"],
            "Architecture": results["model_info"]["architecture"],
            "Classes": results["model_info"]["num_classes"],
            "Parameters": f"{results['model_info']['total_parameters']:,}",
            "Model Size (MB)": results["model_info"]["model_size_mb"],
            "Checkpoint Size (MB)": results["model_info"]["checkpoint_size_mb"],
            "Top-1 Accuracy (%)": results["accuracy_metrics"]["top1_accuracy"],
            "Top-5 Accuracy (%)": results["accuracy_metrics"]["top5_accuracy"],
            "Loss": results["accuracy_metrics"]["avg_loss"],
        }
        
        # Add latency if available
        if "latency_metrics" in results:
            if "cpu_ms_per_image" in results["latency_metrics"]:
                cpu_latency = results["latency_metrics"]["cpu_ms_per_image"]["mean"]
                row["CPU Latency (ms)"] = f"{cpu_latency:.2f}"
            else:
                row["CPU Latency (ms)"] = "N/A"
                
            if "gpu_ms_per_image" in results["latency_metrics"]:
                gpu_latency = results["latency_metrics"]["gpu_ms_per_image"]["mean"]
                row["GPU Latency (ms)"] = f"{gpu_latency:.2f}"
            else:
                row["GPU Latency (ms)"] = "N/A"
        else:
            row["CPU Latency (ms)"] = "N/A"
            row["GPU Latency (ms)"] = "N/A"
        
        # Add timestamp
        row["Timestamp"] = meta.get("timestamp", "unknown")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by Top-1 accuracy (descending)
    if len(df) > 0:
        df = df.sort_values("Top-1 Accuracy (%)", ascending=False)
    
    return df

def print_table(df: pd.DataFrame, format_style: str = "simple"):
    """Print the comparison table."""
    if len(df) == 0:
        print("No experiments found.")
        return
    
    if format_style == "markdown":
        print(df.to_markdown(index=False, tablefmt="github"))
    elif format_style == "latex":
        print(df.to_latex(index=False))
    else:
        # Simple table format
        print("\n" + "="*100)
        print("EXPERIMENT COMPARISON")
        print("="*100)
        
        # Key metrics first
        key_cols = [
            "Experiment", "Top-1 Accuracy (%)", "Top-5 Accuracy (%)", 
            "Model Size (MB)", "GPU Latency (ms)", "CPU Latency (ms)"
        ]
        
        available_cols = [col for col in key_cols if col in df.columns]
        key_df = df[available_cols]
        
        print("\nKey Metrics:")
        print(key_df.to_string(index=False))
        
        # Full details
        print("\nFull Details:")
        print(df.to_string(index=False))
        print()

def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments-dir", type=str, default="results/experiments",
                       help="Directory containing experiments")
    parser.add_argument("--format", choices=["simple", "markdown", "latex"], default="simple",
                       help="Output format")
    parser.add_argument("--save-csv", type=str, default=None,
                       help="Save comparison table as CSV")
    parser.add_argument("--save-json", type=str, default=None,
                       help="Save detailed comparison as JSON")
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Error: Experiments directory {experiments_dir} does not exist")
        return 1
    
    # Find and load experiments
    experiments = find_experiments(experiments_dir)
    
    if not experiments:
        print(f"No experiments found in {experiments_dir}")
        return 1
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['name']}")
    
    # Create comparison table
    comparison_df = create_comparison_table(experiments)
    
    # Print table
    print_table(comparison_df, args.format)
    
    # Save outputs
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(csv_path, index=False)
        print(f"Comparison table saved to: {csv_path}")
    
    if args.save_json:
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        detailed_comparison = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary": {
                "total_experiments": len(experiments),
                "best_accuracy": float(comparison_df["Top-1 Accuracy (%)"].max()) if len(comparison_df) > 0 else None,
                "fastest_gpu": comparison_df["GPU Latency (ms)"].replace("N/A", float('inf')).astype(float).min() if len(comparison_df) > 0 else None
            },
            "experiments": experiments
        }
        
        with open(json_path, 'w') as f:
            json.dump(detailed_comparison, f, indent=2)
        print(f"Detailed comparison saved to: {json_path}")
    
    # Automatically save to comparison directory
    comparison_dir = experiments_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Save latest comparison
    comparison_df.to_csv(comparison_dir / "latest_comparison.csv", index=False)
    
    # Save timestamped comparison
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    comparison_df.to_csv(comparison_dir / f"comparison_{timestamp}.csv", index=False)
    
    print(f"\nComparison files saved to: {comparison_dir}")
    return 0

if __name__ == "__main__":
    exit(main())


