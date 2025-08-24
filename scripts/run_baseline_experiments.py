#!/usr/bin/env python3
"""
Run standardized baseline experiments with fixed seeds for reproducibility.

This script ensures that:
1. Same seed always gives same class selection
2. Comparable experiments across different class counts
3. Consistent training parameters
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

def get_fixed_class_selection(seed: int, num_classes: int):
    """Get the class selection that will be used for a given seed and class count."""
    import random
    
    # Load all available classes from local metadata
    metadata_path = Path(__file__).parent.parent / "data" / "quickdraw_parquet" / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    all_names = metadata["classes"]
    
    # Use same logic as create_dataloaders
    rng = random.Random(seed)
    selected_classes = sorted(rng.sample(all_names, num_classes))
    
    return selected_classes

def run_training(
    classes: int,
    epochs: int, 
    arch: str = "vit_tiny_patch16_224",
    seed: int = 42,
    experiment_name: str = None,
    lr: float = 0.0003,
    batch_size: int = 64,
    per_class_train: int = 1000,
    per_class_val: int = 200,
    warmup_epochs: int = 3,
    dry_run: bool = False
):
    """Run a standardized training experiment."""
    
    if experiment_name is None:
        experiment_name = f"fp32_baseline_{arch.replace('_', '-')}_{classes}c"
    
    # Preview the classes that will be selected
    try:
        selected_classes = get_fixed_class_selection(seed, classes)
        print(f"\nExperiment: {experiment_name}")
        print(f"Classes ({classes}): {', '.join(selected_classes[:5])}{'...' if len(selected_classes) > 5 else ''}")
        print(f"Seed: {seed} (ensures reproducible class selection)")
        print(f"Training: {per_class_train * classes:,} samples, {epochs} epochs")
        print(f"Validation: {per_class_val * classes:,} samples")
    except Exception as e:
        print(f"Could not preview classes: {e}")
    
    # Build command
    cmd = [
        "python", "scripts/train_quickdraw.py",
        "--classes", str(classes),
        "--epochs", str(epochs),
        "--per-class-train", str(per_class_train),
        "--per-class-val", str(per_class_val),
        "--batch-size", str(batch_size),
        "--no-pretrained",
        "--arch", arch,
        "--lr", str(lr),
        "--warmup-epochs", str(warmup_epochs),
        "--seed", str(seed)
    ]
    
    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])
    
    print(f"\nCommand: {' '.join(cmd)}")
    
    if dry_run:
        print("Dry run - not executing")
        return
    
    print("\nStarting training...")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run standardized baseline experiments")
    
    # Experiment selection
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick 10-class experiment")
    parser.add_argument("--main", action="store_true",
                       help="Run main 50-class experiment") 
    parser.add_argument("--full", action="store_true",
                       help="Run full 344-class experiment")
    parser.add_argument("--custom-classes", type=int,
                       help="Custom number of classes")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (auto-selected based on experiment)")
    parser.add_argument("--arch", type=str, default="vit_tiny_patch16_224",
                       help="Model architecture")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Custom experiment name")
    
    # Advanced options
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be run without executing")
    parser.add_argument("--preview-classes", action="store_true",
                       help="Show which classes would be selected")
    
    args = parser.parse_args()
    
    # Determine experiment parameters
    if args.quick:
        classes, epochs = 10, 20
        per_class_train, per_class_val = 500, 100
        batch_size = 32
    elif args.main:
        classes, epochs = 50, 30  
        per_class_train, per_class_val = 1000, 200
        batch_size = 64
    elif args.full:
        classes, epochs = 344, 50
        per_class_train, per_class_val = 1000, 200  
        batch_size = 32  # Smaller batch for memory
    elif args.custom_classes:
        classes = args.custom_classes
        epochs = max(20, min(50, 100 // (classes // 10)))  # Scale epochs
        per_class_train = min(1000, max(200, 5000 // classes))  # Scale samples
        per_class_val = per_class_train // 5
        batch_size = 64 if classes <= 50 else 32
    else:
        print("Must specify --quick, --main, --full, or --custom-classes")
        return 1
    
    # Override epochs if specified
    if args.epochs is not None:
        epochs = args.epochs
    
    # Preview classes mode
    if args.preview_classes:
        try:
            selected_classes = get_fixed_class_selection(args.seed, classes)
            print(f"\nClasses for seed={args.seed}, num_classes={classes}:")
            for i, cls in enumerate(selected_classes, 1):
                print(f"{i:3d}. {cls}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    # Run training
    success = run_training(
        classes=classes,
        epochs=epochs,
        arch=args.arch,
        seed=args.seed,
        experiment_name=args.experiment_name,
        per_class_train=per_class_train,
        per_class_val=per_class_val,
        batch_size=batch_size,
        dry_run=args.dry_run
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
