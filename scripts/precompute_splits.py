#!/usr/bin/env python3
"""
Pre-compute train/val splits for QuickDraw dataset to avoid recomputation on every run.

This script generates stratified splits for different class configurations and saves
them as JSON files for fast loading during training.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import get_all_class_names
from logging_config import setup_logger, log_and_print

def load_class_data_efficiently(data_dir: Path, classes: List[str], max_samples_per_class: int, seed: int) -> Dict[str, List[int]]:
    """
    Load class data efficiently and return class -> indices mapping.
    
    Returns:
        class_indices: Dict mapping class_name -> list of global indices
    """
    
    # Try per-class files first (more efficient)
    per_class_dir = data_dir / "per_class"
    if per_class_dir.exists():
        return load_from_per_class_files(per_class_dir, classes, max_samples_per_class, seed)
    else:
        return load_from_monolithic_file(data_dir, classes, max_samples_per_class, seed)

def load_from_per_class_files(per_class_dir: Path, classes: List[str], max_samples_per_class: int, seed: int) -> Dict[str, List[int]]:
    """Load class indices from per-class parquet files."""
    
    class_indices = {}
    global_index = 0
    
    with open(per_class_dir / "metadata.json", 'r') as f:
        per_class_metadata = json.load(f)
    
    for class_name in tqdm(classes, desc="Loading per-class data"):
        class_file = per_class_dir / f"{class_name.replace(' ', '_')}.parquet"
        
        if not class_file.exists():
            print(f"Warning: File not found for class '{class_name}': {class_file}")
            continue
            
        # Read just the shape to get the number of samples
        df = pd.read_parquet(class_file)
        available_samples = len(df)
        
        # Sample indices with seed for reproducibility
        random.seed(seed + hash(class_name))  # Unique seed per class
        if available_samples <= max_samples_per_class:
            # Use all available samples
            selected_indices = list(range(global_index, global_index + available_samples))
        else:
            # Randomly sample
            local_indices = random.sample(range(available_samples), max_samples_per_class)
            selected_indices = [global_index + idx for idx in local_indices]
        
        class_indices[class_name] = selected_indices
        global_index += len(selected_indices)
        
        print(f"  {class_name}: {len(selected_indices)}/{available_samples} samples")
    
    return class_indices

def load_from_monolithic_file(data_dir: Path, classes: List[str], max_samples_per_class: int, seed: int) -> Dict[str, List[int]]:
    """Load class indices from monolithic parquet file."""
    
    parquet_file = data_dir / "quickdraw_data.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Data file not found: {parquet_file}")
    
    print("Loading data from monolithic file...")
    
    # First pass: read only the 'word' column to get class distribution
    df_meta = pd.read_parquet(parquet_file, columns=['word'])
    
    class_indices = {}
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_mask = df_meta['word'] == class_name
        available_indices = df_meta.index[class_mask].tolist()
        
        if not available_indices:
            print(f"Warning: No samples found for class '{class_name}'")
            continue
            
        # Sample indices with seed for reproducibility
        random.seed(seed + hash(class_name))  # Unique seed per class
        if len(available_indices) <= max_samples_per_class:
            selected_indices = available_indices
        else:
            selected_indices = random.sample(available_indices, max_samples_per_class)
        
        class_indices[class_name] = selected_indices
        print(f"  {class_name}: {len(selected_indices)}/{len(available_indices)} samples")
    
    return class_indices

def create_stratified_split_from_indices(
    class_indices: Dict[str, List[int]], 
    train_samples_per_class: int, 
    val_samples_per_class: int, 
    calib_samples_per_class: int,
    seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create stratified train/val/calibration split from pre-loaded class indices.
    
    Returns:
        train_indices, val_indices, calib_indices
    """
    
    random.seed(seed)
    train_indices = []
    val_indices = []
    calib_indices = []
    
    for class_name, indices in class_indices.items():
        # Shuffle class indices
        class_indices_copy = indices.copy()
        random.shuffle(class_indices_copy)
        
        # Split into train/val/calib
        train_end = min(train_samples_per_class, len(class_indices_copy))
        val_end = min(train_end + val_samples_per_class, len(class_indices_copy))
        calib_end = min(val_end + calib_samples_per_class, len(class_indices_copy))
        
        train_indices.extend(class_indices_copy[:train_end])
        val_indices.extend(class_indices_copy[train_end:val_end])
        calib_indices.extend(class_indices_copy[val_end:calib_end])
        
        print(f"  {class_name}: {train_end} train, {val_end - train_end} val, {calib_end - val_end} calib")
    
    return train_indices, val_indices, calib_indices

def save_split_config(
    output_file: Path,
    classes: List[str],
    train_samples_per_class: int,
    val_samples_per_class: int,
    calib_samples_per_class: int,
    train_indices: List[int],
    val_indices: List[int],
    calib_indices: List[int],
    seed: int,
    metadata: Dict
):
    """Save split configuration to JSON file."""
    
    split_config = {
        'metadata': {
            'classes': classes,
            'num_classes': len(classes),
            'train_samples_per_class': train_samples_per_class,
            'val_samples_per_class': val_samples_per_class,
            'calib_samples_per_class': calib_samples_per_class,
            'total_train_samples': len(train_indices),
            'total_val_samples': len(val_indices),
            'total_calib_samples': len(calib_indices),
            'seed': seed,
            'created_with': 'precompute_splits.py'
        },
        'splits': {
            'train_indices': train_indices,
            'val_indices': val_indices,
            'calib_indices': calib_indices
        },
        'class_info': metadata
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(split_config, f, indent=2)
    
    print(f"Split configuration saved: {output_file}")

def generate_splits_for_config(
    data_dir: Path,
    classes: List[str],
    train_samples_per_class: int,
    val_samples_per_class: int,
    calib_samples_per_class: int,
    seed: int,
    output_dir: Path
):
    """Generate and save splits for a specific configuration."""
    
    print(f"\nGenerating splits for {len(classes)} classes")
    print(f"Classes: {classes[:5]}{'...' if len(classes) > 5 else ''}")
    print(f"Train: {train_samples_per_class}/class, Val: {val_samples_per_class}/class, Calib: {calib_samples_per_class}/class")
    print(f"Seed: {seed}")
    
    # Load class data
    max_samples_per_class = train_samples_per_class + val_samples_per_class + calib_samples_per_class
    class_indices = load_class_data_efficiently(data_dir, classes, max_samples_per_class, seed)
    
    if not class_indices:
        print(f"No data loaded for classes: {classes}")
        return
    
    # Create stratified split
    print("\nCreating stratified split...")
    train_indices, val_indices, calib_indices = create_stratified_split_from_indices(
        class_indices, train_samples_per_class, val_samples_per_class, calib_samples_per_class, seed
    )
    
    # Create filename
    classes_str = f"{len(classes)}c"
    samples_str = f"{train_samples_per_class}+{val_samples_per_class}+{calib_samples_per_class}"
    filename = f"split_{classes_str}_{samples_str}_seed{seed}.json"
    output_file = output_dir / filename
    
    # Save split configuration
    metadata = {
        'class_to_id': {class_name: i for i, class_name in enumerate(classes)},
        'id_to_class': {i: class_name for i, class_name in enumerate(classes)}
    }
    
    save_split_config(
        output_file, classes, train_samples_per_class, val_samples_per_class, calib_samples_per_class,
        train_indices, val_indices, calib_indices, seed, metadata
    )
    
    print(f"Split generated: {len(train_indices)} train, {len(val_indices)} val, {len(calib_indices)} calib")

def main():
    parser = argparse.ArgumentParser(description="Pre-compute train/val splits for QuickDraw dataset")
    
    parser.add_argument("--data-dir", type=str, default="data/quickdraw_parquet",
                       help="Directory containing QuickDraw parquet files")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                       help="Directory to save split configurations")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    # Configuration presets
    parser.add_argument("--quick", action="store_true",
                       help="Generate quick test configurations (3, 10 classes)")
    parser.add_argument("--main", action="store_true",
                       help="Generate main experiment configurations (10, 50 classes)")
    parser.add_argument("--all", action="store_true",
                       help="Generate all useful configurations")
    
    # Custom configuration
    parser.add_argument("--classes", type=int, nargs="+",
                       help="Custom number of classes to generate splits for")
    parser.add_argument("--train-samples", type=int, default=1000,
                       help="Training samples per class")
    parser.add_argument("--val-samples", type=int, default=200,
                       help="Validation samples per class")
    parser.add_argument("--calib-samples", type=int, default=None,
                       help="Calibration samples per class for quantization (auto-computed if not specified)")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Get available classes
    try:
        available_classes = get_all_class_names(str(data_dir))
    except Exception as e:
        print(f"Failed to load class names: {e}")
        return
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Available classes: {len(available_classes)}")
    
    # Determine configurations to generate
    configurations = []
    
    if args.quick:
        configurations.extend([3, 10])
    
    if args.main:
        configurations.extend([10, 50])
    
    if args.all:
        configurations.extend([3, 5, 10, 20, 50, 100])
    
    if args.classes:
        configurations.extend(args.classes)
    
    # Default to main if nothing specified
    if not configurations:
        configurations = [10, 50]
    
    # Remove duplicates and sort
    configurations = sorted(set(configurations))
    
    print(f" Generating splits for: {configurations} classes")
    
    # Generate splits for each configuration
    for num_classes in configurations:
        if num_classes > len(available_classes):
            print(f"Skipping {num_classes} classes (only {len(available_classes)} available)")
            continue
            
        # Auto-compute calibration samples based on README guidance
        if args.calib_samples is None:
            # Target 2048 total calibration samples, distributed across classes
            target_total_calib = 2048
            calib_per_class = max(8, min(200, target_total_calib // num_classes))
            
            # Apply README scaling rules
            if num_classes >= 300:  # All classes case
                calib_per_class = max(6, min(12, target_total_calib // num_classes))
            elif num_classes >= 40:  # Medium class count (40-299)
                calib_per_class = max(40, min(80, target_total_calib // num_classes))
            elif num_classes <= 10:  # Small class count
                calib_per_class = max(100, min(200, target_total_calib // num_classes))
            
            print(f"Auto-computed calibration: {calib_per_class} samples/class ({calib_per_class * num_classes} total)")
        else:
            calib_per_class = args.calib_samples
            
        # Select classes deterministically with seed
        random.seed(args.seed)
        selected_classes = sorted(random.sample(available_classes, num_classes))
        
        generate_splits_for_config(
            data_dir, selected_classes, args.train_samples, args.val_samples, calib_per_class,
            args.seed, output_dir
        )
    
    print(f"\n All splits generated in: {output_dir}")

if __name__ == "__main__":
    main()
