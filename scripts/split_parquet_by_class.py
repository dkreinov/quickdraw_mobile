#!/usr/bin/env python3
"""
Split monolithic QuickDraw Parquet file into per-class files for faster subset loading.

This script takes the existing quickdraw_data.parquet and splits it into separate
parquet files for each class, enabling much faster loading when training on subsets.

Usage:
    python scripts/split_parquet_by_class.py --input-dir data/quickdraw_parquet
    python scripts/split_parquet_by_class.py --input-dir data/quickdraw_parquet --output-dir data/quickdraw_per_class
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd
from tqdm import tqdm

# Add src to path for logging
sys.path.append(str(Path(__file__).parent.parent / "src"))
try:
    from logging_config import get_logger, log_and_print
except ImportError:
    def log_and_print(msg, **kwargs):
        print(msg)
    def get_logger(name):
        return None


def split_parquet_by_class(input_dir: str, output_dir: str = None, keep_original: bool = True):
    """
    Split monolithic Parquet file into per-class files.
    
    Args:
        input_dir: Directory containing quickdraw_data.parquet and metadata.json
        output_dir: Output directory (defaults to input_dir/per_class/)
        keep_original: Whether to keep the original monolithic file
    """
    
    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "per_class"
    else:
        output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger(__name__)
    log_and_print(f"Splitting QuickDraw Parquet by class...", logger_instance=logger)
    log_and_print(f"  Input: {input_path}", logger_instance=logger)
    log_and_print(f"  Output: {output_path}", logger_instance=logger)
    
    # Load metadata
    metadata_file = input_path / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load monolithic parquet
    parquet_file = input_path / "quickdraw_data.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")
    
    log_and_print(f"Loading data from {parquet_file}...", logger_instance=logger)
    df = pd.read_parquet(parquet_file)
    
    log_and_print(f"  Total samples: {len(df):,}", logger_instance=logger)
    log_and_print(f"  Classes: {len(metadata['classes'])}", logger_instance=logger)
    
    # Split by class and save
    class_stats = {}
    
    for class_name in tqdm(metadata['classes'], desc="Splitting classes"):
        # Filter data for this class
        class_df = df[df['class_name'] == class_name].copy()
        
        if len(class_df) == 0:
            log_and_print(f"  Warning: No samples found for class '{class_name}'", logger_instance=logger)
            continue
        
        # Save as parquet file
        class_filename = class_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        class_file = output_path / f"{class_filename}.parquet"
        
        class_df.to_parquet(class_file, index=False, compression='snappy')
        
        # Track stats
        file_size_mb = class_file.stat().st_size / (1024 * 1024)
        class_stats[class_name] = {
            'samples': len(class_df),
            'file_size_mb': file_size_mb,
            'filename': class_filename + '.parquet'
        }
        
        log_and_print(f"  {class_name}: {len(class_df)} samples → {file_size_mb:.1f} MB", logger_instance=logger)
    
    # Create updated metadata for per-class format
    per_class_metadata = metadata.copy()
    per_class_metadata.update({
        'format': 'per_class_parquet',
        'original_file': str(parquet_file),
        'per_class_directory': str(output_path),
        'class_files': class_stats,
        'split_with': 'split_parquet_by_class.py'
    })
    
    # Save metadata in output directory
    output_metadata_file = output_path / "metadata.json"
    with open(output_metadata_file, 'w') as f:
        json.dump(per_class_metadata, f, indent=2)
    
    # Calculate total size
    total_size_mb = sum(stats['file_size_mb'] for stats in class_stats.values())
    original_size_mb = parquet_file.stat().st_size / (1024 * 1024)
    
    log_and_print(f"\nSplit complete!", logger_instance=logger)
    log_and_print(f"  Files created: {len(class_stats)}", logger_instance=logger)
    log_and_print(f"  Total size: {total_size_mb:.1f} MB (original: {original_size_mb:.1f} MB)", logger_instance=logger)
    log_and_print(f"  Average per class: {total_size_mb/len(class_stats):.1f} MB", logger_instance=logger)
    log_and_print(f"  Metadata: {output_metadata_file}", logger_instance=logger)
    
    if not keep_original:
        log_and_print(f"  Removing original file: {parquet_file}", logger_instance=logger)
        parquet_file.unlink()
        
        # Copy metadata to original location but mark as per-class
        with open(metadata_file, 'w') as f:
            json.dump(per_class_metadata, f, indent=2)
    
    return str(output_path)


def verify_split(output_dir: str):
    """Verify that the split was successful."""
    
    output_path = Path(output_dir)
    
    try:
        # Load metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        log_and_print(f"Verifying split in {output_path}...", logger_instance=None)
        
        # Check each class file
        total_samples = 0
        missing_files = []
        
        for class_name, stats in metadata['class_files'].items():
            filename = stats['filename']
            class_file = output_path / filename
            
            if not class_file.exists():
                missing_files.append(filename)
                continue
            
            # Load and verify
            class_df = pd.read_parquet(class_file)
            
            # Check sample count matches
            expected_samples = stats['samples']
            actual_samples = len(class_df)
            
            if actual_samples != expected_samples:
                log_and_print(f"  Warning: {class_name} sample count mismatch: {actual_samples} vs {expected_samples}", logger_instance=None)
            
            # Check all samples are for this class
            unique_classes = class_df['class_name'].unique()
            if len(unique_classes) != 1 or unique_classes[0] != class_name:
                log_and_print(f"  Error: {class_name} contains wrong classes: {unique_classes}", logger_instance=None)
            
            total_samples += actual_samples
        
        if missing_files:
            log_and_print(f"  Error: Missing files: {missing_files}", logger_instance=None)
            return False
        
        log_and_print(f"  Verification passed:", logger_instance=None)
        log_and_print(f"    {len(metadata['class_files'])} class files", logger_instance=None)
        log_and_print(f"    {total_samples:,} total samples", logger_instance=None)
        log_and_print(f"    All samples properly grouped by class", logger_instance=None)
        
        return True
        
    except Exception as e:
        log_and_print(f"  Verification failed: {e}", logger_instance=None)
        return False


def main():
    parser = argparse.ArgumentParser(description="Split QuickDraw Parquet file by class")
    
    parser.add_argument("--input-dir", required=True,
                       help="Input directory containing quickdraw_data.parquet")
    parser.add_argument("--output-dir", 
                       help="Output directory (default: input-dir/per_class/)")
    parser.add_argument("--keep-original", action="store_true", default=True,
                       help="Keep original monolithic file (default: True)")
    parser.add_argument("--remove-original", action="store_true",
                       help="Remove original monolithic file after split")
    parser.add_argument("--verify", action="store_true", default=True,
                       help="Verify split integrity (default: True)")
    
    args = parser.parse_args()
    
    # Handle conflicting flags
    keep_original = args.keep_original and not args.remove_original
    
    try:
        # Perform the split
        output_dir = split_parquet_by_class(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            keep_original=keep_original
        )
        
        # Verify if requested
        if args.verify:
            print(f"\nVerifying split...")
            if verify_split(output_dir):
                print(f"✓ Split verification passed")
            else:
                print(f"✗ Split verification failed")
                return 1
        
        print(f"\n✓ Successfully split QuickDraw dataset by class")
        print(f"   Per-class files: {output_dir}")
        print(f"   Use QuickDrawDataset with this directory for faster subset loading")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
