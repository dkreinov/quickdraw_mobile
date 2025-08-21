#!/usr/bin/env python3
"""
Comprehensive model evaluation script for QuickDraw models.

Usage:
    python scripts/evaluate_model.py \
        --checkpoint results/vit_tiny_patch16_224_best.pt \
        --data-dir data/quickdraw_parquet \
        --classes 10 \
        --output-dir results/evaluation
        
This script will generate:
- baseline_eval.json (metrics summary)
- confusion_matrix.png
- most_confused.txt
- latency_desktop.json
- ckpt_size_mb.txt
- calib_quickdraw.pt (calibration data)
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from data import create_dataloaders
from models import build_model
from evaluation import ModelEvaluator, plot_confusion_matrix, save_most_confused_pairs
from logging_config import setup_logger, get_logger

def create_experiment_name(args, checkpoint_meta):
    """Create a descriptive experiment name based on model and training info."""
    # Extract info from checkpoint or args
    arch = args.arch.replace('_', '-')
    num_classes = args.classes
    
    # Try to determine if this is quantized or not
    if 'quantization' in checkpoint_meta:
        quant_type = checkpoint_meta['quantization'].get('method', 'unknown')
        bits = checkpoint_meta['quantization'].get('bits', 'unknown')
        exp_name = f"{quant_type}-{bits}bit-{arch}-{num_classes}c"
    else:
        # Assume FP32 baseline
        exp_name = f"fp32-baseline-{arch}-{num_classes}c"
    
    # Add timestamp for uniqueness if needed
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    return f"{exp_name}_{timestamp}"

def setup_experiment_directory(base_dir, experiment_name):
    """Setup organized experiment directory structure."""
    base_path = Path(base_dir)
    exp_path = base_path / experiment_name
    
    # Create subdirectories
    (exp_path / "training").mkdir(parents=True, exist_ok=True)
    (exp_path / "evaluation").mkdir(parents=True, exist_ok=True)
    
    # Create comparison directory at base level
    (base_path / "comparison").mkdir(parents=True, exist_ok=True)
    
    return exp_path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QuickDraw model")
    
    # Model and checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--arch", type=str, default="vit_tiny_patch16_224",
                       help="Model architecture")
    
    # Data parameters
    parser.add_argument("--data-dir", type=str, default="data/quickdraw_parquet",
                       help="Path to QuickDraw parquet data")
    parser.add_argument("--classes", type=int, default=10,
                       help="Number of classes to evaluate on")
    parser.add_argument("--per-class-val", type=int, default=100,
                       help="Validation samples per class")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    
    # Evaluation options
    parser.add_argument("--skip-latency", action="store_true",
                       help="Skip latency benchmarking")
    parser.add_argument("--latency-batch-sizes", type=int, nargs="+", default=[1, 8],
                       help="Batch sizes for latency benchmarking")
    parser.add_argument("--calibration-samples", type=int, default=2048,
                       help="Number of samples for calibration dataset")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="results/experiments",
                       help="Base directory for experiment results")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Name for this experiment (auto-generated if not provided)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    setup_logger("INFO")
    logger = get_logger(__name__)
    
    logger.info("=== QuickDraw Model Evaluation ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load data
    logger.info("Loading evaluation data...")
    try:
        train_loader, val_loader, data_meta = create_dataloaders(
            data_dir=args.data_dir,
            num_classes=args.classes,
            train_samples_per_class=1,  # We only need validation data
            val_samples_per_class=args.per_class_val,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        class_names = data_meta['selected_classes']
        num_classes = data_meta['num_classes']
        
        logger.info(f"Dataset loaded: {num_classes} classes, {len(val_loader)} val batches")
        logger.info(f"Classes: {class_names}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Build model
    logger.info(f"Building model: {args.arch}")
    try:
        model = build_model(
            arch=args.arch,
            num_classes=num_classes,
            pretrained=False  # We'll load from checkpoint
        )
        logger.info(f"Model built successfully")
        
    except Exception as e:
        logger.error(f"Failed to build model: {e}")
        return 1
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=class_names
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint_meta = evaluator.load_checkpoint(args.checkpoint)
        logger.info(f"Checkpoint loaded successfully")
        
        # Setup experiment directory structure
        if args.experiment_name:
            experiment_name = args.experiment_name
        else:
            experiment_name = create_experiment_name(args, checkpoint_meta)
        
        exp_dir = setup_experiment_directory(args.output_dir, experiment_name)
        eval_dir = exp_dir / "evaluation"
        training_dir = exp_dir / "training"
        
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Results will be saved to: {exp_dir}")
        
        # Save checkpoint size info
        checkpoint_path = Path(args.checkpoint)
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        size_info = {
            "checkpoint_file": str(checkpoint_path.name),
            "checkpoint_size_mb": round(checkpoint_size_mb, 2),
            "metadata": checkpoint_meta
        }
        
        with open(eval_dir / "ckpt_size_mb.txt", 'w') as f:
            f.write(f"Checkpoint: {checkpoint_path.name}\n")
            f.write(f"Size: {checkpoint_size_mb:.2f} MB\n")
            if checkpoint_meta:
                f.write(f"Metadata: {json.dumps(checkpoint_meta, indent=2)}\n")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 1
    
    # Run evaluation
    logger.info("Running comprehensive evaluation...")
    try:
        results = evaluator.evaluate_dataset(
            val_loader,
            compute_confusion=True,
            save_predictions=True
        )
        
        logger.info(f"Evaluation complete!")
        logger.info(f"Top-1 Accuracy: {results.top1_accuracy:.2f}%")
        logger.info(f"Top-5 Accuracy: {results.top5_accuracy:.2f}%")
        logger.info(f"Average Loss: {results.avg_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    # Latency benchmarking
    if not args.skip_latency:
        logger.info("Running latency benchmarks...")
        try:
            latency_results = evaluator.benchmark_latency(
                input_shape=(1, 1, args.image_size, args.image_size),
                batch_sizes=args.latency_batch_sizes,
                devices=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
            )
            
            # Add latency to results
            if 'cpu' in latency_results and 'batch_1' in latency_results['cpu']:
                results.latency_cpu_ms = latency_results['cpu']['batch_1']
            if 'cuda' in latency_results and 'batch_1' in latency_results['cuda']:
                results.latency_gpu_ms = latency_results['cuda']['batch_1']
            
            # Save detailed latency results
            with open(eval_dir / "latency_desktop.json", 'w') as f:
                json.dump(latency_results, f, indent=2)
            
            logger.info("Latency benchmarks complete")
            
        except Exception as e:
            logger.error(f"Latency benchmarking failed: {e}")
            # Continue without latency data
    
    # Create calibration dataset
    logger.info("Creating calibration dataset...")
    try:
        calib_path = eval_dir / "calib_quickdraw.pt"
        calibration_data = evaluator.create_calibration_dataset(
            val_loader,
            num_samples=args.calibration_samples,
            save_path=calib_path
        )
        
        logger.info(f"Calibration dataset created: {calibration_data.shape}")
        
    except Exception as e:
        logger.error(f"Failed to create calibration dataset: {e}")
        return 1
    
    # Save results
    logger.info("Saving evaluation results...")
    
    # Main results JSON
    results.save_json(eval_dir / "metrics.json")
    
    # Confusion matrix plot
    if results.confusion_matrix is not None:
        plot_confusion_matrix(
            results.confusion_matrix,
            class_names,
            eval_dir / "confusion_matrix.png",
            normalize=True
        )
    
    # Most confused pairs
    if results.most_confused_pairs:
        save_most_confused_pairs(
            results.most_confused_pairs,
            eval_dir / "most_confused.txt"
        )
    
    # Create summary table
    summary_table = {
        "model_info": {
            "architecture": args.arch,
            "num_classes": num_classes,
            "total_parameters": results.total_parameters,
            "trainable_parameters": results.trainable_parameters,
            "model_size_mb": round(results.model_size_mb, 2),
            "checkpoint_size_mb": round(checkpoint_size_mb, 2)
        },
        "accuracy_metrics": {
            "top1_accuracy": round(results.top1_accuracy, 2),
            "top5_accuracy": round(results.top5_accuracy, 2),
            "avg_loss": round(results.avg_loss, 4)
        },
        "latency_metrics": {}
    }
    
    if results.latency_cpu_ms:
        summary_table["latency_metrics"]["cpu_ms_per_image"] = {
            "mean": round(results.latency_cpu_ms['mean_ms'], 2),
            "p50": round(results.latency_cpu_ms['p50_ms'], 2),
            "p95": round(results.latency_cpu_ms['p95_ms'], 2)
        }
    
    if results.latency_gpu_ms:
        summary_table["latency_metrics"]["gpu_ms_per_image"] = {
            "mean": round(results.latency_gpu_ms['mean_ms'], 2),
            "p50": round(results.latency_gpu_ms['p50_ms'], 2),
            "p95": round(results.latency_gpu_ms['p95_ms'], 2)
        }
    
    with open(eval_dir / "summary_table.json", 'w') as f:
        json.dump(summary_table, f, indent=2)
    
    # Copy/link training artifacts to training directory
    try:
        import shutil
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            # Copy checkpoint to training directory
            shutil.copy2(checkpoint_path, training_dir / "model_best.pt")
            
            # Look for related training files in the same directory
            checkpoint_dir = checkpoint_path.parent
            checkpoint_base = checkpoint_path.stem.replace('_best', '').replace('_latest', '')
            
            # Copy history file if it exists
            history_file = checkpoint_dir / f"{checkpoint_base}_history.json"
            if history_file.exists():
                shutil.copy2(history_file, training_dir / "history.json")
                
            # Copy results file if it exists  
            results_file = checkpoint_dir / f"{checkpoint_base}_results.json"
            if results_file.exists():
                shutil.copy2(results_file, training_dir / "config.json")
                
    except Exception as e:
        logger.warning(f"Could not copy training artifacts: {e}")
    
    # Save experiment metadata
    experiment_meta = {
        "experiment_name": experiment_name,
        "timestamp": checkpoint_meta.get('timestamp', 'unknown'),
        "checkpoint_path": str(args.checkpoint),
        "evaluation_args": vars(args),
        "directory_structure": {
            "training": str(training_dir.resolve()),
            "evaluation": str(eval_dir.resolve())
        }
    }
    
    with open(exp_dir / "experiment_meta.json", 'w') as f:
        json.dump(experiment_meta, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Model: {args.arch}")
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Parameters: {results.total_parameters:,}")
    logger.info(f"Model Size: {results.model_size_mb:.2f} MB")
    logger.info(f"Checkpoint Size: {checkpoint_size_mb:.2f} MB")
    logger.info(f"Top-1 Accuracy: {results.top1_accuracy:.2f}%")
    logger.info(f"Top-5 Accuracy: {results.top5_accuracy:.2f}%")
    
    if results.latency_cpu_ms:
        logger.info(f"CPU Latency (mean): {results.latency_cpu_ms['mean_ms']:.2f} ms/image")
    if results.latency_gpu_ms:
        logger.info(f"GPU Latency (mean): {results.latency_gpu_ms['mean_ms']:.2f} ms/image")
    
    logger.info(f"\nExperiment: {experiment_name}")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info(f"Directory structure:")
    logger.info(f"  training/:")
    for file_path in sorted(training_dir.glob("*")):
        logger.info(f"    - {file_path.name}")
    logger.info(f"  evaluation/:")
    for file_path in sorted(eval_dir.glob("*")):
        logger.info(f"    - {file_path.name}")
    logger.info(f"  experiment_meta.json")
    
    logger.info("\nEvaluation completed successfully! 🎉")
    return 0

if __name__ == "__main__":
    exit(main())
