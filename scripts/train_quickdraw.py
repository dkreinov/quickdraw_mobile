#!/usr/bin/env python3
"""
Step 3.3: Complete QuickDraw Training Script

This script demonstrates the full training pipeline:
- Load QuickDraw data with single-channel preprocessing 
- Create ViT-Tiny or MobileViT model adapted for 1-channel input
- Train with proper configuration (AdamW, warmup+cosine, AMP, etc.)
- Save checkpoints and training history
- Report final metrics

Example usage:
    python scripts/train_quickdraw.py --classes 10 --arch vit_tiny_patch16_224 --epochs 5
    python scripts/train_quickdraw.py --classes 20 --arch mobilevitv2_175 --epochs 10
"""

import sys
import argparse
from pathlib import Path
import json
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from data import create_dataloaders, get_all_class_names
from models import build_model
from train_config import TrainingConfig
from trainer import QuickDrawTrainer
from logging_config import setup_logger, log_and_print


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train QuickDraw vision models")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/quickdraw_parquet",
                       help="Path to QuickDraw Parquet data directory")
    parser.add_argument("--classes", type=int, default=10,
                       help="Number of classes to train on (random subset)")
    parser.add_argument("--per-class-train", type=int, default=2000,
                       help="Training samples per class")
    parser.add_argument("--per-class-val", type=int, default=250,
                       help="Validation samples per class")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Input image size")
    
    # Model arguments
    parser.add_argument("--arch", type=str, default="vit_tiny_patch16_224",
                       choices=["vit_tiny_patch16_224", "vit_small_patch16_224", 
                               "mobilevitv2_175", "mobilevitv2_200"],
                       help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--no-pretrained", action="store_true",
                       help="Don't use pretrained weights")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                       help="Number of warmup epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                       help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping norm")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save results")
    parser.add_argument("--no-amp", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        log_file=str(save_dir / "training.log"),
        log_level="INFO"
    )
    
    log_and_print("=== QuickDraw Vision Model Training ===", logger)
    log_and_print(f"Arguments: {vars(args)}", logger)
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    log_and_print(f"Using device: {device}", logger)
    
    # Check data availability
    data_dir = Path(args.data_dir)
    if not (data_dir / "quickdraw_data.parquet").exists():
        log_and_print(f"Data not found at {data_dir}", logger)
        log_and_print("Please run: python scripts/download_quickdraw_direct.py", logger)
        return 1
    
    try:
        # Create data loaders
        log_and_print(f"\nLoading QuickDraw data...", logger)
        train_loader, val_loader, data_meta = create_dataloaders(
            data_dir=str(data_dir),
            num_classes=args.classes,
            train_samples_per_class=args.per_class_train,
            val_samples_per_class=args.per_class_val,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed
        )
        
        log_and_print(f"Dataset created:", logger)
        log_and_print(f"  Classes: {data_meta['num_classes']}", logger)
        log_and_print(f"  Training batches: {len(train_loader)}", logger)
        log_and_print(f"  Validation batches: {len(val_loader)}", logger)
        log_and_print(f"  Image size: {data_meta['image_size']}", logger)
        label_names = data_meta['selected_classes']
        log_and_print(f"  Class names: {label_names[:5]}..." if len(label_names) > 5 else f"  Class names: {label_names}", logger)
        
        # Create model
        use_pretrained = args.pretrained and not args.no_pretrained
        log_and_print(f"\nCreating model: {args.arch} (pretrained: {use_pretrained})", logger)
        model = build_model(
            arch=args.arch,
            num_classes=data_meta['num_classes'],
            pretrained=use_pretrained
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        log_and_print(f"Model created:", logger)
        log_and_print(f"  Architecture: {args.arch}", logger)
        log_and_print(f"  Input size: {model.input_size}", logger)
        log_and_print(f"  Total parameters: {total_params:,}", logger)
        log_and_print(f"  Trainable parameters: {trainable_params:,}", logger)
        log_and_print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB", logger)
        
        # Move model to device
        model = model.to(device)
        log_and_print(f"  Model moved to device: {device}", logger)
        
        # Create training configuration
        config = TrainingConfig(
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            label_smoothing=args.label_smoothing,
            gradient_clip_norm=args.grad_clip,
            use_amp=not args.no_amp and device.type == "cuda",
            seed=args.seed,
            device=device
        )
        
        log_and_print(f"\nTraining configuration:", logger)
        log_and_print(f"  Learning rate: {config.learning_rate}", logger)
        log_and_print(f"  Weight decay: {config.weight_decay}", logger)
        log_and_print(f"  Epochs: {config.total_epochs} (warmup: {config.warmup_epochs})", logger)
        log_and_print(f"  Label smoothing: {config.label_smoothing}", logger)
        log_and_print(f"  Gradient clipping: {config.gradient_clip_norm}", logger)
        log_and_print(f"  Mixed precision: {config.use_amp}", logger)
        
        # Create trainer
        model_name = f"{args.arch}_{args.classes}classes"
        trainer = QuickDrawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_dir=str(save_dir),
            model_name=model_name
        )
        
        # Start training
        log_and_print(f"\nStarting training...", logger)
        start_time = time.time()
        
        training_history = trainer.train()
        
        # Training complete
        total_time = time.time() - start_time
        log_and_print(f"\nTraining completed!", logger)
        log_and_print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)", logger)
        log_and_print(f"Best validation accuracy: {trainer.best_val_acc:.2f}% (epoch {trainer.best_epoch})", logger)
        
        # Save final results summary
        results_summary = {
            "experiment": {
                "model_arch": args.arch,
                "num_classes": data_meta['num_classes'],
                "class_names": data_meta['selected_classes'],
                "image_size": args.image_size,
                "total_epochs": args.epochs,
                "batch_size": args.batch_size
            },
            "model": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": total_params * 4 / (1024**2)
            },
            "training": {
                "best_val_accuracy": trainer.best_val_acc,
                "best_epoch": trainer.best_epoch,
                "total_time_seconds": total_time,
                "training_config": config.get_config_dict()
            },
            "final_metrics": training_history[-1].to_dict() if training_history else None
        }
        
        results_file = save_dir / f"{model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        log_and_print(f"Results saved to: {results_file}", logger)
        
        # Print final summary
        log_and_print(f"\n" + "="*60, logger)
        log_and_print(f"TRAINING SUMMARY", logger)
        log_and_print(f"="*60, logger)
        log_and_print(f"Model: {args.arch} ({total_params:,} parameters)", logger)
        log_and_print(f"Dataset: {data_meta['num_classes']} classes, {args.image_size}x{args.image_size} images", logger)
        log_and_print(f"Best Validation Accuracy: {trainer.best_val_acc:.2f}%", logger)
        log_and_print(f"Training Time: {total_time/60:.1f} minutes", logger)
        log_and_print(f"Checkpoints: {save_dir}/{model_name}_best.pt", logger)
        log_and_print(f"="*60, logger)
        
        return 0
        
    except Exception as e:
        log_and_print(f"Training failed: {e}", logger)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
