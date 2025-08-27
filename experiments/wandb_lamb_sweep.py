#!/usr/bin/env python3
"""
W&B Hyperparameter Sweep for LAMB Optimizer.

This sweep optimizes LAMB optimizer hyperparameters after the breakthrough
discovery that LAMB significantly outperforms AdamW for multi-GPU training.

Target: Optimize beyond the 86.82% validation accuracy baseline.
"""

import wandb
import sys
import torch
import os
from pathlib import Path

# Add the src directory to path so imports work
current_dir = Path.cwd()
if current_dir.name == 'experiments':
    project_root = current_dir.parent
else:
    project_root = current_dir

src_path = str(project_root / "src")
sys.path.insert(0, src_path)
print(f"🔧 LAMB Setup: Added {src_path} to Python path")
print(f"✅ LAMB Available: {(project_root / 'src' / 'optimizers.py').exists()}")

# Import after path setup is done in the function


def train_with_wandb():
    """Training function for LAMB hyperparameter sweep."""
    
    # Import modules after path is set up
    from data import create_dataloaders
    from models import build_model
    from train_config import TrainingConfig
    from trainer import QuickDrawTrainer
    
    # Initialize W&B run
    wandb.init()
    
    # Get hyperparameters from W&B
    config = wandb.config
    
    # Fixed training parameters (optimized from previous experiments)
    args = {
        'data_dir': '../data/quickdraw_parquet',
        'classes': ['aircraft carrier', 'arm', 'asparagus', 'backpack', 'banana', 'basketball', 'bottlecap', 'bread', 'broom', 'bulldozer', 'butterfly', 'camel', 'canoe', 'chair', 'compass', 'cookie', 'drums', 'eyeglasses', 'face', 'fan', 'fence', 'fish', 'flying saucer', 'grapes', 'hand', 'hat', 'horse', 'light bulb', 'lighthouse', 'line', 'marker', 'mountain', 'mouse', 'parachute', 'passport', 'pliers', 'potato', 'sea turtle', 'snowflake', 'spider', 'square', 'steak', 'swing set', 'sword', 'television', 'tennis racquet', 'toothbrush', 'train', 'umbrella', 'washing machine'],
        'per_class_train': 1000,
        'per_class_val': 200,
        'image_size': 224,
        'arch': 'vit_tiny_patch16_224',
        'pretrained': True,  # CRITICAL: Use pretrained weights like successful local run
        'epochs': 30,  # Full training for best results
        'batch_size': 1024,  # Multi-GPU large batch (proven optimal)
        'label_smoothing': 0.1,  # Keep proven values
        'grad_clip': 1.0,
        'device': 'auto',
        'num_workers': 4,
        'save_dir': 'experiments/wandb_lamb_runs',
        'seed': 42,
        'no_amp': False,
        'optimizer': 'lamb',  # Fixed: use LAMB optimizer
        'auto_scale_lr': False,  # Match your successful local command
        'base_batch_size': 64,  # Not used when auto_scale_lr=False
        'schedule_time_unit': 'step'  # Proven better than epoch-based
    }
    
    # Override hyperparameters from sweep
    args['lr'] = config.learning_rate
    args['weight_decay'] = config.weight_decay
    args['warmup_epochs'] = config.warmup_epochs
    
    # Override label smoothing if specified in sweep (batch size kept fixed)
    if hasattr(config, 'label_smoothing'):
        args['label_smoothing'] = config.label_smoothing
    # Batch size remains fixed at 1024 to reproduce baseline
    
    print(f"=== LAMB Sweep Run ===")
    print(f"LR: {config.learning_rate:.6f}")
    print(f"Weight Decay: {config.weight_decay:.4f}")
    print(f"Warmup Epochs: {config.warmup_epochs}")
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Create data loaders
        train_loader, val_loader, data_meta = create_dataloaders(
            data_dir=args['data_dir'],
            classes=args['classes'],
            train_samples_per_class=args['per_class_train'],
            val_samples_per_class=args['per_class_val'],
            image_size=args['image_size'],
            batch_size=args['batch_size'],
            num_workers=args['num_workers'],
            seed=args['seed'],
            splits_dir='../data/splits'
        )
        
        print(f"Dataset: {data_meta['num_classes']} classes, {len(train_loader)} train batches")
        
        # Create model
        model = build_model(
            arch=args['arch'],
            num_classes=data_meta['num_classes'],
            pretrained=args['pretrained']
        )
        
        # Move to device and enable multi-GPU
        model = model.to(device)
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            print(f"Using {torch.cuda.device_count()} GPUs with LAMB optimizer")
            model = torch.nn.DataParallel(model)
        
        # Debug optimizer availability
        print(f"🔍 Debug Info:")
        print(f"   Optimizer requested: {args['optimizer']}")
        try:
            from optimizers import create_optimizer_for_large_batch
            print(f"   ✅ LAMB import successful in wandb script")
        except ImportError as e:
            print(f"   ❌ LAMB import failed in wandb script: {e}")
        
        # Create training config with LAMB
        training_config = TrainingConfig(
            learning_rate=args['lr'],
            weight_decay=args['weight_decay'],
            warmup_epochs=args['warmup_epochs'],
            total_epochs=args['epochs'],
            optimizer_name=args['optimizer'],  # LAMB
            auto_scale_lr=args['auto_scale_lr'],  # CRITICAL: Enable LR scaling
            base_batch_size=args['base_batch_size'],  # Reference batch size
            schedule_time_unit=args['schedule_time_unit'],
            label_smoothing=args['label_smoothing'],
            gradient_clip_norm=args['grad_clip'],
            use_amp=not args['no_amp'] and device.type == "cuda",
            seed=args['seed'],
            device=device
        )
        
        # Debug training config optimizer settings
        print(f"   TrainingConfig optimizer_name: {training_config.optimizer_name}")
        
        # Check if CUSTOM_OPTIMIZERS_AVAILABLE in train_config module
        import train_config
        print(f"   CUSTOM_OPTIMIZERS_AVAILABLE in TrainingConfig: {train_config.CUSTOM_OPTIMIZERS_AVAILABLE}")
        
        # Create trainer
        save_dir = Path(args['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = QuickDrawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            save_dir=str(save_dir),
            model_name=f"lamb_lr_{config.learning_rate:.6f}_wd_{config.weight_decay:.4f}"
        )
        
        # Training loop with W&B logging and smart early stopping
        print(f"Starting LAMB training for {args['epochs']} epochs...")
        
        best_val_acc = 0.0
        best_epoch = 0
        
        # More lenient early stopping - let experiments run longer
        # Only stop truly terrible runs (>10% below your baseline)
        min_acceptable_performance = 75.0  # Very lenient threshold
        patience_epochs = 5  # Reasonable patience - 5 consecutive bad epochs
        consecutive_bad_epochs = 0
        
        for epoch in range(args['epochs']):
            trainer.current_epoch = epoch + 1
            
            # Training
            train_loss, train_top1, train_top5 = trainer.train_epoch()
            
            # Validation
            val_loss, val_top1, val_top5 = trainer.validate_epoch()
            
            # Track best
            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                best_epoch = epoch + 1
            
            # Log to W&B
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/top1_accuracy": train_top1,
                "train/top5_accuracy": train_top5,
                "val/loss": val_loss,
                "val/top1_accuracy": val_top1,
                "val/top5_accuracy": val_top5,
                "val/overfitting_gap": train_top1 - val_top1,
                "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                "best_val_accuracy": best_val_acc
            })
            
            # Very lenient early stopping - only stop truly terrible runs
            current_epoch = epoch + 1
            
            if current_epoch >= 15:  # Give substantial time to develop
                if val_top1 < min_acceptable_performance:
                    consecutive_bad_epochs += 1
                    print(f"⚠️  Very low performance: {val_top1:.2f}% < {min_acceptable_performance}% (Count: {consecutive_bad_epochs}/{patience_epochs})")
                else:
                    consecutive_bad_epochs = 0  # Reset if improving
                
                if consecutive_bad_epochs >= patience_epochs:
                    print(f"🛑 Early stopping: {patience_epochs} consecutive epochs below {min_acceptable_performance}%")
                    print(f"   Current performance: {val_top1:.2f}%")
                    break
            
            print(f"Epoch {epoch+1}/{args['epochs']}: "
                  f"Train={train_top1:.2f}%, Val={val_top1:.2f}%, "
                  f"Best={best_val_acc:.2f}% (E{best_epoch}), "
                  f"Gap={train_top1-val_top1:.2f}%, LR={trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final metrics
        print(f"Final best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        
        # Log summary metrics
        wandb.log({
            "final/best_val_accuracy": best_val_acc,
            "final/best_epoch": best_epoch,
            "final/learning_rate": config.learning_rate,
            "final/weight_decay": config.weight_decay,
            "final/warmup_epochs": config.warmup_epochs
        })
        
        # This is the key metric W&B will optimize
        wandb.log({"objective": best_val_acc})
        
        return best_val_acc
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Log failed run
        wandb.log({"objective": 0.0, "error": str(e)})
        return 0.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="W&B LAMB Sweep")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, help="Warmup epochs")
    parser.add_argument("--label_smoothing", type=float, help="Label smoothing")
    
    # Check if this is being called by wandb agent (has wandb config parameters)
    try:
        args = parser.parse_args()
        # If we have command line args, wandb agent is calling us
        if any(getattr(args, arg) is not None for arg in ['learning_rate', 'weight_decay', 'warmup_epochs', 'label_smoothing']):
            train_with_wandb()
        else:
            print("This script should be run by wandb agent, not directly.")
            print("Use: wandb agent <SWEEP_ID>")
    except SystemExit:
        # argparse called --help or had an error
        pass
