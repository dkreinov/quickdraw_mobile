#!/usr/bin/env python3
"""
W&B Sweep for Finding Optimal Learning Rate for Multi-GPU Training

This script sets up a Weights & Biases sweep to find the optimal learning rate
for multi-GPU training with batch=1024 on the QuickDraw dataset.
"""

import wandb
import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from data import create_dataloaders
from models import build_model
from train_config import TrainingConfig
from trainer import QuickDrawTrainer
from logging_config import setup_logger, log_and_print

# Set W&B API key
# WANDB_API_KEY should be set as environment variable or in ~/.netrc

def train_with_wandb():
    """Training function for W&B sweep."""
    
    # Initialize W&B run
    wandb.init()
    
    # Get hyperparameters from W&B
    config = wandb.config
    
    # Fixed training parameters
    args = {
        'data_dir': '../data/quickdraw_parquet',  # Relative to experiments/ directory
        'classes': ['aircraft carrier', 'arm', 'asparagus', 'backpack', 'banana', 'basketball', 'bottlecap', 'bread', 'broom', 'bulldozer', 'butterfly', 'camel', 'canoe', 'chair', 'compass', 'cookie', 'drums', 'eyeglasses', 'face', 'fan', 'fence', 'fish', 'flying saucer', 'grapes', 'hand', 'hat', 'horse', 'light bulb', 'lighthouse', 'line', 'marker', 'mountain', 'mouse', 'parachute', 'passport', 'pliers', 'potato', 'sea turtle', 'snowflake', 'spider', 'square', 'steak', 'swing set', 'sword', 'television', 'tennis racquet', 'toothbrush', 'train', 'umbrella', 'washing machine'],
        'per_class_train': 1000,
        'per_class_val': 200,
        'image_size': 224,
        'arch': 'vit_tiny_patch16_224',
        'pretrained': False,
        'epochs': 10,  # Shorter for sweep
        'warmup_epochs': 2,
        'batch_size': 1024,  # Fixed multi-GPU batch size
        'weight_decay': 0.05,
        'label_smoothing': 0.1,
        'grad_clip': 1.0,
        'device': 'auto',
        'num_workers': 4,
        'save_dir': 'experiments/wandb_runs',
        'seed': 42,
        'no_amp': False
    }
    
    # Override learning rate from sweep
    args['lr'] = config.learning_rate
    
    print(f"=== W&B Sweep Run: LR={config.learning_rate:.6f} ===")
    
    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
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
            splits_dir='../data/splits'  # Relative to experiments/ directory
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
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        
        # Create training config
        training_config = TrainingConfig(
            learning_rate=args['lr'],
            weight_decay=args['weight_decay'],
            warmup_epochs=args['warmup_epochs'],
            total_epochs=args['epochs'],
            schedule_time_unit="step",  # Use step-based (proven better)
            label_smoothing=args['label_smoothing'],
            gradient_clip_norm=args['grad_clip'],
            use_amp=not args['no_amp'] and device.type == "cuda",
            seed=args['seed'],
            device=device
        )
        
        # Create trainer
        save_dir = Path(args['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = QuickDrawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            save_dir=str(save_dir),
            model_name=f"wandb_lr_{config.learning_rate:.6f}"
        )
        
        # Training loop with W&B logging
        print(f"Starting training for {args['epochs']} epochs...")
        
        for epoch in range(args['epochs']):
            trainer.current_epoch = epoch + 1
            
            # Training
            train_loss, train_top1, train_top5 = trainer.train_epoch()
            
            # Validation
            val_loss, val_top1, val_top5 = trainer.validate_epoch()
            
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
                "learning_rate": trainer.optimizer.param_groups[0]['lr']
            })
            
            # Update best tracking
            if val_top1 > trainer.best_val_acc:
                trainer.best_val_acc = val_top1
                trainer.best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1}/{args['epochs']}: "
                  f"Train={train_top1:.2f}%, Val={val_top1:.2f}%, "
                  f"Gap={train_top1-val_top1:.2f}%, LR={trainer.optimizer.param_groups[0]['lr']:.2e}")
        
        # Final metrics
        final_val_acc = trainer.best_val_acc
        print(f"Best validation accuracy: {final_val_acc:.2f}%")
        
        # Log summary metrics
        wandb.log({
            "final/best_val_accuracy": final_val_acc,
            "final/best_epoch": trainer.best_epoch,
            "final/learning_rate": config.learning_rate
        })
        
        # This is the key metric W&B will optimize
        wandb.log({"objective": final_val_acc})
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Log failure
        wandb.log({"objective": 0.0, "failed": True})
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    train_with_wandb()
