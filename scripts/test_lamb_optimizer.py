#!/usr/bin/env python3
"""
Test script to validate LAMB optimizer with multi-GPU training.

This script runs a quick test to ensure LAMB works correctly and shows
the difference between optimizers for large batch training.
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models import build_model
from optimizers import LAMB, create_optimizer_for_large_batch, get_recommended_lr_for_batch_size
from train_config import TrainingConfig


def test_optimizers_comparison():
    """Test different optimizers with a simple forward/backward pass."""
    
    print("🔬 Testing Optimizers for Large Batch Training")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024 if torch.cuda.is_available() else 64
    num_classes = 50
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create model
    model = build_model('vit_tiny_patch16_224', num_classes=num_classes, pretrained=False)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
        effective_batch_size = batch_size * torch.cuda.device_count()
    else:
        effective_batch_size = batch_size
    
    model = model.to(device)
    
    print(f"Effective batch size: {effective_batch_size}")
    print()
    
    # Test different optimizers
    optimizers_to_test = [
        ("AdamW", "adamw"),
        ("LAMB", "lamb"),
        ("AdamW Large Batch", "adamw_large")
    ]
    
    base_lr = 0.0003
    results = {}
    
    for opt_name, opt_key in optimizers_to_test:
        print(f"Testing {opt_name}...")
        
        try:
            # Get recommended LR for this optimizer
            if opt_key == "lamb":
                scaled_lr = get_recommended_lr_for_batch_size(base_lr, 64, effective_batch_size, "sqrt")
            else:
                scaled_lr = get_recommended_lr_for_batch_size(base_lr, 64, effective_batch_size, "custom")
            
            # Create optimizer
            if opt_key in ["lamb", "adamw_large"]:
                optimizer = create_optimizer_for_large_batch(
                    model,
                    optimizer_name=opt_key,
                    lr=scaled_lr,
                    weight_decay=0.05
                )
            else:
                # Standard AdamW
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=scaled_lr,
                    weight_decay=0.05
                )
            
            # Time a few forward/backward passes
            model.train()
            start_time = time.time()
            
            for i in range(3):
                # Create dummy batch
                x = torch.randn(batch_size, 1, 224, 224).to(device)
                y = torch.randint(0, num_classes, (batch_size,)).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(x)
                loss = torch.nn.CrossEntropyLoss()(outputs, y)
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                if i == 0:
                    first_loss = loss.item()
                elif i == 2:
                    final_loss = loss.item()
            
            elapsed = time.time() - start_time
            
            results[opt_name] = {
                'scaled_lr': scaled_lr,
                'first_loss': first_loss,
                'final_loss': final_loss,
                'time': elapsed,
                'convergence': first_loss - final_loss,
                'optimizer_type': type(optimizer).__name__
            }
            
            print(f"  ✅ {opt_name}: LR={scaled_lr:.6f}, Loss={first_loss:.4f}→{final_loss:.4f}, Time={elapsed:.2f}s")
            
        except Exception as e:
            print(f"  ❌ {opt_name}: Failed - {e}")
            results[opt_name] = {'error': str(e)}
    
    print("\n📊 Results Summary:")
    print("-" * 80)
    print(f"{'Optimizer':<20} {'LR':<10} {'Loss Change':<12} {'Time (s)':<10} {'Type':<15}")
    print("-" * 80)
    
    for opt_name, result in results.items():
        if 'error' in result:
            print(f"{opt_name:<20} {'ERROR':<10} {result['error'][:20]:<12}")
        else:
            loss_change = f"{result['convergence']:.4f}"
            time_str = f"{result['time']:.2f}"
            lr_str = f"{result['scaled_lr']:.6f}"
            print(f"{opt_name:<20} {lr_str:<10} {loss_change:<12} {time_str:<10} {result['optimizer_type']:<15}")
    
    print("-" * 80)
    
    # Recommendations
    print("\n🎯 Recommendations:")
    if torch.cuda.device_count() > 1 and effective_batch_size >= 512:
        print("✅ Multi-GPU with large batch detected - LAMB optimizer recommended")
        print("✅ Use step-based LR scheduling for stability")
        print(f"✅ Suggested command for your setup:")
        print(f"   --optimizer lamb --auto-scale-lr --batch-size {batch_size}")
    else:
        print("ℹ️  Single GPU or small batch - AdamW should work fine")
        print("ℹ️  LAMB benefits are most apparent with large batch multi-GPU training")


def test_training_config_integration():
    """Test that TrainingConfig works with new optimizer options."""
    
    print("\n🔧 Testing TrainingConfig Integration")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {"optimizer_name": "adamw", "auto_scale_lr": False},
        {"optimizer_name": "lamb", "auto_scale_lr": True},
        {"optimizer_name": "adamw_large", "auto_scale_lr": True}
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model('vit_tiny_patch16_224', num_classes=10, pretrained=False)
    model = model.to(device)
    
    for config_params in configs:
        print(f"\nTesting config: {config_params}")
        
        try:
            config = TrainingConfig(
                learning_rate=0.0003,
                total_epochs=5,
                **config_params
            )
            
            # Test optimizer creation
            optimizer = config.create_optimizer(model, current_batch_size=1024)
            
            print(f"  ✅ Created {type(optimizer).__name__}")
            print(f"  ✅ LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


if __name__ == "__main__":
    test_optimizers_comparison()
    test_training_config_integration()
    
    print("\n🚀 Test completed! You can now use:")
    print("   python scripts/train_quickdraw.py --optimizer lamb --auto-scale-lr --batch-size 1024")
