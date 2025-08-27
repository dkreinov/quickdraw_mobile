#!/usr/bin/env python3
"""
Quick test to find optimal batch size for 4 GPU setup.
Tests different batch sizes to find memory limits and speed.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from models import build_model

def test_batch_size(batch_size):
    """Test if a batch size fits in memory and measure speed."""
    print(f"\n=== Testing Batch Size: {batch_size} ===")
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Create model
        model = build_model('vit_tiny_patch16_224', num_classes=50, pretrained=False)
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        model = model.cuda()
        model.train()
        
        # Create dummy batch
        batch_per_gpu = batch_size // torch.cuda.device_count()
        x = torch.randn(batch_size, 1, 224, 224).cuda()
        target = torch.randint(0, 50, (batch_size,)).cuda()
        
        # Forward pass
        outputs = model(x)
        loss = torch.nn.CrossEntropyLoss()(outputs, target)
        
        # Backward pass  
        loss.backward()
        
        # Check memory usage
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        
        print(f"✅ SUCCESS")
        print(f"   Batch per GPU: {batch_per_gpu}")
        print(f"   Total memory: {memory_used:.2f} GB")
        print(f"   Memory per GPU: {memory_used/torch.cuda.device_count():.2f} GB")
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ OUT OF MEMORY")
        else:
            print(f"❌ ERROR: {e}")
        return False
    
    finally:
        torch.cuda.empty_cache()

def main():
    print("🔍 Finding Optimal Batch Size for 4 GPUs")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Test increasing batch sizes
    batch_sizes = [64, 128, 256, 512, 768, 1024, 1536, 2048]
    
    max_working_batch = 64
    
    for batch_size in batch_sizes:
        if test_batch_size(batch_size):
            max_working_batch = batch_size
        else:
            break
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"Maximum batch size: {max_working_batch}")
    print(f"Safe batch size: {max_working_batch * 0.8:.0f}")  # 80% of max for safety
    print(f"Conservative batch size: {max_working_batch * 0.6:.0f}")  # 60% of max
    
    # Learning rate recommendations
    base_lr = 0.0003
    scale_factor = max_working_batch / 64
    
    print(f"\n📚 LEARNING RATE SCALING:")
    print(f"Current LR (batch=64): {base_lr}")
    print(f"Linear scaling (batch={max_working_batch}): {base_lr * scale_factor:.6f}")
    print(f"Sqrt scaling (batch={max_working_batch}): {base_lr * (scale_factor**0.5):.6f}")

if __name__ == "__main__":
    main()
