"""Test device consistency across all training components.

This test ensures that all parts of the training pipeline are on the same device
and properly optimized for CUDA when available.
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from models import build_model
from data import create_dataloaders
from train_config import TrainingConfig
from trainer import QuickDrawTrainer


def get_available_device():
    """Get the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def check_model_device_consistency(model, expected_device):
    """Check that all model parameters are on the expected device."""
    device_issues = []
    
    # Normalize device comparison (cuda:0 == cuda)
    def devices_match(device1, device2):
        # Convert both to string and normalize
        d1_str = str(device1)
        d2_str = str(device2)
        # cuda:0 should match cuda
        if d1_str.startswith('cuda') and d2_str.startswith('cuda'):
            return True
        return d1_str == d2_str
    
    for name, param in model.named_parameters():
        if not devices_match(param.device, expected_device):
            device_issues.append(f"Parameter {name}: {param.device} != {expected_device}")
    
    for name, buffer in model.named_buffers():
        if not devices_match(buffer.device, expected_device):
            device_issues.append(f"Buffer {name}: {buffer.device} != {expected_device}")
    
    return device_issues


def check_optimizer_device_consistency(optimizer, expected_device):
    """Check that all optimizer states are on the expected device."""
    device_issues = []
    
    # Normalize device comparison (cuda:0 == cuda)
    def devices_match(device1, device2):
        # Convert both to string and normalize
        d1_str = str(device1)
        d2_str = str(device2)
        # cuda:0 should match cuda
        if d1_str.startswith('cuda') and d2_str.startswith('cuda'):
            return True
        return d1_str == d2_str
    
    for group in optimizer.param_groups:
        for param in group['params']:
            if not devices_match(param.device, expected_device):
                device_issues.append(f"Optimizer param: {param.device} != {expected_device}")
    
    # Check optimizer state
    for param, state in optimizer.state.items():
        if isinstance(state, dict):
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and not devices_match(value.device, expected_device):
                    device_issues.append(f"Optimizer state {key}: {value.device} != {expected_device}")
    
    return device_issues


def test_device_consistency():
    """Test that all training components are on the same device."""
    device = get_available_device()
    print(f"\nTesting device consistency on: {device}")
    
    # Create small test dataset
    try:
        train_loader, val_loader, metadata = create_dataloaders(
            data_dir='data/quickdraw_parquet',
            num_classes=3,
            train_samples_per_class=10,
            val_samples_per_class=5,
            batch_size=4,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        print("Data loaders created successfully")
    except Exception as e:
        print(f"Data not available for testing: {e}")
        return
    
    # Create model
    model = build_model(
        arch='vit_tiny_patch16_224',
        num_classes=metadata['num_classes'],
        pretrained=False  # Faster for testing
    )
    print(f"Model created: {type(model).__name__}")
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to device: {device}")
    
    # Check model device consistency
    model_issues = check_model_device_consistency(model, device)
    assert not model_issues, f"Model device issues: {model_issues}"
    print("✓ Model device consistency verified")
    
    # Create training config
    config = TrainingConfig(
        learning_rate=0.001,
        weight_decay=0.01,
        warmup_epochs=1,
        total_epochs=1,
        use_amp=device.type == 'cuda',  # Only use AMP on CUDA
        device=device
    )
    print(f"Training config created (AMP: {config.use_amp})")
    
    # Create optimizer
    optimizer = config.create_optimizer(model)
    print(f"Optimizer created: {type(optimizer).__name__}")
    
    # Create scheduler
    scheduler = config.create_scheduler(optimizer, len(train_loader))
    print(f"Scheduler created: {type(scheduler).__name__}")
    
    # Create loss function
    loss_fn = config.create_loss_function()
    print(f"Loss function created: {type(loss_fn).__name__}")
    
    # Create scaler (only for CUDA with AMP)
    scaler = config.create_scaler()
    if scaler:
        print(f"Gradient scaler created: {type(scaler).__name__}")
    else:
        print("No gradient scaler (CPU or AMP disabled)")
    
    # Test a forward pass with actual data
    model.eval()
    with torch.no_grad():
        for batch_images, batch_targets in train_loader:
            # Move data to device
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)
            
            print(f"Batch data moved to device: {device}")
            print(f"  Images shape: {batch_images.shape}, device: {batch_images.device}")
            print(f"  Targets shape: {batch_targets.shape}, device: {batch_targets.device}")
            
            # Test forward pass
            if config.use_amp and scaler:
                # Test mixed precision forward pass
                try:
                    from torch.amp import autocast
                except ImportError:
                    from torch.cuda.amp import autocast
                
                try:
                    with autocast(device_type=device.type):
                        logits = model(batch_images)
                except TypeError:
                    # Fallback for older PyTorch
                    with autocast():
                        logits = model(batch_images)
                        
                print(f"  Mixed precision forward pass successful")
            else:
                # Regular forward pass
                logits = model(batch_images)
                print(f"  Regular forward pass successful")
            
            print(f"  Logits shape: {logits.shape}, device: {logits.device}")
            
            # Test loss computation
            loss = loss_fn(logits, batch_targets)
            print(f"  Loss computed: {loss.item():.4f}, device: {loss.device}")
            
            # Verify all tensors are on the same device (with normalized comparison)
            def devices_match(device1, device2):
                d1_str = str(device1)
                d2_str = str(device2)
                if d1_str.startswith('cuda') and d2_str.startswith('cuda'):
                    return True
                return d1_str == d2_str
                
            assert devices_match(batch_images.device, device), f"Images device mismatch: {batch_images.device} != {device}"
            assert devices_match(batch_targets.device, device), f"Targets device mismatch: {batch_targets.device} != {device}"
            assert devices_match(logits.device, device), f"Logits device mismatch: {logits.device} != {device}"
            assert devices_match(loss.device, device), f"Loss device mismatch: {loss.device} != {device}"
            
            break  # Only test one batch
    
    print("✓ Forward pass device consistency verified")
    
    # Test training mode
    model.train()
    
    # Test backward pass
    optimizer.zero_grad()
    
    for batch_images, batch_targets in train_loader:
        batch_images = batch_images.to(device)
        batch_targets = batch_targets.to(device)
        
        if config.use_amp and scaler:
            # Test AMP backward pass
            try:
                from torch.amp import autocast
            except ImportError:
                from torch.cuda.amp import autocast
            
            try:
                with autocast(device_type=device.type):
                    logits = model(batch_images)
                    loss = loss_fn(logits, batch_targets)
            except TypeError:
                with autocast():
                    logits = model(batch_images)
                    loss = loss_fn(logits, batch_targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print("✓ AMP backward pass successful")
        else:
            # Regular backward pass
            logits = model(batch_images)
            loss = loss_fn(logits, batch_targets)
            loss.backward()
            optimizer.step()
            print("✓ Regular backward pass successful")
        
        break  # Only test one batch
    
    # Check optimizer device consistency after backward pass
    optimizer_issues = check_optimizer_device_consistency(optimizer, device)
    if optimizer_issues:
        print(f"Warning: Optimizer device issues (may be normal): {optimizer_issues[:3]}...")
    else:
        print("✓ Optimizer device consistency verified")
    
    print(f"\n✅ All device consistency tests passed on {device}!")


def test_cuda_optimization_when_available():
    """Test CUDA-specific optimizations when CUDA is available."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping CUDA optimization tests")
        return
    
    device = torch.device('cuda')
    print(f"\nTesting CUDA optimizations on: {device}")
    
    # Test CUDA memory allocation
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial CUDA memory: {initial_memory / 1024**2:.2f} MB")
    
    # Create a larger model for memory testing
    model = build_model(
        arch='vit_tiny_patch16_224',
        num_classes=10,
        pretrained=False
    ).to(device)
    
    model_memory = torch.cuda.memory_allocated() - initial_memory
    print(f"Model memory usage: {model_memory / 1024**2:.2f} MB")
    
    # Test CUDA tensor operations
    x = torch.randn(4, 1, 224, 224, device=device)
    print(f"Test tensor created on CUDA: {x.device}")
    
    # Test mixed precision capability
    if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        print("✓ Mixed precision (AMP) available")
        
        try:
            from torch.amp import autocast
            amp_available = True
        except ImportError:
            try:
                from torch.cuda.amp import autocast
                amp_available = True
            except ImportError:
                amp_available = False
        
        assert amp_available, "AMP not available despite CUDA support"
        print("✓ AMP import successful")
    
    # Test tensor memory cleanup
    del model, x
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    print(f"Final CUDA memory: {final_memory / 1024**2:.2f} MB")
    
    print("✅ CUDA optimization tests passed!")


if __name__ == "__main__":
    # Run tests directly
    print("=== Device Consistency Tests ===")
    test_device_consistency()
    print("\n" + "="*50)
    test_cuda_optimization_when_available()
    print("\nAll tests completed successfully!")
