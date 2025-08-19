"""
Unit tests for training configuration module.
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import timm
from train_config import TrainingConfig, create_training_setup, test_training_config


def test_training_config_creation():
    """Test TrainingConfig creation and parameter validation."""
    print("=== Test 1: TrainingConfig Creation ===")
    
    try:
        # Test default config
        config = TrainingConfig()
        assert config.learning_rate == 3e-4, f"Expected 3e-4, got {config.learning_rate}"
        assert config.weight_decay == 0.05, f"Expected 0.05, got {config.weight_decay}"
        assert config.warmup_epochs == 2, f"Expected 2, got {config.warmup_epochs}"
        assert config.total_epochs == 20, f"Expected 20, got {config.total_epochs}"
        
        print("✅ Default config created successfully")
        
        # Test custom config
        custom_config = TrainingConfig(
            learning_rate=1e-3,
            weight_decay=0.1,
            warmup_epochs=5,
            total_epochs=50,
            label_smoothing=0.2,
            gradient_clip_norm=2.0,
            use_amp=False,
            seed=123
        )
        
        assert custom_config.learning_rate == 1e-3
        assert custom_config.weight_decay == 0.1
        assert custom_config.warmup_epochs == 5
        assert custom_config.total_epochs == 50
        assert custom_config.label_smoothing == 0.2
        assert custom_config.gradient_clip_norm == 2.0
        assert custom_config.use_amp == False
        assert custom_config.seed == 123
        
        print("✅ Custom config created successfully")
        
        # Test config dict
        config_dict = custom_config.get_config_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['learning_rate'] == 1e-3
        
        print("✅ Config dict generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation test failed: {e}")
        return False


def test_optimizer_creation():
    """Test optimizer creation with parameter grouping."""
    print("=== Test 2: Optimizer Creation ===")
    
    try:
        # Create a simple model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        
        config = TrainingConfig()
        optimizer = config.create_optimizer(model)
        
        # Check optimizer type
        assert isinstance(optimizer, torch.optim.AdamW), f"Expected AdamW, got {type(optimizer)}"
        
        # Check parameter groups
        assert len(optimizer.param_groups) == 2, f"Expected 2 param groups, got {len(optimizer.param_groups)}"
        
        # Check weight decay settings
        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]
        
        assert decay_group['weight_decay'] == config.weight_decay
        assert no_decay_group['weight_decay'] == 0.0
        
        print(f"✅ Optimizer created with {len(optimizer.param_groups)} parameter groups")
        print(f"   Decay group: {len(decay_group['params'])} params")
        print(f"   No-decay group: {len(no_decay_group['params'])} params")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer creation test failed: {e}")
        return False


def test_scheduler_creation():
    """Test learning rate scheduler creation."""
    print("=== Test 3: Scheduler Creation ===")
    
    try:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        
        config = TrainingConfig(warmup_epochs=2, total_epochs=10)
        optimizer = config.create_optimizer(model)
        
        steps_per_epoch = 100
        scheduler = config.create_scheduler(optimizer, steps_per_epoch)
        
        # Check scheduler type
        assert hasattr(scheduler, 'step'), "Scheduler should have step method"
        
        # Test initial learning rate (should be reduced due to warmup)
        initial_lr = optimizer.param_groups[0]['lr']
        # With warmup, initial LR should be lower than the configured LR
        assert initial_lr < config.learning_rate, f"With warmup, initial LR should be < {config.learning_rate}, got {initial_lr}"
        assert initial_lr > 0, f"Initial LR should be positive, got {initial_lr}"
        
        print(f"✅ Scheduler created successfully")
        print(f"   Initial LR: {initial_lr}")
        
        # Test warmup phase
        for step in range(50):  # Half of first epoch
            scheduler.step()
        
        warmup_lr = optimizer.param_groups[0]['lr']
        print(f"   Warmup LR (step 50): {warmup_lr:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scheduler creation test failed: {e}")
        return False


def test_loss_function():
    """Test loss function creation."""
    print("=== Test 4: Loss Function ===")
    
    try:
        config = TrainingConfig(label_smoothing=0.1)
        loss_fn = config.create_loss_function()
        
        # Check loss function type
        assert isinstance(loss_fn, nn.CrossEntropyLoss), f"Expected CrossEntropyLoss, got {type(loss_fn)}"
        
        # Test loss computation
        logits = torch.randn(4, 10)  # Batch size 4, 10 classes
        labels = torch.randint(0, 10, (4,))
        
        loss = loss_fn(logits, labels)
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.item() > 0, "Loss should be positive"
        
        print(f"✅ Loss function created and tested")
        print(f"   Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss function test failed: {e}")
        return False


def test_gradient_clipping():
    """Test gradient clipping functionality."""
    print("=== Test 5: Gradient Clipping ===")
    
    try:
        model = nn.Linear(10, 5)
        config = TrainingConfig(gradient_clip_norm=1.0)
        
        # Create some gradients
        dummy_input = torch.randn(2, 10)
        dummy_labels = torch.randint(0, 5, (2,))
        
        output = model(dummy_input)
        loss = nn.CrossEntropyLoss()(output, dummy_labels)
        loss.backward()
        
        # Test gradient clipping
        grad_norm = config.clip_gradients(model)
        
        assert isinstance(grad_norm, float), "Gradient norm should be a float"
        assert grad_norm >= 0, "Gradient norm should be non-negative"
        
        print(f"✅ Gradient clipping tested")
        print(f"   Gradient norm: {grad_norm:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient clipping test failed: {e}")
        return False


def test_complete_training_setup():
    """Test complete training setup creation."""
    print("=== Test 6: Complete Training Setup ===")
    
    try:
        # Create model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        
        # Create config
        config = TrainingConfig(
            learning_rate=1e-3,
            total_epochs=5,
            warmup_epochs=1,
            use_amp=False  # Disable AMP for testing
        )
        
        # Create complete setup
        optimizer, scheduler, loss_fn, scaler = create_training_setup(
            model=model,
            steps_per_epoch=50,
            config=config
        )
        
        # Verify all components
        assert optimizer is not None, "Optimizer should not be None"
        assert scheduler is not None, "Scheduler should not be None" 
        assert loss_fn is not None, "Loss function should not be None"
        # Scaler might be None if AMP is disabled
        
        print(f"✅ Complete training setup created")
        print(f"   Optimizer: {type(optimizer).__name__}")
        print(f"   Scheduler: {type(scheduler).__name__}")
        print(f"   Loss: {type(loss_fn).__name__}")
        print(f"   Scaler: {type(scaler).__name__ if scaler else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all training config tests."""
    print("🧪 Training Configuration Test Suite")
    print("=" * 50)
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(("Config Creation", test_training_config_creation()))
    test_results.append(("Optimizer Creation", test_optimizer_creation()))
    test_results.append(("Scheduler Creation", test_scheduler_creation()))
    test_results.append(("Loss Function", test_loss_function()))
    test_results.append(("Gradient Clipping", test_gradient_clipping()))
    test_results.append(("Complete Setup", test_complete_training_setup()))
    test_results.append(("Built-in Test", test_training_config()))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed_tests = [name for name, result in test_results if result]
    failed_tests = [name for name, result in test_results if not result]
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if failed_tests:
        print(f"\n❌ {len(failed_tests)} tests failed: {failed_tests}")
        return 1
    else:
        print(f"\n🎉 All {len(passed_tests)} tests passed!")
        print("\n🚀 Training configuration is ready!")
        return 0


if __name__ == "__main__":
    exit(main())
