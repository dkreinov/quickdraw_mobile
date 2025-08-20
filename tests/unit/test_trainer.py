"""
Unit tests for the training and validation loops.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.utils.data as data_utils
import timm
from trainer import QuickDrawTrainer, MetricsTracker, EpochMetrics, test_trainer
from train_config import TrainingConfig


def create_dummy_data(num_samples=50, num_classes=5):
    """Create dummy dataset for testing."""
    images = torch.randn(num_samples, 1, 224, 224)  # Single channel
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = data_utils.TensorDataset(images, labels)
    return dataset


def test_metrics_tracker():
    """Test MetricsTracker functionality."""
    print("=== Test 1: MetricsTracker ===")
    
    try:
        tracker = MetricsTracker()
        
        # Test initial state
        loss, top1, top5 = tracker.compute()
        assert loss == 0.0 and top1 == 0.0 and top5 == 0.0, "Initial metrics should be zero"
        
        # Test metric updates
        batch_size = 4
        num_classes = 10
        
        # Create dummy batch
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        loss_val = 1.5
        
        tracker.update(loss_val, logits, targets)
        
        # Compute metrics
        avg_loss, top1_acc, top5_acc = tracker.compute()
        
        assert avg_loss == loss_val, f"Expected loss {loss_val}, got {avg_loss}"
        assert 0 <= top1_acc <= 100, f"Top-1 accuracy should be in [0,100], got {top1_acc}"
        assert 0 <= top5_acc <= 100, f"Top-5 accuracy should be in [0,100], got {top5_acc}"
        assert top5_acc >= top1_acc, f"Top-5 should be >= Top-1, got {top5_acc} vs {top1_acc}"
        
        print("MetricsTracker test passed")
        return True
        
    except Exception as e:
        print(f"MetricsTracker test failed: {e}")
        return False


def test_epoch_metrics():
    """Test EpochMetrics dataclass."""
    print("=== Test 2: EpochMetrics ===")
    
    try:
        metrics = EpochMetrics(
            epoch=1,
            train_loss=1.5,
            train_top1=75.0,
            train_top5=95.0,
            val_loss=1.2,
            val_top1=80.0,
            val_top5=97.0,
            learning_rate=1e-3,
            epoch_time=120.5
        )
        
        # Test dictionary conversion
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict), "to_dict should return a dictionary"
        assert metrics_dict['epoch'] == 1, "Dictionary should contain correct epoch"
        assert metrics_dict['train_top1'] == 75.0, "Dictionary should contain correct train_top1"
        
        print("EpochMetrics test passed")
        return True
        
    except Exception as e:
        print(f"EpochMetrics test failed: {e}")
        return False


def test_trainer_creation():
    """Test QuickDrawTrainer creation."""
    print("=== Test 3: Trainer Creation ===")
    
    try:
        # Create dummy data
        dataset = create_dummy_data(num_samples=20, num_classes=5)
        train_loader = data_utils.DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = data_utils.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=5)
        
        # Create config
        config = TrainingConfig(
            total_epochs=2,
            warmup_epochs=0,
            use_amp=False,
            deterministic=False
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = QuickDrawTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                save_dir=temp_dir,
                model_name="test_model"
            )
            
            # Check trainer attributes
            assert trainer.model is model, "Trainer should store model reference"
            assert trainer.train_loader is train_loader, "Trainer should store train_loader"
            assert trainer.val_loader is val_loader, "Trainer should store val_loader"
            assert trainer.config is config, "Trainer should store config"
            assert trainer.current_epoch == 0, "Initial epoch should be 0"
            assert trainer.best_val_acc == 0.0, "Initial best accuracy should be 0"
            
            # Check training components
            assert trainer.optimizer is not None, "Optimizer should be created"
            assert trainer.scheduler is not None, "Scheduler should be created"
            assert trainer.loss_fn is not None, "Loss function should be created"
            
        print("Trainer creation test passed")
        return True
        
    except Exception as e:
        print(f"Trainer creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_epoch():
    """Test single training and validation epochs."""
    print("=== Test 4: Single Epoch ===")
    
    try:
        # Create dummy data
        dataset = create_dummy_data(num_samples=16, num_classes=3)
        train_loader = data_utils.DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = data_utils.DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Create model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)
        
        # Create config
        config = TrainingConfig(
            learning_rate=1e-3,
            total_epochs=1,
            warmup_epochs=0,
            use_amp=False,
            deterministic=False
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = QuickDrawTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                save_dir=temp_dir,
                model_name="test_model"
            )
            
            # Test training epoch
            train_loss, train_top1, train_top5 = trainer.train_epoch()
            
            assert isinstance(train_loss, float), "Train loss should be float"
            assert isinstance(train_top1, float), "Train top1 should be float"
            assert isinstance(train_top5, float), "Train top5 should be float"
            assert train_loss > 0, "Train loss should be positive"
            assert 0 <= train_top1 <= 100, "Train top1 should be in [0,100]"
            assert 0 <= train_top5 <= 100, "Train top5 should be in [0,100]"
            
            # Test validation epoch
            val_loss, val_top1, val_top5 = trainer.validate_epoch()
            
            assert isinstance(val_loss, float), "Val loss should be float"
            assert isinstance(val_top1, float), "Val top1 should be float"
            assert isinstance(val_top5, float), "Val top5 should be float"
            assert val_loss > 0, "Val loss should be positive"
            assert 0 <= val_top1 <= 100, "Val top1 should be in [0,100]"
            assert 0 <= val_top5 <= 100, "Val top5 should be in [0,100]"
            
        print(f"Single epoch test passed")
        print(f"  Train: Loss={train_loss:.4f}, Top1={train_top1:.2f}%, Top5={train_top5:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Top1={val_top1:.2f}%, Top5={val_top5:.2f}%")
        return True
        
    except Exception as e:
        print(f"Single epoch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_training():
    """Test complete training loop."""
    print("=== Test 5: Full Training Loop ===")
    
    try:
        # Create dummy data
        dataset = create_dummy_data(num_samples=32, num_classes=4)
        train_loader = data_utils.DataLoader(dataset, batch_size=8, shuffle=True)
        val_loader = data_utils.DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Create model
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=4)
        
        # Create config for short training
        config = TrainingConfig(
            learning_rate=1e-3,
            total_epochs=3,
            warmup_epochs=1,
            use_amp=False,
            deterministic=False
        )
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = QuickDrawTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                save_dir=temp_dir,
                model_name="test_full_training"
            )
            
            # Run training
            history = trainer.train()
            
            # Check training results
            assert len(history) == 3, f"Expected 3 epochs, got {len(history)}"
            assert trainer.current_epoch == 3, f"Expected current epoch 3, got {trainer.current_epoch}"
            assert trainer.best_val_acc > 0, f"Best validation accuracy should be > 0, got {trainer.best_val_acc}"
            
            # Check saved files
            save_path = Path(temp_dir)
            assert (save_path / "test_full_training_latest.pt").exists(), "Latest checkpoint should exist"
            assert (save_path / "test_full_training_best.pt").exists(), "Best checkpoint should exist"
            assert (save_path / "test_full_training_history.json").exists(), "History file should exist"
            
            # Check history file
            with open(save_path / "test_full_training_history.json", 'r') as f:
                history_data = json.load(f)
            
            assert history_data['total_epochs'] == 3, "History should record 3 epochs"
            assert len(history_data['metrics']) == 3, "History should contain 3 metric entries"
            assert 'best_val_acc' in history_data, "History should contain best validation accuracy"
            
        print(f"Full training test passed")
        print(f"  Epochs trained: {len(history)}")
        print(f"  Best val accuracy: {trainer.best_val_acc:.2f}%")
        return True
        
    except Exception as e:
        print(f"Full training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading():
    """Test checkpoint saving and loading."""
    print("=== Test 6: Checkpoint Loading ===")
    
    try:
        # Create dummy data and model
        dataset = create_dummy_data(num_samples=16, num_classes=3)
        train_loader = data_utils.DataLoader(dataset, batch_size=8)
        val_loader = data_utils.DataLoader(dataset, batch_size=8)
        
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=3)
        config = TrainingConfig(total_epochs=1, use_amp=False, deterministic=False)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = QuickDrawTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                save_dir=temp_dir,
                model_name="checkpoint_test"
            )
            
            # Train one epoch to create checkpoint
            trainer.train(num_epochs=1)
            
            # Load checkpoint
            checkpoint_path = Path(temp_dir) / "checkpoint_test_best.pt"
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify checkpoint contents
            required_keys = [
                'epoch', 'model_state_dict', 'optimizer_state_dict', 
                'scheduler_state_dict', 'best_val_acc', 'config', 'metrics'
            ]
            
            for key in required_keys:
                assert key in checkpoint, f"Checkpoint should contain {key}"
            
            assert checkpoint['epoch'] == 1, "Checkpoint should record epoch 1"
            assert checkpoint['model_name'] == "checkpoint_test", "Checkpoint should contain model name"
            
        print("Checkpoint loading test passed")
        return True
        
    except Exception as e:
        print(f"Checkpoint loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all trainer tests."""
    print("QuickDraw Trainer Test Suite")
    print("=" * 50)
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(("MetricsTracker", test_metrics_tracker()))
    test_results.append(("EpochMetrics", test_epoch_metrics()))
    test_results.append(("Trainer Creation", test_trainer_creation()))
    test_results.append(("Single Epoch", test_single_epoch()))
    test_results.append(("Full Training", test_full_training()))
    test_results.append(("Checkpoint Loading", test_checkpoint_loading()))
    test_results.append(("Built-in Test", test_trainer()))
    
    # Summary
    print("\nTest Results Summary:")
    print("=" * 50)
    
    passed_tests = [name for name, result in test_results if result]
    failed_tests = [name for name, result in test_results if not result]
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"   {status} {test_name}")
    
    if failed_tests:
        print(f"\n{len(failed_tests)} tests failed: {failed_tests}")
        return 1
    else:
        print(f"\nAll {len(passed_tests)} tests passed!")
        print("\nTrainer is ready for use!")
        return 0


if __name__ == "__main__":
    exit(main())
