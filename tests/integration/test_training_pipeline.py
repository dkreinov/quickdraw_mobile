#!/usr/bin/env python3
"""
Integration test for the complete training pipeline.

Tests the integration of:
- Data loading (per-class parquet)
- Model creation (ViT/MobileViT with single-channel adaptation)
- Training configuration
- Training loop execution
- Checkpointing and metrics

This is a full end-to-end test to ensure all components work together.
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import time

# Add src to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch

from data import create_dataloaders
from models import build_model
from train_config import TrainingConfig
from trainer import QuickDrawTrainer


class TestTrainingPipelineIntegration(unittest.TestCase):
    """Integration test for the complete training pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test parameters."""
        cls.data_dir = "data/quickdraw_parquet"
        cls.test_classes = 3
        cls.train_samples = 20  # Very small for fast testing
        cls.val_samples = 5
        cls.batch_size = 4
        cls.epochs = 1  # Just one epoch
        
        # Check if data exists
        data_path = Path(cls.data_dir)
        if not data_path.exists():
            raise unittest.SkipTest(f"Test data not found at {cls.data_dir}")
    
    def setUp(self):
        """Set up each test with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
    
    def test_data_model_integration(self):
        """Test that data loading and model creation work together."""
        print("\n=== Test 1: Data + Model Integration ===")
        
        # Load data
        start_time = time.time()
        train_loader, val_loader, meta = create_dataloaders(
            data_dir=self.data_dir,
            num_classes=self.test_classes,
            train_samples_per_class=self.train_samples,
            val_samples_per_class=self.val_samples,
            batch_size=self.batch_size,
            num_workers=0  # No multiprocessing for testing
        )
        data_time = time.time() - start_time
        
        self.assertGreater(len(train_loader), 0, "Train loader should have batches")
        self.assertGreater(len(val_loader), 0, "Val loader should have batches")
        
        print(f"  Data loaded in {data_time:.2f}s")
        print(f"  Classes: {meta['num_classes']}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Create model
        start_time = time.time()
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=meta['num_classes'],
            pretrained=False  # No pretrained for testing
        )
        model_time = time.time() - start_time
        
        print(f"  Model created in {model_time:.2f}s")
        print(f"  Model input size: {model.input_size}")
        
        # Test forward pass with data
        batch_images, batch_labels = next(iter(train_loader))
        
        self.assertEqual(batch_images.shape[1], 1, "Should be single channel")
        self.assertEqual(batch_images.shape[2], model.input_size, "Height should match model")
        self.assertEqual(batch_images.shape[3], model.input_size, "Width should match model")
        
        model.eval()
        with torch.no_grad():
            outputs = model(batch_images)
        
        self.assertEqual(outputs.shape[0], batch_images.shape[0], "Batch size should match")
        self.assertEqual(outputs.shape[1], meta['num_classes'], "Output classes should match")
        
        print(f"  Forward pass successful: {batch_images.shape} -> {outputs.shape}")
        
    def test_training_config_integration(self):
        """Test training configuration with model and data."""
        print("\n=== Test 2: Training Config Integration ===")
        
        # Create minimal components
        train_loader, val_loader, meta = create_dataloaders(
            data_dir=self.data_dir,
            num_classes=self.test_classes,
            train_samples_per_class=self.train_samples,
            val_samples_per_class=self.val_samples,
            batch_size=self.batch_size,
            num_workers=0
        )
        
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=meta['num_classes'],
            pretrained=False
        )
        
        # Create training config
        config = TrainingConfig(
            learning_rate=1e-3,
            total_epochs=1,
            warmup_epochs=0,
            use_amp=False,  # Disable AMP for testing
            deterministic=False  # Faster testing
        )
        
        # Test config components
        optimizer = config.create_optimizer(model)
        scheduler = config.create_scheduler(optimizer, len(train_loader))
        loss_fn = config.create_loss_function()
        scaler = config.create_scaler()
        
        self.assertIsNotNone(optimizer, "Optimizer should be created")
        self.assertIsNotNone(scheduler, "Scheduler should be created")
        self.assertIsNotNone(loss_fn, "Loss function should be created")
        # Scaler can be None if AMP is disabled
        
        print(f"  Optimizer: {type(optimizer).__name__}")
        print(f"  Scheduler: {type(scheduler).__name__}")
        print(f"  Loss function: {type(loss_fn).__name__}")
        print(f"  AMP Scaler: {type(scaler).__name__ if scaler else 'None'}")
        
        # Test one training step
        model.train()
        batch_images, batch_labels = next(iter(train_loader))
        
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        
        grad_norm = config.clip_gradients(model)
        optimizer.step()
        scheduler.step()
        
        self.assertGreater(loss.item(), 0, "Loss should be positive")
        self.assertGreater(grad_norm, 0, "Gradient norm should be positive")
        
        print(f"  Training step successful: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    def test_full_training_pipeline(self):
        """Test the complete training pipeline end-to-end."""
        print("\n=== Test 3: Full Training Pipeline ===")
        
        # Create data loaders
        train_loader, val_loader, meta = create_dataloaders(
            data_dir=self.data_dir,
            num_classes=self.test_classes,
            train_samples_per_class=self.train_samples,
            val_samples_per_class=self.val_samples,
            batch_size=self.batch_size,
            num_workers=0
        )
        
        # Create model
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=meta['num_classes'],
            pretrained=False
        )
        
        # Create config for very fast training
        config = TrainingConfig(
            learning_rate=1e-3,
            total_epochs=1,
            warmup_epochs=0,
            use_amp=False,
            deterministic=False
        )
        
        # Create trainer
        trainer = QuickDrawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_dir=self.temp_dir,
            model_name="test_integration"
        )
        
        # Run training
        start_time = time.time()
        history = trainer.train(num_epochs=1)
        training_time = time.time() - start_time
        
        # Verify training results
        self.assertEqual(len(history), 1, "Should have 1 epoch of history")
        self.assertGreater(trainer.best_val_acc, 0, "Should have some validation accuracy")
        
        # Check saved files
        save_path = Path(self.temp_dir)
        self.assertTrue((save_path / "test_integration_latest.pt").exists(), "Latest checkpoint should exist")
        self.assertTrue((save_path / "test_integration_best.pt").exists(), "Best checkpoint should exist")
        self.assertTrue((save_path / "test_integration_history.json").exists(), "History file should exist")
        
        print(f"  Training completed in {training_time:.2f}s")
        print(f"  Final train accuracy: {history[0].train_top1:.2f}%")
        print(f"  Final val accuracy: {history[0].val_top1:.2f}%")
        print(f"  Best val accuracy: {trainer.best_val_acc:.2f}%")
        print(f"  Checkpoints saved to: {self.temp_dir}")
        
    def test_different_architectures(self):
        """Test training with different model architectures."""
        print("\n=== Test 4: Different Architectures ===")
        
        architectures = ["vit_tiny_patch16_224"]  # Start with one that doesn't need download
        
        for arch in architectures:
            print(f"\n  Testing {arch}...")
            
            try:
                # Create minimal setup
                train_loader, val_loader, meta = create_dataloaders(
                    data_dir=self.data_dir,
                    num_classes=2,  # Even smaller for architecture testing
                    train_samples_per_class=10,
                    val_samples_per_class=5,
                    batch_size=4,
                    num_workers=0
                )
                
                model = build_model(
                    arch=arch,
                    num_classes=meta['num_classes'],
                    pretrained=False
                )
                
                # Test forward pass
                batch_images, batch_labels = next(iter(train_loader))
                model.eval()
                with torch.no_grad():
                    outputs = model(batch_images)
                
                self.assertEqual(outputs.shape[1], meta['num_classes'], f"Output shape mismatch for {arch}")
                print(f"    {arch}: Forward pass successful")
                
            except Exception as e:
                self.fail(f"Architecture {arch} failed: {e}")


def main():
    """Run integration tests."""
    print("QuickDraw Training Pipeline Integration Tests")
    print("=" * 60)
    
    # Import torch here to avoid import overhead in other tests
    import torch
    
    # Check if data is available
    data_dir = Path("data/quickdraw_parquet")
    if not data_dir.exists():
        print(f"Skipping integration tests: Data not found at {data_dir}")
        print("Please run: python scripts/download_quickdraw_direct.py")
        return 1
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\nIntegration tests completed!")
    return 0


if __name__ == "__main__":
    exit(main())
