"""Step 3.3: Training and Validation Loops with Metrics.

This module provides a comprehensive trainer for QuickDraw vision models with:
- Training and validation loops with progress tracking
- Top-1 and Top-5 accuracy metrics
- Loss tracking and logging
- AMP support for efficient training
- Early stopping and best model tracking
- Training history logging to JSON
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

# Try to import logging config, fallback to basic logging if not available
try:
    from .logging_config import get_logger, log_and_print
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"): print(msg)

# Try relative import first, fallback to absolute import
try:
    from .train_config import TrainingConfig
except ImportError:
    from train_config import TrainingConfig


@dataclass 
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_top1: float
    train_top5: float
    val_loss: float
    val_top1: float
    val_top5: float
    learning_rate: float
    epoch_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsTracker:
    """Tracks and computes training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for new epoch."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.correct_top1 = 0
        self.correct_top5 = 0
    
    def update(self, loss: float, logits: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        batch_size = targets.size(0)
        
        # Loss
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Top-k accuracy
        with torch.no_grad():
            _, predicted = logits.topk(5, dim=1, largest=True, sorted=True)
            targets_expanded = targets.view(-1, 1).expand_as(predicted)
            
            # Top-1 accuracy
            self.correct_top1 += predicted[:, 0].eq(targets).sum().item()
            
            # Top-5 accuracy  
            self.correct_top5 += predicted.eq(targets_expanded).sum().item()
    
    def compute(self) -> Tuple[float, float, float]:
        """Compute average metrics."""
        if self.total_samples == 0:
            return 0.0, 0.0, 0.0
            
        avg_loss = self.total_loss / self.total_samples
        top1_acc = 100.0 * self.correct_top1 / self.total_samples
        top5_acc = 100.0 * self.correct_top5 / self.total_samples
        
        return avg_loss, top1_acc, top5_acc


class QuickDrawTrainer:
    """
    Comprehensive trainer for QuickDraw vision models.
    
    Features:
    - Training and validation loops with metrics
    - Mixed precision support
    - Learning rate scheduling  
    - Best model tracking
    - Training history logging
    - Progress reporting
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_dir: str = "results",
        model_name: str = "quickdraw_model"
    ):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup training components
        self.optimizer = self.config.create_optimizer(self.model)
        self.scheduler = self.config.create_scheduler(self.optimizer, len(self.train_loader))
        self.loss_fn = self.config.create_loss_function()
        self.scaler = self.config.create_scaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.training_history: List[EpochMetrics] = []
        
        # Logging
        self.logger = get_logger(__name__)
        
        log_and_print(f"Trainer initialized:", self.logger)
        log_and_print(f"  Model: {model_name}", self.logger)
        log_and_print(f"  Train batches: {len(train_loader)}", self.logger)
        log_and_print(f"  Val batches: {len(val_loader)}", self.logger)
        log_and_print(f"  Save directory: {save_dir}", self.logger)
    
    def train_epoch(self) -> Tuple[float, float, float]:
        """
        Run one training epoch.
        
        Returns:
            Tuple of (avg_loss, top1_acc, top5_acc)
        """
        
        self.model.train()
        metrics = MetricsTracker()
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with autocast(self.config.device.type):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                grad_norm = self.config.clip_gradients(self.model)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision
                logits = self.model(images)
                loss = self.loss_fn(logits, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                grad_norm = self.config.clip_gradients(self.model)
                
                # Optimizer step
                self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            metrics.update(loss.item(), logits, targets)
            
            # Log progress periodically
            if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_and_print(
                    f"  Batch {batch_idx:4d}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"GradNorm: {grad_norm:.3f}",
                    self.logger
                )
        
        return metrics.compute()
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """
        Run validation epoch.
        
        Returns:
            Tuple of (avg_loss, top1_acc, top5_acc)
        """
        
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                # Move to device
                images = images.to(self.config.device)
                targets = targets.to(self.config.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast(self.config.device.type):
                        logits = self.model(images)
                        loss = self.loss_fn(logits, targets)
                else:
                    logits = self.model(images)
                    loss = self.loss_fn(logits, targets)
                
                # Update metrics
                metrics.update(loss.item(), logits, targets)
        
        return metrics.compute()
    
    def save_checkpoint(self, metrics: EpochMetrics, is_best: bool = False):
        """Save model checkpoint with training state."""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'config': self.config.get_config_dict(),
            'metrics': metrics.to_dict(),
            'model_name': self.model_name
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            log_and_print(f"  Saved best model: {best_path}", self.logger)
    
    def save_training_history(self):
        """Save training history to JSON file."""
        
        history_data = {
            'model_name': self.model_name,
            'config': self.config.get_config_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.training_history),
            'metrics': [m.to_dict() for m in self.training_history]
        }
        
        history_path = self.save_dir / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        log_and_print(f"Training history saved: {history_path}", self.logger)
    
    def train(self, num_epochs: Optional[int] = None) -> List[EpochMetrics]:
        """
        Run complete training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config.total_epochs if None)
            
        Returns:
            List of epoch metrics
        """
        
        if num_epochs is None:
            num_epochs = self.config.total_epochs
        
        log_and_print(f"Starting training for {num_epochs} epochs", self.logger)
        log_and_print(f"Device: {self.config.device}", self.logger)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            log_and_print(f"\nEpoch {self.current_epoch}/{num_epochs}", self.logger)
            log_and_print("-" * 50, self.logger)
            
            # Training phase
            log_and_print("Training...", self.logger)
            train_loss, train_top1, train_top5 = self.train_epoch()
            
            # Validation phase
            log_and_print("Validating...", self.logger)
            val_loss, val_top1, val_top5 = self.validate_epoch()
            
            # Compute epoch time
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create epoch metrics
            metrics = EpochMetrics(
                epoch=self.current_epoch,
                train_loss=train_loss,
                train_top1=train_top1,
                train_top5=train_top5,
                val_loss=val_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                learning_rate=current_lr,
                epoch_time=epoch_time
            )
            
            self.training_history.append(metrics)
            
            # Check if best model
            is_best = val_top1 > self.best_val_acc
            if is_best:
                self.best_val_acc = val_top1
                self.best_epoch = self.current_epoch
            
            # Save checkpoint
            self.save_checkpoint(metrics, is_best)
            
            # Log epoch summary
            log_and_print(f"\nEpoch {self.current_epoch} Summary:", self.logger)
            log_and_print(f"  Train: Loss={train_loss:.4f}, Top1={train_top1:.2f}%, Top5={train_top5:.2f}%", self.logger)
            log_and_print(f"  Val:   Loss={val_loss:.4f}, Top1={val_top1:.2f}%, Top5={val_top5:.2f}%", self.logger)
            log_and_print(f"  LR: {current_lr:.2e}, Time: {epoch_time:.1f}s", self.logger)
            if is_best:
                log_and_print(f"  New best validation accuracy!", self.logger)
        
        # Training complete
        total_time = time.time() - start_time
        log_and_print(f"\nTraining complete!", self.logger)
        log_and_print(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)", self.logger)
        log_and_print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})", self.logger)
        
        # Save training history
        self.save_training_history()
        
        return self.training_history


def test_trainer():
    """Test function for the trainer."""
    
    print("=== Testing QuickDraw Trainer ===")
    
    try:
        # Create dummy data
        import torch.utils.data as data_utils
        
        # Dummy dataset
        dummy_images = torch.randn(100, 1, 224, 224)  # Single channel
        dummy_labels = torch.randint(0, 10, (100,))
        dummy_dataset = data_utils.TensorDataset(dummy_images, dummy_labels)
        
        train_loader = data_utils.DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        val_loader = data_utils.DataLoader(dummy_dataset, batch_size=16, shuffle=False)
        
        # Create model
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        
        # Create config for fast testing
        config = TrainingConfig(
            learning_rate=1e-3,
            warmup_epochs=0,  # No warmup for testing
            total_epochs=2,   # Short training
            use_amp=False,    # Disable AMP for testing
            deterministic=False  # Faster testing
        )
        
        # Create trainer
        trainer = QuickDrawTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            save_dir="test_results",
            model_name="test_model"
        )
        
        print("Trainer created successfully")
        
        # Test single epoch
        print("Testing single training epoch...")
        train_loss, train_top1, train_top5 = trainer.train_epoch()
        print(f"Train metrics: Loss={train_loss:.4f}, Top1={train_top1:.2f}%, Top5={train_top5:.2f}%")
        
        print("Testing single validation epoch...")
        val_loss, val_top1, val_top5 = trainer.validate_epoch()
        print(f"Val metrics: Loss={val_loss:.4f}, Top1={val_top1:.2f}%, Top5={val_top5:.2f}%")
        
        print("All trainer tests passed!")
        return True
        
    except Exception as e:
        print(f"Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_trainer()
