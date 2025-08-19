"""Step 3.2: Training Configuration for QuickDraw Vision Models.

This module provides comprehensive training configuration including:
- AdamW optimizer with warmup + cosine decay
- Cross-entropy loss with label smoothing
- Mixed precision (AMP) support
- Gradient clipping and regularization
- Deterministic training setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
try:
    # Try new API first (PyTorch 2.0+)
    from torch.amp import GradScaler
    USE_NEW_AMP_API = True
except ImportError:
    # Fall back to old API
    from torch.cuda.amp import GradScaler
    USE_NEW_AMP_API = False
from typing import Dict, Any, Optional, Tuple
import math
import random
import numpy as np

# Try to import logging config, fallback to basic logging if not available
try:
    from .logging_config import get_logger, log_and_print
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"): print(msg)


class TrainingConfig:
    """
    Comprehensive training configuration for vision transformers.
    
    Follows modern best practices for ViT/MobileViT training:
    - AdamW optimizer with proper weight decay
    - Warmup + cosine annealing schedule
    - Label smoothing for regularization
    - Mixed precision for efficiency
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        # Optimization
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 2,
        total_epochs: int = 20,
        
        # Regularization
        label_smoothing: float = 0.1,
        gradient_clip_norm: float = 1.0,
        
        # Precision & Performance
        use_amp: bool = True,  # Automatic Mixed Precision
        
        # Reproducibility
        seed: int = 402,
        deterministic: bool = True,
        
        # Device
        device: Optional[str] = None
    ):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        
        self.use_amp = use_amp
        
        self.seed = seed
        self.deterministic = deterministic
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger = get_logger(__name__)
        
        # Setup reproducibility
        if self.deterministic:
            self._setup_deterministic()
    
    def _setup_deterministic(self):
        """Setup deterministic training for reproducibility."""
        log_and_print(f"Setting up deterministic training (seed={self.seed})", self.logger)
        
        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
            # Deterministic operations (slower but reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Set deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Create AdamW optimizer with proper weight decay groups.
        
        Follows best practices:
        - No weight decay on bias and normalization layers
        - Different weight decay for different parameter groups
        """
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Don't apply weight decay to bias and norm layers
            if param.ndim <= 1 or 'bias' in name or 'norm' in name or 'bn' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {
                'params': decay_params,
                'weight_decay': self.weight_decay,
                'name': 'decay'
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
                'name': 'no_decay'
            }
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        log_and_print(f"Created AdamW optimizer:", self.logger)
        log_and_print(f"   Learning rate: {self.learning_rate}", self.logger)
        log_and_print(f"   Weight decay: {self.weight_decay}", self.logger)
        log_and_print(f"   Decay params: {len(decay_params)}", self.logger)
        log_and_print(f"   No-decay params: {len(no_decay_params)}", self.logger)
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, steps_per_epoch: int) -> optim.lr_scheduler.LRScheduler:
        """
        Create warmup + cosine annealing scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            steps_per_epoch: Number of training steps per epoch
            
        Returns:
            Combined warmup + cosine scheduler
        """
        
        warmup_steps = self.warmup_epochs * steps_per_epoch
        total_steps = self.total_epochs * steps_per_epoch
        cosine_steps = total_steps - warmup_steps
        
        if warmup_steps > 0:
            # Warmup scheduler: linear increase from small fraction to learning_rate
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of LR
                end_factor=1.0,     # End at full LR
                total_iters=warmup_steps
            )
            
            # Cosine annealing scheduler: cosine decay after warmup
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7  # Minimum learning rate
            )
            
            # Combine both schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            # Just cosine annealing if no warmup
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
        
        log_and_print(f"Created learning rate scheduler:", self.logger)
        log_and_print(f"   Warmup steps: {warmup_steps}", self.logger)
        log_and_print(f"   Total steps: {total_steps}", self.logger)
        log_and_print(f"   Steps per epoch: {steps_per_epoch}", self.logger)
        
        return scheduler
    
    def create_loss_function(self) -> nn.Module:
        """Create cross-entropy loss with label smoothing."""
        
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing
        )
        
        log_and_print(f"Created loss function:", self.logger)
        log_and_print(f"   Type: CrossEntropyLoss", self.logger)
        log_and_print(f"   Label smoothing: {self.label_smoothing}", self.logger)
        
        return loss_fn
    
    def create_scaler(self) -> Optional[GradScaler]:
        """Create gradient scaler for mixed precision training."""
        
        if self.use_amp and self.device.type == 'cuda':
            # Use appropriate API based on PyTorch version
            if USE_NEW_AMP_API:
                scaler = GradScaler('cuda')
            else:
                scaler = GradScaler()
            log_and_print(f"Created AMP gradient scaler", self.logger)
            return scaler
        else:
            log_and_print(f"AMP disabled (use_amp={self.use_amp}, device={self.device})", self.logger)
            return None
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients by norm.
        
        Returns:
            grad_norm: The gradient norm before clipping
        """
        
        if self.gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.gradient_clip_norm
            )
            return grad_norm.item()
        else:
            # Calculate norm without clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for logging/saving."""
        
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'label_smoothing': self.label_smoothing,
            'gradient_clip_norm': self.gradient_clip_norm,
            'use_amp': self.use_amp,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'device': str(self.device)
        }
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        config = self.get_config_dict()
        lines = ["TrainingConfig:"]
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def create_training_setup(
    model: nn.Module,
    steps_per_epoch: int,
    config: Optional[TrainingConfig] = None
) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler, nn.Module, Optional[GradScaler]]:
    """
    Create complete training setup.
    
    Args:
        model: The model to train
        steps_per_epoch: Number of training steps per epoch
        config: Training configuration (uses defaults if None)
        
    Returns:
        Tuple of (optimizer, scheduler, loss_fn, scaler)
    """
    
    if config is None:
        config = TrainingConfig()
    
    logger = get_logger(__name__)
    log_and_print("\nCreating training setup:", logger)
    log_and_print(str(config), logger)
    
    # Move model to device
    model = model.to(config.device)
    
    # Create components
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer, steps_per_epoch)
    loss_fn = config.create_loss_function()
    scaler = config.create_scaler()
    
    log_and_print(f"\nTraining setup complete!", logger)
    
    return optimizer, scheduler, loss_fn, scaler


# Test function to be used in unit tests
def test_training_config():
    """Test function for training configuration."""
    
    print("=== Testing Training Configuration ===")
    
    try:
        # Create dummy model
        import timm
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
        
        # Test default config
        config = TrainingConfig(total_epochs=5, warmup_epochs=1)
        print(f"Created config: {config.learning_rate} LR, {config.total_epochs} epochs")
        
        # Test training setup
        optimizer, scheduler, loss_fn, scaler = create_training_setup(
            model=model,
            steps_per_epoch=100,
            config=config
        )
        
        print(f"Created optimizer: {type(optimizer).__name__}")
        print(f"Created scheduler: {type(scheduler).__name__}")
        print(f"Created loss: {type(loss_fn).__name__}")
        print(f"Created scaler: {type(scaler).__name__ if scaler else 'None'}")
        
        # Test gradient clipping
        dummy_input = torch.randn(2, 3, 224, 224).to(config.device)
        dummy_labels = torch.randint(0, 10, (2,)).to(config.device)
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = loss_fn(output, dummy_labels)
        loss.backward()
        
        grad_norm = config.clip_gradients(model)
        print(f"Gradient norm: {grad_norm:.4f}")
        
        optimizer.step()
        scheduler.step()
        
        print(f"Training step completed")
        print(f"All training config tests passed!")
        
        return True
        
    except Exception as e:
        print(f"Training config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_training_config()
