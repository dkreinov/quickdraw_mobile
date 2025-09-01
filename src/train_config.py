"""Step 3.2: Training Configuration for QuickDraw Vision Models.

This module provides comprehensive training configuration including:
- Multiple optimizer support (AdamW, LAMB for large batch training)
- Warmup + cosine decay scheduling
- Cross-entropy loss with label smoothing
- Mixed precision (AMP) support
- Gradient clipping and regularization
- Deterministic training setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import math
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

# Try to import custom optimizers for large batch training
try:
    from .optimizers import create_optimizer_for_large_batch, get_recommended_lr_for_batch_size
    CUSTOM_OPTIMIZERS_AVAILABLE = True
except ImportError:
    # Try absolute import if relative import fails
    try:
        from optimizers import create_optimizer_for_large_batch, get_recommended_lr_for_batch_size
        CUSTOM_OPTIMIZERS_AVAILABLE = True
    except ImportError:
        CUSTOM_OPTIMIZERS_AVAILABLE = False


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
        optimizer_name: str = "adamw",  # "adamw", "lamb", "adamw_large"
        auto_scale_lr: bool = False,  # Auto-scale LR for large batches
        base_batch_size: int = 64,    # Reference batch size for LR scaling
        
        # Scheduler
        schedule_time_unit: str = "step",
        
        # Regularization
        label_smoothing: float = 0.1,
        gradient_clip_norm: float = 1.0,
        
        # Early Stopping
        early_stopping_patience: int = 5,  # Stop if no improvement for N epochs (0 = disabled)
        
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
        self.optimizer_name = optimizer_name.lower()
        self.auto_scale_lr = auto_scale_lr
        self.base_batch_size = base_batch_size
        self.schedule_time_unit = schedule_time_unit
        
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        
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
    
    def create_optimizer(self, model: nn.Module, current_batch_size: Optional[int] = None) -> optim.Optimizer:
        """
        Create optimizer with support for large batch training.
        
        Supports:
        - AdamW: Standard optimizer for small-medium batches
        - LAMB: Layer-wise adaptive optimizer for large batches
        - AdamW Large: Enhanced AdamW for large batch training
        
        Args:
            model: PyTorch model
            current_batch_size: Current effective batch size for LR scaling
        """
        
        # Auto-scale learning rate if enabled
        effective_lr = self.learning_rate
        if self.auto_scale_lr and current_batch_size is not None:
            if CUSTOM_OPTIMIZERS_AVAILABLE:
                scaling_method = "sqrt" if self.optimizer_name == "lamb" else "custom"
                effective_lr = get_recommended_lr_for_batch_size(
                    self.learning_rate,
                    self.base_batch_size,
                    current_batch_size,
                    scaling_method
                )
                log_and_print(f"Auto-scaled LR: {self.learning_rate:.6f} -> {effective_lr:.6f} "
                            f"(batch {self.base_batch_size} -> {current_batch_size})", self.logger)
        
        # Use custom optimizers if available and requested
        if CUSTOM_OPTIMIZERS_AVAILABLE and self.optimizer_name in ["lamb", "adamw_large"]:
            optimizer = create_optimizer_for_large_batch(
                model,
                optimizer_name=self.optimizer_name,
                lr=effective_lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-6 if self.optimizer_name == "lamb" else 1e-8
            )
            
            log_and_print(f"Created {self.optimizer_name.upper()} optimizer:", self.logger)
            log_and_print(f"   Learning rate: {effective_lr:.6f}", self.logger)
            log_and_print(f"   Weight decay: {self.weight_decay}", self.logger)
            log_and_print(f"   Optimizer type: {type(optimizer).__name__}", self.logger)
            
            return optimizer
        
        # Fallback to standard AdamW
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
            lr=effective_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        optimizer_name = self.optimizer_name if self.optimizer_name == "adamw" else f"{self.optimizer_name} (fallback to AdamW)"
        log_and_print(f"Created {optimizer_name} optimizer:", self.logger)
        log_and_print(f"   Learning rate: {effective_lr:.6f}", self.logger)
        log_and_print(f"   Weight decay: {self.weight_decay}", self.logger)
        log_and_print(f"   Decay params: {len(decay_params)}", self.logger)
        log_and_print(f"   No-decay params: {len(no_decay_params)}", self.logger)
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer, steps_per_epoch: int) -> optim.lr_scheduler.LRScheduler:
        """
        Create warmup + cosine annealing scheduler.
        
        Modes:
          - 'epoch': schedule progresses per epoch (warmup_epochs/total_epochs)
          - 'step' : schedule progresses per optimizer step (original behavior)
        """
        unit = getattr(self, "schedule_time_unit", "epoch")
        if unit not in ("epoch", "step"):
            unit = "epoch"
        
        if unit == "epoch":
            warmup_iters = max(0, int(self.warmup_epochs))
            total_iters = max(1, int(self.total_epochs))
            
            if warmup_iters > 0:
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_iters
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, total_iters - warmup_iters),
                    eta_min=1e-7
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iters]
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_iters,
                    eta_min=1e-7
                )
            
            log_and_print(f"Created learning rate scheduler (epoch-based):", self.logger)
            log_and_print(f"   Warmup epochs: {warmup_iters}", self.logger)
            log_and_print(f"   Total epochs: {total_iters}", self.logger)
            log_and_print(f"   Steps per epoch: {steps_per_epoch}", self.logger)
            return scheduler
        
        # step-based
        warmup_steps = int(self.warmup_epochs * steps_per_epoch)
        total_steps = int(self.total_epochs * steps_per_epoch)
        cosine_steps = max(1, total_steps - warmup_steps)
        
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=1e-7
            )
        
        log_and_print(f"Created learning rate scheduler (step-based):", self.logger)
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
            'schedule_time_unit': self.schedule_time_unit,
            'label_smoothing': self.label_smoothing,
            'gradient_clip_norm': self.gradient_clip_norm,
            'early_stopping_patience': self.early_stopping_patience,
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
