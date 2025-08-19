
"""Step 3: MobileViT/ViT model factory for single-channel QuickDraw classification.

This module provides vision transformer models optimized for mobile deployment:
- Single-channel input (grayscale) instead of RGB
- Configurable number of classes
- Support for ViT-Tiny and MobileViT architectures
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import timm

# Try to import logging config, fallback to basic logging if not available
try:
    from .logging_config import get_logger, log_and_print
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
    def log_and_print(msg, logger_instance=None, level="INFO"): print(msg)


class SingleChannelViT(nn.Module):
    """
    Vision Transformer/MobileViT adapted for single-channel (grayscale) input.
    
    This wrapper modifies ViT or MobileViT models to accept 1-channel input instead of 3-channel RGB.
    Perfect for QuickDraw doodles which are naturally grayscale.
    
    Supports:
    - ViT models (vit_tiny, vit_small, vit_base)
    - MobileViT models (mobilevitv2_175, mobilevitv2_200)
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        num_classes: int = 344,
        pretrained: bool = True,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        
        logger = get_logger(__name__)
        log_and_print(f"Creating {model_name} with {num_classes} classes", logger_instance=logger)
        log_and_print(f"   Single-channel input: 1 → 3 channel conversion", logger_instance=logger)
        log_and_print(f"   Pretrained: {pretrained}", logger_instance=logger)
        log_and_print(f"   Drop path rate: {drop_path_rate}", logger_instance=logger)
        
        # Load base model (will be 3-channel input initially)
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate
        )
        
        # Get the original first layer - different for ViT vs MobileViT
        if hasattr(self.base_model, 'patch_embed'):
            # Standard ViT structure
            original_conv = self.base_model.patch_embed.proj
            self.model_type = "vit"
        elif hasattr(self.base_model, 'stem') and hasattr(self.base_model.stem, 'conv'):
            # MobileViT structure
            original_conv = self.base_model.stem.conv
            self.model_type = "mobilevit"
        else:
            raise ValueError(f"Unknown model structure for {model_name}. Expected ViT or MobileViT.")
        
        # Create new first layer that accepts 1 channel
        self.single_channel_conv = nn.Conv2d(
            in_channels=1,  # Single channel input
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize weights intelligently
        if pretrained:
            # Average the RGB weights to initialize single-channel weights
            with torch.no_grad():
                rgb_weights = original_conv.weight  # Shape: (out_channels, 3, H, W)
                # Average across the 3 input channels
                avg_weights = rgb_weights.mean(dim=1, keepdim=True)  # Shape: (out_channels, 1, H, W)
                self.single_channel_conv.weight.copy_(avg_weights)
                
                if original_conv.bias is not None:
                    self.single_channel_conv.bias.copy_(original_conv.bias)
                    
            logger.info("Initialized single-channel weights from pretrained RGB weights")
            print("   Initialized single-channel weights from pretrained RGB weights")
        else:
            logger.info("Random initialization for single-channel weights")
            print("   Random initialization for single-channel weights")
        
        # Replace the original first layer (different path for each model type)
        if self.model_type == "vit":
            self.base_model.patch_embed.proj = self.single_channel_conv
        elif self.model_type == "mobilevit":
            self.base_model.stem.conv = self.single_channel_conv
        
        # Store model info
        self.model_name = model_name
        self.num_classes = num_classes
        # Set input size based on model type
        if self.model_type == "mobilevit":
            self.input_size = 256  # MobileViT uses 256x256
        else:
            self.input_size = 224  # ViT uses 224x224
        
    def forward(self, x):
        """
        Forward pass for single-channel input.
        
        Args:
            x: Tensor of shape (B, 1, H, W) - single channel grayscale images
            
        Returns:
            logits: Tensor of shape (B, num_classes) - class predictions
        """
        return self.base_model(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging and checkpointing."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'input_channels': 1,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }


def build_model(
    arch: str = "vit_tiny_patch16_224",
    in_channels: int = 1,
    num_classes: int = 344,
    pretrained: bool = True,
    drop_path_rate: float = 0.1
) -> SingleChannelViT:
    """
    Build a vision model for QuickDraw classification.
    
    Args:
        arch: Model architecture name (currently supports ViT variants)
        in_channels: Number of input channels (should be 1 for grayscale)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        drop_path_rate: Drop path rate for regularization
        
    Returns:
        model: Configured model ready for training
    """
    
    if in_channels != 1:
        raise ValueError(f"This implementation is optimized for single-channel input, got {in_channels}")
    
    # Supported architectures
    supported_archs = [
        # ViT models
        "vit_tiny_patch16_224",
        "vit_small_patch16_224", 
        "vit_base_patch16_224",
        # MobileViT models
        "mobilevitv2_175",
        "mobilevitv2_200"
    ]
    
    if arch not in supported_archs:
        raise ValueError(f"Architecture {arch} not supported. Available: {supported_archs}")
    
    logger = get_logger(__name__)
    log_and_print(f"\nBuilding model:", logger_instance=logger)
    log_and_print(f"   Architecture: {arch}", logger_instance=logger)
    log_and_print(f"   Input channels: {in_channels} (grayscale)", logger_instance=logger)
    log_and_print(f"   Output classes: {num_classes}", logger_instance=logger)
    
    # Create the model
    model = SingleChannelViT(
        model_name=arch,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate
    )
    
    # Print model info
    info = model.get_model_info()
    log_and_print(f"\nModel info:", logger_instance=logger)
    log_and_print(f"   Total parameters: {info['total_params']:,}", logger_instance=logger)
    log_and_print(f"   Trainable parameters: {info['trainable_params']:,}", logger_instance=logger)
    log_and_print(f"   Model size: {info['model_size_mb']:.2f} MB", logger_instance=logger)
    
    return model


def test_model_forward(model: nn.Module, batch_size: int = 2, verbose: bool = True):
    """
    Test model forward pass with dummy input.
    
    Args:
        model: Model to test
        batch_size: Batch size for test
        verbose: Whether to print details
    """
    
    if verbose:
        logger = get_logger(__name__)
        log_and_print(f"\nTesting model forward pass...", logger_instance=logger)
    
    # Create dummy input (B, 1, 224, 224)
    dummy_input = torch.randn(batch_size, 1, 224, 224)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    if verbose:
        log_and_print(f"   Input shape: {dummy_input.shape}", logger_instance=logger)
        log_and_print(f"   Output shape: {output.shape}", logger_instance=logger)
        log_and_print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]", logger_instance=logger)
        log_and_print(f"   Forward pass successful!", logger_instance=logger)
    
    return output
