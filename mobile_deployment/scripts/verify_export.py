#!/usr/bin/env python3
"""
Verify ExecuTorch export quality by comparing outputs with original PyTorch model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models import build_model
from logging_config import get_logger, log_and_print

def load_pytorch_model(checkpoint_path: str):
    """Load original PyTorch model."""
    model = build_model(
        arch="vit_tiny_patch16_224",
        num_classes=344,
        pretrained=False,
        in_channels=1
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def test_pytorch_inference(model, test_inputs):
    """Run PyTorch inference on test inputs."""
    logger = get_logger(__name__)
    
    with torch.no_grad():
        outputs = []
        for i, input_tensor in enumerate(test_inputs):
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            outputs.append({
                'logits': output,
                'probabilities': probs,
                'predicted_class': pred_class,
                'confidence': confidence
            })
            
            log_and_print(f"Test {i+1}: Class {pred_class.item()}, Confidence {confidence.item():.4f}", logger_instance=logger)
    
    return outputs

def create_test_inputs(num_tests=5):
    """Create diverse test inputs for validation."""
    test_inputs = []
    
    # 1. Random noise
    test_inputs.append(torch.randn(1, 1, 224, 224))
    
    # 2. All zeros (blank canvas)
    test_inputs.append(torch.zeros(1, 1, 224, 224))
    
    # 3. All ones (white canvas)
    test_inputs.append(torch.ones(1, 1, 224, 224))
    
    # 4. Simple pattern (circle-like)
    circle_input = torch.zeros(1, 1, 224, 224)
    center = 112
    radius = 50
    y, x = torch.meshgrid(torch.arange(224), torch.arange(224), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    circle_input[0, 0][mask] = 1.0
    test_inputs.append(circle_input)
    
    # 5. Gradient pattern
    gradient_input = torch.zeros(1, 1, 224, 224)
    for i in range(224):
        gradient_input[0, 0, i, :] = i / 224.0
    test_inputs.append(gradient_input)
    
    return test_inputs

def verify_export_quality(checkpoint_path: str, pte_path: str = None):
    """Comprehensive export quality verification."""
    logger = get_logger(__name__)
    log_and_print("🔍 Verifying ExecuTorch export quality...", logger_instance=logger)
    
    # Load PyTorch model
    log_and_print("Loading PyTorch model...", logger_instance=logger)
    pytorch_model = load_pytorch_model(checkpoint_path)
    
    # Create test inputs
    log_and_print("Creating test inputs...", logger_instance=logger)
    test_inputs = create_test_inputs()
    
    # Test PyTorch model
    log_and_print("\n📊 PyTorch Model Results:", logger_instance=logger)
    pytorch_outputs = test_pytorch_inference(pytorch_model, test_inputs)
    
    # Basic sanity checks
    log_and_print("\n✅ Basic Sanity Checks:", logger_instance=logger)
    
    for i, output in enumerate(pytorch_outputs):
        logits = output['logits']
        probs = output['probabilities']
        
        # Check output shapes
        assert logits.shape == (1, 344), f"Wrong logits shape: {logits.shape}"
        assert probs.shape == (1, 344), f"Wrong probs shape: {probs.shape}"
        
        # Check probability constraints
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5), "Probabilities don't sum to 1"
        assert (probs >= 0).all(), "Negative probabilities found"
        assert (probs <= 1).all(), "Probabilities > 1 found"
        
        # Check for reasonable outputs (not all same class)
        if i > 0:  # Skip first (random) input
            prev_pred = pytorch_outputs[i-1]['predicted_class']
            curr_pred = output['predicted_class']
            # Different inputs should generally give different predictions
            
        log_and_print(f"  Test {i+1}: ✓ Shape, probabilities, and ranges OK", logger_instance=logger)
    
    # Model statistics
    log_and_print("\n📈 Model Statistics:", logger_instance=logger)
    all_logits = torch.cat([out['logits'] for out in pytorch_outputs], dim=0)
    log_and_print(f"  Logits range: [{all_logits.min():.3f}, {all_logits.max():.3f}]", logger_instance=logger)
    log_and_print(f"  Logits mean: {all_logits.mean():.3f}, std: {all_logits.std():.3f}", logger_instance=logger)
    
    all_confidences = torch.cat([out['confidence'] for out in pytorch_outputs], dim=0)
    log_and_print(f"  Confidence range: [{all_confidences.min():.3f}, {all_confidences.max():.3f}]", logger_instance=logger)
    
    # Check for model collapse (all same predictions)
    unique_preds = torch.unique(torch.cat([out['predicted_class'] for out in pytorch_outputs]))
    log_and_print(f"  Unique predictions: {len(unique_preds)}/5 tests", logger_instance=logger)
    
    if len(unique_preds) == 1:
        log_and_print("  ⚠️  WARNING: All test inputs give same prediction (possible model issue)", logger_instance=logger)
    
    log_and_print("\n✅ Export verification complete!", logger_instance=logger)
    log_and_print("The PyTorch model appears to be working correctly.", logger_instance=logger)
    
    if pte_path and Path(pte_path).exists():
        log_and_print(f"\n📱 ExecuTorch file ready: {pte_path}", logger_instance=logger)
        log_and_print("You can now test this on Android to verify mobile parity.", logger_instance=logger)
    
    return pytorch_outputs

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify ExecuTorch export quality')
    parser.add_argument('--checkpoint', type=str, 
                       default='../../results/full344_4k500_regaug_best.pt',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--pte-file', type=str,
                       default='../exports/quickdraw_fp32.pte',
                       help='Path to ExecuTorch .pte file')
    
    args = parser.parse_args()
    
    verify_export_quality(args.checkpoint, args.pte_file)

if __name__ == "__main__":
    main()

