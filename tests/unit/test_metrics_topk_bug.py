"""Test that reproduces and fixes the top-k metrics bug.

When num_classes < 5, trying to compute top-5 accuracy fails with:
"RuntimeError: selected index k out of range"
"""

import torch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from trainer import MetricsTracker


def test_topk_metrics_with_few_classes():
    """Test that metrics work correctly when num_classes < 5."""
    
    # Test case 1: 3 classes (should fail with current implementation)
    print("\n=== Testing with 3 classes ===")
    tracker = MetricsTracker()
    
    # Create logits for 3 classes, batch size 2
    logits = torch.tensor([
        [0.8, 0.1, 0.1],  # Should predict class 0
        [0.1, 0.9, 0.0]   # Should predict class 1
    ], dtype=torch.float32)
    
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = 0.25
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets: {targets}")
    print(f"Attempting top-k with k=5, but only {logits.shape[1]} classes available")
    
    # This should fail with current implementation
    try:
        tracker.update(loss, logits, targets)
        print("✗ UNEXPECTED: No error occurred (this should fail with current code)")
        # If we get here, the bug is already fixed
        loss_avg, top1_acc, top5_acc = tracker.compute()
        print(f"Results: loss={loss_avg:.3f}, top1={top1_acc:.1f}%, top5={top5_acc:.1f}%")
        assert top1_acc == 100.0, f"Expected 100% top1 accuracy, got {top1_acc}%"
        assert top5_acc == 100.0, f"Expected 100% top5 accuracy, got {top5_acc}% (should equal top1 when num_classes < 5)"
    except RuntimeError as e:
        if "selected index k out of range" in str(e):
            print(f"✓ EXPECTED ERROR: {e}")
            print("This confirms the bug exists")
        else:
            print(f"✗ UNEXPECTED ERROR: {e}")
            raise
    
    # Test case 2: 1 class (edge case)
    print("\n=== Testing with 1 class ===")
    tracker2 = MetricsTracker()
    
    logits_1class = torch.tensor([
        [0.9],  # Only one class
        [0.8]
    ], dtype=torch.float32)
    
    targets_1class = torch.tensor([0, 0], dtype=torch.long)
    
    print(f"Logits shape: {logits_1class.shape}")
    print(f"Attempting top-k with k=5, but only {logits_1class.shape[1]} classes available")
    
    try:
        tracker2.update(loss, logits_1class, targets_1class)
        print("✗ UNEXPECTED: No error occurred")
    except RuntimeError as e:
        if "selected index k out of range" in str(e):
            print(f"✓ EXPECTED ERROR: {e}")
        else:
            print(f"✗ UNEXPECTED ERROR: {e}")
            raise
    
    # Test case 3: 5+ classes (should work fine)
    print("\n=== Testing with 5+ classes ===")
    tracker3 = MetricsTracker()
    
    logits_5class = torch.tensor([
        [0.5, 0.2, 0.15, 0.1, 0.05],  # 5 classes
        [0.1, 0.6, 0.15, 0.1, 0.05]
    ], dtype=torch.float32)
    
    targets_5class = torch.tensor([0, 1], dtype=torch.long)
    
    print(f"Logits shape: {logits_5class.shape}")
    print(f"This should work fine with k=5")
    
    try:
        tracker3.update(loss, logits_5class, targets_5class)
        loss_avg, top1_acc, top5_acc = tracker3.compute()
        print(f"✓ SUCCESS: loss={loss_avg:.3f}, top1={top1_acc:.1f}%, top5={top5_acc:.1f}%")
        assert top1_acc == 100.0, f"Expected 100% top1 accuracy, got {top1_acc}%"
        assert top5_acc == 100.0, f"Expected 100% top5 accuracy, got {top5_acc}%"
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR with 5 classes: {e}")
        raise


def test_fixed_topk_metrics():
    """Test the fixed implementation that handles num_classes < 5."""
    print("\n=== Testing Fixed Implementation ===")
    
    # We'll implement the fix and test it works for all cases
    class FixedMetricsTracker:
        """Fixed version of MetricsTracker that handles num_classes < 5."""
        
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
            num_classes = logits.size(1)
            
            # Loss
            self.total_loss += loss * batch_size
            self.total_samples += batch_size
            
            # Top-k accuracy (handle case where num_classes < 5)
            with torch.no_grad():
                # Use min(5, num_classes) to avoid out-of-range error
                k = min(5, num_classes)
                _, predicted = logits.topk(k, dim=1, largest=True, sorted=True)
                targets_expanded = targets.view(-1, 1).expand(-1, k)
                
                # Top-1 accuracy
                self.correct_top1 += predicted[:, 0].eq(targets).sum().item()
                
                # Top-5 accuracy (or top-k if k < 5)
                self.correct_top5 += predicted.eq(targets_expanded).any(dim=1).sum().item()
        
        def compute(self):
            """Compute average metrics."""
            if self.total_samples == 0:
                return 0.0, 0.0, 0.0
                
            avg_loss = self.total_loss / self.total_samples
            top1_acc = 100.0 * self.correct_top1 / self.total_samples
            top5_acc = 100.0 * self.correct_top5 / self.total_samples
            
            return avg_loss, top1_acc, top5_acc
    
    # Test the fixed implementation
    test_cases = [
        ("3 classes", torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0]]), torch.tensor([0, 1])),
        ("1 class", torch.tensor([[0.9], [0.8]]), torch.tensor([0, 0])),
        ("2 classes", torch.tensor([[0.7, 0.3], [0.4, 0.6]]), torch.tensor([0, 1])),
        ("5 classes", torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05], [0.1, 0.6, 0.15, 0.1, 0.05]]), torch.tensor([0, 1])),
    ]
    
    for name, logits, targets in test_cases:
        print(f"\nTesting {name}:")
        tracker = FixedMetricsTracker()
        
        try:
            tracker.update(0.25, logits, targets)
            loss_avg, top1_acc, top5_acc = tracker.compute()
            print(f"  ✓ SUCCESS: loss={loss_avg:.3f}, top1={top1_acc:.1f}%, top5={top5_acc:.1f}%")
            
            # Verify correctness
            assert top1_acc == 100.0, f"Expected 100% top1 accuracy for {name}, got {top1_acc}%"
            # For cases with < 5 classes, top5 should equal top1 (all predictions are within the available classes)
            if logits.size(1) < 5:
                assert top5_acc == top1_acc, f"For {name}, top5 should equal top1 when num_classes < 5"
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            raise
    
    print("\n✅ All fixed implementation tests passed!")


if __name__ == "__main__":
    print("=== Top-K Metrics Bug Reproduction and Fix Test ===")
    
    print("\n1. Testing current implementation (should show the bug):")
    test_topk_metrics_with_few_classes()
    
    print("\n2. Testing fixed implementation:")
    test_fixed_topk_metrics()
    
    print("\n=== Test completed ===")
    print("Next step: Apply the fix to src/trainer.py")
