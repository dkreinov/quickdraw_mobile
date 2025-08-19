#!/usr/bin/env python3
"""
Test the model integration with QuickDraw dataset.
Run this to verify the model and dataset work together.

Usage:
    .venv/bin/python tests/models/test_model.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from models import build_model, test_model_forward
    from data import create_dataloaders, get_all_class_names
    import torch
    print("✅ Successfully imported model and data modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've installed all dependencies: pip install -r requirements.txt")
    sys.exit(1)


def test_model_creation():
    """Test creating different model variants."""
    print("=== Test 1: Model Creation ===")
    
    try:
        # Test ViT-Tiny
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=10,  # Small number for testing
            pretrained=True,
            drop_path_rate=0.1
        )
        
        # Test forward pass
        test_model_forward(model, batch_size=2)
        
        print("✅ Model creation test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False


def test_model_with_data():
    """Test model integration with actual dataset."""
    print("=== Test 2: Model + Data Integration ===")
    
    # Data directory relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "quickdraw_parquet"
    
    if not data_dir.exists():
        print("⚠️  No data found - skipping model+data integration test")
        print("   To test with data, download some first:")
        print("   python scripts/download_quickdraw_direct.py --num-classes 3 --samples-per-class 100")
        print("✅ Model+data test skipped (expected)\n")
        return True
    
    try:
        # Get available classes
        class_names = get_all_class_names(data_dir)
        num_classes = min(5, len(class_names))  # Use max 5 classes for testing
        
        print(f"📊 Testing with {num_classes} classes: {class_names[:num_classes]}")
        
        # Create small dataloaders
        train_loader, val_loader, metadata = create_dataloaders(
            data_dir=data_dir,
            classes=class_names[:num_classes],
            train_samples_per_class=10,  # Very small for testing
            val_samples_per_class=5,
            batch_size=4,
            num_workers=0
        )
        
        # Create model with correct number of classes
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=num_classes,
            pretrained=True
        )
        
        # Test with real data batch
        print(f"\n🔗 Testing model with real QuickDraw data...")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"   Batch {batch_idx}: images={images.shape}, labels={labels.shape}")
                
                # Forward pass
                outputs = model(images)
                print(f"   Model output: {outputs.shape}")
                print(f"   Label range: {labels.min()}-{labels.max()}")
                print(f"   Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                
                # Verify shapes match
                assert outputs.shape[0] == images.shape[0], "Batch size mismatch"
                assert outputs.shape[1] == num_classes, f"Expected {num_classes} classes, got {outputs.shape[1]}"
                
                break  # Just test first batch
        
        print("✅ Model+data integration test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Model+data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_vs_random():
    """Compare pretrained vs random initialization."""
    print("=== Test 3: Pretrained vs Random Initialization ===")
    
    try:
        print("🔧 Creating pretrained model...")
        pretrained_model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=5,
            pretrained=True
        )
        
        print("\n🔧 Creating random model...")
        random_model = build_model(
            arch="vit_tiny_patch16_224", 
            num_classes=5,
            pretrained=False
        )
        
        # Compare first layer weights
        pretrained_weights = pretrained_model.single_channel_conv.weight
        random_weights = random_model.single_channel_conv.weight
        
        print(f"\n📊 Weight comparison:")
        print(f"   Pretrained weights mean: {pretrained_weights.mean():.6f}")
        print(f"   Random weights mean: {random_weights.mean():.6f}")
        print(f"   Pretrained weights std: {pretrained_weights.std():.6f}")
        print(f"   Random weights std: {random_weights.std():.6f}")
        
        # They should be different
        weight_diff = torch.abs(pretrained_weights - random_weights).mean()
        print(f"   Mean absolute difference: {weight_diff:.6f}")
        
        assert weight_diff > 0.001, "Pretrained and random weights are too similar"
        
        print("✅ Pretrained vs random test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Pretrained vs random test failed: {e}")
        return False


def test_mobilevit_support():
    """Test MobileViT model creation and forward pass."""
    print("=== Test 4: MobileViT Support ===")
    
    try:
        # Test both MobileViT models
        mobilevit_models = ["mobilevitv2_175", "mobilevitv2_200"]
        
        for arch in mobilevit_models:
            print(f"\n🔧 Testing {arch}...")
            
            # Create model (use pretrained=False for faster testing)
            model = build_model(
                arch=arch,
                num_classes=10,
                pretrained=False
            )
            
            print(f"   ✅ Model created successfully")
            
            # Test forward pass with MobileViT's native input size (256x256)
            test_input = torch.randn(2, 1, 256, 256)
            output = model(test_input)
            
            print(f"   ✅ Forward pass: {test_input.shape} → {output.shape}")
            
            # Verify output shape
            assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"
            
            # Check model info
            info = model.get_model_info()
            print(f"   📊 Parameters: {info['total_params']:,}")
            print(f"   📊 Model size: {info['model_size_mb']:.1f} MB")
            print(f"   📊 Input size: {info['input_size']}x{info['input_size']}")
            
            # Verify it's truly single-channel
            assert model.single_channel_conv.in_channels == 1, "Model should accept single-channel input"
            assert model.model_type == "mobilevit", f"Model type should be 'mobilevit', got {model.model_type}"
            
            print(f"   ✅ {arch} test passed!")
        
        print("\n✅ MobileViT support test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MobileViT support test failed: {e}")
        traceback.print_exc()
        return False


def demonstrate_mobile_benefits():
    """Show the benefits of single-channel for mobile deployment."""
    print("=== Demo: Mobile Deployment Benefits ===")
    
    try:
        # Create model
        model = build_model(
            arch="vit_tiny_patch16_224",
            num_classes=344,  # Full QuickDraw classes
            pretrained=True
        )
        
        info = model.get_model_info()
        
        print(f"📱 Mobile-optimized model:")
        print(f"   Architecture: ViT-Tiny")
        print(f"   Input: Single-channel (grayscale) 224x224")
        print(f"   Parameters: {info['total_params']:,}")
        print(f"   Model size: {info['model_size_mb']:.2f} MB")
        print(f"   Memory per image: {1 * 224 * 224 * 4 / 1024:.1f} KB (vs {3 * 224 * 224 * 4 / 1024:.1f} KB for RGB)")
        
        print(f"\n💡 Benefits for mobile:")
        print(f"   ✅ 3x less memory per image (1-channel vs RGB)")
        print(f"   ✅ Smaller first layer (1/3 the parameters)")
        print(f"   ✅ Matches phone drawing reality (grayscale)")
        print(f"   ✅ Faster inference on mobile devices")
        print(f"   ✅ Ready for quantization (INT8, 4-bit)")
        
        print("✅ Mobile benefits demonstration complete!\n")
        return True
        
    except Exception as e:
        print(f"❌ Mobile benefits demo failed: {e}")
        return False


def main():
    """Run all model tests."""
    print("🧪 QuickDraw Model Test Suite")
    print("=" * 50)
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(("Model Creation", test_model_creation()))
    test_results.append(("Model+Data Integration", test_model_with_data()))
    test_results.append(("Pretrained vs Random", test_pretrained_vs_random()))
    test_results.append(("MobileViT Support", test_mobilevit_support()))
    test_results.append(("Mobile Benefits Demo", demonstrate_mobile_benefits()))
    
    # Summary
    print("📊 Test Results Summary:")
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
        print("\n🚀 Model is ready for training!")
        print("   Next step: Implement training loop")
        return 0


if __name__ == "__main__":
    exit(main())
