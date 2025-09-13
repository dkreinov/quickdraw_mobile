# Mobile Deployment for QuickDraw Classifier

This folder contains everything needed to deploy the trained QuickDraw model to mobile devices (Android/iOS).

## 📁 Folder Structure

```
mobile_deployment/
├── README.md                           # This file
├── ANDROID_EXECUTORCH_FP32.md         # Step-by-step Android guide
├── scripts/
│   ├── export_executorch.py           # Export PyTorch → ExecuTorch .pte
│   └── create_labels_file.py          # Generate labels.txt for apps
├── android/                           # Android app projects (future)
└── ios/                              # iOS app projects (future)
```

## 🚀 Quick Start

### 1. Export Model to Mobile Format

```bash
# Export your trained model to ExecuTorch .pte format
cd mobile_deployment/scripts
python export_executorch.py --checkpoint ../../results/full344_4k500_regaug_best.pt --output-dir ../exports

# Create labels file for the app
python create_labels_file.py --output ../exports/labels.txt --classes 344
```

This creates:
- `exports/quickdraw_fp32.pte` - ExecuTorch model for Android
- `exports/labels.txt` - Class names (344 lines)
- `exports/export_metadata.json` - Model info and preprocessing details

### 2. Follow Platform Guide

- **Android**: See `ANDROID_EXECUTORCH_FP32.md` for complete step-by-step instructions
  - Android project location: `C:\Users\dkreinov\AndroidStudioProjects\QuickDrawClassifier\`
- **iOS**: Coming soon (Core ML or TensorFlow Lite path)

## 📋 Prerequisites

### Desktop (Export)
- Python 3.10+, PyTorch 2.3+
- ExecuTorch: `pip install executorch`
- Your trained checkpoint (e.g., `full344_4k500_regaug_best.pt`)

### Android
- Android Studio Flamingo+
- SDK 34, NDK 26+
- Device with Android 10+ (arm64-v8a)

### iOS (Future)
- Xcode 15+
- iOS 14+ device
- Core ML or TensorFlow Lite runtime

## 🎯 Model Details

- **Architecture**: ViT-Tiny (5.5M parameters)
- **Input**: 1×224×224 grayscale (NCHW)
- **Output**: 344 class logits
- **Accuracy**: 76.42% on validation set
- **Size**: ~22 MB (FP32), ~6 MB (INT8 quantized)

## 📱 Performance Results

### Pixel 7 Emulator (x86_64)
- **Debug**: 4437ms avg, 0.23 images/sec
- **Release**: 4096ms avg, 0.24 images/sec (🚀 8.3% faster)

### Samsung SM-G780F (Real Device)
- **Debug**: 2152ms avg, 0.46 images/sec  
- **Release**: 3171ms avg, 0.32 images/sec

### Performance Targets (Future Optimizations)
- **CPU (4 threads)**: ~50-100ms per inference
- **NNAPI**: ~20-50ms (device dependent)
- **GPU**: ~15-40ms (device dependent)

### iOS (Estimated)
- **Core ML (ANE)**: ~10-30ms
- **Core ML (GPU)**: ~20-50ms
- **CPU**: ~50-100ms

## 🔧 Troubleshooting

### Export Issues
- **ExecuTorch not found**: `pip install executorch`
- **torch.export fails**: Ensure PyTorch 2.1+ and model is in eval() mode
- **ONNX export alternative**: Use `export_onnx.py` (TensorFlow Lite path)

### Android Issues
- **Module.load() fails**: Check .pte file path and ExecuTorch runtime integration
- **Wrong predictions**: Verify preprocessing matches training (grayscale, normalization)
- **Slow inference**: Try NNAPI delegate or reduce input resolution

## 📚 Additional Resources

- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [Android ExecuTorch Integration](https://pytorch.org/executorch/stable/using-executorch-android.html)
- [TensorFlow Lite Android](https://www.tensorflow.org/lite/android) (alternative path)

## 🎨 Demo App Features

The mobile apps will include:
- **Drawing Canvas**: Sketch recognition interface
- **Real-time Inference**: Live classification as you draw
- **Performance Metrics**: Latency and throughput display
- **Benchmark Mode**: Measure performance on fixed inputs
- **Model Comparison**: FP32 vs INT8 quantized models
