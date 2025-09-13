## FP32 on Android with ExecuTorch — Practical Guide

### Overview

- **Goal**: Run the trained FP32 QuickDraw classifier on Android using ExecuTorch with correct pre/post‑processing, sanity checks, and basic benchmarking.
- **Why ExecuTorch**: PyTorch‑native export pipeline, small runtime, strong CPU via XNNPACK, and straightforward Android APIs.

### 0) Lock down model invariants

- **Input tensor**: NCHW, 1×H×W (grayscale). H=224 (ViT‑Tiny) or 256 (MobileViT), as defined in `src/models.py`.
- **Normalization**: Same grayscale mean/std used in training.
- **Class labels**: Stable 344‑label list used in training/eval.

Why: Export and Android pre/post must match training exactly or accuracy will drop.

### 1) Prepare environments

- **Desktop/export**: Python 3.10+, PyTorch 2.3+, `timm`, your repo; ExecuTorch Python exporter installed.
- **Android**: Android Studio Flamingo+, SDK 34, NDK 26+, a device (Android 10+, arm64‑v8a) with ADB.

Why: Ensures compatible exporter/runtime and reproducible builds.

### 2) Load the trained checkpoint (desktop)

- Build model with exact `arch` and `num_classes`, load `state_dict` (strip `module.` if present), set `eval()`, on CPU.

Why: Export captures an inference‑mode, shape‑fixed graph.

### 3) Ensure an export‑friendly forward

- Single input `forward(input: Tensor)` where `input.shape == (1, 1, H, W)`.
- Avoid data‑dependent Python control flow, randomness, or `.item()` in forward.

Why: `torch.export` needs a static, capture‑friendly graph.

### 4) Export the model graph

```python
import torch
from torch.export import export

model.eval()
dummy = torch.randn(1, 1, H, W)  # H=224 for ViT, 256 for MobileViT
aten_graph = export(model, (dummy,), strict=True)
```

Why: Produces a stable program prepared for edge lowering; `strict=True` prevents silent fallbacks.

### 5) Lower to ExecuTorch and save `.pte`

```python
from executorch.exir import to_edge

edge = to_edge(aten_graph)
executorch_program = edge.to_executorch()
with open("quickdraw_fp32.pte", "wb") as f:
    f.write(executorch_program.buffer)
```

Why: `.pte` is the portable ExecuTorch program the Android runtime loads.

### 6) Verify parity offline (recommended)

- Compare logits/argmax of PyTorch vs exported program (using ExecuTorch host runner if available) on fixed inputs.

Why: Catches preprocessing or op lowering issues before device testing.

### 7) Create/prepare the Android app

- New project or demo; `minSdk >= 24`, `compileSdk 34`, Kotlin 1.9+.
- Restrict ABI to `arm64-v8a` in Gradle for smaller APKs.

Why: Ensures device compatibility and simpler native packaging.

### 8) Add ExecuTorch runtime

- Use official Gradle artifacts if provided for your release, or build AARs from source and add to `app/libs`.
- Add Gradle dependency for ExecuTorch runtime.

Why: Provides `Module`, `EValue`, `Tensor` APIs to load and run `.pte` models.

### 9) Package model and labels

- Place `quickdraw_fp32.pte` and `labels.txt` (344 lines) in `app/src/main/assets/`.

Why: Assets are easy to ship and copy to an accessible file path at runtime.

### 10) Implement preprocessing (Android)

- Capture a sketch via Canvas → `Bitmap` (H×W = 224 or 256, ARGB_8888).
- Convert to grayscale and normalize with training mean/std.
- Reorder to NCHW float32 buffer.

Why: Exact preprocessing parity is critical for accuracy.

### 11) Load and run inference (Android)

```kotlin
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor

val module = Module.load(modelFilePath) // path to copied .pte
val input = Tensor.fromBlob(floats, longArrayOf(1, 1, H.toLong(), W.toLong()))
val outputs = module.forward(EValue.from(input))
val logits = outputs[0].toTensor().dataAsFloatArray
```

Why: Loads the ExecuTorch program and performs forward inference with NCHW float32 input.

### 12) Minimal benchmarking

```kotlin
repeat(10) { module.forward(EValue.from(input)) } // warmup
val runs = 100
val t0 = android.os.SystemClock.elapsedRealtimeNanos()
repeat(runs) { module.forward(EValue.from(input)) }
val t1 = android.os.SystemClock.elapsedRealtimeNanos()
val avgMs = (t1 - t0) / 1e6 / runs
val imgsPerSec = 1000.0 / avgMs
```

Why: Warmup avoids first‑run overhead; average latency and throughput provide a stable baseline.

### 13) UI essentials

- Show top‑1 class (+ optional confidence), latency (ms), and a Benchmark button.
- Provide a Clear button and reuse last drawing for repeated runs.

Why: Improves manual validation and performance testing workflow.

### 14) On‑device accuracy validation

- Bundle a few known sketches and expected labels; verify predictions match desktop.

Why: Confirms full parity including Android preprocessing.

### 15) Performance methodology

- Airplane mode, fixed brightness, cool device, close background apps.
- Report median and p90 over ≥100 runs; note warmup and thread settings.

Why: Reduces variability and yields comparable numbers across devices.

### 16) Next steps (optional)

- NNAPI / QNN backends: Depending on ExecuTorch version and device support.
- Quantization: Export an INT8 `.pte` in parallel and A/B test.

### 17) Benchmark-Only Implementation (Quick Path)

For throughput measurement without drawing UI, follow these exact Android Studio steps:

#### **Step 17a: Add ExecuTorch Dependency**
In `app/build.gradle.kts` (working configuration):
```kotlin
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
}

android {
    namespace = "com.dkreinov.quickdraw"
    compileSdk = 34
    ndkVersion = "26.1.10909125"

    defaultConfig {
        applicationId = "com.dkreinov.quickdraw"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Only ship 64-bit (ExecuTorch provides arm64-v8a, x86_64)
        ndk { abiFilters += listOf("arm64-v8a") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // (Nice-to-have) don't compress .pte so loads are faster
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    aaptOptions { noCompress += "pte" } // for AGP < 8; remove if not recognized

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions { jvmTarget = "1.8" }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)

    // ExecuTorch (Java API + JNI) - WORKING VERSION
    implementation("org.pytorch:executorch-android:0.7.0")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
```

#### **Step 17b: Copy Assets**
1. Create `app/src/main/assets/` directory
2. Copy `quickdraw_fp32.pte` to `app/src/main/assets/`
3. Copy `labels.txt` to `app/src/main/assets/`

#### **Step 17c: Create Benchmark Activity**
Replace `MainActivity.kt` content:
```kotlin
package com.dkreinov.quickdraw

import android.app.Activity
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : Activity() {
    
    private fun assetFilePath(assetName: String): String {
        val outFile = File(filesDir, assetName)
        if (!outFile.exists()) {
            assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output -> 
                    input.copyTo(output) 
                }
            }
        }
        return outFile.absolutePath
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val TAG = "QuickDrawBenchmark"
        val H = 224  // ViT-Tiny input size
        val W = 224

        try {
            Log.i(TAG, "Starting ExecuTorch benchmark...")
            
            // 1) Load model
            val modelPath = assetFilePath("quickdraw_fp32.pte")
            Log.i(TAG, "Loading model from: $modelPath")
            val module = Module.load(modelPath)
            Log.i(TAG, "✓ Model loaded successfully")

            // 2) Create dummy input (zeros for benchmarking)
            val inputFloats = FloatArray(1 * 1 * H * W) { 0f }
            val input = Tensor.fromBlob(inputFloats, longArrayOf(1, 1, H.toLong(), W.toLong()))
            val eInput = EValue.from(input)
            Log.i(TAG, "✓ Input tensor created: [1, 1, $H, $W]")

            // 3) Warmup runs
            Log.i(TAG, "Warming up...")
            repeat(10) { module.forward(eInput) }
            Log.i(TAG, "✓ Warmup complete")

            // 4) Benchmark measurement
            val runs = 100
            val timesMs = DoubleArray(runs)
            
            Log.i(TAG, "Starting benchmark ($runs runs)...")
            val t0 = SystemClock.elapsedRealtimeNanos()
            
            repeat(runs) { i ->
                val start = SystemClock.elapsedRealtimeNanos()
                module.forward(eInput)
                val end = SystemClock.elapsedRealtimeNanos()
                timesMs[i] = (end - start) / 1e6 // Convert to milliseconds
            }
            
            val t1 = SystemClock.elapsedRealtimeNanos()

            // 5) Calculate statistics
            fun percentile(arr: DoubleArray, p: Double): Double {
                val sorted = arr.clone()
                sorted.sort()
                val index = (sorted.size - 1) * p
                val lower = sorted[index.toInt()]
                val upper = sorted[kotlin.math.ceil(index).toInt()]
                return lower + (upper - lower) * (index - kotlin.math.floor(index))
            }

            val avgMs = timesMs.average()
            val medianMs = percentile(timesMs, 0.5)
            val p90Ms = percentile(timesMs, 0.9)
            val totalMs = (t1 - t0) / 1e6
            val throughputImgsPerSec = 1000.0 / avgMs

            // 6) Log results
            Log.i(TAG, "🏁 BENCHMARK RESULTS:")
            Log.i(TAG, "   Runs: $runs")
            Log.i(TAG, "   Average latency: %.2f ms".format(avgMs))
            Log.i(TAG, "   Median latency: %.2f ms".format(medianMs))
            Log.i(TAG, "   P90 latency: %.2f ms".format(p90Ms))
            Log.i(TAG, "   Total time: %.1f ms".format(totalMs))
            Log.i(TAG, "   Throughput: %.2f images/sec".format(throughputImgsPerSec))
            
            // 7) Test inference output
            val output = module.forward(eInput)
            val logits = output[0].toTensor().dataAsFloatArray
            Log.i(TAG, "   Output shape: ${logits.size} classes")
            Log.i(TAG, "   Output range: [%.3f, %.3f]".format(logits.minOrNull(), logits.maxOrNull()))

        } catch (e: Exception) {
            Log.e(TAG, "❌ Benchmark failed: ${e.message}", e)
        }
    }
}
```

#### **Step 17d: Update AndroidManifest.xml**
Ensure `MainActivity` is the launcher:
```xml
<activity
    android:name=".MainActivity"
    android:exported="true">
    <intent-filter>
        <action android:name="android.intent.action.MAIN" />
        <category android:name="android.intent.category.LAUNCHER" />
    </intent-filter>
</activity>
```

#### **Step 17e: Build and Run**
1. **Connect Android device** with USB debugging enabled
2. **Build → Make Project** (Ctrl+F9)
3. **Run → Run 'app'** (Shift+F10)
4. **Open Logcat** (View → Tool Windows → Logcat)
5. **Filter by "QuickDrawBenchmark"** to see results

#### **Expected Output:**
```
QuickDrawBenchmark: Starting ExecuTorch benchmark...
QuickDrawBenchmark: ✓ Model loaded successfully
QuickDrawBenchmark: ✓ Input tensor created: [1, 1, 224, 224]
QuickDrawBenchmark: ✓ Warmup complete
QuickDrawBenchmark: Starting benchmark (100 runs)...
QuickDrawBenchmark: 🏁 BENCHMARK RESULTS:
QuickDrawBenchmark:    Runs: 100
QuickDrawBenchmark:    Average latency: 4123.80 ms
QuickDrawBenchmark:    Median latency: 3908.85 ms  
QuickDrawBenchmark:    P90 latency: 5470.29 ms
QuickDrawBenchmark:    Total time: 412379.8 ms
QuickDrawBenchmark:    Throughput: 0.24 images/sec
QuickDrawBenchmark:    Output shape: 344 classes
QuickDrawBenchmark:    Output range: [-2.131, 3.132]
```

**Performance Note:** Emulator results (~4 seconds/inference) are much slower than real device performance. Physical Android devices typically achieve 20-100ms latency.

#### **Troubleshooting:**
- **"Module not found"**: Check ExecuTorch dependency version
- **"Asset not found"**: Verify `.pte` and `.txt` files in `assets/` folder
- **Native crashes**: Ensure NDK version matches and `arm64-v8a` ABI filter set
- **No output**: Check device logs in Logcat, filter by your app package name

### 18) Deliverables checklist

- Desktop: `quickdraw_fp32.pte`, `labels.txt`, `export_fp32.py`, mean/std docs.
- Android: Benchmark app with ExecuTorch integration, asset loader, inference timing.
- Validation: Latency/throughput results logged to Android Logcat.


