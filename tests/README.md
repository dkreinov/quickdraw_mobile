# Tests

This directory contains all tests for the QuickDraw MobileViT project, organized by functionality.

## Test Organization

```
tests/
├── run_tests.py            # Main test runner (run this!)
├── data/                   # Dataset and data pipeline tests
│   └── test_dataset.py     # Dataset loading, transforms, dataloaders
├── models/                 # Model architecture tests  
│   └── test_model.py       # Model creation, forward pass, integration
├── unit/                   # Unit tests for individual functions
├── integration/            # End-to-end integration tests
└── README.md              # This file
```

## Running Tests

### 🚀 Quick Start (Recommended)
```bash
# Run all tests with organized output
.venv/bin/python tests/run_tests.py

# Run specific category
.venv/bin/python tests/run_tests.py --category data
.venv/bin/python tests/run_tests.py --category models
```

### 🔧 Individual Tests
```bash
# Data tests (dataset, dataloaders, transforms)
.venv/bin/python tests/data/test_dataset.py

# Model tests (architecture, forward pass, integration)
.venv/bin/python tests/models/test_model.py
```

## Test Categories

### 📊 Data Tests (`tests/data/`)
Tests for dataset loading and preprocessing:
- **Dataset loading** - Parquet files, metadata validation
- **Transforms** - Image preprocessing, augmentation  
- **DataLoaders** - Batching, stratified splits
- **Class filtering** - Subset selection, sampling

### 🧠 Model Tests (`tests/models/`)
Tests for model architecture and integration:
- **Architecture** - Single-channel ViT creation
- **Weight initialization** - Pretrained vs random weights
- **Forward pass** - Input/output shapes, value ranges
- **Dataset integration** - Model + QuickDraw data compatibility

### 🔧 Unit Tests (`tests/unit/`)
Tests for individual components:
- Individual function testing
- Utility function validation
- Edge case handling
- No external dependencies

### 🔗 Integration Tests (`tests/integration/`)  
End-to-end workflow tests:
- Complete training pipelines
- Export and deployment workflows
- Multi-component interactions

## Test Data Requirements

Tests are designed to work with or without downloaded data:

**With data** (full testing):
```bash
# Download test data first
python scripts/download_quickdraw_direct.py --num-classes 3 --samples-per-class 100

# Then run tests
.venv/bin/python tests/run_tests.py
```

**Without data** (partial testing):
```bash
# Tests will skip data-dependent portions gracefully
.venv/bin/python tests/run_tests.py
```

## Test Design Principles

Our tests follow these principles:

- ✅ **Graceful degradation** - Skip tests when data unavailable
- ✅ **Clear reporting** - Descriptive output and error messages  
- ✅ **Fast execution** - Use small datasets for quick feedback
- ✅ **Self-contained** - Tests don't depend on each other
- ✅ **Well-organized** - Clear separation by functionality
- ✅ **Helpful errors** - Show exactly how to fix issues

## Sample Test Output

```
🧪 QuickDraw MobileViT Test Suite
==================================================

🧪 Running data tests...
==================================================

▶️  Running test_dataset...
✅ test_dataset PASSED

🧪 Running models tests...
==================================================

▶️  Running test_model...
✅ test_model PASSED

📊 TEST SUMMARY
======================================================================
DATA TESTS:
  ✅ PASS test_dataset

MODELS TESTS:
  ✅ PASS test_model

🎯 OVERALL: 2/2 tests passed
🎉 All tests passed!
```

## Writing New Tests

1. **Choose the right category**:
   - `data/` - Dataset, transforms, data loading
   - `models/` - Model architecture, training, inference
   - `unit/` - Individual functions, utilities
   - `integration/` - End-to-end workflows

2. **Follow naming convention**: `test_*.py`

3. **Include clear documentation** with usage examples

4. **Handle missing dependencies** gracefully with helpful messages

5. **Use the test runner** for consistent output formatting

## Adding to CI/CD

The test runner (`tests/run_tests.py`) is designed for automated environments:
- ✅ Clear exit codes (0 = success, 1 = failure)  
- ✅ No interactive prompts
- ✅ Structured output for parsing
- ✅ Graceful handling of missing data/dependencies