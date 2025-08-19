# Tests

This directory contains the test suite for the QuickDraw MobileViT quantization project.

## Structure

```
tests/
├── __init__.py              # Tests package
├── unit/                    # Unit tests
│   ├── __init__.py
│   └── test_logging.py     # Logging configuration tests
├── integration/            # Integration tests (future)
└── README.md               # This file
```

## Running Tests

### All tests with pytest
```bash
python -m pytest tests/ -v
```

### Specific test file
```bash
python -m pytest tests/unit/test_logging.py -v
```

### Individual test directly
```bash
python tests/unit/test_logging.py
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Fast execution
- No external dependencies (mock external services)

### Integration Tests (`tests/integration/`)
- Test component interactions
- May require test data or external services
- Slower execution

## Writing Tests

1. Place unit tests in `tests/unit/`
2. Name test files with `test_` prefix
3. Use descriptive test method names
4. Include docstrings for test classes and methods
5. Follow the AAA pattern: Arrange, Act, Assert

## Test Artifacts

Test outputs and artifacts are gitignored but the test files themselves are tracked:
- `test_*.log` - Test-specific log files
- `temp_*` - Temporary test files
- `__pycache__/` - Python cache files
- Coverage reports and other test outputs
