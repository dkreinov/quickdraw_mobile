#!/usr/bin/env python3
"""
Test runner for QuickDraw MobileViT project.

This script runs all tests in the appropriate order and provides a summary.
Organizes tests by category and shows clear results.

Usage:
    .venv/bin/python tests/run_tests.py
    .venv/bin/python tests/run_tests.py --category data
    .venv/bin/python tests/run_tests.py --category models
    .venv/bin/python tests/run_tests.py --category all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_test_file(test_file: Path) -> Tuple[bool, str]:
    """
    Run a single test file and return result.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        (success, output): Test result and output
    """
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent  # Run from project root
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
        
    except Exception as e:
        return False, f"Error running test: {e}"


def run_category_tests(category: str) -> List[Tuple[str, bool, str]]:
    """
    Run tests for a specific category.
    
    Args:
        category: Test category (data, models, unit, integration)
        
    Returns:
        List of (test_name, success, output) tuples
    """
    
    test_dir = Path(__file__).parent / category
    results = []
    
    if not test_dir.exists():
        print(f"⚠️  No tests found for category: {category}")
        return results
    
    # Find all test files
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print(f"⚠️  No test files found in {test_dir}")
        return results
    
    print(f"\n🧪 Running {category} tests...")
    print("=" * 50)
    
    for test_file in sorted(test_files):
        test_name = test_file.stem
        print(f"\n▶️  Running {test_name}...")
        
        success, output = run_test_file(test_file)
        results.append((test_name, success, output))
        
        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
            print(f"Output:\n{output}")
    
    return results


def print_summary(all_results: List[Tuple[str, List[Tuple[str, bool, str]]]]):
    """Print a summary of all test results."""
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    total_passed = 0
    
    for category, results in all_results:
        if not results:
            continue
            
        passed = sum(1 for _, success, _ in results if success)
        failed = len(results) - passed
        
        total_tests += len(results)
        total_passed += passed
        
        print(f"\n{category.upper()} TESTS:")
        for test_name, success, _ in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        print(f"  Summary: {passed}/{len(results)} passed")
    
    print(f"\n🎯 OVERALL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_tests - total_passed} tests failed")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run QuickDraw project tests")
    parser.add_argument(
        "--category", 
        choices=["data", "models", "unit", "integration", "all"],
        default="all",
        help="Test category to run (default: all)"
    )
    
    args = parser.parse_args()
    
    print("🧪 QuickDraw MobileViT Test Suite")
    print("=" * 50)
    
    if args.category == "all":
        categories = ["data", "models", "unit", "integration"]
    else:
        categories = [args.category]
    
    all_results = []
    
    for category in categories:
        results = run_category_tests(category)
        all_results.append((category, results))
    
    return print_summary(all_results)


if __name__ == "__main__":
    exit(main())
