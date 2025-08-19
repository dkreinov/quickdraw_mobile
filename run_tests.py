#!/usr/bin/env python3
"""
Test runner script for QuickDraw MobileViT quantization project.

This script provides a convenient way to run different types of tests
with proper configuration and reporting.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print("=" * len(description))
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, 
                              capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run tests for QuickDraw MobileViT project")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--file", help="Run specific test file")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    original_cwd = Path.cwd()
    
    try:
        project_root = project_root.resolve()
        print(f"Running tests from: {project_root}")
        
        success = True
        
        if args.file:
            # Run specific test file
            cmd = f"python -m pytest {args.file}"
            if args.verbose:
                cmd += " -v"
            success = run_command(cmd, f"Running {args.file}")
            
        elif args.unit:
            # Run unit tests only
            cmd = "python -m pytest tests/unit/"
            if args.verbose:
                cmd += " -v"
            if args.coverage:
                cmd = f"python -m pytest tests/unit/ --cov=src --cov-report=html --cov-report=term"
            success = run_command(cmd, "Running unit tests")
            
        elif args.integration:
            # Run integration tests only
            cmd = "python -m pytest tests/integration/"
            if args.verbose:
                cmd += " -v"
            success = run_command(cmd, "Running integration tests")
            
        else:
            # Run all tests
            cmd = "python -m pytest tests/"
            if args.verbose:
                cmd += " -v"
            if args.coverage:
                cmd = f"python -m pytest tests/ --cov=src --cov-report=html --cov-report=term"
            success = run_command(cmd, "Running all tests")
        
        if args.coverage and success:
            print("\nCoverage report generated in htmlcov/")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
