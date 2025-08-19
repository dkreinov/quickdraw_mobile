#!/usr/bin/env python3
"""Unit tests for logging configuration."""

import sys
import unittest
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from logging_config import get_logger, log_and_print, setup_logger


class TestLoggingConfiguration(unittest.TestCase):
    """Test cases for logging configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_file = os.path.join(self.temp_dir, "test.log")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
        os.rmdir(self.temp_dir)
    
    def test_get_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = get_logger(__name__)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, __name__)
    
    def test_setup_logger_with_custom_file(self):
        """Test logger setup with custom log file."""
        logger = setup_logger(
            log_file=self.test_log_file,
            console_output=False  # Disable console for clean test output
        )
        
        # Test logging
        test_message = "Test log message"
        logger.info(test_message)
        
        # Verify file was created and contains message
        self.assertTrue(os.path.exists(self.test_log_file))
        
        with open(self.test_log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
            self.assertIn("INFO", log_content)
    
    def test_log_and_print_function(self):
        """Test the log_and_print utility function."""
        logger = setup_logger(
            log_file=self.test_log_file,
            console_output=False
        )
        
        test_message = "Test log and print message"
        
        # Capture stdout to verify print functionality
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            log_and_print(test_message, logger_instance=logger)
            printed_output = captured_output.getvalue().strip()
        finally:
            sys.stdout = sys.__stdout__
        
        # Verify message was printed
        self.assertEqual(printed_output, test_message)
        
        # Verify message was logged
        with open(self.test_log_file, 'r') as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)


def test_logging_integration():
    """Integration test for logging functionality."""
    print("Running logging integration test...")
    
    logger = get_logger(__name__)
    
    # Test basic logging
    logger.info("Integration test: info message")
    logger.warning("Integration test: warning message")
    
    # Test log_and_print function
    log_and_print("Integration test: log and print message", logger_instance=logger)
    
    print("✓ Logging integration test completed successfully!")
    print("Check logs/quickdraw.log for the logged messages")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n" + "="*50)
    test_logging_integration()
