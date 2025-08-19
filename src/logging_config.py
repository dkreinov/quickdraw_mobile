"""
Logging configuration for the QuickDraw project.

Provides consistent logging across all modules with both file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        console_output: Whether to output to console
        
    Returns:
        Configured root logger
    """
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        )
        root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_and_print(message: str, logger_instance: Optional[logging.Logger] = None, level: str = "INFO"):
    """
    Log a message and also print it to console.
    
    Args:
        message: Message to log and print
        logger_instance: Optional logger instance
        level: Log level (INFO, WARNING, ERROR, etc.)
    """
    
    # Print to console
    print(message)
    
    # Log to file if logger provided
    if logger_instance:
        log_func = getattr(logger_instance, level.lower(), logger_instance.info)
        log_func(message)


# Alias for backward compatibility
setup_logging = setup_logger