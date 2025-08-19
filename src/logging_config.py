"""
Centralized logging configuration for the QuickDraw MobileViT quantization project.

This module provides a consistent logging setup that can be used across all modules.
It supports both file and console logging with appropriate formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        level: Logging level (default: INFO)
        log_file: Custom log file path. If None, uses 'logs/quickdraw.log'
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            log_file = "logs/quickdraw.log"
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get or create a logger with the standard configuration.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
        
    Returns:
        Logger instance
    """
    return setup_logger(name)


# Default logger for this module
logger = get_logger(__name__)


def log_and_print(message: str, level: int = logging.INFO, logger_instance: Optional[logging.Logger] = None):
    """
    Log a message and also print it to screen for user visibility.
    
    This function is useful when you want both logging (for debugging/audit trail)
    and immediate screen output (for user feedback).
    
    Args:
        message: Message to log and print
        level: Logging level
        logger_instance: Logger to use. If None, uses default logger
    """
    if logger_instance is None:
        logger_instance = logger
        
    # Log the message
    logger_instance.log(level, message)
    
    # Also print to screen (remove any emoji/icons for clean output)
    clean_message = message
    print(clean_message)
