"""
Logging utility for the AI Game Player system.
"""
import logging
from typing import Optional
import os
from datetime import datetime

def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the project.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: logging.INFO)
        log_file: Optional path to log file. If None, logs to console only.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
default_logger = setup_logger(
    'ai_game_player',
    log_file=f'logs/game_player_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
) 