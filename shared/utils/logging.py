"""
Logging utilities for Hidden Layer projects.

Provides standardized logging configuration across all projects.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

# Default format for logs
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
SIMPLE_FORMAT = "%(levelname)-8s | %(message)s"
DEBUG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


def setup_logging(
    level: int | str = logging.INFO,
    format_style: Literal["default", "simple", "debug"] = "default",
    log_file: str | Path | None = None,
    name: str | None = None,
) -> logging.Logger:
    """Configure logging for a Hidden Layer project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style - "default", "simple", or "debug"
        log_file: Optional file path to log to
        name: Logger name (None for root logger)

    Returns:
        Configured logger

    Usage:
        from shared.utils.logging import setup_logging

        # Simple setup
        logger = setup_logging()
        logger.info("Starting experiment")

        # Debug mode with file logging
        logger = setup_logging(
            level="DEBUG",
            format_style="debug",
            log_file="experiment.log",
            name="my_experiment"
        )
    """
    # Get format string
    if format_style == "simple":
        fmt = SIMPLE_FORMAT
    elif format_style == "debug":
        fmt = DEBUG_FORMAT
    else:
        fmt = DEFAULT_FORMAT

    # Parse level if string
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(DEBUG_FORMAT))
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Usage:
        from shared.utils.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Processing data...")
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for temporary logging level changes.

    Usage:
        logger = get_logger(__name__)

        with LogContext(logger, logging.DEBUG):
            logger.debug("This will be logged")
        # Back to original level
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self.logger = logger
        self.new_level = level
        self.original_level = logger.level

    def __enter__(self) -> logging.Logger:
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, *args: object) -> None:
        self.logger.setLevel(self.original_level)


def silence_logger(name: str) -> None:
    """Silence a specific logger (useful for noisy libraries).

    Args:
        name: Logger name to silence
    """
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)


def configure_library_logging(quiet: bool = True) -> None:
    """Configure logging for common noisy libraries.

    Args:
        quiet: If True, silence noisy library loggers
    """
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "openai",
        "anthropic",
    ]

    level = logging.WARNING if quiet else logging.INFO
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(level)
