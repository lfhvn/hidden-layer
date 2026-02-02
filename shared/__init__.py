"""
Shared resources for Hidden Layer projects.

Includes:
- Concept vectors
- Datasets
- Common utilities
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export utilities for convenience
from shared.utils import (
    Timer,
    timed,
    ensure_dir,
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    truncate_text,
    word_count,
    extract_json_from_text,
    setup_logging,
    get_logger,
    LogContext,
    silence_logger,
    configure_library_logging,
)

__all__ = [
    # Timing
    "Timer",
    "timed",
    # File handling
    "ensure_dir",
    "load_json",
    "save_json",
    "load_yaml",
    "save_yaml",
    # Text processing
    "truncate_text",
    "word_count",
    "extract_json_from_text",
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "silence_logger",
    "configure_library_logging",
]
