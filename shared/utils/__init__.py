"""
Shared utilities for Hidden Layer projects.

Common functionality used across research projects:
- File handling
- JSON/YAML utilities
- Text processing
- Timing and profiling
- Logging
"""

from __future__ import annotations

from shared.utils.timing import Timer, timed
from shared.utils.files import ensure_dir, load_json, load_yaml, save_json, save_yaml
from shared.utils.text import truncate_text, word_count, extract_json_from_text
from shared.utils.logging import (
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
