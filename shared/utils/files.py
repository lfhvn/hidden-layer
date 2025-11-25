"""
File handling utilities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Optional yaml support
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    """Load JSON from a file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as JSON to a file.

    Args:
        data: Data to serialize
        path: Output file path
        indent: Indentation level (default 2)
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(path: str | Path) -> Any:
    """Load YAML from a file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML data

    Raises:
        ImportError: If PyYAML is not installed
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Any, path: str | Path) -> None:
    """Save data as YAML to a file.

    Args:
        data: Data to serialize
        path: Output file path

    Raises:
        ImportError: If PyYAML is not installed
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")

    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def read_text(path: str | Path) -> str:
    """Read text from a file.

    Args:
        path: Path to text file

    Returns:
        File contents as string
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(content: str, path: str | Path) -> None:
    """Write text to a file.

    Args:
        content: Text content to write
        path: Output file path
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
