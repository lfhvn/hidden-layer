"""
Text processing utilities.
"""

from __future__ import annotations

import json
import re
from typing import Any


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length (default 100)
        suffix: Suffix to add if truncated (default "...")

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Number of words
    """
    return len(text.split())


def extract_json_from_text(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract JSON from text that may contain other content.

    Looks for JSON objects or arrays in the text, handling common
    cases like markdown code blocks.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON or None if not found
    """
    # Try to find JSON in markdown code block
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to find bare JSON object
    obj_pattern = r"\{[\s\S]*\}"
    matches = re.findall(obj_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try to find bare JSON array
    arr_pattern = r"\[[\s\S]*\]"
    matches = re.findall(arr_pattern, text)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace in text.

    - Removes leading/trailing whitespace
    - Collapses multiple spaces to single space
    - Collapses multiple newlines to double newline

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Strip leading/trailing
    text = text.strip()
    # Collapse multiple spaces
    text = re.sub(r" +", " ", text)
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent each line of text.

    Args:
        text: Text to indent
        spaces: Number of spaces to indent (default 4)

    Returns:
        Indented text
    """
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in text.split("\n"))


def extract_between(text: str, start: str, end: str) -> str | None:
    """Extract text between two markers.

    Args:
        text: Text to search
        start: Start marker
        end: End marker

    Returns:
        Text between markers, or None if not found
    """
    start_idx = text.find(start)
    if start_idx == -1:
        return None

    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        return None

    return text[start_idx:end_idx]
