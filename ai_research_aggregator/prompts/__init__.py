"""
Prompt template loader for the AI Research Aggregator.

Loads prompt templates from .txt files in this directory, with optional
user overrides from the config directory.
"""

import os
from functools import lru_cache
from typing import Optional

PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# User override directory
USER_PROMPTS_DIR = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "ai-research-aggregator",
    "prompts",
)


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """
    Load a prompt template by name.

    Checks user override directory first, then falls back to built-in prompts.
    The persona is automatically injected into templates containing {persona}.

    Args:
        name: Prompt name without extension (e.g., "ranking", "opportunity").

    Returns:
        The prompt template string.
    """
    filename = f"{name}.txt"

    # Check user override first
    user_path = os.path.join(USER_PROMPTS_DIR, filename)
    if os.path.exists(user_path):
        with open(user_path) as f:
            return f.read()

    # Fall back to built-in
    builtin_path = os.path.join(PROMPTS_DIR, filename)
    if os.path.exists(builtin_path):
        with open(builtin_path) as f:
            return f.read()

    raise FileNotFoundError(
        f"Prompt template '{name}' not found in {USER_PROMPTS_DIR} or {PROMPTS_DIR}"
    )


def get_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt template and format it with the given kwargs.

    The {persona} placeholder is automatically resolved from persona.txt.
    All other placeholders must be provided as kwargs.

    Args:
        name: Prompt name (e.g., "ranking").
        **kwargs: Format variables for the template.

    Returns:
        The fully formatted prompt string.
    """
    template = load_prompt(name)

    # Auto-inject persona if the template uses it
    if "{persona}" in template and "persona" not in kwargs:
        kwargs["persona"] = load_prompt("persona")

    return template.format(**kwargs)
