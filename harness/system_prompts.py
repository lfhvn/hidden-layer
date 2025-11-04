"""
System Prompt Management - load and manage reusable system prompts
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Path to system prompts directory
SYSTEM_PROMPTS_DIR = Path(__file__).parent.parent.parent / "config" / "system_prompts"


class SystemPromptMetadata:
    """Metadata for a system prompt"""

    def __init__(self, data: dict):
        self.name = data.get("name", "unknown")
        self.description = data.get("description", "")
        self.author = data.get("author", "")
        self.created = data.get("created", "")
        self.version = data.get("version", "1.0")
        self.tags = data.get("tags", [])
        self.recommended_models = data.get("recommended_models", [])
        self.temperature_range = data.get("temperature_range", [0.7, 0.9])
        self.best_for = data.get("best_for", [])
        self.example_tasks = data.get("example_tasks", [])
        self.notes = data.get("notes", "")

    def __repr__(self):
        return f"SystemPromptMetadata(name='{self.name}', description='{self.description}')"


def load_system_prompt(name: str) -> str:
    """
    Load a system prompt from the config/system_prompts directory.

    Args:
        name: Name of the prompt file (without .md extension)

    Returns:
        The prompt text as a string

    Examples:
        # Load researcher prompt
        prompt = load_system_prompt("researcher")

        # Use with llm_call
        from harness import llm_call
        response = llm_call("Question", system_prompt=prompt)

        # Or pass name directly (will auto-load)
        response = llm_call("Question", system_prompt="researcher")
    """
    # Handle both "researcher" and "researcher.md"
    if name.endswith(".md"):
        name = name[:-3]

    prompt_path = SYSTEM_PROMPTS_DIR / f"{name}.md"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt '{name}' not found at {prompt_path}\n"
            f"Available prompts: {', '.join(list_system_prompts())}"
        )

    with open(prompt_path, "r") as f:
        return f.read()


def load_system_prompt_metadata(name: str) -> Optional[SystemPromptMetadata]:
    """
    Load metadata for a system prompt (from optional .yaml file).

    Args:
        name: Name of the prompt

    Returns:
        SystemPromptMetadata object if .yaml exists, None otherwise

    Examples:
        metadata = load_system_prompt_metadata("researcher")
        if metadata:
            print(f"Best for: {metadata.best_for}")
            print(f"Recommended temp: {metadata.temperature_range}")
    """
    if name.endswith(".md"):
        name = name[:-3]

    metadata_path = SYSTEM_PROMPTS_DIR / f"{name}.yaml"

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r") as f:
        data = yaml.safe_load(f)

    return SystemPromptMetadata(data)


def list_system_prompts() -> List[str]:
    """
    List all available system prompts.

    Returns:
        List of prompt names (without .md extension)

    Examples:
        prompts = list_system_prompts()
        print(f"Available: {', '.join(prompts)}")
    """
    if not SYSTEM_PROMPTS_DIR.exists():
        return []

    prompts = []
    for file in SYSTEM_PROMPTS_DIR.glob("*.md"):
        # Skip README
        if file.name.lower() != "readme.md":
            prompts.append(file.stem)

    return sorted(prompts)


def get_system_prompt_info(name: str) -> Dict:
    """
    Get complete information about a system prompt.

    Args:
        name: Name of the prompt

    Returns:
        Dictionary with prompt text and metadata (if available)

    Examples:
        info = get_system_prompt_info("researcher")
        print(f"Prompt: {info['text'][:100]}...")
        print(f"Description: {info['metadata'].description}")
    """
    text = load_system_prompt(name)
    metadata = load_system_prompt_metadata(name)

    return {"name": name, "text": text, "metadata": metadata, "has_metadata": metadata is not None}


def resolve_system_prompt(prompt: Optional[str]) -> Optional[str]:
    """
    Resolve a system prompt - either load from file or return as-is.

    This is used internally by llm_call to handle both:
    - Named prompts: "researcher" -> loads from file
    - Inline prompts: "You are..." -> returns as-is
    - None -> returns None

    Args:
        prompt: Either a prompt name, inline prompt text, or None

    Returns:
        The resolved prompt text, or None
    """
    if prompt is None:
        return None

    # If it's short and doesn't contain newlines, try loading as a named prompt
    if len(prompt) < 100 and "\n" not in prompt:
        try:
            return load_system_prompt(prompt)
        except FileNotFoundError:
            # Not a file, treat as inline prompt
            pass

    # Return as-is (inline prompt)
    return prompt


# Convenience function for interactive use
def show_prompt(name: str):
    """
    Print a system prompt to console (useful in notebooks).

    Args:
        name: Name of the prompt to display

    Examples:
        show_prompt("researcher")
    """
    info = get_system_prompt_info(name)

    print(f"=== System Prompt: {name} ===\n")
    print(info["text"])

    if info["metadata"]:
        meta = info["metadata"]
        print("\n=== Metadata ===")
        print(f"Description: {meta.description}")
        if meta.tags:
            print(f"Tags: {', '.join(meta.tags)}")
        if meta.recommended_models:
            print(f"Recommended models: {', '.join(meta.recommended_models)}")
        if meta.temperature_range:
            print(f"Temperature range: {meta.temperature_range}")
        if meta.best_for:
            print("\nBest for:")
            for use in meta.best_for:
                print(f"  - {use}")
