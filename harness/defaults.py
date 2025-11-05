"""
Default configuration values for the harness.

The harness auto-detects available providers and prioritizes local models (MLX, Ollama).
To override, change DEFAULT_PROVIDER and DEFAULT_MODEL below.
"""

import shutil

# Auto-detect best available provider (prioritize local models)
def _detect_default_provider():
    """Detect and return best available provider."""
    # 1. Try MLX (Apple Silicon) first - best for local inference
    try:
        import mlx.core
        return "mlx", None  # None means use provider's default model
    except ImportError:
        pass

    # 2. Try Ollama second - good for local inference
    if shutil.which("ollama"):
        return "ollama", "llama3.2:latest"

    # 3. Fall back to ollama anyway (user can configure)
    return "ollama", "gpt-oss:20b"

# Default model and provider (auto-detected or set manually)
DEFAULT_PROVIDER, DEFAULT_MODEL = _detect_default_provider()

# To manually override, uncomment and set these:
# DEFAULT_PROVIDER = "mlx"  # or "ollama", "anthropic", "openai"
# DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # or your preferred model

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# Default debate parameters
DEFAULT_N_DEBATERS = 2
DEFAULT_N_ROUNDS = 1
