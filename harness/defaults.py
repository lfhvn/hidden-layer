"""
Default configuration values for the harness.

The baseline is Inkling 2 Mini on a local OpenAI-compatible server (such as
mlx_lm.server). Override with environment variables:

    HARNESS_PROVIDER   "local", "mlx", "ollama", "anthropic" or "openai"
    HARNESS_MODEL      model name for that provider
    HARNESS_LOCAL_URL  server URL for the "local" provider
"""

import os

BASELINE_MODEL = "pipenetwork/Inkling-2-Mini-MLX-bf16"

DEFAULT_PROVIDER = os.environ.get("HARNESS_PROVIDER", "local")
# The baseline model only makes sense on the local server; other providers
# need HARNESS_MODEL or an explicit model.
DEFAULT_MODEL = os.environ.get("HARNESS_MODEL") or (BASELINE_MODEL if DEFAULT_PROVIDER == "local" else None)
LOCAL_URL = os.environ.get("HARNESS_LOCAL_URL", "http://127.0.0.1:8080/v1")

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# Default debate parameters
DEFAULT_N_DEBATERS = 2
DEFAULT_N_ROUNDS = 1
