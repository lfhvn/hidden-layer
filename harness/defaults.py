"""
Default configuration values for the harness.

Change DEFAULT_MODEL and DEFAULT_PROVIDER here to apply globally.
"""

# Default model and provider
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "gpt-oss:20b"  # Change this to your preferred model

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# Default debate parameters
DEFAULT_N_DEBATERS = 2
DEFAULT_N_ROUNDS = 1
