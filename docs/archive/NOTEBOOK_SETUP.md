# Notebook Setup Guide

## Provider Configuration

All notebooks support multiple LLM providers. Here's how to configure them:

### Option 1: Use MLX (Apple Silicon - Recommended for Local)

If you have MLX models installed:

1. **Set as default** in `harness/defaults.py`:
```python
DEFAULT_PROVIDER = "mlx"
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Or your preferred MLX model
```

2. **Override in notebooks**:
```python
# At the top of any notebook
PROVIDER = "mlx"
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Then use in calls
result = run_strategy("single", "Your prompt", provider=PROVIDER, model=MODEL)
```

### Option 2: Use Ollama

If you have Ollama running:

1. **Set as default** in `harness/defaults.py`:
```python
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2:latest"  # Or your preferred Ollama model
```

2. **Override in notebooks**:
```python
PROVIDER = "ollama"
MODEL = "llama3.2:latest"
```

### Option 3: Use API Providers (Anthropic/OpenAI)

1. **Set environment variables**:
```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

2. **Use in notebooks**:
```python
# For Anthropic
PROVIDER = "anthropic"
MODEL = "claude-3-5-sonnet-20241022"

# For OpenAI
PROVIDER = "openai"
MODEL = "gpt-4o"
```

## Provider Detection (Automatic)

The quickstart notebook (`00_quickstart.ipynb`) automatically detects available providers. You can copy this pattern:

```python
import os
import shutil

# Detect available providers
providers = []

# Check for MLX (prioritized for local)
try:
    import mlx.core
    providers.append("mlx")
except ImportError:
    pass

# Check for Ollama
if shutil.which("ollama"):
    providers.append("ollama")

# Check for API providers
if os.getenv("ANTHROPIC_API_KEY"):
    providers.append("anthropic")
if os.getenv("OPENAI_API_KEY"):
    providers.append("openai")

# Use first available
if providers:
    PROVIDER = providers[0]
    MODEL = None  # Use default for provider
    print(f"Using provider: {PROVIDER}")
else:
    print("No providers detected!")
```

## Recommended Setup for MLX Users

Since you have MLX models, we recommend:

1. **Update `harness/defaults.py`**:
```python
DEFAULT_PROVIDER = "mlx"
DEFAULT_MODEL = None  # Will use harness defaults for MLX
```

2. **Run quickstart** to verify:
```bash
jupyter notebook communication/multi-agent/notebooks/00_quickstart.ipynb
```

3. **All other notebooks will now use MLX by default**

## Per-Notebook Override

If you want to use a different provider for a specific notebook, just set these variables after imports:

```python
# After imports, before any LLM calls
PROVIDER = "mlx"  # or "ollama", "anthropic", "openai"
MODEL = "your-model-name"  # or None for default

# Then use these variables in all calls
response = llm_call("Your prompt", provider=PROVIDER, model=MODEL)
result = run_strategy("debate", "Your prompt", provider=PROVIDER, model=MODEL)
```

## Checking Which Provider is Being Used

Most notebooks will print the provider and model being used. You can also check manually:

```python
from harness.defaults import DEFAULT_PROVIDER, DEFAULT_MODEL
print(f"Default provider: {DEFAULT_PROVIDER}")
print(f"Default model: {DEFAULT_MODEL}")
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'mlx'"
- MLX is not installed. Install with: `pip install mlx-lm`
- Or switch to Ollama: Set `DEFAULT_PROVIDER = "ollama"` in `harness/defaults.py`

### "Ollama not responding"
- Make sure Ollama is running: `ollama serve`
- Check if models are available: `ollama list`
- Or switch to MLX if available

### "API key not found"
- Set environment variable for your API provider
- Or use local providers (MLX/Ollama)
