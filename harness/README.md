# Hidden Layer Harness

**Core infrastructure for LLM research**

A standalone library providing unified abstractions for working with language models across providers, with built-in experiment tracking, evaluation utilities, and reproducibility features.

## Features

- **Unified LLM Provider**: Seamlessly switch between Ollama, MLX, Anthropic Claude, OpenAI GPT
- **Experiment Tracking**: Automatic logging, metrics, reproducibility
- **Evaluation Utilities**: Exact match, keyword match, LLM-as-judge, benchmarks
- **Model Configuration**: Named presets, system prompts, hyperparameter management
- **Benchmark Integration**: Load and evaluate on standard datasets

## Quick Start

```python
from harness import llm_call, get_tracker, ExperimentConfig

# Simple LLM call (any provider)
response = llm_call(
    "What is the capital of France?",
    provider="ollama",
    model="llama3.2:latest"
)

print(response.text)

# With experiment tracking
config = ExperimentConfig(
    experiment_name="capitals_test",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

tracker = get_tracker()
tracker.start_experiment(config)

response = llm_call(
    "What is the capital of France?",
    provider=config.provider,
    model=config.model
)

tracker.log_result(response.text)
tracker.finish_experiment()
```

## Supported Providers

- **Ollama**: Local models via Ollama
- **MLX**: Apple Silicon optimized (MLX framework)
- **Anthropic**: Claude models via API
- **OpenAI**: GPT models via API

## Installation

### As part of Hidden Layer lab

Already included - import directly:

```python
from harness import llm_call
```

### Standalone installation (future)

```bash
pip install hidden-layer-harness
```

## Core Modules

### LLM Provider (`llm_provider.py`)

Unified interface to any LLM:

```python
from harness import llm_call

# Local
response = llm_call("Question?", provider="ollama", model="llama3.2:latest")

# API
response = llm_call("Question?", provider="anthropic", model="claude-3-5-sonnet-20241022")
```

### Experiment Tracker (`experiment_tracker.py`)

Automatic experiment logging:

```python
from harness import get_tracker, ExperimentConfig

config = ExperimentConfig(experiment_name="my_experiment")
tracker = get_tracker()
tracker.start_experiment(config)

# ... run experiments ...

tracker.finish_experiment()

# Outputs to: experiments/my_experiment_{timestamp}_{hash}/
```

### Evaluation (`evals.py`)

Built-in evaluation functions:

```python
from harness import exact_match, llm_judge

score = exact_match(output="Paris", expected="Paris")  # 1.0
score = llm_judge(output="Paris is capital", expected="Paris")  # Uses LLM
```

### Benchmarks (`benchmarks.py`)

Load standard benchmarks:

```python
from harness import load_benchmark, BENCHMARKS

# List available
print(BENCHMARKS.keys())

# Load one
tombench = load_benchmark('tombench', split='test')
```

### Model Configuration (`model_config.py`)

Named model presets:

```python
from harness import get_model_config

config = get_model_config("gpt-oss-20b-reasoning")
# Returns: {provider: "ollama", model: "...", temperature: 0.1, ...}
```

### System Prompts (`system_prompts.py`)

Reusable system prompts:

```python
from harness import load_system_prompt

prompt = load_system_prompt("researcher")
# Returns persona/instructions for frontier researcher
```

## Philosophy

**Provider Agnostic**: Work with any LLM provider - local or API

**Reproducible**: Every experiment is logged and rerunnable

**Extensible**: Easy to add new providers, evaluations, benchmarks

**Minimal**: No heavy dependencies, simple to understand

## Use Cases

1. **Research Projects**: Use across all Hidden Layer projects
2. **Rapid Prototyping**: Switch providers easily during development
3. **Benchmarking**: Compare models across providers
4. **Production**: Track experiments in research → production pipeline

## Documentation

- **LLM Providers**: [/docs/infrastructure/llm-providers.md](../docs/infrastructure/llm-providers.md)
- **Experiment Tracking**: [/docs/infrastructure/experiment-tracking.md](../docs/infrastructure/experiment-tracking.md)
- **Benchmarks**: [/docs/workflows/benchmarking.md](../docs/workflows/benchmarking.md)

## Development

Adding a new provider:

```python
# In llm_provider.py

class MyNewProvider(LLMProvider):
    def __call__(self, prompt, **kwargs):
        # Implementation
        return LLMResponse(...)

# Register
PROVIDERS["mynew"] = MyNewProvider
```

## License

MIT - Can be used independently of Hidden Layer projects

## Status

**Version**: 0.2.0

**Stability**: Used in production research at Hidden Layer

**Roadmap**:
- [ ] PyPI package
- [ ] Additional providers (Cohere, Gemini, etc.)
- [ ] Streaming support for all providers
- [ ] Advanced caching
- [ ] Cost optimization tools

---

**Part of [Hidden Layer Research Lab](../)** - Can be used standalone
