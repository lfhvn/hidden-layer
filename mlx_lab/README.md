# MLX Lab - CLI Tool for MLX Model Management

**MLX Lab** is a command-line tool for managing MLX models, benchmarking performance, and integrating with Hidden Layer's research infrastructure.

## Features

- **Model Management**: Download, list, remove, and get info about MLX models
- **Performance Benchmarking**: Test model speed, memory usage, and latency on your hardware
- **Concept Browser**: Browse and manage concept vectors for activation steering
- **Config Integration**: Integrates with harness configuration and validates setup
- **Setup Wizard**: Interactive first-time setup

## Installation

MLX Lab is included when you install the hidden-layer package:

```bash
# From the repository root
pip install -e .

# Or with MLX support
pip install -e .[mlx]
```

This will install the `mlx-lab` command globally.

## Quick Start

### 1. Validate Setup

```bash
mlx-lab config validate
```

### 2. Download a Model

```bash
# Download recommended model
mlx-lab models download qwen3-8b-4bit

# Or download by full repo ID
mlx-lab models download mlx-community/Qwen3-8B-4bit
```

### 3. List Downloaded Models

```bash
mlx-lab models list
```

### 4. Test Model Performance

```bash
mlx-lab models test qwen3-8b-4bit
```

## Commands

### Setup

```bash
mlx-lab setup                    # Run interactive setup wizard
mlx-lab setup --non-interactive  # Run without prompts
```

### Model Management

```bash
mlx-lab models list                              # List downloaded models
mlx-lab models download <name>                   # Download a model
mlx-lab models remove <name>                     # Remove a model
mlx-lab models info <name>                       # Show model details
mlx-lab models test <name>                       # Test model performance
mlx-lab models test <name> --no-cache            # Re-run performance test
mlx-lab models compare <name1> <name2> [...]    # Compare multiple models
```

### Concept Management

```bash
mlx-lab concepts list            # List concept vectors
mlx-lab concepts info <name>     # Show concept details
```

### Configuration

```bash
mlx-lab config show              # Show current configuration
mlx-lab config validate          # Validate setup
```

## Recommended Models

MLX Lab includes shortcuts for recommended models:

| Shortcut | Full Repo ID | Description | RAM | Use Case |
|----------|--------------|-------------|-----|----------|
| `qwen3-8b-4bit` | `mlx-community/Qwen3-8B-4bit` | Fast, capable general-purpose model | ~5GB | Interactive experiments |
| `gpt-oss-20b-4bit` | `mlx-community/gpt-oss-20b-reasoning-4bit` | Powerful reasoning model | ~12GB | Complex reasoning tasks |
| `llama3.2-3b-4bit` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Lightweight, fast model | ~2GB | Quick testing |

## Performance Benchmarking

MLX Lab benchmarks models on **your specific hardware** to help you make informed choices.

### What is Measured

- **Speed**: Tokens per second (generation speed)
- **Memory**: Actual RAM usage during inference
- **Latency**: Time to first token (responsiveness)

### Example Output

```bash
$ mlx-lab models test qwen3-8b-4bit

Testing qwen3-8b-4bit...
Loading model...
Running performance test...
✅ Benchmark complete

Performance Test: mlx-community/Qwen3-8B-4bit
======================================================================
Speed:        47.3 tokens/sec
Memory:       5.2 GB
First token:  148 ms
Total time:   2.11 sec
Tested:       2025-11-06 10:30:15

✅ Fast - Good for interactive experiments
```

### Comparing Models

```bash
$ mlx-lab models compare qwen3-8b-4bit gpt-oss-20b-4bit

Comparing 2 models...

Model Performance Comparison
======================================================================

Model                          Speed           Memory     Latency
----------------------------------------------------------------------
Qwen3-8B-4bit                  47.3 tok/s      5.2 GB     148 ms
gpt-oss-20b-reasoning-4bit     15.1 tok/s      12.1 GB    312 ms

Tested on: 2025-11-06 10:35:22
```

### Benchmark Caching

Results are cached in `~/.mlx-lab/benchmarks.json`. Use `--no-cache` to re-run tests.

## Concept Vectors

MLX Lab integrates with the introspection research by managing concept vectors used in activation steering.

```bash
$ mlx-lab concepts list

Concept Vectors (in /home/user/hidden-layer/shared/concepts):
======================================================================
  • emotions_layer15
    Type: Concept Library (5 concepts)
    Size: 12.3 KB

  • honesty
    Dimensions: 4096
    Layer: 15
    Source: mlx-community/Qwen3-8B-4bit
    Size: 32.1 KB

======================================================================
Total: 2 concept vector files
```

## Integration with Harness

MLX Lab integrates seamlessly with the harness:

- **Uses same HuggingFace cache**: Models downloaded via `mlx-lab` are available to `mlx_lm.load()`
- **Reads harness config**: Shows default provider and model from `harness/defaults.py`
- **Validates environment**: Checks that harness, MLX, and mlx-lm are installed correctly

## File Locations

- **Models**: `~/.cache/huggingface/hub/` (standard HuggingFace cache)
- **Concepts**: `<repo>/shared/concepts/`
- **Benchmarks**: `~/.mlx-lab/benchmarks.json`
- **Config**: `<repo>/config/models.yaml`

## Requirements

- Python 3.10+
- MLX and mlx-lm (macOS with Apple Silicon only)
- harness package (automatically installed)

## Troubleshooting

### "MLX not installed"

Install MLX support:
```bash
pip install -e .[mlx]
```

Note: MLX only works on macOS with Apple Silicon (M1/M2/M3/M4).

### "No models found"

Download a model first:
```bash
mlx-lab models download qwen3-8b-4bit
```

### "Model not found" when testing

Make sure the model name matches a downloaded model:
```bash
mlx-lab models list  # See available models
```

## Development

MLX Lab is part of the Hidden Layer research infrastructure. See:
- `CLAUDE.md` - Lab development guide
- `docs/hardware/mlx-models.md` - Model selection guide
- `theory-of-mind/introspection/` - Activation steering research

## License

MIT License - Part of Hidden Layer research lab.
