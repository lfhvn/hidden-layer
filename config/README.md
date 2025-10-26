# Model Configuration Management

This directory contains model configuration presets that make it easy to manage different model setups and hyperparameters.

## Quick Start

### List available configurations

```bash
python code/cli.py --list-configs
```

### Use a configuration

```bash
# Use a preset config
python code/cli.py "Your question here" --config gpt-oss-20b-reasoning

# With debate strategy
python code/cli.py "Your question" --config gpt-oss-20b-creative --strategy debate --n-debaters 3
```

### Override config parameters

```bash
# Use config but override temperature
python code/cli.py "Question" --config gpt-oss-20b-reasoning --temperature 0.9

# Override thinking budget
python code/cli.py "Question" --config gpt-oss-20b-reasoning --thinking-budget 3000
```

## Configuration File Format

Configurations are stored in `config/models.yaml`:

```yaml
my-config-name:
  provider: ollama           # Required: "ollama", "mlx", "anthropic", "openai"
  model: gpt-oss:20b        # Required: Model identifier
  temperature: 0.7           # Optional: Sampling temperature (0.0-1.0)
  max_tokens: 2048          # Optional: Maximum output tokens
  thinking_budget: 2000      # Optional: Reasoning token budget
  num_ctx: 4096             # Optional: Context window size
  top_k: 40                 # Optional: Top-K sampling
  top_p: 0.9                # Optional: Top-P (nucleus) sampling
  repeat_penalty: 1.1        # Optional: Repetition penalty
  seed: 42                  # Optional: Random seed for reproducibility
  system_prompt: "Custom..." # Optional: System prompt for this config
  description: "Description" # Optional: Human-readable description
  tags:                     # Optional: Tags for organization
    - reasoning
    - large
```

## Built-in Configurations

### Reasoning Models

- **gpt-oss-20b-reasoning**: Extended thinking budget for complex problems
- **gpt-oss-20b-precise**: Low temperature, deterministic for factual tasks

### Creative Models

- **gpt-oss-20b-creative**: High temperature, diverse outputs

### Fast Models

- **llama3.2-fast**: Small, fast model for quick iteration

### API Models

- **claude-sonnet**: High-quality Claude 3.5 Sonnet
- **claude-haiku**: Fast, cost-effective Claude 3.5 Haiku

## Using in Python/Notebooks

```python
from harness import get_model_config, run_strategy, llm_call

# Load a configuration
config = get_model_config("gpt-oss-20b-reasoning")

# Use with run_strategy
result = run_strategy(
    "single",
    "Your question",
    **config.to_kwargs()
)

# Use with llm_call
response = llm_call(
    "Your prompt",
    **config.to_kwargs()
)

# Override specific parameters
kwargs = config.to_kwargs()
kwargs['temperature'] = 0.9
result = run_strategy("single", "Question", **kwargs)
```

## Creating Custom Configurations

### Method 1: Edit YAML file

Add your configuration to `config/models.yaml`:

```yaml
my-custom-config:
  provider: ollama
  model: my-model:latest
  temperature: 0.8
  thinking_budget: 1500
  description: My custom configuration for specific tasks
  tags:
    - custom
```

### Method 2: Programmatically

```python
from harness import ModelConfig, get_config_manager

# Create a new configuration
config = ModelConfig(
    name="my-custom-config",
    provider="ollama",
    model="my-model:latest",
    temperature=0.8,
    thinking_budget=1500,
    description="My custom configuration",
    tags=["custom"]
)

# Save it
manager = get_config_manager()
manager.add(config, save=True)
```

## Use Cases

### Rapid Experimentation

Use presets to quickly switch between different setups:

```bash
# Try reasoning approach
python code/cli.py "Question" --config gpt-oss-20b-reasoning

# Try creative approach
python code/cli.py "Question" --config gpt-oss-20b-creative

# Try fast iteration
python code/cli.py "Question" --config llama3.2-fast
```

### Reproducible Research

Lock in configurations for experiments:

```python
# experiments/my_experiment.py
from harness import get_model_config, run_strategy

# Use consistent configuration
config = get_model_config("gpt-oss-20b-precise")  # seed=42, temp=0.3
for task in tasks:
    result = run_strategy("single", task, **config.to_kwargs())
```

### Task-Specific Configs

Create specialized configurations for different task types:

```yaml
coding-assistant:
  provider: ollama
  model: gpt-oss:20b
  temperature: 0.5
  system_prompt: "You are an expert programmer..."
  thinking_budget: 2500

creative-writer:
  provider: ollama
  model: gpt-oss:20b
  temperature: 0.9
  top_p: 0.95
  system_prompt: "You are a creative storyteller..."

math-solver:
  provider: ollama
  model: gpt-oss:20b
  temperature: 0.3
  thinking_budget: 3000
  system_prompt: "Solve step-by-step, showing all work..."
```

## Best Practices

1. **Name configs descriptively**: `model-task-style` (e.g., `gpt-oss-20b-reasoning`)
2. **Use tags for organization**: Group by model size, cost, speed, etc.
3. **Document with descriptions**: Explain when to use each config
4. **Set reasonable defaults**: Start conservative, adjust based on results
5. **Version control your configs**: Commit `models.yaml` to git
6. **Use different configs for different strategies**: Debate may need different settings than single-model

## Configuration Priority

When using `--config` with CLI flags, the priority is:

1. **CLI flags** (highest priority)
2. **Config file values**
3. **Default values** (lowest priority)

Example:
```bash
# Config has temperature=0.7, but CLI overrides to 0.9
python code/cli.py "Question" --config gpt-oss-20b-reasoning --temperature 0.9
```

## Tips

- **Thinking budget**: Start with 1000-2000 for reasoning tasks, increase if needed
- **Temperature**: 0.3-0.5 for factual, 0.7-0.8 for balanced, 0.9+ for creative
- **Context window**: Increase `num_ctx` for tasks requiring long context
- **Seed**: Set for reproducibility in experiments

## Troubleshooting

**Config not found?**
```bash
python code/cli.py --list-configs  # Check available configs
```

**Config not loading?**
- Check YAML syntax in `config/models.yaml`
- Ensure file is in the correct location
- Check for proper indentation (YAML is whitespace-sensitive)

**Parameters not working?**
- Some parameters are provider-specific (e.g., `thinking_budget` for Ollama)
- Check model documentation for supported parameters
