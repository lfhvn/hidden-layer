# System Prompts Library

This directory contains reusable system prompts that define personas and behaviors for LLMs across all Hidden Layer experiments.

## Quick Start

### Use in Python/Notebooks

```python
from harness import llm_call, load_system_prompt

# Method 1: Reference by name (auto-loads from this directory)
response = llm_call(
    "Design a novel architecture",
    system_prompt="researcher"
)

# Method 2: Load explicitly and inspect
researcher = load_system_prompt("researcher")
print(researcher)  # View the full prompt
response = llm_call("Your task", system_prompt=researcher)

# Method 3: Use with model configs
from harness import get_model_config, run_strategy

config = get_model_config("claude-researcher")  # Has researcher prompt built-in
result = run_strategy("single", "Task", **config.to_kwargs())
```

### Use from CLI

```bash
# Use a named system prompt
python code/cli.py "Your question" --system-prompt researcher

# Use with a model config that includes a system prompt
python code/cli.py "Your question" --config claude-researcher

# Override system prompt in a config
python code/cli.py "Your question" --config claude-sonnet --system-prompt researcher
```

## Available System Prompts

### `researcher.md`
**Purpose**: Frontier AI research and paradigm-shifting thinking

**Best for**:
- Novel architecture design
- Theoretical framework development
- Research hypothesis generation
- Paradigm-level analysis
- Scientific writing and proposals

**Example tasks**:
- "Propose a new paradigm beyond transformer scaling laws"
- "Design an architecture unifying active inference and predictive coding"
- "Generate testable hypotheses about emergent reasoning"

### `default.md`
**Purpose**: General-purpose helpful assistant

**Best for**:
- Standard Q&A
- General reasoning tasks
- Baseline comparisons
- Quick iterations

**Example tasks**:
- General questions
- Simple reasoning
- Factual queries

## Creating Custom System Prompts

### Method 1: Create a Markdown File

Create a new `.md` file in this directory:

```markdown
# My Custom Prompt

You are a [role description].

## Key Behaviors
- Behavior 1
- Behavior 2

## Expertise
- Domain 1
- Domain 2

## Output Format
Describe how to structure responses.
```

Then use it:
```python
response = llm_call("Task", system_prompt="my_custom_prompt")
```

### Method 2: Create with Metadata (YAML)

Create a `.yaml` file alongside your `.md` file:

```yaml
name: my_custom_prompt
description: Brief description of when to use this prompt
author: Your Name
created: 2025-01-15
tags:
  - custom
  - research
  - creative
recommended_models:
  - claude-3-5-sonnet-20241022
  - gpt-4o
temperature_range: [0.7, 0.9]
best_for:
  - Task type 1
  - Task type 2
```

### Method 3: Inline (No File)

```python
custom_prompt = """
You are an expert in [domain].
Your goal is to [objective].
"""

response = llm_call("Task", system_prompt=custom_prompt)
```

## Integration with Model Configs

Add system prompts to your model configurations in `config/models.yaml`:

```yaml
claude-researcher:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096
  system_prompt: researcher  # References researcher.md
  description: Claude Sonnet with AI researcher persona

gpt-creative-writer:
  provider: openai
  model: gpt-4o
  temperature: 0.9
  system_prompt: creative_writer
  description: GPT-4o optimized for creative writing
```

## Best Practices

### 1. Name Files Descriptively
Use lowercase with underscores: `researcher.md`, `code_reviewer.md`, `creative_writer.md`

### 2. Structure Prompts Clearly
- **Role definition**: Who/what the model is
- **Mission/objectives**: What it should accomplish
- **Expertise areas**: What it knows deeply
- **Method/approach**: How it should think
- **Output format**: How to structure responses

### 3. Use Metadata Files
Create matching `.yaml` files to document:
- When to use the prompt
- Recommended models/temperatures
- Author and versioning info
- Tags for organization

### 4. Version Control
Commit all system prompts to git for reproducibility across experiments.

### 5. Test Variations
Create variations for different use cases:
- `researcher_concise.md` vs `researcher_detailed.md`
- `coder_python.md` vs `coder_javascript.md`

### 6. Document Examples
Include example tasks in the README or YAML metadata.

## How System Prompts Work

System prompts are passed differently to each provider:

- **Anthropic (Claude)**: Uses `system` parameter in API call
- **OpenAI (GPT)**: Added as first `system` message in chat
- **Ollama**: Prepended to user prompt with formatting
- **MLX**: Prepended to user prompt (model-dependent)

The harness handles these differences automatically.

## Tips for Effective System Prompts

### Do:
- Be specific about role and expertise
- Define clear output formats
- Specify reasoning methods
- Include example invocations
- Use structured markdown

### Don't:
- Make prompts unnecessarily long
- Include task-specific instructions (those go in the user prompt)
- Contradict model capabilities
- Use conflicting directives

### Temperature Guidelines

Different prompts work better at different temperatures:

- **Researcher**: 0.7-0.9 (creative hypotheses, diverse ideas)
- **Code reviewer**: 0.3-0.5 (deterministic, precise)
- **Creative writer**: 0.9-1.0 (maximum diversity)
- **Math solver**: 0.1-0.3 (deterministic, step-by-step)

Specify recommendations in your YAML metadata.

## Experiments and Reproducibility

For reproducible experiments:

1. **Lock the system prompt**: Commit to version control
2. **Document in experiment logs**: System prompt is auto-logged by experiment tracker
3. **Version your prompts**: Use git tags or filename versions (`researcher_v2.md`)
4. **Compare systematically**: Run same tasks with different prompts

Example experiment:
```python
from harness import run_strategy, get_tracker, load_system_prompt

prompts = ["researcher", "default", "creative_writer"]
tasks = ["Task 1", "Task 2", "Task 3"]

for prompt_name in prompts:
    tracker = get_tracker()
    tracker.start_experiment(...)

    for task in tasks:
        result = run_strategy(
            "single",
            task,
            system_prompt=prompt_name,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022"
        )
        tracker.log_result(...)
```

## Troubleshooting

**Prompt not loading?**
```python
from harness import list_system_prompts
print(list_system_prompts())  # See all available prompts
```

**Prompt not working as expected?**
- Check that you're using a compatible model (some prompts designed for specific capabilities)
- Try adjusting temperature
- Verify the prompt is being passed (check experiment logs)

**Want to see what prompt is being used?**
```python
from harness import load_system_prompt
print(load_system_prompt("researcher"))
```

## Examples

### Comparative Analysis

Compare how different personas handle the same task:

```python
task = "Explain the attention mechanism in transformers"

# Researcher perspective
r1 = llm_call(task, system_prompt="researcher")

# Default assistant
r2 = llm_call(task, system_prompt="default")

# Compare outputs
print("Researcher:", r1.text[:200])
print("Default:", r2.text[:200])
```

### Multi-Agent with Different Personas

```python
from harness import run_strategy

# Debate with different personas
result = run_strategy(
    "debate",
    "Should we pursue scaling laws or architectural innovation?",
    n_debaters=3,
    system_prompt="researcher",  # All debaters use researcher persona
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)
```

### Task-Specific Workflows

```python
# Code generation
code = llm_call(
    "Implement a transformer attention layer in PyTorch",
    system_prompt="coder_python",
    temperature=0.3
)

# Research analysis
analysis = llm_call(
    "Analyze this architecture for novelty and impact",
    system_prompt="researcher",
    temperature=0.8
)

# Documentation
docs = llm_call(
    "Write documentation for this code",
    system_prompt="technical_writer",
    temperature=0.5
)
```

## Contributing

To add new system prompts to the library:

1. Create your `.md` file with clear structure
2. Optionally create matching `.yaml` metadata
3. Test with multiple models and temperatures
4. Document recommended use cases
5. Update this README with a description
6. Commit to version control

## References

- **Anthropic System Prompts**: https://docs.anthropic.com/claude/docs/system-prompts
- **OpenAI System Messages**: https://platform.openai.com/docs/guides/text-generation
- **Prompt Engineering Guide**: https://www.promptingguide.ai/

---

**Remember**: System prompts define *who* the model is, while user prompts define *what* it should do. Keep them separate for maximum flexibility and reusability.
