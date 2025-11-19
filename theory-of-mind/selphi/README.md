# SELPHI - Theory of Mind

Study of Epistemic and Logical Processing in Human-AI Interaction

## Prerequisites

- Python 3.10+ with Hidden Layer repository set up
- At least one LLM provider configured:
  - **Local**: Ollama or MLX (for rapid iteration)
  - **API**: Anthropic Claude or OpenAI GPT (recommended for benchmark comparisons)

**New to Hidden Layer?** See [/QUICKSTART.md](../../QUICKSTART.md) for initial setup.

## Installation

This project uses the shared harness infrastructure. From the repository root:

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python check_setup.py
```

## Quick Start

### Python API

```python
from harness import llm_call
from theory_of_mind.selphi import SALLY_ANNE, evaluate_scenario

# Run Sally-Anne test
response = llm_call(
    SALLY_ANNE.get_prompt(),
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Evaluate
result = evaluate_scenario(SALLY_ANNE, response.text)
print(f"Score: {result['average_score']:.2f}")
```

### Jupyter Notebooks

The easiest way to explore ToM scenarios is with the provided notebooks:

```bash
# From repository root
jupyter lab theory-of-mind/selphi/notebooks/01_basic_tom_tests.ipynb
```

Available notebooks:
- `01_basic_tom_tests.ipynb` - Run individual ToM scenarios
- `02_benchmark_evaluation.ipynb` - Evaluate on ToMBench, OpenToM, SocialIQA

## Example Output

### Running a ToM Scenario

```python
from harness import llm_call
from theory_of_mind.selphi import SALLY_ANNE, evaluate_scenario

# Run Sally-Anne test
response = llm_call(
    SALLY_ANNE.get_prompt(),
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Example response:
"""
Sally will look for her marble in the basket.

Reasoning: Sally placed her marble in the basket before leaving the room.
While she was away, Anne moved the marble to the box. However, Sally did
not witness this change. Therefore, Sally still believes the marble is in
the basket where she left it. This demonstrates understanding of false
belief - Sally's belief differs from reality.
"""

# Evaluate the response
result = evaluate_scenario(SALLY_ANNE, response.text)
print(f"Correct: {result['correct']}")  # True
print(f"Score: {result['average_score']:.2f}")  # 1.0
```

### Benchmark Evaluation Example

```python
from theory_of_mind.selphi import run_benchmark_evaluation

# Evaluate on ToMBench
results = run_benchmark_evaluation(
    benchmark="tombench",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    sample_size=100  # Subset for quick testing
)

# Example results:
"""
ToMBench Evaluation Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: claude-3-5-sonnet-20241022
Samples: 100/388
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

By ToM Type:
  False Belief:        87.5% (35/40)
  Second-Order:        82.1% (23/28)
  Epistemic States:    91.2% (31/34)
  Perspective-Taking:  88.9% (16/18)

Overall Accuracy: 86.0%
Average Score: 0.86
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
```

## Features

- **9+ ToM Scenarios**: Sally-Anne, Chocolate Bar, Birthday Puppy, etc.
- **7 ToM Types**: False belief, second-order belief, epistemic states, etc.
- **3 Benchmarks**: ToMBench (388), OpenToM (696), SocialIQA (38k)
- **Multiple Evaluation Methods**: Semantic matching, LLM-as-judge

## Project Structure

```
selphi/
├── code/
│   ├── scenarios.py       # ToM scenarios
│   ├── evals.py           # ToM evaluation
│   └── benchmarks.py      # Benchmark loaders
├── notebooks/             # Experiments
└── CLAUDE.md              # Development guide
```

## Troubleshooting

### Scenario Import Errors

**Problem**: `AttributeError: module 'theory_of_mind.selphi' has no attribute 'SALLY_ANNE'`

**Solution**:
```python
# Import scenarios directly from the code module
from theory_of_mind.selphi.code.scenarios import (
    SALLY_ANNE,
    CHOCOLATE_BAR,
    BIRTHDAY_PUPPY,
    # ... other scenarios
)

# Or import all scenarios
from theory_of_mind.selphi.code import scenarios
scenario = scenarios.SALLY_ANNE
```

### Benchmark Loading Issues

**Problem**: Benchmark datasets not found

**Solution**:
```bash
# Benchmarks are loaded automatically from HuggingFace
# Make sure you have internet connection and datasets library
pip install datasets

# Test benchmark loading
python -c "from theory_of_mind.selphi.code.benchmarks import load_benchmark; load_benchmark('tombench')"
```

### Evaluation Accuracy Issues

**Problem**: All scenarios scoring 0.0 or incorrectly

**Solution**:
- Check that the model response format matches expected format
- Use `evaluate_scenario` with `verbose=True` to see evaluation details
- Try LLM-as-judge evaluation method for more flexible matching:

```python
result = evaluate_scenario(
    SALLY_ANNE,
    response.text,
    eval_method="llm_judge",  # Instead of "exact_match"
    judge_provider="anthropic",
    judge_model="claude-3-5-sonnet-20241022"
)
```

### Performance Issues with Large Benchmarks

**Problem**: Evaluation takes too long on full benchmarks

**Solution**:
```python
# Use sample_size parameter for quick testing
results = run_benchmark_evaluation(
    benchmark="tombench",
    provider="ollama",
    sample_size=50,  # Test on 50 samples instead of all 388
    random_seed=42   # For reproducibility
)

# Or run in parallel (if using API providers)
results = run_benchmark_evaluation(
    benchmark="opentom",
    provider="anthropic",
    parallel=True,
    max_workers=5  # Concurrent requests
)
```

### API Rate Limiting

**Problem**: `RateLimitError` when evaluating benchmarks

**Solution**:
```python
# Add delays between requests
results = run_benchmark_evaluation(
    benchmark="tombench",
    provider="anthropic",
    delay_between_requests=1.0,  # 1 second delay
    retry_on_error=True
)

# Or use local models to avoid rate limits
results = run_benchmark_evaluation(
    benchmark="tombench",
    provider="ollama",  # No rate limits!
    model="llama3.2:latest"
)
```

For more help:
- [SELPHI Development Guide](CLAUDE.md)
- [Harness Documentation](../../harness/README.md)
- [Main QUICKSTART](../../QUICKSTART.md)

## Research Questions

- Which ToM types are hardest for LLMs?
- How does ToM scale with model size?
- Connection to deception and alignment?

See [CLAUDE.md](CLAUDE.md) for details.
