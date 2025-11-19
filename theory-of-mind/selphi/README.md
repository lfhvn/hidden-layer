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

## Research Questions

- Which ToM types are hardest for LLMs?
- How does ToM scale with model size?
- Connection to deception and alignment?

See [CLAUDE.md](CLAUDE.md) for details.
