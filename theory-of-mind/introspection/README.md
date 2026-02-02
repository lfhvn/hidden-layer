# Introspection

Model introspection experiments inspired by Anthropic's findings.

## Prerequisites

- Python 3.10+ with Hidden Layer repository set up
- For **local model steering** (activation-based):
  - MLX (macOS with Apple Silicon) - Python 3.10-3.12 only
  - OR PyTorch + HuggingFace Transformers (any platform)
  - GPU recommended for non-MLX platforms
- For **API introspection** (prompt-based):
  - Anthropic API key (Claude) or OpenAI API key (GPT)

**New to Hidden Layer?** See [/QUICKSTART.md](../../QUICKSTART.md) for initial setup.

## Installation

This project uses the shared harness infrastructure:

```bash
# From repository root
pip install -r requirements.txt

# For local model steering with MLX (macOS Apple Silicon only)
pip install mlx mlx-lm

# OR for local model steering with PyTorch (any platform)
pip install torch transformers

# Verify setup
python check_setup.py
```

## Quick Start

### Activation Steering (Local Models)

```python
from theory_of_mind.introspection import ActivationSteerer, ConceptLibrary
from mlx_lm import load

# Load concept library
library = ConceptLibrary.load("../../shared/concepts/emotions_layer15.pkl")

# Use with local model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Steer toward happiness
output = steerer.generate_with_steering(
    prompt="Write a story",
    concept_vector=library.get("happiness"),
    strength=2.0
)
```

### API Introspection (Claude/GPT)

```python
from theory_of_mind.introspection import run_introspection_task
from harness import llm_call

# Test model's self-knowledge
result = run_introspection_task(
    task="emotional_state",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

print(f"Accuracy: {result['accuracy']:.2%}")
```

### Jupyter Notebooks

```bash
# From repository root
jupyter lab theory-of-mind/introspection/notebooks/01_concept_vectors.ipynb
```

Available notebooks:
- `01_concept_vectors.ipynb` - Extract and use concept vectors
- `02_activation_steering.ipynb` - Steer model behavior with activations

## Features

- **Activation Steering**: Capture and modify model activations
- **Concept Vectors**: Extract and use concept representations
- **Introspection Tasks**: Test self-knowledge accuracy
- **API Introspection**: Prompt-based steering for frontier models

## Project Structure

```
introspection/
├── code/
│   ├── activation_steering.py  # Activation capture/steering
│   ├── concept_vectors.py      # Concept representation
│   ├── introspection_tasks.py  # Introspection evaluation
│   └── introspection_api.py    # API-based introspection
└── CLAUDE.md                    # Development guide
```

## Research Questions

- Can models accurately report their internal states?
- How does introspection relate to alignment?
- Can we detect deceptive self-reporting?

See [CLAUDE.md](CLAUDE.md) and [/shared/concepts/README.md](../../shared/concepts/README.md) for details.
