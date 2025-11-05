# Introspection

Model introspection experiments inspired by Anthropic's findings.

## Quick Start

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
