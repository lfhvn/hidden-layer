# SELPHI - Theory of Mind

Study of Epistemic and Logical Processing in Human-AI Interaction

## Quick Start

```python
from harness import llm_call
from code.scenarios import SALLY_ANNE
from code.evals import evaluate_scenario

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
