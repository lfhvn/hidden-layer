# Multi-Agent Architecture

Research platform for multi-agent coordination and collective intelligence.

## Quick Start

```python
from harness import get_tracker, ExperimentConfig
from code.strategies import run_strategy

# Run 3-agent debate
result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

print(result.output)
```

## Strategies

- **Single**: Baseline single-model inference
- **Debate**: n-agent debate with judge
- **CRIT**: Multi-perspective design critique
- **Self-Consistency**: Sample multiple times, aggregate
- **Manager-Worker**: Decompose → parallel → synthesize
- **Consensus**: Multiple agents find agreement

## Project Structure

```
multi-agent/
├── code/
│   ├── strategies.py      # Strategy implementations
│   ├── rationale.py       # Reasoning extraction
│   ├── cli.py             # Command-line interface
│   └── crit/              # Design critique
├── notebooks/             # Experiment notebooks
├── config/                # Model configurations
├── tests/                 # Tests
└── CLAUDE.md              # Development guide
```

## Documentation

- **Development Guide**: [CLAUDE.md](CLAUDE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) (if exists)
- **Lab Overview**: [/README.md](../../README.md)

## Research Questions

- When do multi-agent strategies outperform single models?
- Why do they outperform (coverage, diversity, synthesis)?
- What are the tradeoffs (latency, cost)?

See [CLAUDE.md](CLAUDE.md) for details.
