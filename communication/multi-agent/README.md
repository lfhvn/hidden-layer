# Multi-Agent Architecture

Research platform for multi-agent coordination and collective intelligence.

## Prerequisites

- Python 3.10+ with Hidden Layer repository set up
- At least one LLM provider configured:
  - **Local**: Ollama or MLX (recommended for iteration)
  - **API**: Anthropic Claude or OpenAI GPT (for frontier capabilities)

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
from communication.multi_agent import run_strategy
from harness import ExperimentConfig, get_tracker

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

### Jupyter Notebooks

The easiest way to get started is with the provided notebooks:

```bash
# From repository root
jupyter lab communication/multi-agent/notebooks/00_quickstart.ipynb
```

Available notebooks:
- `00_quickstart.ipynb` - Basic usage and examples
- `01_baseline_experiments.ipynb` - Strategy comparisons
- `02_multi_agent_comparison.ipynb` - Performance benchmarks
- `03_introspection_experiments.ipynb` - Agent reasoning analysis

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
├── multi_agent/
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
