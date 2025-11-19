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

## Example Output

Here's what you can expect when running a debate strategy:

```python
result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

# result.output will contain the final synthesized answer:
"""
After careful deliberation, the consensus is to invest in renewable energy.

Key arguments in favor:
- Long-term cost savings despite higher initial investment
- Environmental benefits and climate change mitigation
- Energy independence and security
- Job creation in emerging sectors

Considerations:
- Transition costs and infrastructure requirements
- Need for energy storage solutions
- Balancing with existing energy mix
"""

# result.metadata contains strategy details:
print(result.metadata["strategy"])  # "debate"
print(result.metadata["n_debaters"])  # 3
print(result.metadata["rounds"])  # Number of debate rounds
```

### Strategy Comparison Example

```python
from communication.multi_agent import run_strategy
from harness import get_tracker

tracker = get_tracker()
question = "What are the pros and cons of remote work?"

# Compare strategies
strategies = ["single", "debate", "self_consistency", "consensus"]
results = {}

for strategy in strategies:
    result = run_strategy(strategy, task_input=question)
    results[strategy] = result
    tracker.log_result(strategy, result)

# View logged experiments
tracker.summarize()
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

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'communication.multi_agent'`

**Solution**:
```bash
# Make sure you're running from the repository root
cd /path/to/hidden-layer

# Verify harness is installed
python -c "from harness import llm_call; print('✓ Harness OK')"

# Test multi-agent import
python -c "from communication.multi_agent import run_strategy; print('✓ Multi-agent OK')"
```

### Provider Errors

**Problem**: `OllamaConnectionError` or model not found

**Solution**:
```bash
# Make sure Ollama is running
ollama serve  # In a separate terminal

# Pull the model you want to use
ollama pull llama3.2:latest

# Test the model
ollama run llama3.2:latest "Hello"
```

**Problem**: API key errors (Anthropic/OpenAI)

**Solution**:
```bash
# Check your .env file exists
ls .env

# Make sure API keys are set
cat .env | grep API_KEY

# Or export directly
export ANTHROPIC_API_KEY="your-key-here"
```

### Performance Issues

**Problem**: Debate strategy is slow

**Solution**:
- Use local models (Ollama/MLX) for faster iteration
- Reduce number of debaters: `n_debaters=2` instead of 3+
- Use smaller models for development: `llama3.2:latest` (3B) instead of larger models
- Check `config/models.yaml` for timeout settings

### Experiment Tracking Issues

**Problem**: Results not being logged

**Solution**:
```python
from harness import get_tracker

# Initialize tracker explicitly
tracker = get_tracker()

# Log results manually
result = run_strategy("debate", task_input="question")
tracker.log_result("debate_experiment", result)

# Check logged experiments
tracker.summarize()
```

### Strategy-Specific Issues

**Problem**: CRIT strategy not working

**Solution**:
```python
# Make sure you're passing the right parameters
from communication.multi_agent.crit import run_crit

result = run_crit(
    task_input="Design a mobile app",
    perspectives=["UX designer", "Engineer", "Product Manager"],
    provider="ollama"
)
```

For more help, see:
- [Harness documentation](../../harness/README.md)
- [QUICKSTART.md](../../QUICKSTART.md) for setup issues
- [CLAUDE.md](CLAUDE.md) for development details

## Research Questions

- When do multi-agent strategies outperform single models?
- Why do they outperform (coverage, diversity, synthesis)?
- What are the tradeoffs (latency, cost)?

See [CLAUDE.md](CLAUDE.md) for details.
