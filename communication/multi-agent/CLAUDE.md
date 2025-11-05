# Multi-Agent Architecture - Development Guide

## Project Overview

Research platform for multi-agent coordination and collective intelligence.

**Research Question**: When and why do multi-agent systems outperform single models?

**Strategies Implemented**:
- **Debate**: n-agent debate with judge
- **CRIT**: Multi-perspective design critique
- **Self-Consistency**: Sample multiple times, aggregate
- **Manager-Worker**: Decompose → parallel execution → synthesis
- **Consensus**: Multiple agents find agreement
- **XFN Teams**: Cross-functional team coordination (planned)

**Uses**: `harness/` for LLM provider abstraction and experiment tracking

---

## Architecture

### Core Components

**Strategies** (`code/strategies.py`):
- Strategy registry and implementation
- Each strategy is a hypothesis about coordination
- Easy to add new strategies

**Rationale** (`code/rationale.py`):
- Extract reasoning from agent outputs
- Understand *why* strategies work

**CRIT** (`code/crit/`):
- Design critique evaluation
- 8 design problems across domains
- 9 expert perspectives
- 4 critique strategies

---

## Development Workflows

### Adding a New Strategy

```python
# In code/strategies.py

def my_new_strategy(task_input: str, **kwargs) -> StrategyResult:
    """
    Your custom strategy.

    Design consideration: What coordination mechanism are you testing?
    What hypothesis about multi-agent dynamics does this represent?
    """
    from harness import llm_call

    provider = kwargs.get('provider', 'ollama')
    model = kwargs.get('model', 'llama3.2:latest')

    # Implement your coordination logic
    response = llm_call(task_input, provider=provider, model=model)

    return StrategyResult(
        output=response.text,
        strategy_name="my_new",
        latency_s=response.latency_s,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        metadata={"custom": "data"}
    )

# Register it
STRATEGIES["my_new"] = my_new_strategy
```

### Running Experiments

```python
from harness import get_tracker, ExperimentConfig
from code.strategies import run_strategy

# Configure experiment
config = ExperimentConfig(
    experiment_name="energy_debate",
    strategy="debate",
    provider="ollama",
    model="llama3.2:latest"
)

# Start tracking
tracker = get_tracker()
tracker.start_experiment(config)

# Run strategy
result = run_strategy(
    "debate",
    "Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama"
)

print(result.output)
tracker.finish_experiment()
```

### Using the CLI

```bash
# Run debate with 3 agents
python code/cli.py "Complex question?" --strategy debate --n-debaters 3

# Use model config
python code/cli.py "Question?" --config gpt-oss-20b-reasoning

# Use system prompt
python code/cli.py "Design task" --config claude-researcher
```

---

## Key Files

- `code/strategies.py` - Strategy implementations
- `code/rationale.py` - Reasoning extraction
- `code/cli.py` - Command-line interface
- `code/crit/` - CRIT design critique
- `config/models.yaml` - Model configurations
- `config/system_prompts/` - Reusable prompts
- `notebooks/` - Experiment notebooks

---

## Testing

```bash
# Run tests
cd communication/multi-agent
pytest tests/

# Test specific strategy
python -c "from communication.multi_agent import run_strategy; print(run_strategy('single', 'test'))"
```

---

## Research Questions

1. **When** do multi-agent strategies outperform?
   - Task types (reasoning, creative, planning)
   - Complexity levels
   - Domain-specific patterns

2. **Why** do they outperform?
   - Coverage (more perspectives)
   - Diversity (different approaches)
   - Synthesis (combining insights)
   - Error correction (catching mistakes)

3. **What** are the tradeoffs?
   - Latency vs. quality
   - Cost vs. improvement
   - Diminishing returns

---

## Integration Points

**With AI-to-AI Communication**:
- Can agents use latent messaging instead of natural language?
- Does latent communication improve coordination efficiency?

**With SELPHI**:
- Do multi-agent systems exhibit better theory of mind?
- Can debate improve perspective-taking?

**With Introspection**:
- What activations correspond to successful coordination?
- Can we steer agents toward better collaboration?

---

## See Also

- Lab-wide infrastructure: `/docs/infrastructure/`
- Research methodology: `/docs/workflows/research-methodology.md`
- Benchmark usage: `/docs/workflows/benchmarking.md`
- Top-level README: `/README.md`
