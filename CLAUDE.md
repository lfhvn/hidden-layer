# Claude Development Guide - Hidden Layer Project

## Project Overview

Hidden Layer is a comprehensive research platform for exploring multi-agent LLM systems on Apple Silicon (M4 Max with 128GB RAM). The project consists of three major integrated subsystems designed for rapid experimentation and interpretability research.

**Core Research Question**: When and why do multi-agent strategies outperform single models, and what can we learn from their internal representations?

**System Components**:
1. **Harness** - Core multi-agent infrastructure (1,900 LOC)
2. **CRIT** - Collective Reasoning for Iterative Testing (1,500+ LOC)
3. **SELPHI** - Study of Epistemic and Logical Processing (1,200+ LOC)

**Total**: 6,613 lines of Python + 5,641 lines of documentation

## Quick Reference

For detailed architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
For getting started, see [QUICKSTART.md](QUICKSTART.md).

## System Structure

All three subsystems share common infrastructure:

- **Unified LLM Provider**: Same interface across harness, CRIT, SELPHI
- **Experiment Tracking**: Common logging format
- **Model Configuration**: Shared YAML presets
- **Benchmark Interface**: Unified access to all datasets

**Import Pattern**:
```python
# Harness
from harness import run_strategy, llm_call, load_benchmark

# CRIT
from crit import run_critique_strategy, MOBILE_CHECKOUT

# SELPHI
from selphi import run_scenario, SALLY_ANNE
```

## Development Workflows

### Adding a New Strategy (Harness)

```python
# In code/harness/strategies.py

def my_new_strategy(task_input: str, **kwargs) -> StrategyResult:
    """Your custom multi-agent strategy."""
    provider = kwargs.get('provider', 'ollama')
    model = kwargs.get('model', 'llama3.2:latest')

    # Implement your logic
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

### Adding a New Problem (CRIT)

```python
# In code/crit/problems.py

MY_PROBLEM = DesignProblem(
    name="my_problem",
    domain=DesignDomain.UI_UX,
    description="...",
    current_design="...",
    success_criteria=[...],
    difficulty="medium"
)
```

### Adding a New Scenario (SELPHI)

```python
# In code/selphi/scenarios.py

MY_SCENARIO = ToMScenario(
    name="my_scenario",
    tom_type=ToMType.FALSE_BELIEF,
    scenario_text="...",
    question="...",
    correct_answer="...",
    difficulty="easy"
)
```

### Adding a New Evaluation Function

```python
# Works in any subsystem (harness, crit, selphi)

def my_eval(output: str, expected: Any) -> float:
    """Return score 0-1."""
    # Implementation
    return score

# Register in appropriate module
EVAL_FUNCTIONS["my_eval"] = my_eval
```

## Common Development Tasks

### Testing Locally

```bash
# Activate environment
source venv/bin/activate

# Start Ollama
ollama serve &

# Test harness
python -c "from harness import llm_call; print(llm_call('Hi!', provider='ollama').text)"

# Test CRIT
python -c "from crit import MOBILE_CHECKOUT; print('CRIT ready')"

# Test SELPHI
python -c "from selphi import SALLY_ANNE; print('SELPHI ready')"
```

### Running Experiments

**Harness**:
```bash
python code/cli.py "Question?" --strategy debate --n-debaters 3
```

**CRIT**:
```python
from crit import run_critique_strategy, MOBILE_CHECKOUT

result = run_critique_strategy("multi_perspective", MOBILE_CHECKOUT)
```

**SELPHI**:
```python
from selphi import run_multiple_scenarios, get_scenarios_by_difficulty

scenarios = get_scenarios_by_difficulty("medium")
results = run_multiple_scenarios(scenarios, provider="ollama")
```

### Cross-Subsystem Integration

```python
# Use harness experiment tracking with CRIT
from harness import get_tracker, ExperimentConfig
from crit import run_critique_strategy, MOBILE_CHECKOUT

config = ExperimentConfig(
    experiment_name="crit_experiments",
    task_type="design_critique"
)

tracker = get_tracker()
tracker.start_experiment(config)

result = run_critique_strategy("multi_perspective", MOBILE_CHECKOUT)

tracker.log_result(...)
tracker.finish_experiment()
```

## Hardware Optimization (M4 Max 128GB)

### What You Can Run

- **70B models**: ~35GB (4-bit quantized)
- **3-4 agents in parallel**: 3x 7B models (~12GB total)
- **Fine-tune 13B**: LoRA training (~20GB)

### Recommended Configurations

```python
# Fast iteration
provider="ollama"
model="llama3.2:3b"  # ~2GB, ~120 tok/s

# Quality experiments
provider="ollama"
model="llama3.1:70b"  # ~35GB, ~15 tok/s

# Multi-agent (3-4 agents)
model="llama3.1:8b"  # ~4GB each

# API baseline
provider="anthropic"
model="claude-3-5-sonnet-20241022"
```

## Important Conventions

### Provider Names
- `"ollama"`, `"mlx"`, `"anthropic"`, `"openai"`

### Model Names
- Ollama: `"llama3.2:latest"`
- MLX: `"mlx-community/Llama-3.2-3B-Instruct-4bit"`
- API: `"claude-3-5-sonnet-20241022"`

### Strategy Names
- Use lowercase with underscores: `"single"`, `"debate"`, `"multi_perspective"`

### Temperature Guidelines
- **0.1-0.3**: Deterministic tasks
- **0.7-0.8**: Balanced reasoning
- **0.9+**: Creative outputs

## Key Files by Subsystem

### Harness (1,900 LOC)
- `llm_provider.py` (527 LOC) - Unified LLM interface
- `strategies.py` (749 LOC) - Multi-agent strategies
- `experiment_tracker.py` - Logging
- `evals.py` - Evaluation functions
- `benchmarks.py` - Unified benchmark interface
- `rationale.py` (285 LOC) - Reasoning extraction
- `model_config.py` - YAML config

### CRIT (1,500+ LOC)
- `problems.py` (584 LOC) - 8 design problems
- `strategies.py` - 4 critique strategies
- `evals.py` - Critique quality metrics
- `benchmarks.py` - UICrit dataset (11,344 critiques)

### SELPHI (1,200+ LOC)
- `scenarios.py` - 9+ ToM scenarios
- `evals.py` - ToM evaluation
- `benchmarks.py` - ToMBench, OpenToM, SocialIQA

## Research Direction

### Core Questions
1. **When** do multi-agent strategies outperform?
2. **Why** do they outperform?
3. **What** are the tradeoffs?

### CRIT Questions
- Do multi-perspective critiques cover more design issues?
- Is iterative refinement better than one-shot?
- Can adversarial critique find edge cases?

### SELPHI Questions
- Which ToM types are hardest for LLMs?
- Do larger models have better ToM?
- Can fine-tuning improve ToM?

## Status

### ✅ Completed
- Core harness with 5 strategies
- CRIT with 8 problems, 4 strategies
- SELPHI with 9+ scenarios, 3 benchmarks
- Experiment tracking
- Model configuration
- Unified benchmark interface
- Rationale extraction

### 🚧 In Progress
- Baseline experiments
- Performance benchmarking
- Documentation

### 📋 Planned
- Fine-tuning workflows (MLX + LoRA)
- Interpretability tools
- Advanced visualization

## Performance Tips

1. **Start small**: 3B/7B for iteration, 70B for quality
2. **Use configs**: Leverage model presets
3. **Batch wisely**: 1-4 items per batch
4. **Cache models**: Ollama/HF cache for speed
5. **Monitor memory**: Use Activity Monitor

## Troubleshooting

**Ollama**:
```bash
killall ollama && ollama serve &
ollama list
```

**Imports**:
```python
import sys
sys.path.append('../code')  # In notebooks
```

**MLX**:
```python
import mlx.core as mx
print(mx.__version__)
```

## Documentation Map

- **README.md** - Project overview
- **ARCHITECTURE.md** - Deep dive (all subsystems)
- **QUICKSTART.md** - Cheat sheet
- **SETUP.md** - Installation
- **BENCHMARKS.md** - Benchmark datasets
- **CLAUDE.md** - This file (development guide)

Subsystem docs:
- **code/crit/README.md** - CRIT details
- **code/selphi/README.md** - SELPHI details
- **config/README.md** - Model configs

## References

- MLX: https://github.com/ml-explore/mlx
- Ollama: https://ollama.ai
- MLX Models: https://huggingface.co/mlx-community
- UICrit: https://github.com/google-research-datasets/uicrit
- ToMBench: https://github.com/wadimiusz/ToMBench

## Development Philosophy

**Remember**: This is a research tool for rapid experimentation, not production.

Key principles:
1. Local-first (MLX, Ollama)
2. Notebook-centric
3. Reproducible (auto-logging)
4. Hackable (simple code)
5. Modular (clear subsystem boundaries)
6. Extensible (registry patterns)

Ask yourself:
- Is this local-first?
- Is this notebook-friendly?
- Is this reproducible?
- Is this extensible?
- Does this help answer research questions?

---

**For detailed architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
**For quick start**: See [QUICKSTART.md](QUICKSTART.md)
**For setup**: See [SETUP.md](SETUP.md)
