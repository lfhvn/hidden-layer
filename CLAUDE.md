# Claude Development Guide - Hidden Layer Project

## Project Overview

This is a research experimentation harness for comparing single-model and multi-agent LLM strategies on Apple Silicon (M4 Max with 128GB RAM). The project enables notebook-first research workflows with proper experiment tracking, designed for rapid iteration on multi-agent architectures and interpretability research.

**Core Research Question**: When and why do multi-agent strategies outperform single models, and what can we learn from their internal representations?

## Project Architecture

### Design Philosophy

1. **Local-First**: Prioritize MLX and Ollama for fast, cost-effective iteration
2. **Hybrid Approach**: Seamlessly switch between local and API providers for comparison
3. **Notebook-Centric**: All core functions work directly in Jupyter with minimal boilerplate
4. **Reproducible**: Automatic experiment logging, configuration tracking, git hash capture
5. **Hackable**: Simple, well-commented code over heavy frameworks

### Core Components

#### 1. LLM Provider (`code/harness/llm_provider.py`)
- **Purpose**: Unified interface for all LLM providers
- **Providers**: MLX (Apple Silicon), Ollama, Anthropic Claude, OpenAI
- **Features**: Automatic cost tracking, token counting, easy provider switching
- **Key Function**: `llm_call(prompt, provider, model, **kwargs)` - universal LLM interface

#### 2. Multi-Agent Strategies (`code/harness/strategies.py`)
- **Purpose**: Implement and compare different multi-agent approaches
- **Available Strategies**:
  - `single`: Single-model baseline
  - `debate`: n-agent debate with judge
  - `self_consistency`: Sample multiple times and aggregate
  - `manager_worker`: Decompose, execute in parallel, synthesize
- **Key Function**: `run_strategy(strategy_name, task_input, **kwargs)` - execute any strategy
- **Extensible**: Easy to add new strategies to registry

#### 3. Experiment Tracking (`code/harness/experiment_tracker.py`)
- **Purpose**: Automatic logging and reproducibility
- **Features**:
  - Unique experiment IDs (name + timestamp + hash)
  - JSON/JSONL streaming logs
  - Metrics aggregation (latency, tokens, cost)
  - Comparison across runs
- **Output Structure**: `experiments/{name}_{timestamp}_{hash}/`
  - `config.json`: Experiment configuration
  - `results.jsonl`: Streaming results (one per line)
  - `summary.json`: Aggregated metrics
  - `README.md`: Human-readable summary

#### 4. Evaluation Suite (`code/harness/evals.py`)
- **Purpose**: Systematic evaluation of outputs
- **Methods**: Exact match, keyword match, numeric match, LLM-as-judge
- **Metrics**: Win-rate comparison, coherence scoring
- **Extensible**: Easy to add custom evaluation functions

#### 5. Configuration Management (`config/models.yaml`, `config/README.md`)
- **Purpose**: Manage model presets and hyperparameters
- **Features**: Named configurations, parameter overrides, task-specific setups
- **Built-in Configs**:
  - Reasoning models (extended thinking budget)
  - Creative models (high temperature)
  - Fast models (quick iteration)
  - API models (Claude, GPT)
- **Usage**: `--config gpt-oss-20b-reasoning` in CLI or `get_model_config()` in Python

### File Structure

```
hidden-layer/
├── code/
│   ├── harness/              # Core library (import from notebooks)
│   │   ├── __init__.py       # Package exports
│   │   ├── llm_provider.py   # Unified LLM interface
│   │   ├── strategies.py     # Multi-agent strategies
│   │   ├── experiment_tracker.py  # Experiment logging
│   │   └── evals.py          # Evaluation functions
│   └── cli.py                # Command-line interface
│
├── notebooks/                # Experimentation notebooks
│   ├── 01_baseline_experiments.ipynb
│   ├── 02_debate_experiments.ipynb
│   └── (future notebooks)
│
├── config/                   # Model configurations
│   ├── models.yaml           # Named model presets
│   └── README.md             # Config documentation
│
├── experiments/              # Auto-generated experiment logs
│   └── {name}_{timestamp}_{hash}/
│
├── README.md                 # Project overview and features
├── SETUP.md                  # Installation guide for M4 Max
├── QUICKSTART.md            # Cheat sheet for common operations
├── IMPLEMENTATION.md        # What was built and why
├── START_HERE.md            # Quick start for new users
├── CLAUDE.md                # This file - development guide
└── requirements.txt         # Python dependencies
```

## Development Principles

### When Making Changes

1. **Maintain Backward Compatibility**: Notebooks depend on stable APIs
2. **Log Everything**: Use experiment tracker for all runs
3. **Document Decisions**: Update relevant .md files
4. **Test Locally First**: Use small models for rapid iteration
5. **Version Control Configs**: Commit `models.yaml` changes

### Code Patterns

#### Adding a New Strategy

```python
# In code/harness/strategies.py

def my_new_strategy(task_input: str, **kwargs) -> StrategyResult:
    """Your custom strategy."""
    provider = kwargs.get('provider', 'ollama')
    model = kwargs.get('model', 'llama3.2:latest')

    # Implement your logic
    response = llm_call(task_input, provider=provider, model=model, **kwargs)

    return StrategyResult(
        output=response.text,
        strategy_name="my_new",
        latency_s=response.latency_s,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        cost_usd=response.cost_usd,
        metadata={"custom": "data"}
    )

# Add to registry
STRATEGIES["my_new"] = my_new_strategy
```

#### Adding a New Evaluation Function

```python
# In code/harness/evals.py

def my_eval(output: str, expected: Any) -> float:
    """Your custom evaluation."""
    # Return score 0-1
    score = calculate_score(output, expected)
    return score

# Add to registry
EVAL_FUNCTIONS["my_eval"] = my_eval
```

#### Using in Notebooks

```python
import sys
sys.path.append('../code')

from harness import (
    run_strategy,
    get_tracker,
    evaluate_task,
    get_model_config
)

# Load configuration
config = get_model_config("gpt-oss-20b-reasoning")

# Start tracking
tracker = get_tracker()
experiment_dir = tracker.start_experiment(ExperimentConfig(
    experiment_name="my_experiment",
    strategy="debate",
    **config.to_dict()
))

# Run and log
result = run_strategy("debate", task_input, **config.to_kwargs())
tracker.log_result(ExperimentResult(...))

# Finish
summary = tracker.finish_experiment()
```

## Common Tasks

### Quick Testing
```bash
# Activate environment
source venv/bin/activate

# Start Ollama
ollama serve &

# Test with CLI
python code/cli.py "What is 2+2?" --strategy single --provider ollama

# Test with config
python code/cli.py "Complex question" --config gpt-oss-20b-reasoning
```

### Running Experiments
```bash
# Debate with 3 agents
python code/cli.py "Should we invest in solar?" --strategy debate --n-debaters 3

# Self-consistency with 5 samples
python code/cli.py "What is..." --strategy self_consistency --n-samples 5

# Manager-worker decomposition
python code/cli.py "Plan a research project" --strategy manager_worker --n-workers 3
```

### Working with Notebooks
```bash
cd notebooks
jupyter notebook
# Open 01_baseline_experiments.ipynb
```

### Adding New Models
```bash
# Ollama
ollama pull model-name:latest

# MLX (downloaded automatically on first use)
# Models from: https://huggingface.co/mlx-community
```

## Key Context for Claude

### Hardware Capabilities (M4 Max 128GB)
- Can run models up to 70B (4-bit quantized)
- Can run 3-4 7B models simultaneously for multi-agent
- Can fine-tune 13B models with LoRA
- MLX optimized for unified memory architecture

### Research Direction
- Focus on **when** multi-agent helps (task types, complexity)
- Focus on **why** it helps (interpretability, hidden layers)
- Cost/latency tradeoffs vs quality improvements
- Fine-tuning impact on multi-agent performance

### Current Status
- ✅ Core harness implemented and tested
- ✅ Baseline experiments established
- ✅ Debate strategy validated
- ✅ Configuration management in place
- 🚧 Advanced interpretability features (future)
- 🚧 Fine-tuning workflows (future)
- 🚧 Hidden layer analysis (future)

### Important Conventions

1. **Provider Naming**: Use consistent provider strings
   - `"ollama"` for Ollama
   - `"mlx"` for MLX
   - `"anthropic"` for Claude
   - `"openai"` for GPT

2. **Model Naming**:
   - Ollama: `"model-name:tag"` (e.g., `"llama3.2:latest"`)
   - MLX: Full HF path (e.g., `"mlx-community/Llama-3.2-3B-Instruct-4bit"`)
   - API: Official model IDs (e.g., `"claude-3-5-sonnet-20241022"`)

3. **Strategy Naming**: Use lowercase with underscores
   - `"single"`, `"debate"`, `"self_consistency"`, `"manager_worker"`

4. **Temperature Guidelines**:
   - 0.1-0.3: Deterministic, factual tasks
   - 0.7-0.8: Balanced, general reasoning
   - 0.9+: Creative, diverse outputs

5. **Thinking Budget**: For reasoning-capable models
   - 1000-2000: Standard reasoning tasks
   - 2500-3000: Complex multi-step problems
   - 5000+: Extended chain-of-thought

### Performance Optimization

1. **Start Small**: Use 3B/7B models for iteration, scale up for quality
2. **Use Configs**: Leverage presets to avoid repeating parameters
3. **Batch Wisely**: Keep batch_size 1-4 for most experiments
4. **Cache Models**: Ollama and HF cache models for faster loading
5. **Monitor Memory**: Use Activity Monitor to track RAM usage

### Troubleshooting

**Ollama Issues**:
```bash
killall ollama && ollama serve &
ollama list  # Check available models
```

**MLX Issues**:
```python
import mlx.core as mx
print(mx.__version__)  # Verify installation
```

**Import Issues**:
```python
# In notebooks, ensure path is correct
import sys
sys.path.append('../code')
```

## Integration Points

### For New Features

1. **New Provider**: Extend `llm_provider.py` with new provider class
2. **New Strategy**: Add function to `strategies.py` and register
3. **New Eval**: Add function to `evals.py` and register
4. **New Config**: Add entry to `config/models.yaml`
5. **New Notebook**: Create in `notebooks/` with imports from harness

### For External Tools

- Experiment logs are JSON/JSONL (easy to parse)
- CLI can be scripted for automation
- Harness can be imported as Python package

## References

- **MLX**: https://github.com/ml-explore/mlx
- **MLX-LM**: https://github.com/ml-explore/mlx-examples/tree/main/llms
- **Ollama**: https://ollama.ai
- **MLX Models**: https://huggingface.co/mlx-community

## Next Steps for Development

Based on the roadmap, prioritize:

1. **Immediate**: Run more baseline experiments, expand task suite
2. **Short-term**: Analyze multi-agent vs single performance systematically
3. **Medium-term**: Add fine-tuning workflows, interpretability probes
4. **Long-term**: Build hidden layer analysis tools, draft findings

## Questions to Keep in Mind

While developing, constantly ask:
1. Does this maintain the local-first philosophy?
2. Is this notebook-friendly?
3. Is this reproducible (logged, versioned)?
4. Is this extensible (easy for future additions)?
5. Does this help answer the core research questions?

---

**Remember**: This is a research tool for rapid experimentation, not a production system. Favor simplicity and hackability over robustness and scale. Make it easy to try new ideas quickly.
