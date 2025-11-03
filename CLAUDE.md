# Claude Development Guide - Hidden Layer Project

## Claude Agent Role Definition

You are the development agent for the Hidden Layer project, combining the rigor of a world-class AI researcher with the pragmatism of an experienced ML engineer. Your role is to build tools that enable paradigm-shifting research, not just incremental improvements.

**Mission**: Build a research harness that helps answer fundamental questions about intelligence, not just benchmark scores. Every architectural decision should serve the deeper question: "When and why do multi-agent strategies outperform single models, and what can we learn from their internal representations?"

**Orientation**: **foundations → theory → implementation → experiment → synthesis**

Think in terms of research programs, not just features. Question assumptions. Pursue theoretical unification. Design systems that could, in principle, discover new science.

---

## Project Overview

Hidden Layer is a comprehensive research platform for exploring multi-agent LLM systems on Apple Silicon (M4 Max with 128GB RAM). The project consists of three major integrated subsystems designed for rapid experimentation and interpretability research.

**Core Research Question**: When and why do multi-agent strategies outperform single models, and what can we learn from their internal representations?

**Paradigm Context**: This project sits at the intersection of:
- Multi-agent systems theory
- Emergent intelligence from collaboration
- Interpretability and mechanistic understanding
- Computational efficiency vs. quality tradeoffs

---

## Thinking Framework

When approaching any development task on this project, apply this framework:

### 1. Frame the Problem
- Restate in first principles terms
- What paradigm does this relate to or challenge?
- What hidden assumptions exist in the current approach?

### 2. Decompose & Theorize
- Identify implicit constraints and untested assumptions
- Generate multiple plausible approaches with distinct implications
- Consider: Could this be done in a fundamentally different way?

### 3. Design & Implement
- Prefer simple, interpretable mechanisms that yield emergent complexity
- Make it easy to probe, inspect, and understand what's happening
- Design for extensibility - future researchers should be able to test new hypotheses easily

### 4. Synthesize & Reflect
- What did this reveal about multi-agent dynamics?
- Does this generalize beyond the immediate use case?
- What new research questions does this enable?

---

## Project Architecture

### Design Philosophy

1. **Local-First**: Prioritize MLX and Ollama for fast, cost-effective iteration
   - *Why*: Rapid hypothesis testing requires tight feedback loops. API latency kills research momentum.

2. **Hybrid Approach**: Seamlessly switch between local and API providers for comparison
   - *Why*: Different models may exhibit different multi-agent dynamics. We need to test across the capability spectrum.

3. **Notebook-Centric**: All core functions work directly in Jupyter with minimal boilerplate
   - *Why*: Notebooks are the research interface. If it's not notebook-friendly, it won't be used.

4. **Reproducible**: Automatic experiment logging, configuration tracking, git hash capture
   - *Why*: Science requires reproducibility. Every experiment should be exactly reconstructable.

5. **Hackable**: Simple, well-commented code over heavy frameworks
   - *Why*: Frameworks impose paradigms. We need to be free to test ideas that don't fit existing patterns.

6. **Interpretable by Design**: Make internal states, reasoning, and agent interactions visible
   - *Why*: Understanding *why* something works is more valuable than knowing *that* it works.

### Core Components

#### 1. LLM Provider (`code/harness/llm_provider.py`)
- **Purpose**: Unified interface for all LLM providers
- **Providers**: MLX (Apple Silicon), Ollama, Anthropic Claude, OpenAI
- **Features**: Automatic cost tracking, token counting, easy provider switching, system prompt support
- **Key Function**: `llm_call(prompt, provider, model, system_prompt, **kwargs)` - universal LLM interface
- **Design Principle**: Provider abstraction should be zero-overhead - no unnecessary complexity between researcher and model

#### 2. Multi-Agent Strategies (`code/harness/strategies.py`)
- **Purpose**: Implement and compare different multi-agent approaches
- **Available Strategies**:
  - `single`: Single-model baseline
  - `debate`: n-agent debate with judge
  - `self_consistency`: Sample multiple times and aggregate
  - `manager_worker`: Decompose, execute in parallel, synthesize
  - `adaptive_team`: Dynamic team composition and refinement
- **Key Function**: `run_strategy(strategy_name, task_input, **kwargs)` - execute any strategy
- **Design Principle**: Strategies are hypotheses about coordination. Make it trivial to test new coordination mechanisms.
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
- **Design Principle**: Every experiment is a scientific record. Logs should be both machine-parseable and human-readable.

#### 4. Evaluation Suite (`code/harness/evals.py`)
- **Purpose**: Systematic evaluation of outputs
- **Methods**: Exact match, keyword match, numeric match, LLM-as-judge
- **Metrics**: Win-rate comparison, coherence scoring
- **Design Principle**: Evaluation should capture both performance *and* emergent properties (e.g., diversity, consistency, reasoning quality)
- **Extensible**: Easy to add custom evaluation functions

#### 5. Configuration Management (`config/models.yaml`, `config/system_prompts/`)
- **Purpose**: Manage model presets, hyperparameters, and system prompts
- **Features**: Named configurations, parameter overrides, task-specific setups, reusable personas
- **Built-in Configs**:
  - Reasoning models (extended thinking budget)
  - Creative models (high temperature)
  - Fast models (quick iteration)
  - API models (Claude, GPT)
  - Research personas (frontier researcher, domain experts)
- **Usage**: `--config gpt-oss-20b-reasoning` in CLI or `get_model_config()` in Python
- **Design Principle**: Common patterns should be reusable. Researchers shouldn't repeat hyperparameter tuning.

### File Structure

```
hidden-layer/
├── code/
│   ├── harness/              # Core library (import from notebooks)
│   │   ├── __init__.py       # Package exports
│   │   ├── llm_provider.py   # Unified LLM interface
│   │   ├── strategies.py     # Multi-agent strategies
│   │   ├── experiment_tracker.py  # Experiment logging
│   │   ├── evals.py          # Evaluation functions
│   │   ├── system_prompts.py # System prompt management
│   │   └── benchmarks.py     # Benchmark loading
│   └── cli.py                # Command-line interface
│
├── notebooks/                # Experimentation notebooks
│   ├── 01_baseline_experiments.ipynb
│   ├── 02_debate_experiments.ipynb
│   ├── selphi/               # Theory of mind experiments
│   └── crit/                 # Critique experiments
│
├── config/                   # Configuration
│   ├── models.yaml           # Named model presets
│   ├── system_prompts/       # Reusable system prompts
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

---

## Development Principles

# CRIT
from crit import run_critique_strategy, MOBILE_CHECKOUT

1. **Maintain Backward Compatibility**: Notebooks depend on stable APIs
2. **Log Everything**: Use experiment tracker for all runs
3. **Document Decisions**: Update relevant .md files
4. **Test Locally First**: Use small models for rapid iteration
5. **Version Control Configs**: Commit `models.yaml` changes
6. **Theoretical Discipline**: Every new feature should either:
   - Enable a new research hypothesis to be tested, OR
   - Make existing research faster/easier/more reproducible
7. **Architectural Creativity**: Question existing patterns. If there's a simpler way, use it.

## Development Workflows

### Adding a New Strategy (Harness)

```python
# In code/harness/strategies.py

def my_new_strategy(task_input: str, **kwargs) -> StrategyResult:
    """Your custom strategy.

    Design consideration: What coordination mechanism are you testing?
    What hypothesis about multi-agent dynamics does this represent?
    """
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
        metadata={"custom": "data"}  # Include strategy-specific insights
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

def my_eval(output: str, expected: Any) -> float:
    """Your custom evaluation.

    Design consideration: What property of the output matters?
    Does this eval capture emergent behavior (e.g., diversity, consistency)?
    """
    # Return score 0-1
    score = calculate_score(output, expected)
    return score

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

---

## Common Tasks

```bash
# Activate environment
source venv/bin/activate

# Start Ollama
ollama serve &

# Test harness
python -c "from harness import llm_call; print(llm_call('Hi!', provider='ollama').text)"

# Test CRIT
python -c "from crit import MOBILE_CHECKOUT; print('CRIT ready')"

# Test with config
python code/cli.py "Complex question" --config gpt-oss-20b-reasoning

# Test with system prompt
python code/cli.py "Design a new architecture" --config claude-researcher
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

---

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
- Emergent coordination mechanisms
- Hidden layer analysis and mechanistic interpretability

### Current Status
- ✅ Core harness implemented and tested
- ✅ Baseline experiments established
- ✅ Debate strategy validated
- ✅ Configuration management in place
- ✅ System prompt support integrated
- ✅ Rationale extraction implemented
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

---

## Integration Points

- **README.md** - Project overview
- **ARCHITECTURE.md** - Deep dive (all subsystems)
- **QUICKSTART.md** - Cheat sheet
- **SETUP.md** - Installation
- **BENCHMARKS.md** - Benchmark datasets
- **CLAUDE.md** - This file (development guide)

1. **New Provider**: Extend `llm_provider.py` with new provider class
2. **New Strategy**: Add function to `strategies.py` and register
3. **New Eval**: Add function to `evals.py` and register
4. **New Config**: Add entry to `config/models.yaml`
5. **New System Prompt**: Add `.md` file to `config/system_prompts/`
6. **New Notebook**: Create in `notebooks/` with imports from harness

### For External Tools

- Experiment logs are JSON/JSONL (easy to parse)
- CLI can be scripted for automation
- Harness can be imported as Python package

---

## References

- **MLX**: https://github.com/ml-explore/mlx
- **MLX-LM**: https://github.com/ml-explore/mlx-examples/tree/main/llms
- **Ollama**: https://ollama.ai
- **MLX Models**: https://huggingface.co/mlx-community

---

## Next Steps for Development

## Development Philosophy

**Remember**: This is a research tool for rapid experimentation, not production.

---

## Questions to Keep in Mind

While developing, constantly ask:

### Technical Level
1. Does this maintain the local-first philosophy?
2. Is this notebook-friendly?
3. Is this reproducible (logged, versioned)?
4. Is this extensible (easy for future additions)?
5. Does this help answer the core research questions?

### Paradigm Level
6. What hidden assumptions am I encoding in this design?
7. Could multi-agent dynamics work fundamentally differently than I'm assuming?
8. Does this tool enable testing hypotheses we haven't thought of yet?
9. What would a researcher need to falsify their theory about multi-agent intelligence?
10. Am I building a benchmark harness or a discovery engine?

### Research Impact
11. Will this help us understand *why* multi-agent works, not just *that* it works?
12. Does this make it easier to inspect internal representations and agent interactions?
13. Could this approach generalize to understanding biological or social intelligence?
14. What new research questions does this unlock?

---

## Guiding Principles

- **Radical Curiosity**: Never accept the status quo; question everything, even the question.
- **Theoretical Discipline**: Every claim must connect to measurable or falsifiable evidence.
- **Paradigm Awareness**: Understand how existing frameworks are situated within history — and how to leap beyond them.
- **Architectural Creativity**: Design learning systems that could, in principle, invent new science.
- **Empirical Elegance**: Prefer simple, interpretable mechanisms that yield emergent complexity.

---

**For detailed architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)
**For quick start**: See [QUICKSTART.md](QUICKSTART.md)
**For setup**: See [SETUP.md](SETUP.md)
