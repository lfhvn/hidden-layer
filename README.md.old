# Hidden Layer - Multi-Agent LLM Research Platform

A comprehensive research platform for exploring multi-agent LLM systems, consisting of three integrated subsystems optimized for local-first experimentation on Apple Silicon.

## 🎯 What is Hidden Layer?

Hidden Layer is a toolkit for researching **when and why multi-agent LLM strategies outperform single models**. It includes:

- **Harness** - Core multi-agent infrastructure with 5 strategies and experiment tracking
- **CRIT** - Collective Reasoning for Iterative Testing (design critique evaluation)
- **SELPHI** - Study of Epistemic and Logical Processing (theory of mind testing)

**Total**: 6,613 lines of Python + 5,641 lines of documentation

## 🚀 Quick Start

```bash
# 1. Setup environment
source venv/bin/activate
ollama serve &

# 2. Test the harness
python -c "from harness import llm_call; print(llm_call('Hi!', provider='ollama').text)"

# 3. Test CRIT
python -c "from crit import MOBILE_CHECKOUT, run_critique_strategy; print('CRIT ready!')"

# 4. Test SELPHI
python -c "from selphi import SALLY_ANNE, run_scenario; print('SELPHI ready!')"

# 5. Open notebooks
cd notebooks && jupyter notebook
```

## 📦 System Overview

### Subsystem 1: Harness (Core Infrastructure)

Multi-agent infrastructure for LLM experimentation.

**Features**:
- 🔌 **Unified LLM Interface**: MLX, Ollama, Anthropic, OpenAI
- 🤖 **5 Multi-Agent Strategies**: single, debate, self-consistency, manager-worker, consensus
- 📊 **Experiment Tracking**: Automatic logging, metrics, reproducibility
- ⚡ **M4 Max Optimized**: 128GB RAM, run multiple 7B models in parallel

```python
from harness import run_strategy

# Run a debate with 3 agents
result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

print(f"Answer: {result.output}")
print(f"Latency: {result.latency_s:.2f}s")
```

**Available Strategies**:
- `single` - Baseline single-model inference
- `debate` - n-agent debate with judge
- `self_consistency` - Sample multiple times + majority vote
- `manager_worker` - Decompose → parallel execution → synthesis
- `consensus` - Multiple agents find agreement

### Subsystem 2: CRIT (Design Critique)

Testing collective design critique from multiple expert perspectives.

**Research Question**: Can multi-perspective critique produce better design solutions than a single generalist?

**Features**:
- 📐 **8 Design Problems**: UI/UX, API design, system architecture, data modeling
- 👥 **9 Expert Perspectives**: usability, security, accessibility, performance, etc.
- 🔄 **4 Critique Strategies**: single, multi-perspective, iterative, adversarial
- 📊 **UICrit Benchmark**: 11,344 critiques from Google Research (UIST 2024)

```python
from crit import run_critique_strategy, MOBILE_CHECKOUT

# Multi-perspective critique with synthesis
result = run_critique_strategy(
    "multi_perspective",
    MOBILE_CHECKOUT,
    perspectives=["usability", "security", "accessibility"],
    synthesize=True
)

print(result.critique)
```

**Design Domains**:
- UI/UX Design (mobile checkout, dashboards)
- API Design (REST versioning, GraphQL)
- System Architecture (microservices, caching)
- Data Modeling (permissions)
- Workflow Design (approvals)

### Subsystem 3: SELPHI (Theory of Mind)

Evaluating theory of mind and epistemic reasoning in LLMs.

**Research Question**: How well can language models understand mental states, beliefs, and perspective-taking?

**Features**:
- 🧠 **9+ ToM Scenarios**: False belief, knowledge attribution, perspective taking
- 📚 **3 Major Benchmarks**: ToMBench (388), OpenToM (696), SocialIQA (38k)
- 🎯 **7 ToM Types**: False belief, second-order beliefs, epistemic states, etc.
- 📊 **Multiple Evaluation Methods**: Semantic matching, LLM-as-judge

```python
from selphi import run_scenario, SALLY_ANNE, evaluate_scenario

# Run classic false belief test
result = run_scenario(SALLY_ANNE, provider="ollama")

# Evaluate response
eval_result = evaluate_scenario(SALLY_ANNE, result.model_response)
print(f"Score: {eval_result['average_score']:.2f}")
```

**ToM Types Covered**:
- False Belief (Sally-Anne, Chocolate Bar)
- Second-Order Belief (Birthday Puppy)
- Knowledge Attribution (Museum Trip)
- Perspective Taking (Painted Room)
- Belief Updating (Ice Cream Van)
- Epistemic States (Library Book)
- Pragmatic Reasoning (Restaurant Bill)

## 📂 Project Structure

```
hidden-layer/
├── code/
│   ├── harness/          # Core infrastructure (1,900 LOC)
│   │   ├── llm_provider.py       # Unified LLM interface
│   │   ├── strategies.py         # Multi-agent strategies
│   │   ├── experiment_tracker.py # Logging & tracking
│   │   ├── evals.py              # Evaluation functions
│   │   ├── benchmarks.py         # Unified benchmark interface
│   │   ├── rationale.py          # Reasoning extraction
│   │   └── model_config.py       # YAML config management
│   │
│   ├── crit/             # Design critique (1,500+ LOC)
│   │   ├── problems.py           # 8 design problems
│   │   ├── strategies.py         # 4 critique strategies
│   │   ├── evals.py              # Critique quality metrics
│   │   └── benchmarks.py         # UICrit dataset loader
│   │
│   └── selphi/           # Theory of Mind (1,200+ LOC)
│       ├── scenarios.py          # 9+ ToM scenarios
│       ├── evals.py              # ToM evaluation
│       └── benchmarks.py         # ToMBench, OpenToM, SocialIQA
│
├── notebooks/            # Jupyter notebooks
├── experiments/          # Auto-generated experiment logs
├── config/              # Model configurations (YAML)
│
├── README.md            # This file - Project overview
├── QUICKSTART.md        # Cheat sheet for common tasks
├── ARCHITECTURE.md      # Deep dive into all subsystems
└── SETUP.md             # Installation guide for M4 Max
```

## 💡 Usage Examples

### Example 1: Basic Multi-Agent Debate

```python
from harness import run_strategy, get_tracker, ExperimentConfig

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

# Run debate
result = run_strategy(
    "debate",
    "Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama"
)

print(result.output)

# See individual arguments
for i, arg in enumerate(result.metadata['arguments']):
    print(f"\nDebater {i+1}: {arg}")
```

### Example 2: Design Critique Comparison

```python
from crit import run_critique_strategy, MOBILE_CHECKOUT, evaluate_critique

# Single critic baseline
single_result = run_critique_strategy("single", MOBILE_CHECKOUT)

# Multi-perspective critique
multi_result = run_critique_strategy(
    "multi_perspective",
    MOBILE_CHECKOUT,
    perspectives=["usability", "security", "accessibility", "performance"]
)

# Evaluate both
single_metrics = evaluate_critique(single_result.critique, MOBILE_CHECKOUT)
multi_metrics = evaluate_critique(multi_result.critique, MOBILE_CHECKOUT)

print(f"Single coverage: {single_metrics['coverage_score']:.2f}")
print(f"Multi coverage: {multi_metrics['coverage_score']:.2f}")
```

### Example 3: Theory of Mind Testing

```python
from selphi import (
    run_multiple_scenarios,
    get_scenarios_by_difficulty,
    evaluate_batch,
    compare_models
)

# Get all medium difficulty scenarios
scenarios = get_scenarios_by_difficulty("medium")

# Compare two models
comparison = compare_models(
    scenarios,
    providers=["ollama", "anthropic"],
    models=["llama3.2:latest", "claude-3-5-sonnet-20241022"]
)

print(f"Ollama avg score: {comparison['ollama']['avg_score']:.2f}")
print(f"Claude avg score: {comparison['anthropic']['avg_score']:.2f}")
```

### Example 4: Benchmark Evaluation

```python
from harness import load_benchmark, get_baseline_scores

# List all available benchmarks
from harness import BENCHMARKS
for name, info in BENCHMARKS.items():
    print(f"{name}: {info.description}")

# Load ToMBench
tombench = load_benchmark('tombench', split='test')

# Get baseline scores
scores = get_baseline_scores('tombench')
print(f"Human performance: {scores['human_performance']:.2f}")
print(f"GPT-4 performance: {scores['gpt4_performance']:.2f}")

# Run your model
from selphi import run_multiple_scenarios
results = run_multiple_scenarios(tombench.problems, provider="ollama")
```

## 🔬 Research Questions

This platform is designed to explore:

### Core Questions
1. **When do multi-agent strategies outperform single models?**
   - Problem types (reasoning, creative, planning)
   - Complexity levels
   - Domain-specific patterns

2. **Why do they outperform?**
   - Coverage (more perspectives)
   - Diversity (different approaches)
   - Synthesis (combining insights)
   - Error correction (catching mistakes)

3. **What are the tradeoffs?**
   - Latency (3x agents = 3x time?)
   - Cost (local vs API)
   - Quality improvements
   - Diminishing returns

### Subsystem-Specific Questions

**CRIT**:
- Do multi-perspective critiques cover more design issues?
- Is iterative refinement better than one-shot critique?
- Can adversarial critique find edge cases?

**SELPHI**:
- Which ToM types are hardest for LLMs?
- Do larger models have better ToM?
- Can fine-tuning improve ToM performance?

## 🛠 Hardware Optimization (M4 Max)

### Memory Usage

With 128GB unified memory, you can:

- **Run 70B models**: ~35GB (4-bit quantized)
- **3-4 agents in parallel**: 3x 7B models (~12GB total)
- **Fine-tune 13B models**: LoRA training (~20GB)

### Recommended Configurations

| Use Case | Model | Memory | Speed |
|----------|-------|--------|-------|
| Fast iteration | llama3.2:3b | 2GB | ~120 tok/s |
| Quality experiments | llama3.1:70b | 35GB | ~15 tok/s |
| Multi-agent (3x) | llama3.1:8b | 12GB | ~80 tok/s each |
| API baseline | claude-3-5-sonnet | N/A | Variable |

### Performance Tips

1. **Start small**: Use 3B/7B for iteration, 70B for quality
2. **Parallel experiments**: Run multiple small models simultaneously
3. **Mixed approach**: Local debaters + API judge
4. **Cache models**: Ollama caches for faster loading

## 📊 Experiment Tracking

All experiments auto-log to `experiments/{name}_{timestamp}_{hash}/`:

```
experiments/debate_energy_20251103_143022_a3f9/
├── config.json          # Experiment configuration
├── results.jsonl        # Streaming results (one per line)
├── summary.json         # Aggregated metrics
└── README.md            # Human-readable summary
```

**Load and compare**:
```python
from harness import compare_experiments

comparison = compare_experiments(
    ["experiments/run1/", "experiments/run2/"],
    metric="latency_s"
)
```

## 📚 Documentation

- **README.md** (this file) - Project overview and quick start
- **QUICKSTART.md** - Cheat sheet for common operations
- **ARCHITECTURE.md** - Deep dive into all subsystems
- **SETUP.md** - Detailed installation guide for M4 Max
- **BENCHMARKS.md** - Benchmark datasets and usage
- **CLAUDE.md** - Development guide for extending the platform

Subsystem-specific:
- **code/crit/README.md** - CRIT design critique details
- **code/selphi/README.md** - SELPHI theory of mind details
- **config/README.md** - Model configuration guide

## 🔮 Roadmap

### ✅ Completed
- Core harness with 5 multi-agent strategies
- CRIT subsystem with 8 problems, 4 strategies
- SELPHI subsystem with 9+ scenarios, 3 benchmarks
- Experiment tracking and reproducibility
- Model configuration management
- Unified benchmark interface

### 🚧 In Progress
- Baseline experiments across all subsystems
- Performance benchmarking (local vs API)
- Fine-tuning workflows with MLX

### 📋 Planned
- Hidden layer interpretability tools
- Advanced visualization (attention, confidence)
- Production API serving (if needed)
- Distributed execution for large experiments

## 🤝 Contributing

This is a research lab, but feel free to:
- Fork and extend for your own experiments
- Submit issues for bugs
- Share interesting findings

## 📄 License

MIT (code) / CC-BY (documentation)

## 🙏 Acknowledgments

- **Apple MLX Team** - Apple Silicon optimization
- **Ollama** - Making local models accessible
- **Google Research** - UICrit dataset (UIST 2024)
- **ToM Research Community** - ToMBench, OpenToM benchmarks
- **Meta, Mistral AI** - Open-weight models (Llama, Mistral)
- **Anthropic, OpenAI** - API access for comparison baselines

## 🔗 Resources

- MLX Framework: https://github.com/ml-explore/mlx
- Ollama: https://ollama.ai
- MLX Models: https://huggingface.co/mlx-community
- UICrit Dataset: https://github.com/google-research-datasets/uicrit
- ToMBench: https://github.com/wadimiusz/ToMBench

---

**Ready to start?** See [QUICKSTART.md](QUICKSTART.md) for common workflows, or [SETUP.md](SETUP.md) for installation.
