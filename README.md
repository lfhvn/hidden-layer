# Research Lab - Agentic Simulation Harness

A notebook-first toolkit for comparing single-model and multi-agent LLM strategies on your local Mac (M4 Max) with MLX and Ollama.

## Quick Start

```bash
# 1. Setup (see SETUP.md for details)
source venv/bin/activate
ollama serve &

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run a test
python -c "from code.harness import llm_call; print(llm_call('Hi!', provider='ollama'))"

# 4. Open notebooks
cd notebooks && jupyter notebook
```

## Project Structure

```
.
├── code/
│   └── harness/              # Core simulation harness
│       ├── llm_provider.py   # Unified LLM interface (MLX, Ollama, APIs)
│       ├── strategies.py     # Multi-agent strategies
│       └── experiment_tracker.py  # Experiment logging
│
├── notebooks/
│   ├── 01_baseline_experiments.ipynb       # Single-model baselines
│   ├── 02_debate_experiments.ipynb         # Debate & consensus strategies
│   ├── 03_consensus_experiments.ipynb      # Consensus (no judge)
│   ├── 04_benchmark_evaluation.ipynb       # Test on standard benchmarks
│   ├── 05_interdisciplinary_team.ipynb     # Domain expert collaboration
│   └── 06_design_critique.ipynb            # Iterative feedback & refinement
│
├── experiments/              # Auto-generated experiment logs
│
├── SETUP.md                  # Detailed setup instructions
└── requirements.txt          # Python dependencies
```

## Features

### ✅ Hybrid LLM Support
- **Local**: MLX (Apple Silicon optimized), Ollama
- **API**: Anthropic Claude, OpenAI
- Unified interface: switch providers with one parameter

### ✅ Multi-Agent Strategies
- Single-model baseline
- Debate (n-agent with judge)
- Consensus (agents build agreement without judge)
- Self-consistency (sample & aggregate)
- Manager-worker (decompose, execute, synthesize)
- **Design critique** (iterative feedback & refinement)
- **Interdisciplinary team** (domain experts collaborate)

### ✅ Benchmark Evaluation
- 10 established benchmarks (GSM8K, MMLU, GPQA, HumanEval, and more)
- Compare strategies against published SOTA baselines
- Test custom strategies on standard datasets
- Automatic accuracy, latency, and cost tracking

### ✅ Experiment Tracking
- Automatic logging to disk (JSON/JSONL)
- Metrics: latency, tokens, cost
- Compare across runs
- Notebook-friendly API

### ✅ M4 Max Optimized
- Local inference with 128GB RAM
- Fine-tuning support (LoRA)
- Run 3-4 small models simultaneously
- Handle up to 70B models

## Usage Examples

### Basic LLM Call

```python
from harness import llm_call

# Local Ollama
response = llm_call("What is 2+2?", provider="ollama", model="llama3.2:latest")
print(response.text)

# Local MLX
response = llm_call("What is 2+2?", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
print(response.text)

# API (for comparison)
response = llm_call("What is 2+2?", provider="anthropic", model="claude-3-5-haiku-20241022")
print(f"Cost: ${response.cost_usd:.4f}")
```

### Run Multi-Agent Strategy

```python
from harness import run_strategy

# Debate strategy: 3 agents + judge
result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)

print(f"Final answer: {result.output}")
print(f"Latency: {result.latency_s:.2f}s")
print(f"Tokens: {result.tokens_in + result.tokens_out}")

# See all debater arguments
for i, arg in enumerate(result.metadata['arguments']):
    print(f"Debater {i+1}: {arg[:100]}...")
```

### Track Experiments

```python
from harness import ExperimentConfig, ExperimentResult, get_tracker

# Configure experiment
config = ExperimentConfig(
    experiment_name="baseline_reasoning",
    task_type="reasoning",
    strategy="single",
    provider="ollama",
    model="llama3.2:latest",
    temperature=0.7
)

# Start tracking
tracker = get_tracker()
run_dir = tracker.start_experiment(config)

# Run tasks and log results
for task in tasks:
    result = run_strategy(config.strategy, task, ...)
    
    exp_result = ExperimentResult(
        config=config,
        task_input=task,
        output=result.output,
        latency_s=result.latency_s,
        ...
    )
    
    tracker.log_result(exp_result)

# Finish and get summary
summary = tracker.finish_experiment()
print(f"Average latency: {summary['avg_latency_s']:.2f}s")
```

## Research Questions

This harness is designed to explore:

1. **When do multi-agent strategies outperform single models?**
   - Debate for reasoning tasks?
   - Manager-worker for complex planning?
   - Self-consistency for math/logic?

2. **What are the cost/latency tradeoffs?**
   - 3 local models vs 1 API call?
   - Quality vs speed?

3. **How does fine-tuning impact results?**
   - LoRA on task-specific data
   - Compare base vs fine-tuned in multi-agent setting

## Notebooks

### `01_baseline_experiments.ipynb`
- Test local model setup
- Run seed tasks with single-model strategy
- Establish performance baselines
- Generate metrics dashboard

### `02_multi_agent_comparison.ipynb`
- Compare strategies on same tasks
- Debate vs single model
- Analyze when multi-agent helps
- Cost/latency analysis

### Coming Soon
- `03_fine_tuning_with_mlx.ipynb` - LoRA fine-tuning workflow
- `04_hidden_layer_interpretability.ipynb` - Probe attention, confidence
- `05_eval_deep_dive.ipynb` - LLM-as-judge, human eval

## Experiment Outputs

All experiments auto-log to `experiments/{name}_{timestamp}_{hash}/`:

```
experiments/baseline_llama32_20251026_143022_a3f9/
├── config.json          # Experiment configuration
├── results.jsonl        # One result per line (streaming)
├── summary.json         # Aggregated metrics
└── README.md            # Human-readable summary
```

Load and compare experiments:

```python
from harness import compare_experiments

comparison = compare_experiments(
    ["experiments/run1/", "experiments/run2/"],
    metric="latency_s"
)
```

## Tips for M4 Max

**Model Selection:**
- Start small: Llama 3.2 3B (fast iteration)
- Production quality: Mistral 7B, Llama 3.1 8B
- High-end baseline: Llama 3.1 70B (you have the RAM!)

**Multi-Agent:**
- 3x 7B models run comfortably in parallel
- Mix local + API: debate with local, judge with Claude

**Fine-Tuning:**
- 7B models: ~2-4 hours for LoRA
- 13B models: ~6-8 hours
- Use 4-bit quantization for speed

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[SETUP.md](SETUP.md)** - Detailed environment setup
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Technical architecture
- **[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** - Using benchmark evaluation
- **[CUSTOM_STRATEGIES_GUIDE.md](CUSTOM_STRATEGIES_GUIDE.md)** - Design critique & XFN team strategies
- **[DEBATE_GUIDE.md](DEBATE_GUIDE.md)** - Debate and consensus strategies
- **[CLAUDE.md](CLAUDE.md)** - Development guide for AI assistants

## Contributing

This is a personal research lab, but feel free to:
- Fork and extend for your own experiments
- Submit issues for bugs
- Share interesting findings

## License

MIT (code) / CC-BY (documentation)

## Acknowledgments

- MLX team at Apple
- Ollama for making local models accessible
- Meta (Llama), Mistral AI for open weights
