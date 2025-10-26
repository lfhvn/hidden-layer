# Quick Reference - Harness Cheat Sheet

## Setup
```bash
source venv/bin/activate
ollama serve &
cd notebooks && jupyter notebook
```

## Basic LLM Call
```python
from harness import llm_call

# Local
response = llm_call("Hello!", provider="ollama", model="llama3.2:latest")

# MLX
response = llm_call("Hello!", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")

# API
response = llm_call("Hello!", provider="anthropic", model="claude-3-5-haiku-20241022")
```

## Run Strategies
```python
from harness import run_strategy

# Single model
result = run_strategy("single", "What is 2+2?", provider="ollama")

# Debate (3 agents + judge)
result = run_strategy("debate", "Should we...", n_debaters=3, provider="ollama")

# Self-consistency (5 samples)
result = run_strategy("self_consistency", "What is...", n_samples=5, provider="ollama")

# Manager-worker
result = run_strategy("manager_worker", "Plan a...", n_workers=3, provider="ollama")
```

## Experiment Tracking
```python
from harness import ExperimentConfig, ExperimentResult, get_tracker

# Config
config = ExperimentConfig(
    experiment_name="my_experiment",
    task_type="reasoning",
    strategy="single",
    provider="ollama",
    model="llama3.2:latest"
)

# Track
tracker = get_tracker()
tracker.start_experiment(config)

# Log result
exp_result = ExperimentResult(config=config, task_input="...", output="...", ...)
tracker.log_result(exp_result)

# Finish
summary = tracker.finish_experiment()
```

## Evaluation
```python
from harness import evaluate_task, llm_judge

# Auto-evaluate based on task type
task = {"input": "What is 2+2?", "expected": "4", "eval_type": "numeric"}
scores = evaluate_task(task, output="The answer is 4")
# Returns: {'coherence': 1.0, 'accuracy': 1.0}

# LLM judge
result = llm_judge(
    task_input="Write a story",
    output="Once upon a time...",
    criteria="creativity"
)
# Returns: {'score': 0.8, 'reasoning': '...'}
```

## Model Recommendations

### Fast Iteration
- `llama3.2:latest` (Ollama, 3B)
- `mistral:latest` (Ollama, 7B)

### High Quality
- `mlx-community/Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Meta-Llama-3.1-70B-Instruct-4bit`

### API Comparison
- `claude-3-5-haiku-20241022` (fast, cheap)
- `claude-3-5-sonnet-20241022` (best quality)

## Common Patterns

### Compare strategies
```python
configs = [
    ("single", {"provider": "ollama"}),
    ("debate", {"provider": "ollama", "n_debaters": 3}),
]

for strategy, kwargs in configs:
    result = run_strategy(strategy, task_input, **kwargs)
    print(f"{strategy}: {result.latency_s:.2f}s")
```

### Batch evaluation
```python
tasks = [...]
tracker = get_tracker()
tracker.start_experiment(config)

for task in tasks:
    result = run_strategy(...)
    scores = evaluate_task(task, result.output)
    
    exp_result = ExperimentResult(...)
    exp_result.eval_scores = scores
    tracker.log_result(exp_result)

summary = tracker.finish_experiment()
```

### Compare models
```python
models = ["llama3.2:latest", "mistral:latest"]
for model in models:
    result = run_strategy("single", task, provider="ollama", model=model)
    print(f"{model}: {result.output[:50]}...")
```

## File Locations

```
experiments/              # Auto-generated logs
├── run_name_date_hash/
│   ├── config.json      # Experiment config
│   ├── results.jsonl    # Streaming results
│   └── summary.json     # Aggregated metrics

notebooks/               # Your experiments
code/harness/           # Core library
```

## Troubleshooting

```bash
# Ollama not responding?
killall ollama && ollama serve &

# Check which models you have
ollama list

# Pull a new model
ollama pull llama3.2

# Clear cache
rm -rf ~/.ollama/models/
rm -rf ~/.cache/huggingface/
```

## Performance Tips

- Start with small models (3B/7B) for rapid iteration
- Use 4-bit quantization for speed
- Batch size 1-4 for most experiments
- Keep temperature 0.7-0.9 for diverse outputs
- Use temperature 0.1-0.3 for deterministic eval

## Next Steps

1. Run `notebooks/01_baseline_experiments.ipynb`
2. Try different strategies on your tasks
3. Compare local vs API models
4. Fine-tune with MLX (see SETUP.md)

## Resources

- Harness code: `code/harness/`
- Setup guide: `SETUP.md`
- Full README: `README.md`
- MLX examples: https://github.com/ml-explore/mlx-examples
