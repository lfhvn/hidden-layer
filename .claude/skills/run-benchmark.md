# Skill: Run Benchmark

You are an expert at running systematic benchmark evaluations for the Hidden Layer research platform.

## Task

When given:
- `benchmark`: Benchmark name (tombench, opentom, socialiqa, uicrit)
- `models`: Comma-separated list of models to test
- `strategies` (optional): Comma-separated strategies (default: single)

Do:

1. **Validate inputs**:
   - Check benchmark exists: `from harness import BENCHMARKS; check if benchmark in BENCHMARKS`
   - Parse model list
   - Parse strategy list

2. **Load benchmark**:
   ```python
   from harness import load_benchmark
   dataset = load_benchmark(benchmark_name)
   ```

3. **For each model** in the model list:
   - **For each strategy** in the strategy list:
     - Set up experiment tracking
     - Run the benchmark (all tasks or sample)
     - Track metrics
     - Save results

4. **Calculate aggregate metrics** for each model/strategy:
   - Accuracy (mean across all tasks)
   - Average latency
   - Total cost
   - Success rate

5. **Compare to published baselines**:
   ```python
   from harness import get_baseline_scores
   baselines = get_baseline_scores(benchmark_name)
   ```

6. **Generate comparison table**:
   - Show all models side-by-side
   - Include baselines for context
   - Highlight best performer

7. **Analyze results**:
   - Which model/strategy performed best?
   - Cost-effectiveness analysis
   - Latency vs quality tradeoffs
   - Patterns in failures

8. **Save results**:
   - Create directory: `experiments/benchmarks/{benchmark}_{date}/`
   - Save comparison table as markdown
   - Save raw results as JSON
   - Include visualizations if helpful

## Code Template

```python
from harness import load_benchmark, run_strategy, get_baseline_scores, BENCHMARKS
from harness import ExperimentConfig, get_tracker
import json
from datetime import datetime

# Configuration
benchmark_name = "{benchmark}"
models = "{models}".split(",")
strategies = "{strategies}".split(",")

# Load benchmark
print(f"Loading {benchmark_name}...")
dataset = load_benchmark(benchmark_name)

# Get baselines
baselines = get_baseline_scores(benchmark_name)

# Run evaluation
results = {}

for model in models:
    provider, model_name = model.split(":") if ":" in model else ("ollama", model)

    for strategy in strategies:
        print(f"\\nEvaluating {model} with {strategy} strategy...")

        config = ExperimentConfig(
            experiment_name=f"{benchmark_name}_{strategy}_{model_name}",
            strategy=strategy,
            provider=provider,
            model=model_name
        )

        tracker = get_tracker(use_wandb=True)
        tracker.start_experiment(config)

        # Run all tasks (or sample for quick test)
        task_results = []
        for i, problem in enumerate(dataset.problems[:10]):  # Sample first 10
            result = run_strategy(strategy, problem, provider=provider, model=model_name)
            task_results.append(result)
            tracker.log_result(result)

            if i % 5 == 0:
                print(f"  Completed {i+1}/{len(dataset.problems)} tasks")

        tracker.finish_experiment()

        # Calculate metrics
        results[f"{model}_{strategy}"] = {
            "model": model,
            "strategy": strategy,
            "accuracy": calculate_accuracy(task_results),
            "avg_latency": sum(r.latency_s for r in task_results) / len(task_results),
            "total_cost": sum(r.cost_usd for r in task_results),
        }

# Generate report
# ... (see Report Template below)
```

## Report Template

```markdown
# Benchmark Evaluation: {benchmark_name}

**Date**: {date}
**Benchmark**: {benchmark_name} ({num_tasks} tasks)
**Models Tested**: {models}
**Strategies Tested**: {strategies}

## Results Summary

### Performance Comparison

| Model | Strategy | Accuracy | Avg Latency (s) | Total Cost (USD) |
|-------|----------|----------|-----------------|------------------|
{for each result}
| {model} | {strategy} | {accuracy}% | {latency} | ${cost} |
{end}

### Comparison to Published Baselines

| Source | Accuracy |
|--------|----------|
| **Your Best** | {best_accuracy}% |
| Human | {baseline.human}% |
| GPT-4 | {baseline.gpt4}% |
| Published Baseline | {baseline.published}% |
| Current SOTA | {baseline.sota}% |

## Key Findings

1. **Best Performer**: {model} with {strategy} achieved {accuracy}%
2. **Cost-Effectiveness**: {analysis}
3. **Latency Analysis**: {analysis}

## Detailed Analysis

### Accuracy by Model

{For each model}
**{model}**:
- Accuracy: {accuracy}%
- Comparison to baseline: {+X% or -X%}
- Strengths: {where it excelled}
- Weaknesses: {where it struggled}

### Strategy Comparison

{For each strategy}
**{strategy}**:
- Average improvement over single: {+X%}
- Latency overhead: {Xs}
- When to use: {recommendations}

## Recommendations

Based on evaluation:

1. **For accuracy**: Use {model} with {strategy}
2. **For cost-effectiveness**: Use {model}
3. **For speed**: Use {model}
4. **For balance**: {recommendation}

## Next Steps

- [ ] Test on full benchmark (not just sample)
- [ ] Try additional strategies
- [ ] Investigate failure cases
- [ ] Compare to other benchmarks

## Raw Results

See `results.json` for complete data.

## Visualizations

{If created charts}
- See `comparison.png` for visual comparison
- See `latency_vs_accuracy.png` for tradeoff analysis
```

## Important Notes

- For quick tests, use sample of tasks (first 10-20)
- For full evaluation, run all tasks (may take hours)
- Save intermediate results in case of crashes
- Use W&B if available for live tracking
- Compare apples-to-apples (same temperature, same tasks)

## Example Usage

```
User: "Use run-benchmark skill with benchmark=tombench models=llama3.2:3b,llama3.1:8b,claude-3-5-sonnet strategies=single,debate"

You:
1. Load ToMBench
2. For each model (llama3.2:3b, llama3.1:8b, claude-3-5-sonnet):
   - For each strategy (single, debate):
     - Run benchmark
     - Track metrics
3. Generate comparison table
4. Save to experiments/benchmarks/tombench_{date}/
```

## Quick Test Mode

For rapid iteration:

```
User: "Quick test benchmark=tombench models=llama3.2:3b sample=10"

You:
- Run only first 10 tasks
- Use single strategy
- Generate quick comparison
- Note this is a sample (not full evaluation)
```

## Error Handling

- If benchmark loading fails: Check benchmark name, provide available list
- If model fails: Continue with other models, note failure
- If out of API credits: Use local models, warn user
- Save partial results if interrupted
