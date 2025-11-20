# ACE Experiments

Benchmark integration and evaluation scripts for the ACE framework.

---

## Available Benchmarks

### Agent Benchmarks

**SimpleAgentBenchmark**: Multi-step reasoning and planning tasks
- Prioritization, scheduling, file management
- Multi-step workflows
- Constraint satisfaction
- 8 test tasks + 3 training tasks

**ToolUseBenchmark**: Function calling and tool selection
- Calculator, search, file operations
- Multi-tool composition
- 4 test tasks

**ReasoningBenchmark**: Logic and causal reasoning
- Logical deduction
- Mathematical reasoning
- Causal inference
- Planning and scheduling
- 4 test tasks

### Domain Benchmarks

**MathBenchmark**: Mathematical problem solving
- Arithmetic, fractions, percentages
- Word problems
- Algebra
- Multi-step problems
- 8 test tasks + 3 training tasks

**FinanceBenchmark**: Finance domain reasoning
- ROI and interest calculations
- Budget analysis
- Financial terminology
- Entity recognition (FiNER-inspired)
- 7 test tasks

**CodeBenchmark**: Code understanding and generation
- Code comprehension
- Bug identification
- Code completion
- Algorithm selection
- Complexity analysis
- 5 test tasks

---

## Running Experiments

### Offline ACE (Pre-deployment Optimization)

Optimize a context on training tasks, then evaluate on test tasks:

```bash
# Single benchmark
cd alignment/ace
python experiments/offline_ace.py --benchmark math --iterations 3

# Available benchmarks: agent, tools, reasoning, math, finance, code

# Custom configuration
python experiments/offline_ace.py \
  --benchmark finance \
  --iterations 5 \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --save-dir results/offline
```

**Output**:
- Baseline vs optimized performance
- Learned strategies and pitfalls
- Context evolution over iterations
- Results saved to `results/offline/{benchmark}/`

### Online ACE (Continuous Adaptation)

Adapt context continuously during task execution:

```bash
# Single benchmark
python experiments/online_ace.py --benchmark math --buffer-size 10

# Custom configuration
python experiments/online_ace.py \
  --benchmark agent \
  --buffer-size 10 \
  --update-frequency 5 \
  --save-dir results/online
```

**Output**:
- Performance improvement over time
- Context updates and adaptations
- Final evolved context
- Results saved to `results/online/{benchmark}/`

### All Benchmarks

Run offline ACE on all benchmarks:

```bash
python experiments/run_all_benchmarks.py --iterations 3

# Generates comparison report across all benchmarks
```

**Output**:
- Summary table comparing all benchmarks
- Average improvement across domains
- Results saved to `results/all_benchmarks/{timestamp}/`

---

## Experiment Options

### Common Arguments

- `--iterations` / `--buffer-size`: Number of iterations or buffer size
- `--provider`: LLM provider (`anthropic`, `openai`, `ollama`)
- `--model`: Model name (e.g., `claude-3-5-sonnet-20241022`)
- `--save-dir`: Directory to save results

### Offline ACE Specific

- `--tasks-per-iteration`: Limit tasks per iteration (None = all)

### Online ACE Specific

- `--update-frequency`: Update context every N tasks

---

## Results Structure

```
results/
├── offline/
│   ├── math/
│   │   ├── context_iter_0.json
│   │   ├── context_iter_1.json
│   │   ├── final_context.yaml
│   │   ├── trajectories_iter_0.json
│   │   ├── delta_iter_0.json
│   │   └── history.json
│   └── ...
├── online/
│   ├── agent/
│   │   ├── context_update_1.json
│   │   ├── final_context.yaml
│   │   └── history.json
│   └── ...
└── all_benchmarks/
    └── 20250120_143052/
        ├── summary.json
        ├── math/
        ├── finance/
        └── ...
```

---

## Adding Custom Benchmarks

### 1. Create Benchmark Class

```python
from experiments.benchmarks.base import Benchmark, Task

class MyBenchmark(Benchmark):
    def __init__(self):
        super().__init__("MyBenchmark")

    def load_tasks(self) -> List[Task]:
        return [
            Task(
                id="task_1",
                description="Task description",
                expected_output="expected answer",
                metadata={"split": "test"},
                difficulty="medium"
            ),
            # ... more tasks
        ]

    def evaluate(self, task: Task, output: str) -> Dict[str, Any]:
        correct = task.expected_output.lower() in output.lower()
        return {
            "correct": correct,
            "feedback": "Correct" if correct else f"Expected: {task.expected_output}"
        }
```

### 2. Add to Experiment Scripts

```python
from experiments.benchmarks import MyBenchmark

benchmarks = {
    "my_benchmark": MyBenchmark(),
    # ... existing benchmarks
}
```

### 3. Run Experiments

```bash
python experiments/offline_ace.py --benchmark my_benchmark
```

---

## Evaluating Custom Tasks

Use benchmarks with ACE directly:

```python
from alignment.ace.src import ACEFramework, Context
from alignment.ace.experiments.benchmarks import MathBenchmark

# Initialize
benchmark = MathBenchmark()
ace = ACEFramework(provider="anthropic", model="claude-3-5-sonnet-20241022")

# Create context
context = Context(version=0, domain="math", base_prompt="You are a math expert.")

# Get tasks and evaluator
tasks = benchmark.get_tasks(split="train")
task_descriptions = [t.description for t in tasks]
evaluator = benchmark.create_evaluator()

# Run offline ACE
optimized_context = ace.run_offline(
    tasks=task_descriptions,
    initial_context=context,
    num_iterations=3,
    evaluator=evaluator
)

# Evaluate on test set
test_tasks = benchmark.get_tasks(split="test")
test_descriptions = [t.description for t in test_tasks]
results = ace.evaluate(test_descriptions, optimized_context, evaluator)

print(f"Test accuracy: {results['success_rate']:.1%}")
```

---

## Benchmark Statistics

| Benchmark | Train Tasks | Test Tasks | Categories | Difficulty Range |
|-----------|-------------|------------|------------|------------------|
| SimpleAgent | 3 | 8 | 6 | Easy - Hard |
| ToolUse | 0 | 4 | 3 | Easy - Hard |
| Reasoning | 0 | 4 | 4 | Easy - Medium |
| Math | 3 | 8 | 6 | Easy - Medium |
| Finance | 0 | 7 | 6 | Easy - Hard |
| Code | 0 | 5 | 5 | Easy - Medium |

---

## Expected Performance

Based on the ACE paper (Zhang et al., 2025):

- **Agent benchmarks**: +10.6% improvement expected
- **Domain benchmarks**: +8.6% improvement expected
- **Efficiency**: 75-90% reduction in tokens/latency vs baselines

Your results may vary depending on:
- Model capability
- Task difficulty
- Number of iterations
- Quality of training tasks

---

## Troubleshooting

### Low Improvement

- Increase `--iterations` (try 5-7)
- Check if training tasks are representative
- Ensure evaluator is working correctly
- Try different models

### Context Collapse

- ACE should prevent this automatically via structured deltas
- Check `context_size` in history.json
- Verify strategies aren't being over-pruned

### High Cost

- Use local models (`--provider ollama`)
- Reduce `--iterations`
- Limit `--tasks-per-iteration`
- Use smaller models for initial experiments

---

## Citation

If you use these benchmarks, please cite:

```bibtex
@article{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and others},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025}
}
```
