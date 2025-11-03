# Weights & Biases (W&B) Integration Guide

**Last Updated**: 2025-11-03
**Purpose**: Professional experiment tracking with beautiful dashboards

---

## What is Weights & Biases?

W&B is the **industry-standard platform** for ML experiment tracking. Think of it as GitHub for your experiments.

**Why you need it**:
- 📊 **Beautiful dashboards** - Interactive visualizations
- 🔍 **Easy comparison** - Compare 100s of experiments visually
- 📈 **Track metrics** - Latency, cost, accuracy, etc.
- 🌐 **Shareable** - Public links to results
- 💾 **Artifact versioning** - Track models, datasets
- 🆓 **Free** for individuals and academics

---

## What W&B Solves For You

### Before W&B

**Your current workflow**:
```python
# Run experiment
result = run_strategy("debate", task)

# Results saved to JSON
experiments/debate_20251103_143022/results.jsonl

# To analyze:
- Open file
- Load JSON
- Parse data
- Create plots manually
- Compare experiments manually
```

**Finding insights**:
```python
# Which strategy is best?
import pandas as pd
import glob

results = []
for file in glob('experiments/*/results.jsonl'):
    # Load, parse, aggregate...
    # 30 lines of code
```

### After W&B

**Your new workflow**:
```python
# Run experiment (same code, 3 lines added)
result = run_strategy("debate", task)

# W&B automatically tracks everything
# Go to wandb.ai → See dashboard with:
```

**Finding insights**:
- Open browser → wandb.ai
- See all experiments
- Interactive charts automatically
- Filter by strategy, model, date
- Compare side-by-side
- Share with link

---

## Setup (10 Minutes)

### Step 1: Install W&B

```bash
# Activate your environment
source venv/bin/activate

# Install W&B
pip install wandb

# Or add to requirements.txt
echo "wandb>=0.16.0" >> requirements.txt
pip install -r requirements.txt
```

### Step 2: Create W&B Account

1. Go to https://wandb.ai/signup
2. Sign up (free for individuals)
3. Verify email

**Academic/Research?** You can upgrade to free team account:
- Go to Settings → Billing
- Apply for academic account
- Get unlimited storage + features

### Step 3: Login

```bash
wandb login
```

This will:
1. Open browser
2. Show your API key
3. Paste key into terminal (or hit enter to auto-paste)

**Done!** You're authenticated.

**Finding your key later**:
- Go to https://wandb.ai/settings
- Copy API key
- Run `wandb login` and paste

---

## Integration With Hidden Layer

I've added W&B integration to your experiment tracker. Here's what changed:

### Changes Made

**1. Updated `code/harness/experiment_tracker.py`**:
```python
# Added W&B support (optional, controlled by flag)
import wandb

class ExperimentTracker:
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        # ... existing code

    def start_experiment(self, config):
        if self.use_wandb:
            wandb.init(
                project="hidden-layer",
                name=config.experiment_name,
                config={
                    "strategy": config.strategy,
                    "provider": config.provider,
                    "model": config.model,
                    "temperature": config.temperature
                }
            )
        # ... existing code

    def log_result(self, result):
        if self.use_wandb:
            wandb.log({
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_usd": result.cost_usd
            })
        # ... existing code
```

**2. Backward Compatible**:
- Existing code works exactly the same
- W&B is **optional** - defaults to off
- Enable with `use_wandb=True`

---

## How To Use

### Basic Usage

**Option 1: Enable globally**:
```python
from harness import get_tracker

# Enable W&B for all experiments
tracker = get_tracker(use_wandb=True)
```

**Option 2: Enable per-experiment**:
```python
from harness import ExperimentTracker, ExperimentConfig

config = ExperimentConfig(
    experiment_name="debate_energy",
    strategy="debate",
    provider="ollama"
)

# Enable W&B for this experiment
tracker = ExperimentTracker(use_wandb=True)
tracker.start_experiment(config)

# Run experiments
result = run_strategy("debate", task, provider="ollama")
tracker.log_result(result)

tracker.finish_experiment()
```

**Option 3: Environment variable** (recommended):
```bash
# In your terminal or .env file
export WANDB_ENABLED=true

# Or in .env
echo "WANDB_ENABLED=true" >> .env
```

```python
import os
use_wandb = os.getenv('WANDB_ENABLED', 'false').lower() == 'true'
tracker = get_tracker(use_wandb=use_wandb)
```

### Running Experiments With W&B

**Example 1: Single Experiment**:
```python
from harness import run_strategy, get_tracker, ExperimentConfig

# Configure
config = ExperimentConfig(
    experiment_name="debate_renewable_energy",
    strategy="debate",
    provider="ollama",
    model="llama3.2:latest"
)

# Start with W&B
tracker = get_tracker(use_wandb=True)
tracker.start_experiment(config)

# Run
result = run_strategy(
    "debate",
    "Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama"
)

# Track
tracker.log_result(result)
tracker.finish_experiment()

# Visit wandb.ai to see results!
```

**Example 2: Benchmark Evaluation**:
```python
from harness import load_benchmark, run_strategy, get_tracker, ExperimentConfig
from selphi import run_multiple_scenarios

# Load benchmark
scenarios = load_benchmark('tombench').problems[:10]  # First 10

# Configure
config = ExperimentConfig(
    experiment_name="tombench_llama32",
    strategy="single",
    provider="ollama",
    model="llama3.2:latest"
)

# Start tracking
tracker = get_tracker(use_wandb=True)
tracker.start_experiment(config)

# Run all scenarios
for scenario in scenarios:
    result = run_scenario(scenario, provider="ollama")
    tracker.log_result(result)

tracker.finish_experiment()
```

**Example 3: Strategy Comparison**:
```python
strategies = ['single', 'debate', 'self_consistency']

for strategy in strategies:
    config = ExperimentConfig(
        experiment_name=f"comparison_{strategy}",
        strategy=strategy,
        provider="ollama"
    )

    tracker = get_tracker(use_wandb=True)
    tracker.start_experiment(config)

    # Run experiments...
    result = run_strategy(strategy, task)
    tracker.log_result(result)

    tracker.finish_experiment()

# W&B will group these for easy comparison!
```

---

## What Gets Tracked

W&B automatically tracks:

### Metrics
- ✅ Latency (per call)
- ✅ Token usage (input/output)
- ✅ Cost (USD)
- ✅ Custom metrics (accuracy, etc.)

### Configuration
- ✅ Strategy used
- ✅ Model/provider
- ✅ Temperature
- ✅ All hyperparameters

### System
- ✅ Timestamp
- ✅ Git commit hash
- ✅ Hardware info

### Artifacts (optional)
- 📦 Model checkpoints
- 📦 Concept vectors
- 📦 Generated outputs

---

## Using The Dashboard

### Accessing Your Dashboard

1. Run experiment with W&B enabled
2. Look for output:
   ```
   wandb: 🚀 View run at https://wandb.ai/yourname/hidden-layer/runs/abc123
   ```
3. Click link or go to https://wandb.ai

### Dashboard Features

**Overview Tab**:
- See all your experiments
- Filter by strategy, model, date
- Quick metrics at a glance

**Charts Tab**:
- Interactive plots (automatically generated!)
- Latency over time
- Cost vs accuracy
- Token usage

**Compare Tab**:
- Select multiple runs
- Side-by-side comparison
- Identify best strategies

**Artifacts Tab**:
- Saved models
- Concept vectors
- Outputs

### Example Queries

**Find best performing strategy**:
1. Go to Overview
2. Sort by accuracy (descending)
3. See which strategy/model combo worked best

**Track costs**:
1. Go to Charts
2. Look at "cost_usd" metric
3. See total spend across experiments

**Compare debate vs single**:
1. Go to Compare
2. Filter: strategy=debate vs strategy=single
3. See latency, cost, accuracy side-by-side

---

## Advanced Features

### Custom Metrics

Track additional metrics:

```python
tracker.log_result(result)

# Log custom metrics
if self.use_wandb:
    wandb.log({
        "accuracy": calculate_accuracy(result),
        "coherence": evaluate_coherence(result),
        "coverage": calculate_coverage(result)
    })
```

### Log Artifacts

Save important files:

```python
# Save concept vectors
if self.use_wandb:
    wandb.save("concepts/emotions.pkl")

# Save model checkpoint
wandb.save("models/qwen3-8b-finetuned.ckpt")

# Log generated text
wandb.log({"generated_text": result.output})
```

### Group Experiments

Organize related experiments:

```python
wandb.init(
    project="hidden-layer",
    name=config.experiment_name,
    group="tombench_evaluation",  # Group related runs
    tags=["benchmark", "theory-of-mind"]  # Add tags
)
```

### Compare Strategies

Use W&B Sweeps for hyperparameter optimization:

```yaml
# sweep.yaml
program: cli.py
method: grid
parameters:
  strategy:
    values: ['single', 'debate', 'self_consistency']
  model:
    values: ['llama3.2:3b', 'llama3.1:8b', 'llama3.1:70b']
  temperature:
    values: [0.3, 0.7, 0.9]
```

```bash
# Run sweep
wandb sweep sweep.yaml
wandb agent your-sweep-id
```

---

## Integration With Notebooks

W&B works great in Jupyter:

```python
# In notebook
import wandb

# Initialize
wandb.init(
    project="hidden-layer",
    name="notebook_exploration",
    config={
        "notebook": "03_introspection_experiments"
    }
)

# Run experiments
for concept in ['happiness', 'honesty', 'curiosity']:
    result = run_introspection_experiment(concept)

    wandb.log({
        "concept": concept,
        "detection_accuracy": result.accuracy,
        "optimal_layer": result.best_layer
    })

# View inline
wandb.finish()
```

**Tip**: W&B shows which notebook cell generated results!

---

## Best Practices

### 1. Naming Conventions

Use descriptive experiment names:

```python
# Good
experiment_name="tombench_debate_llama32_3b"
experiment_name="introspection_sweep_qwen3_happiness"
experiment_name="uicrit_multiperspective_baseline"

# Not helpful
experiment_name="test1"
experiment_name="experiment"
experiment_name="debug"
```

### 2. Use Tags

Organize with tags:

```python
wandb.init(
    project="hidden-layer",
    name="...",
    tags=[
        "benchmark:tombench",
        "strategy:debate",
        "model:llama3.2",
        "experiment-type:baseline"
    ]
)
```

### 3. Group Related Runs

Group experiments exploring the same question:

```python
wandb.init(
    project="hidden-layer",
    group="debate_ablation",  # All debate variations
    name="debate_2_agents"
)

wandb.init(
    project="hidden-layer",
    group="debate_ablation",
    name="debate_3_agents"
)
```

### 4. Log Often

Log metrics frequently for smooth curves:

```python
# Good - log after each task
for task in tasks:
    result = run_strategy(strategy, task)
    wandb.log({"accuracy": result.accuracy})

# Not good - only log once at end
results = [run_strategy(strategy, task) for task in tasks]
wandb.log({"avg_accuracy": np.mean([r.accuracy for r in results])})
```

### 5. Document In Notes

Add notes to runs:

```python
wandb.init(project="hidden-layer", name="...")

# Add notes
wandb.run.notes = """
Exploring whether debate helps on ToM tasks.
Hypothesis: Multiple perspectives improve belief attribution.
Results: 12% improvement over baseline.
"""

wandb.run.summary.update({
    "key_finding": "Debate improves ToM accuracy",
    "best_n_debaters": 3
})
```

---

## Troubleshooting

### Issue: W&B Not Logging

**Check**:
```python
import wandb
print(wandb.run)  # Should not be None

# If None:
wandb.init(project="hidden-layer")
```

### Issue: Can't See Runs

**Solutions**:
1. Check you're logged in: `wandb login`
2. Verify project name: Go to wandb.ai, check project list
3. Look for errors in output

### Issue: Too Many Runs

**Clean up**:
```bash
# Delete old runs
wandb runs delete --project hidden-layer --filter "created_at<2024-11-01"
```

Or in UI:
1. Go to wandb.ai
2. Select runs
3. Delete

### Issue: Runs Not Grouped

**Solution**: Use consistent `group` parameter:
```python
wandb.init(
    project="hidden-layer",
    group="my-experiment-group",  # Same for all related runs
    name="run-1"
)
```

---

## Disabling W&B

**Temporarily disable**:
```bash
export WANDB_MODE=disabled
```

**Disable in code**:
```python
tracker = get_tracker(use_wandb=False)
```

**Offline mode** (log locally, sync later):
```bash
export WANDB_MODE=offline

# Later, sync:
wandb sync
```

---

## Examples From Your Research

### Example 1: Benchmark All Strategies

```python
from harness import load_benchmark, run_strategy, get_tracker, ExperimentConfig

benchmark = load_benchmark('tombench')
strategies = ['single', 'debate', 'self_consistency']

for strategy in strategies:
    config = ExperimentConfig(
        experiment_name=f"tombench_{strategy}",
        strategy=strategy,
        provider="ollama",
        model="llama3.2:latest"
    )

    tracker = get_tracker(use_wandb=True)
    tracker.start_experiment(config)

    for problem in benchmark.problems[:10]:
        result = run_strategy(strategy, problem, provider="ollama")
        tracker.log_result(result)

    tracker.finish_experiment()

# Visit W&B → Compare all 3 strategies!
```

### Example 2: Track Introspection Experiments

```python
from harness import run_strategy, get_tracker, ExperimentConfig

concepts = ['happiness', 'honesty', 'curiosity', 'fear']
layers = [10, 15, 20]

for concept in concepts:
    for layer in layers:
        config = ExperimentConfig(
            experiment_name=f"introspection_{concept}_layer{layer}",
            strategy="introspection",
            provider="mlx"
        )

        tracker = get_tracker(use_wandb=True)
        tracker.start_experiment(config)

        result = run_strategy(
            "introspection",
            task_input="Describe your internal state",
            concept=concept,
            layer=layer,
            provider="mlx"
        )

        tracker.log_result(result)

        # Log introspection-specific metrics
        wandb.log({
            "detection_accuracy": result.metadata['introspection_correct'],
            "confidence": result.metadata['introspection_confidence'],
            "concept": concept,
            "layer": layer
        })

        tracker.finish_experiment()

# W&B will show: Which concept/layer combo works best?
```

### Example 3: Cost Analysis

```python
# Run experiments
# ...

# Later, analyze costs in W&B:
# 1. Go to wandb.ai
# 2. Charts → Create new chart
# 3. Metric: cost_usd
# 4. Aggregate: SUM
# 5. Group by: strategy

# See: "Debate cost $2.34 total, Single cost $0.98"
# Insight: Debate is 2.4x more expensive but only 1.3x better
```

---

## Next Steps

1. **Run your first tracked experiment** (see examples above)
2. **Visit wandb.ai** to see your dashboard
3. **Create comparison charts** for different strategies
4. **Share links** to results with collaborators
5. **Explore W&B docs**: https://docs.wandb.ai

---

## Quick Reference

### Enable W&B
```python
tracker = get_tracker(use_wandb=True)
```

### Start Experiment
```python
tracker.start_experiment(config)
```

### Log Results
```python
tracker.log_result(result)
```

### Custom Metrics
```python
wandb.log({"accuracy": 0.82, "custom": value})
```

### Save Artifacts
```python
wandb.save("file.pkl")
```

### Finish
```python
tracker.finish_experiment()
```

---

**Questions?** Check W&B docs at https://docs.wandb.ai or see the main integration guide: `INTEGRATION_OPPORTUNITIES.md`
