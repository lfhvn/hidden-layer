# Integration Opportunities - Claude Skills, MCP Servers, and Tools

**Date**: 2025-11-03
**Purpose**: Identify opportunities to enhance the Hidden Layer research platform with advanced integrations

---

## 🎯 Your Vision & Current State

### What You Have
- **3 Subsystems**: Harness (6 strategies), CRIT (design critique), SELPHI (theory of mind)
- **Local-First**: MLX + Ollama for rapid experimentation
- **Interpretability**: Activation steering, introspection testing
- **Benchmarking**: 4 major datasets (UICrit, ToMBench, OpenToM, SocialIQA)
- **Experiment Tracking**: JSON/JSONL logging, automatic metrics

### What You're Building Toward
- Multi-agent LLM research platform
- Interpretability and introspection research
- Understanding when/why multi-agent strategies outperform
- Hidden layer analysis and visualization
- Production-quality research workflows

---

## 1️⃣ Claude Skills Integration

### What Are Claude Skills?
Claude Skills are reusable automation modules that extend Claude Code's capabilities. They're like mini-agents specialized for specific tasks.

### 🌟 High-Impact Skills for Your Research

#### A. **Experiment Analysis Skill**
**Purpose**: Automatically analyze experiment results and generate insights

```yaml
# .claude/skills/analyze-experiments.md
name: analyze-experiments
description: Analyze experiment results from the experiments/ directory

prompt: |
  You are an expert at analyzing multi-agent LLM experiment results.

  1. Read all results from experiments/{experiment_name}/
  2. Parse results.jsonl and summary.json
  3. Calculate key metrics:
     - Strategy performance comparison
     - Cost-effectiveness analysis
     - Latency vs quality tradeoffs
  4. Generate insights about when multi-agent strategies outperform
  5. Create visualization recommendations
  6. Write findings to experiments/{experiment_name}/analysis.md
```

**Use Case**:
```bash
# After running experiments
make analyze-experiments experiment=debate_energy_20251103
```

#### B. **Benchmark Runner Skill**
**Purpose**: Run systematic benchmark evaluations across models

```yaml
# .claude/skills/run-benchmark.md
name: run-benchmark
description: Run benchmarks across multiple models and strategies

prompt: |
  You are a benchmark orchestrator for LLM research.

  Given a benchmark name and list of models:
  1. Load the benchmark using harness.load_benchmark()
  2. Run each model on all benchmark tasks
  3. Track metrics (accuracy, latency, cost)
  4. Compare to baseline scores
  5. Generate comparison report
  6. Save results to experiments/benchmarks/{benchmark}_{timestamp}/
```

**Use Case**:
```python
# Run ToMBench across 5 models
/run-benchmark tombench --models llama3.2:3b,llama3.1:8b,llama3.1:70b,claude-3-5-sonnet,gpt-4
```

#### C. **Introspection Experiment Skill**
**Purpose**: Run systematic introspection experiments

```yaml
# .claude/skills/introspection-sweep.md
name: introspection-sweep
description: Run introspection experiments across layers and concepts

prompt: |
  You are an introspection researcher.

  Given a model and list of concepts:
  1. For each concept (happiness, honesty, etc.):
     a. Extract concept vectors from layers 10, 15, 20
     b. Test steering strength 0.5, 1.0, 1.5, 2.0
     c. Run detection tasks
     d. Run generation tasks
  2. Track detection accuracy and generation quality
  3. Find optimal layer/strength combinations
  4. Generate introspection profile for this model
  5. Save to concepts/{model_name}/profile.json
```

**Use Case**:
```bash
# Comprehensive introspection profile
/introspection-sweep --model qwen3-8b --concepts happiness,honesty,curiosity,fear
```

#### D. **Paper Writer Skill**
**Purpose**: Generate research papers from experimental results

```yaml
# .claude/skills/write-paper.md
name: write-paper
description: Generate research paper sections from experiment data

prompt: |
  You are a research paper writer specializing in LLM research.

  Given experiment results:
  1. Read all experiment data from specified directories
  2. Analyze findings and identify key insights
  3. Generate LaTeX sections:
     - Abstract (summarize findings)
     - Introduction (research questions)
     - Methods (experimental setup)
     - Results (tables, graphs descriptions)
     - Discussion (interpret findings)
     - Conclusion (implications)
  4. Include proper citations
  5. Suggest figure placements
  6. Save to papers/{paper_name}/
```

---

## 2️⃣ MCP (Model Context Protocol) Server Integration

### What Are MCP Servers?
MCP servers give Claude access to external data sources and services through a standardized protocol.

### 🌟 High-Impact MCP Servers for Your Research

#### A. **File System MCP** (Essential)
**Purpose**: Direct access to experiment logs, results, notebooks

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/hidden-layer"]
    }
  }
}
```

**Benefits**:
- Read experiment results without manual file operations
- Analyze notebooks directly
- Search through logs efficiently
- Monitor experiments in real-time

**Use Cases**:
```python
# Claude can now directly:
# - Read all experiment results
# - Compare across experiments
# - Generate reports from logs
# - Track experiment progress
```

#### B. **SQLite MCP** (Highly Recommended)
**Purpose**: Structured storage for experiment results

**Setup**:
```bash
# Create experiments database
sqlite3 experiments.db < schema.sql
```

```sql
-- schema.sql
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT,
    strategy TEXT,
    provider TEXT,
    model TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status TEXT
);

CREATE TABLE results (
    id INTEGER PRIMARY KEY,
    experiment_id TEXT,
    task_input TEXT,
    output TEXT,
    latency_s REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    cost_usd REAL,
    metadata JSON,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    benchmark_name TEXT,
    model TEXT,
    accuracy REAL,
    avg_latency_s REAL,
    total_cost_usd REAL,
    run_date TIMESTAMP
);
```

```json
// Add to claude_desktop_config.json
{
  "mcpServers": {
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/path/to/experiments.db"]
    }
  }
}
```

**Benefits**:
- Query experiment results with SQL
- Fast aggregations and comparisons
- Track experiments over time
- Generate reports easily

**Use Cases**:
```sql
-- Find best performing strategy
SELECT strategy, AVG(accuracy)
FROM results
GROUP BY strategy
ORDER BY AVG(accuracy) DESC;

-- Cost analysis
SELECT model, SUM(cost_usd) as total_cost, COUNT(*) as runs
FROM results
GROUP BY model;

-- Latency vs quality tradeoff
SELECT model, AVG(latency_s), AVG(accuracy)
FROM results
WHERE strategy = 'debate'
GROUP BY model;
```

#### C. **GitHub MCP** (Useful)
**Purpose**: Manage experiments via GitHub issues/PRs

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your_token"
      }
    }
  }
}
```

**Benefits**:
- Create issues for failed experiments
- Track research progress via projects
- Version experiment configurations
- Collaborate with team

#### D. **Slack/Discord MCP** (Team Collaboration)
**Purpose**: Send experiment notifications

```json
{
  "mcpServers": {
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": {
        "SLACK_TOKEN": "your_token"
      }
    }
  }
}
```

**Benefits**:
- Get notified when long experiments complete
- Share findings with team
- Alert on experiment failures

#### E. **Brave Search MCP** (Research)
**Purpose**: Research related work and latest papers

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_key"
      }
    }
  }
}
```

**Benefits**:
- Find related papers automatically
- Stay updated on latest research
- Discover new benchmarks
- Find similar work for citations

---

## 3️⃣ Other Tool Integrations

### A. **Weights & Biases (W&B)** (Highly Recommended)

**Purpose**: Professional experiment tracking and visualization

**Setup**:
```bash
pip install wandb
wandb login
```

**Integration**:
```python
# code/harness/experiment_tracker.py

import wandb

class ExperimentTracker:
    def __init__(self, use_wandb=True):
        self.use_wandb = use_wandb

    def start_experiment(self, config):
        if self.use_wandb:
            wandb.init(
                project="hidden-layer",
                name=config.experiment_name,
                config={
                    "strategy": config.strategy,
                    "provider": config.provider,
                    "model": config.model
                }
            )
        # ... existing code

    def log_result(self, result):
        if self.use_wandb:
            wandb.log({
                "latency": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost": result.cost_usd
            })
        # ... existing code
```

**Benefits**:
- Beautiful interactive dashboards
- Compare experiments visually
- Share results publicly
- Track hyperparameters
- Artifact versioning

**Use Cases**:
- Track all experiments in one place
- Generate plots automatically
- Compare strategies side-by-side
- Share findings with collaborators

### B. **Ray** (For Parallel Experiments)

**Purpose**: Distributed execution for large-scale experiments

**Setup**:
```bash
pip install ray
```

**Integration**:
```python
# code/harness/distributed.py

import ray

@ray.remote
def run_experiment_remote(task, strategy, model):
    """Run experiment in parallel"""
    from harness import run_strategy
    return run_strategy(strategy, task, model=model)

def run_experiments_parallel(tasks, strategies, models):
    """Run experiments in parallel across all combinations"""
    ray.init()

    futures = []
    for task in tasks:
        for strategy in strategies:
            for model in models:
                future = run_experiment_remote.remote(task, strategy, model)
                futures.append(future)

    results = ray.get(futures)
    return results
```

**Benefits**:
- Run 100s of experiments in parallel
- Utilize all CPU cores
- Scale to multiple machines
- Fault tolerance

**Use Cases**:
```python
# Run massive hyperparameter sweep
tasks = load_benchmark('tombench').problems
strategies = ['single', 'debate', 'self_consistency']
models = ['llama3.2:3b', 'llama3.1:8b', 'llama3.1:70b']

# 388 tasks × 3 strategies × 3 models = 3,492 experiments
# Run in parallel instead of sequentially (days → hours)
results = run_experiments_parallel(tasks, strategies, models)
```

### C. **Hydra** (Configuration Management)

**Purpose**: Manage complex experiment configurations

**Setup**:
```bash
pip install hydra-core
```

**Structure**:
```
config/
├── config.yaml           # Main config
├── strategy/
│   ├── single.yaml
│   ├── debate.yaml
│   └── introspection.yaml
├── model/
│   ├── llama3_3b.yaml
│   ├── llama3_70b.yaml
│   └── claude_sonnet.yaml
└── experiment/
    ├── tombench.yaml
    ├── uicrit.yaml
    └── introspection_sweep.yaml
```

**Usage**:
```python
# cli_hydra.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    from harness import run_strategy

    result = run_strategy(
        cfg.strategy.name,
        task_input=cfg.task,
        **cfg.strategy.params,
        provider=cfg.model.provider,
        model=cfg.model.name
    )
```

```bash
# Run with different configs
python cli_hydra.py strategy=debate model=llama3_70b
python cli_hydra.py strategy=introspection model=claude_sonnet experiment=tombench
```

### D. **DVC (Data Version Control)**

**Purpose**: Version control for large datasets and models

**Setup**:
```bash
pip install dvc dvc-gdrive
dvc init
```

**Structure**:
```bash
# Track large files
dvc add experiments/large_results.jsonl
dvc add concepts/emotion_library.pkl
dvc add uicrit/uicrit_public.csv

# Push to remote storage
dvc remote add -d storage gdrive://your-folder-id
dvc push
```

**Benefits**:
- Version control for data
- Share datasets with team
- Track model checkpoints
- Reproducible experiments

### E. **Streamlit** (Interactive Dashboard)

**Purpose**: Build interactive experiment viewer

**Setup**:
```bash
pip install streamlit plotly
```

**Example**:
```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from harness import compare_experiments, BENCHMARKS

st.title("Hidden Layer Experiment Dashboard")

# Sidebar filters
benchmark = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
strategy = st.sidebar.multiselect("Strategy", ['single', 'debate', 'self_consistency'])

# Load and display results
experiments = load_experiment_results()
df = pd.DataFrame(experiments)

# Visualization
fig = px.scatter(df, x='latency_s', y='accuracy', color='strategy',
                 hover_data=['model', 'cost_usd'])
st.plotly_chart(fig)

# Comparison table
st.dataframe(df.groupby('strategy').agg({
    'accuracy': 'mean',
    'latency_s': 'mean',
    'cost_usd': 'sum'
}))
```

```bash
streamlit run dashboard.py
```

### F. **MLflow** (ML Experiment Platform)

**Purpose**: End-to-end ML experiment tracking

**Setup**:
```bash
pip install mlflow
mlflow ui  # Start UI on http://localhost:5000
```

**Integration**:
```python
import mlflow

def run_tracked_experiment(strategy, task, model):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("strategy", strategy)
        mlflow.log_param("model", model)

        # Run experiment
        result = run_strategy(strategy, task, model=model)

        # Log metrics
        mlflow.log_metric("latency", result.latency_s)
        mlflow.log_metric("cost", result.cost_usd)

        # Log artifacts
        mlflow.log_artifact("experiments/results.jsonl")

        return result
```

---

## 📋 Recommended Implementation Roadmap

### Phase 1: Foundation (Week 1)
1. **Set up MCP servers**:
   - ✅ File system MCP (essential for accessing experiments)
   - ✅ SQLite MCP (structured experiment storage)

2. **Create 2 essential skills**:
   - Experiment analyzer skill
   - Benchmark runner skill

3. **Add requirements**:
   ```bash
   pip install wandb ray[default] hydra-core streamlit plotly mlflow
   ```

### Phase 2: Enhanced Tracking (Week 2)
1. **Integrate W&B** into experiment tracker
2. **Set up Hydra** for configuration management
3. **Create Streamlit dashboard** for visualization
4. **Add GitHub MCP** for project management

### Phase 3: Scale & Automation (Week 3)
1. **Integrate Ray** for parallel experiments
2. **Add DVC** for data versioning
3. **Create introspection sweep skill**
4. **Set up MLflow** for ML ops

### Phase 4: Research Workflows (Week 4)
1. **Add paper writer skill**
2. **Set up Brave Search MCP** for research
3. **Add Slack notifications** for long experiments
4. **Create automated benchmark pipeline**

---

## 💡 Quick Wins (Start Today)

### 1. File System MCP (15 minutes)
```bash
# Add to ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/hidden-layer"]
    }
  }
}
```

Now Claude can directly read your experiment results!

### 2. Create Experiment Analyzer Skill (30 minutes)
```bash
mkdir -p .claude/skills
cat > .claude/skills/analyze-experiments.md << 'EOF'
You are an experiment analyzer for multi-agent LLM research.

Given an experiment directory:
1. Read config.json, results.jsonl, summary.json
2. Analyze performance metrics
3. Compare to baseline scores
4. Identify insights (when did strategy help? cost-effectiveness?)
5. Generate markdown report with findings
6. Save to analysis.md in experiment directory
EOF
```

### 3. Add W&B (10 minutes)
```bash
pip install wandb
wandb login
```

Add 5 lines to your experiment tracker to get beautiful dashboards!

---

## 🎯 Best Integrations for Your Vision

Based on your research goals, prioritize:

1. **SQLite MCP** ⭐⭐⭐⭐⭐ - Essential for querying experiments
2. **File System MCP** ⭐⭐⭐⭐⭐ - Access all your data
3. **W&B** ⭐⭐⭐⭐⭐ - Best-in-class experiment tracking
4. **Experiment Analyzer Skill** ⭐⭐⭐⭐⭐ - Automate analysis
5. **Ray** ⭐⭐⭐⭐ - Scale experiments massively
6. **Hydra** ⭐⭐⭐⭐ - Clean configuration management
7. **Streamlit** ⭐⭐⭐⭐ - Interactive visualization
8. **GitHub MCP** ⭐⭐⭐ - Project management
9. **Brave Search MCP** ⭐⭐⭐ - Research assistance
10. **MLflow** ⭐⭐⭐ - Alternative to W&B

---

## 📚 Next Steps

1. **Read this document** and identify 2-3 integrations to start with
2. **Set up File System MCP** (15 min) - instant value
3. **Create one skill** - try the experiment analyzer
4. **Add W&B** - professional tracking with minimal code
5. **Iterate** - add more as you see value

Would you like me to help implement any of these integrations? I can:
- Set up MCP servers
- Create Claude Skills
- Integrate W&B/Ray/Hydra
- Build Streamlit dashboard
- Write SQL schemas for experiments

Just let me know which integration excites you most! 🚀
