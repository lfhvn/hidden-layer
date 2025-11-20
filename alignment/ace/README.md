# ACE: Agentic Context Engineering

**Reproduction of "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models"**
Zhang et al., 2025 (arXiv:2510.04618)

---

## Overview

ACE (Agentic Context Engineering) is a framework that treats contexts as **evolving playbooks** that accumulate, refine, and organize strategies through a modular process of **generation**, **reflection**, and **curation**.

Unlike traditional approaches that rely on weight updates or fine-tuning, ACE adapts LLMs through structured context modifications that:
- **Prevent brevity bias**: Maintains detailed domain insights rather than over-summarizing
- **Avoid context collapse**: Uses incremental updates to preserve knowledge over time
- **Scale with long-context models**: Leverages modern LLMs' extended context windows

---

## Key Innovations

### Three-Role Architecture

1. **Generator**: Produces reasoning trajectories for queries
   - Surfaces effective strategies and recurring pitfalls
   - Generates execution traces with feedback

2. **Reflector**: Distills insights from execution traces
   - Critiques successes and failures
   - Extracts concrete, actionable lessons
   - Can refine insights over multiple iterations

3. **Curator**: Integrates insights into structured contexts
   - Synthesizes lessons into compact delta entries
   - Merges deltas deterministically using lightweight logic (non-LLM)
   - Maintains organized, evolving playbooks

### Structured Context Updates

- **Delta-based updates**: Incremental additions rather than complete rewrites
- **Deterministic merging**: Reproducible context evolution
- **Modular organization**: Strategies organized by topic/category
- **Prevents collapse**: Structured format preserves details over iterations

---

## Results (Original Paper)

### Performance Improvements
- **Agent benchmarks**: +10.6% improvement
- **Finance tasks**: +8.6% improvement
- **AppWorld (ReAct + ACE)**: 59.4% accuracy (matches production IBM CUGA at 60.3%)
  - Using DeepSeek-V3.1 (open-source) vs GPT-4.1 (IBM CUGA)
  - Online adaptation: +8.4% TGC, +0.7% SGC on test-challenge split

### Efficiency Gains
**Offline (AppWorld)**:
- -82.3% latency vs GEPA
- -75.1% rollouts vs GEPA

**Online (FiNER)**:
- -91.5% latency vs Dynamic Cheatsheet
- -83.6% token cost vs Dynamic Cheatsheet

---

## Project Structure

```
alignment/ace/
├── README.md              # This file
├── CLAUDE.md              # Development guide
├── src/
│   ├── __init__.py
│   ├── generator.py       # Reasoning trajectory generation
│   ├── reflector.py       # Insight extraction & critique
│   ├── curator.py         # Context delta synthesis & merging
│   ├── context.py         # Context management & playbook structure
│   └── ace.py             # Main ACE framework orchestration
├── experiments/
│   ├── offline_ace.py     # Offline context optimization
│   ├── online_ace.py      # Online agent memory adaptation
│   └── benchmarks/        # Benchmark implementations
│       ├── agent_tasks.py
│       └── domain_tasks.py
├── configs/
│   ├── ace_config.yaml    # ACE framework configuration
│   └── models.yaml        # Model configurations
├── data/
│   └── benchmarks/        # Benchmark datasets
└── results/               # Experiment results
```

---

## Quick Start

### Installation

```bash
cd alignment/ace
pip install -e .
```

### Basic Usage

```python
from src.ace import ACEFramework
from harness import llm_call

# Initialize ACE
ace = ACEFramework(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

# Run offline ACE (optimize system prompt)
optimized_context = ace.run_offline(
    task_examples=[...],
    initial_context="You are a helpful assistant..."
)

# Run online ACE (agent memory)
agent_memory = ace.run_online(
    task_stream=[...],
    update_frequency=5  # Update context every 5 tasks
)
```

---

## Relation to Hidden Layer Research

### Alignment Research
- **Context-based steerability**: Alternative to vector-based steering
- **Reliable adaptation**: Consistent behavior modification without retraining
- **Adherence metrics**: Measuring how well contexts guide behavior

### Theory of Mind
- **Self-improvement**: Models learn from their own execution traces
- **Meta-cognition**: Reflection on reasoning trajectories
- **Introspection**: Understanding what strategies work and why

### Communication
- **Multi-agent coordination**: Shared evolving playbooks
- **Strategy accumulation**: Collective learning across agents
- **Structured knowledge**: Organized vs unstructured agent memory

---

## Benchmarks

We provide 6 integrated benchmarks for evaluating ACE:

### Agent Benchmarks
- **SimpleAgent**: Multi-step planning, prioritization, scheduling (8 test + 3 train tasks)
- **ToolUse**: Function calling and tool selection (4 test tasks)
- **Reasoning**: Logic, causal, and planning reasoning (4 test tasks)

### Domain Benchmarks
- **Math**: Arithmetic, word problems, algebra (8 test + 3 train tasks)
- **Finance**: ROI, budgeting, entity recognition, FiNER-inspired (7 test tasks)
- **Code**: Bug finding, completion, complexity analysis (5 test tasks)

### Running Benchmarks

**Offline ACE** (pre-deployment optimization):
```bash
python experiments/offline_ace.py --benchmark math --iterations 3
```

**Online ACE** (continuous adaptation):
```bash
python experiments/online_ace.py --benchmark agent --buffer-size 10
```

**All benchmarks** (comprehensive evaluation):
```bash
python experiments/run_all_benchmarks.py --iterations 3
```

See `experiments/README.md` for detailed benchmark documentation.

---

## Key Research Questions

1. **Generalization**: Do ACE-optimized contexts transfer across models?
2. **Robustness**: How do contexts handle distribution shift?
3. **Interpretability**: Can we understand what strategies ACE discovers?
4. **Composability**: Can we merge contexts from different domains?
5. **Limits**: When does context-based adaptation outperform fine-tuning?

---

## Implementation Status

- [x] Core ACE components (Generator, Reflector, Curator)
- [x] Context management system
- [x] Delta-based merging logic
- [x] Offline ACE optimization
- [x] Online ACE optimization
- [x] Agent benchmark integration (SimpleAgent, ToolUse, Reasoning)
- [x] Domain benchmark integration (Math, Finance, Code)
- [x] Experiment tracking and metrics
- [x] Benchmark evaluation framework
- [ ] Comparative evaluation vs baselines (static prompts, fine-tuning)
- [ ] Cross-model transfer experiments
- [ ] Real AppWorld/FiNER dataset integration

---

## References

- **Paper**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
- **Authors**: Zhang et al., 2025
- **Related Work**:
  - Context optimization (DSPy, GEPA)
  - Agent memory systems (Dynamic Cheatsheet)
  - Meta-learning for LLMs

---

## Contact & Contributions

This is a research reproduction project at Hidden Layer Lab.

For questions or contributions:
- Open an issue on GitHub
- See `CLAUDE.md` for development guidelines
