# SELPHI - Theory of Mind Development Guide

## Project Overview

**SELPHI**: Study of Epistemic and Logical Processing in Human-AI Interaction

**Research Question**: How well do language models understand mental states, beliefs, and perspective-taking?

**Core Focus**:
- Theory of mind evaluation
- False belief reasoning
- Perspective-taking
- Epistemic states
- Connection to deception and alignment

**Uses**: `harness/` for LLM provider abstraction and experiment tracking

---

## Architecture

### Core Components

**Scenarios** (`code/scenarios.py`):
- 9+ ToM scenarios (Sally-Anne, Chocolate Bar, Birthday Puppy, etc.)
- Difficulty levels: easy, medium, hard
- 7 ToM types covered

**Evaluation** (`code/evals.py`):
- Semantic matching
- LLM-as-judge
- ToM-specific metrics

**Benchmarks** (`code/benchmarks.py`):
- ToMBench (388 scenarios)
- OpenToM (696 scenarios)
- SocialIQA (38k questions)

---

## Development Workflows

### Adding a ToM Scenario

```python
# In code/scenarios.py

from code.scenarios import ToMScenario, ToMType

MY_SCENARIO = ToMScenario(
    name="my_scenario",
    tom_type=ToMType.FALSE_BELIEF,
    scenario_text="...",
    question="...",
    correct_answer="...",
    difficulty="medium",
    explanation="Why this tests ToM..."
)
```

### Running Experiments

```python
from harness import llm_call, get_tracker
from code.scenarios import SALLY_ANNE
from code.evals import evaluate_scenario

# Run scenario
response = llm_call(
    SALLY_ANNE.get_prompt(),
    provider="ollama",
    model="llama3.2:latest"
)

# Evaluate
result = evaluate_scenario(SALLY_ANNE, response.text)
print(f"Score: {result['average_score']:.2f}")
```

### Benchmark Evaluation

```python
from harness import load_benchmark
from code import run_multiple_scenarios

# Load ToMBench
tombench = load_benchmark('tombench', split='test')

# Run experiments
results = run_multiple_scenarios(
    tombench.problems,
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)
```

---

## Research Questions

1. **Which ToM types** are hardest for LLMs?
   - False belief vs. second-order belief
   - Epistemic states vs. perspective-taking

2. **How does ToM scale** with model size?
   - Does larger = better ToM?
   - What capabilities emerge at what scale?

3. **Connection to alignment**:
   - Can models with good ToM deceive more effectively?
   - Is ToM necessary for alignment?

4. **Fine-tuning impact**:
   - Can we improve ToM through training?
   - What training signals help?

---

## Integration Points

**With Introspection**:
- How does understanding others relate to understanding self?
- Is there a unified ToM mechanism?

**With Multi-Agent**:
- Do multi-agent systems exhibit better ToM?
- Can debate improve perspective-taking?

**With Alignment**:
- Detecting deceptive ToM reasoning
- Using ToM for alignment verification

---

## Key Files

- `code/scenarios.py` - ToM scenarios
- `code/evals.py` - ToM evaluation
- `code/benchmarks.py` - Benchmark loaders
- `notebooks/` - Experiment notebooks

---

## Testing

```bash
# Run tests
cd projects/selphi
pytest tests/ -v

# Quick test
python -c "from code.scenarios import SALLY_ANNE; print(SALLY_ANNE.get_prompt())"
```

---

## See Also

- Lab-wide infrastructure: `/docs/infrastructure/`
- Research themes: `/RESEARCH.md`
- Benchmark guide: `/docs/workflows/benchmarking.md`
