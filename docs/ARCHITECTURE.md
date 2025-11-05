# Hidden Layer - System Architecture

**Version**: 0.1.0
**Last Updated**: 2025-11-05

## Overview

Hidden Layer is an independent research lab investigating multi-agent architectures, theory of mind, steerability, interpretability, alignment, and human computer and human AI interaction. Current research areas:

1. **Harness** - Core multi-agent infrastructure and experiment tracking
2. **CRIT** - Collective Reasoning for Iterative Testing (design critique)
3. **SELPHI** - Study of Epistemic and Logical Processing (theory of mind)
4. **Introspection** - assessing LLM ability to introspect
5. **Reasoning and Rationale** exploring explainability
6. **Latent lens and Laten topologies** investigating the geometry and internal representation of concepts in latent space
7. **AI to AI communication** Non-linguistic communication between different LLMs

All subsystems are designed for local-first experimentation on Apple Silicon (M4 Max with 128GB RAM) using MLX and Ollama, with seamless fallback to API providers (Anthropic, OpenAI) for comparison.

---

## System Architecture

```
hidden-layer/
├── harness/             # Core infrastructure library
├── shared/              # Shared resources (concepts, datasets, utils)
├── communication/       # Agent communication research
├── theory-of-mind/      # Theory of mind & self-knowledge research
├── representations/     # Latent space & interpretability research
│   └── latent-space/
│       ├── lens/        # SAE interpretability web app
│       └── topologies/  # Mobile latent space exploration
├── alignment/           # Alignment & steerability research
├── notebooks/           # Jupyter notebooks for experimentation
├── experiments/         # Auto-generated experiment logs
└── config/             # Model configurations and presets
```

**Total**: 6,613 lines of Python code + 5,641 lines of documentation

---

## Subsystem 1: Harness

**Purpose**: Core infrastructure for multi-agent LLM experimentation

### Components

#### 1.1 LLM Provider (`harness/llm_provider.py` - 527 LOC)

Unified interface to all LLM providers:

```python
from harness import llm_call

# Local inference
response = llm_call("Question?", provider="mlx", model="mlx-community/Llama-3.2-3B-Instruct-4bit")
response = llm_call("Question?", provider="ollama", model="llama3.2:latest")

# API inference
response = llm_call("Question?", provider="anthropic", model="claude-3-5-sonnet-20241022")
response = llm_call("Question?", provider="openai", model="gpt-4")
```

**Features**:
- Automatic cost tracking for API calls
- Token counting across all providers
- Streaming support
- Easy provider switching

**Providers**:
- `mlx` - Apple Silicon optimized (Metal acceleration)
- `ollama` - Local model server
- `anthropic` - Claude API
- `openai` - GPT API

#### 1.2 Multi-Agent Strategies (`harness/strategies.py` - 749 LOC)

Five different multi-agent strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `single` | Baseline single-model | Standard inference |
| `debate` | n-agent debate + judge | Reasoning, argumentation |
| `self_consistency` | Sample multiple times + vote | Math, logic problems |
| `manager_worker` | Decompose → parallel → synthesize | Complex planning |
| `consensus` | Multiple agents → find agreement | Decision-making |

```python
from harness import run_strategy

result = run_strategy(
    "debate",
    task_input="Should we invest in renewable energy?",
    n_debaters=3,
    provider="ollama",
    model="llama3.2:latest"
)
```

#### 1.3 Experiment Tracking (`harness/experiment_tracker.py`)

Automatic experiment logging with reproducibility:

```python
from harness import ExperimentConfig, get_tracker

config = ExperimentConfig(
    experiment_name="baseline_reasoning",
    strategy="debate",
    provider="ollama",
    model="llama3.2:latest"
)

tracker = get_tracker()
run_dir = tracker.start_experiment(config)

# ... run experiments ...

summary = tracker.finish_experiment()
```

**Output Structure**:
```
experiments/{name}_{timestamp}_{hash}/
├── config.json          # Experiment configuration
├── results.jsonl        # Streaming results (one per line)
├── summary.json         # Aggregated metrics
└── README.md            # Human-readable summary
```

#### 1.4 Evaluation Suite (`harness/evals.py`)

8+ evaluation methods:

- `exact_match` - Exact string matching
- `keyword_match` - Check for required keywords
- `numeric_match` - Compare numerical answers
- `llm_judge` - LLM-as-judge evaluation
- `win_rate_comparison` - Head-to-head comparison
- Custom evaluators (extensible)

#### 1.5 Model Configuration (`harness/model_config.py`)

YAML-based model presets:

```yaml
gpt-oss-20b-reasoning:
  provider: ollama
  model: llama3.1:latest
  temperature: 0.3
  max_tokens: 2000
  thinking_budget: 2000
```

#### 1.6 Rationale Extraction (`harness/rationale.py` - 285 LOC)

Extract reasoning chains from model responses:

```python
from harness import ask_with_reasoning

response = ask_with_reasoning(
    "Why is the sky blue?",
    provider="ollama"
)

print(response.rationale)  # Step-by-step reasoning
print(response.answer)     # Final answer
```

#### 1.7 Benchmark Integration (`harness/benchmarks.py`)

Unified interface to all benchmarks across subsystems:

```python
from harness import load_benchmark, get_baseline_scores, BENCHMARKS

# List all available benchmarks
print(BENCHMARKS)  # {'uicrit', 'tombench', 'opentom', 'socialiqa'}

# Load a benchmark
dataset = load_benchmark('tombench')

# Get baseline performance
scores = get_baseline_scores('tombench')
# {'human_performance': 0.95, 'gpt4_performance': 0.76, ...}
```

---

## Subsystem 2: CRIT (Design Critique)

**Purpose**: Testing collective design critique reasoning

### Research Question

**Can collective design critique from multiple expert perspectives produce better solutions than a single generalist critic?**

### Components

#### 2.1 Design Problems (`crit/problems.py` - 584 LOC)

8 pre-defined design problems across 5 domains:

| Domain | Problems |
|--------|----------|
| **UI/UX** | Mobile checkout flow, Analytics dashboard |
| **API Design** | REST versioning, GraphQL schema |
| **System Architecture** | Microservices decomposition, Caching strategy |
| **Data Modeling** | Permission system |
| **Workflow** | Multi-stage approval |

```python
from crit import MOBILE_CHECKOUT, API_VERSIONING, MICROSERVICES

problem = MOBILE_CHECKOUT
print(problem.domain)           # DesignDomain.UI_UX
print(problem.description)      # "Improve mobile checkout..."
print(problem.success_criteria) # ["< 30s checkout time", ...]
```

#### 2.2 Critique Strategies (`crit/strategies.py`)

4 different critique approaches:

**1. Single Critic** (baseline)
```python
from crit import run_critique_strategy, MOBILE_CHECKOUT

result = run_critique_strategy("single", MOBILE_CHECKOUT, provider="ollama")
```

**2. Multi-Perspective** (9 expert viewpoints)
```python
result = run_critique_strategy(
    "multi_perspective",
    MOBILE_CHECKOUT,
    perspectives=["usability", "security", "accessibility", "performance"],
    synthesize=True
)
```

Perspectives: usability, security, accessibility, performance, aesthetics, scalability, maintainability, cost_efficiency, user_delight

**3. Iterative Critique** (cycles of improvement)
```python
result = run_critique_strategy(
    "iterative",
    MOBILE_CHECKOUT,
    iterations=2
)
```

**4. Adversarial Critique** (proposer vs challenger)
```python
result = run_critique_strategy(
    "adversarial",
    MOBILE_CHECKOUT
)
```

#### 2.3 Evaluation (`crit/evals.py`)

Comprehensive critique quality metrics:

- **Coverage**: How many perspectives are addressed?
- **Depth**: How specific and detailed are critiques?
- **Actionability**: Can recommendations be implemented?
- **Quality Score**: Overall critique effectiveness

```python
from crit import evaluate_critique

metrics = evaluate_critique(critique_text, problem)
# {
#   'coverage_score': 0.85,
#   'depth_score': 0.72,
#   'actionability_score': 0.90,
#   'overall_quality': 0.82
# }
```

#### 2.4 Benchmarks (`crit/benchmarks.py`)

**UICrit Dataset** (Google Research, UIST 2024):
- 11,344 design critiques
- 1,000 mobile UIs from RICO dataset
- Expert human critiques + LLM-generated critiques
- Quality ratings across multiple dimensions

```python
from crit.benchmarks import load_uicrit, compare_to_experts

dataset = load_uicrit(min_quality_rating=7.0)
comparison = compare_to_experts(your_critiques, expert_critiques)
```

---

## Subsystem 3: SELPHI (Theory of Mind)

**Purpose**: Evaluating theory of mind and epistemic reasoning in LLMs

### Research Question

**How well can language models understand mental states, beliefs, and perspective-taking?**

### Components

#### 3.1 Scenarios (`selphi/scenarios.py`)

9+ pre-defined Theory of Mind test scenarios:

| Scenario | ToM Type | Difficulty |
|----------|----------|------------|
| Sally-Anne | False belief | Easy |
| Chocolate Bar | False belief | Easy |
| Ice Cream Van | Belief updating | Medium |
| Birthday Puppy | Second-order belief | Hard |
| Museum Trip | Knowledge attribution | Medium |
| Painted Room | Perspective taking | Medium |
| Restaurant Bill | Pragmatic reasoning | Medium |
| Library Book | Epistemic state | Medium |
| Coffee Shop | Multi-character | Hard |

```python
from selphi import SALLY_ANNE, BIRTHDAY_PUPPY

scenario = SALLY_ANNE
print(scenario.scenario_text)  # Story
print(scenario.question)       # Question about belief
print(scenario.correct_answer) # Expected answer
```

#### 3.2 ToM Types (`selphi/scenarios.py`)

7 different types of Theory of Mind reasoning:

- **False Belief**: Understanding beliefs that differ from reality
- **Knowledge Attribution**: Knowing what others know
- **Perspective Taking**: Reasoning from another's viewpoint
- **Belief Updating**: How beliefs change with new information
- **Second-Order Beliefs**: Beliefs about beliefs ("Alice thinks Bob believes...")
- **Epistemic States**: Understanding knowledge vs. ignorance
- **Pragmatic Reasoning**: Understanding communicative intentions

#### 3.3 Evaluation (`selphi/evals.py`)

Multiple evaluation methods:

```python
from selphi import evaluate_scenario

result = evaluate_scenario(
    scenario=SALLY_ANNE,
    model_response="Basket",
    method="semantic_match"  # or "llm_judge"
)

# {
#   'average_score': 1.0,
#   'reasoning': 'Correct understanding of false belief',
#   'evaluation_method': 'semantic_match'
# }
```

#### 3.4 Benchmarks (`selphi/benchmarks.py`)

Three major ToM benchmarks:

**1. ToMBench** (388 test cases)
- Multiple levels: first-order, second-order, third-order beliefs
- Source: https://github.com/wadimiusz/ToMBench

**2. OpenToM** (696 questions)
- Location, multihop, attitude questions
- Source: https://github.com/seacowx/OpenToM

**3. SocialIQA** (38,000 questions)
- Commonsense reasoning about social situations
- Source: https://leaderboard.allenai.org/socialiqa

```python
from selphi.benchmarks import load_tombench, load_opentom

dataset = load_tombench(split='test')
# BenchmarkDataset(name='ToMBench', problems=[...], ...)
```

#### 3.5 Batch Processing

Run scenarios at scale:

```python
from selphi import run_multiple_scenarios, get_scenarios_by_difficulty

# Get all medium difficulty scenarios
scenarios = get_scenarios_by_difficulty("medium")

# Run with multiple models
results = run_multiple_scenarios(
    scenarios,
    provider="ollama",
    verbose=True
)

# Compare models
from selphi import compare_models
comparison = compare_models(
    scenarios,
    providers=["ollama", "anthropic"],
    models=["llama3.2:latest", "claude-3-5-sonnet-20241022"]
)
```

---

## Integration Between Subsystems

### Shared Infrastructure

All three subsystems share:

1. **LLM Provider** - Same interface for calling models
2. **Experiment Tracker** - Unified logging format
3. **Model Configs** - Shared YAML configuration
4. **Evaluation Framework** - Common eval patterns

### Cross-Subsystem Workflows

```python
# Use harness experiment tracking with CRIT
from harness import get_tracker, ExperimentConfig
from crit import run_critique_strategy, MOBILE_CHECKOUT

config = ExperimentConfig(
    experiment_name="crit_multi_perspective",
    task_type="design_critique",
    strategy="multi_perspective"
)

tracker = get_tracker()
tracker.start_experiment(config)

result = run_critique_strategy("multi_perspective", MOBILE_CHECKOUT)
tracker.log_result(...)
tracker.finish_experiment()
```

### Unified Benchmarks

The `harness.benchmarks` module provides a single interface to access benchmarks from all subsystems:

```python
from harness import load_benchmark, BENCHMARKS

# BENCHMARKS contains: uicrit (CRIT), tombench/opentom/socialiqa (SELPHI)
for name, info in BENCHMARKS.items():
    print(f"{name}: {info.subsystem} - {info.description}")
```

---

## Hardware Optimization (M4 Max)

### Memory Utilization

With 128GB unified memory:

- **Single 70B model**: ~35GB (4-bit quantized)
- **3x 7B models**: ~10GB (parallel multi-agent)
- **Fine-tuning 13B**: ~20GB (LoRA)

### Recommended Configurations

**Fast Iteration** (development):
```python
provider="ollama"
model="llama3.2:3b"  # 3B parameter model
# ~2GB memory, ~100 tokens/sec
```

**Quality Experiments** (research):
```python
provider="ollama"
model="llama3.1:70b"  # 70B parameter model
# ~35GB memory, ~15 tokens/sec
```

**Multi-Agent** (parallel):
```python
# 3-4 agents with 7B models simultaneously
n_debaters=3
model="llama3.1:8b"
# ~12GB total, good throughput
```

**Comparison Baseline** (API):
```python
provider="anthropic"
model="claude-3-5-sonnet-20241022"
# $3/MTok input, $15/MTok output
```

---

## Extensibility

### Adding New Strategies

**Harness**:
```python
# In harness/strategies.py
def my_new_strategy(task_input: str, **kwargs) -> StrategyResult:
    # Implementation
    return StrategyResult(...)

STRATEGIES["my_new"] = my_new_strategy
```

**CRIT**:
```python
# In crit/strategies.py
def my_critique_strategy(problem: DesignProblem, **kwargs) -> CritiqueResult:
    # Implementation
    return CritiqueResult(...)
```

### Adding New Problems/Scenarios

**CRIT**:
```python
from crit import DesignProblem, DesignDomain

MY_PROBLEM = DesignProblem(
    name="my_problem",
    domain=DesignDomain.UI_UX,
    description="...",
    current_design="...",
    success_criteria=[...],
    difficulty="medium"
)
```

**SELPHI**:
```python
from selphi import ToMScenario, ToMType

MY_SCENARIO = ToMScenario(
    name="my_scenario",
    tom_type=ToMType.FALSE_BELIEF,
    scenario_text="...",
    question="...",
    correct_answer="...",
    difficulty="easy"
)
```

### Adding New Evaluations

All subsystems use a registry pattern:

```python
# Custom evaluator
def my_eval(output: str, expected: Any) -> float:
    # Return score 0-1
    return score

# Register it
from harness.evals import EVAL_FUNCTIONS
EVAL_FUNCTIONS["my_eval"] = my_eval
```

---

## Data Flow

### Typical Experiment Flow

```
1. Define Configuration
   └─> ExperimentConfig(name, strategy, provider, model, ...)

2. Start Tracking
   └─> tracker.start_experiment(config)
   └─> Creates experiments/{name}_{timestamp}/ directory

3. Run Experiments
   └─> For Harness: run_strategy(...)
   └─> For CRIT: run_critique_strategy(...)
   └─> For SELPHI: run_scenario(...)

4. Log Results
   └─> tracker.log_result(...)
   └─> Appends to results.jsonl

5. Finish Experiment
   └─> tracker.finish_experiment()
   └─> Generates summary.json, README.md

6. Compare Experiments
   └─> compare_experiments([dir1, dir2], metric="latency_s")
```

### File Formats

**Configuration** (`config.json`):
```json
{
  "experiment_name": "baseline_reasoning",
  "strategy": "debate",
  "provider": "ollama",
  "model": "llama3.2:latest",
  "temperature": 0.7,
  "timestamp": "2025-11-03T14:30:22"
}
```

**Results** (`results.jsonl`):
```jsonl
{"task_input": "...", "output": "...", "latency_s": 2.3, "tokens_in": 150, "tokens_out": 200}
{"task_input": "...", "output": "...", "latency_s": 1.8, "tokens_in": 120, "tokens_out": 180}
```

**Summary** (`summary.json`):
```json
{
  "total_tasks": 10,
  "avg_latency_s": 2.1,
  "total_tokens": 3500,
  "total_cost_usd": 0.15
}
```

---

## Performance Characteristics

### Latency Benchmarks (M4 Max)

| Model Size | Provider | Tokens/sec | First Token | Memory |
|------------|----------|------------|-------------|--------|
| 3B (4-bit) | MLX | ~120 | ~100ms | 2GB |
| 7B (4-bit) | Ollama | ~80 | ~150ms | 4GB |
| 13B (4-bit) | Ollama | ~50 | ~200ms | 8GB |
| 70B (4-bit) | Ollama | ~15 | ~800ms | 35GB |
| Claude 3.5 Sonnet | API | Variable | ~500ms | N/A |

### Cost Comparison

**Local** (one-time):
- No per-token cost
- Electricity: ~$0.01/hour (M4 Max at 50W)

**API** (per-use):
- Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
- GPT-4: $30/MTok input, $60/MTok output

**Break-even**: ~1M tokens processed (depending on model)

---

## Design Principles

### 1. Local-First
- Prioritize MLX and Ollama for cost-effective iteration
- API providers as comparison baseline, not default

### 2. Notebook-Centric
- All core functions work directly in Jupyter
- Minimal boilerplate, maximum interactivity

### 3. Reproducible
- Automatic experiment logging
- Git hash capture
- Configuration tracking

### 4. Hackable
- Simple, well-commented code
- Registry patterns for extensibility
- No heavy frameworks

### 5. Modular
- Clear subsystem boundaries
- Shared infrastructure where it makes sense
- Independent evolution of each subsystem

---

## Future Directions

### Planned Enhancements

1. **Interpretability Tools** (representations/latent-space/topologies integration)
   - Hidden layer visualization
   - Attention pattern analysis
   - Confidence calibration

2. **Fine-Tuning Workflows**
   - LoRA training with MLX
   - Dataset generation from experiments
   - Performance comparison (base vs fine-tuned)

3. **Advanced Strategies**
   - Chain-of-thought prompting
   - Tree-of-thought search
   - Mixture-of-agents

4. **Benchmark Expansion**
   - MMLU (general knowledge)
   - GSM8K (math reasoning)
   - HumanEval (code generation)

5. **Production Features** (if needed)
   - Distributed execution
   - Model caching/quantization
   - Production API serving

---

## References

### Papers & Datasets

**CRIT**:
- UICrit: Duan et al., "UICrit: 11,344 Design Critiques for Mobile UIs", UIST 2024

**SELPHI**:
- ToMBench: Nemirovsky et al., 2023
- OpenToM: Ma et al., 2023
- SocialIQA: Sap et al., EMNLP 2019

**MLX**:
- MLX Framework: https://github.com/ml-explore/mlx
- MLX-LM: https://github.com/ml-explore/mlx-examples/tree/main/llms

### External Resources

- Ollama: https://ollama.ai
- MLX Models: https://huggingface.co/mlx-community
- Anthropic API: https://anthropic.com
- OpenAI API: https://openai.com

---

**Document Status**: Living document, updated as architecture evolves.
