# ACE Development Guide

**Project**: Agentic Context Engineering (ACE) Framework Reproduction
**Paper**: arXiv:2510.04618 (Zhang et al., 2025)
**Location**: `alignment/ace/`

---

## Project Overview

This project reproduces the ACE (Agentic Context Engineering) framework from the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models".

**Core Idea**: Instead of fine-tuning model weights, ACE adapts LLMs through structured, evolving contexts (playbooks) that accumulate and organize strategies over time.

---

## Architecture

### Three Core Components

#### 1. Generator
**Purpose**: Produce reasoning trajectories for tasks

**Responsibilities**:
- Execute tasks using current context
- Generate detailed reasoning traces
- Capture successes and failures
- Surface effective strategies and pitfalls

**Implementation**: `src/generator.py`

**Key Methods**:
- `generate_trajectory(task, context) -> Trajectory`
- `execute_with_feedback(task, context) -> (result, feedback)`

#### 2. Reflector
**Purpose**: Extract insights from execution traces

**Responsibilities**:
- Critique reasoning trajectories
- Identify what worked and what didn't
- Distill concrete, actionable lessons
- Optionally refine insights over iterations

**Implementation**: `src/reflector.py`

**Key Methods**:
- `reflect(trajectory) -> List[Insight]`
- `critique(trajectory) -> Critique`
- `extract_lessons(critiques) -> List[Lesson]`
- `refine_insights(insights) -> List[Insight]`

#### 3. Curator
**Purpose**: Integrate insights into structured contexts

**Responsibilities**:
- Synthesize lessons into compact delta entries
- Merge deltas into existing context (deterministically)
- Organize strategies by topic/category
- Maintain playbook structure

**Implementation**: `src/curator.py`

**Key Methods**:
- `synthesize_delta(insights) -> ContextDelta`
- `merge_delta(context, delta) -> Context`
- `organize_strategies(context) -> OrganizedContext`

---

## Context Structure

### Playbook Format

Contexts are structured as **playbooks** with:

```yaml
metadata:
  version: int
  last_updated: timestamp
  domain: string

strategies:
  - id: string
    category: string
    description: string
    when_to_use: string
    examples: List[Example]
    success_rate: float

pitfalls:
  - id: string
    description: string
    how_to_avoid: string
    related_strategies: List[string]

history:
  - delta_id: string
    timestamp: timestamp
    changes: List[Change]
```

### Delta Format

```yaml
delta_id: string
timestamp: timestamp
source_trajectories: List[string]
insights: List[Insight]

additions:
  strategies: List[Strategy]
  pitfalls: List[Pitfall]

modifications:
  - strategy_id: string
    changes: Dict[string, Any]

removals:
  - strategy_id: string
    reason: string
```

---

## Workflow

### Offline ACE (System Prompt Optimization)

**Goal**: Optimize a fixed system prompt or context before deployment

**Process**:
1. **Collect trajectories**: Run tasks with initial context
2. **Reflect**: Extract insights from all trajectories
3. **Curate**: Synthesize and merge deltas
4. **Iterate**: Repeat with updated context
5. **Evaluate**: Test on held-out tasks

**Use Cases**:
- Pre-deployment prompt engineering
- Domain adaptation without fine-tuning
- Creating specialized agents

**Implementation**: `experiments/offline_ace.py`

### Online ACE (Agent Memory)

**Goal**: Continuously adapt context during deployment

**Process**:
1. **Execute**: Agent performs tasks with current context
2. **Accumulate**: Buffer recent trajectories
3. **Periodic update**: When buffer is full or threshold reached:
   - Reflect on recent trajectories
   - Curate delta
   - Merge into agent memory
4. **Continue**: Agent uses updated context

**Use Cases**:
- Long-running agent deployments
- Personalization to specific users/domains
- Continuous improvement from experience

**Implementation**: `experiments/online_ace.py`

---

## Implementation Guidelines

### Using the Harness

All LLM calls should use the harness:

```python
from harness import llm_call, get_tracker, ExperimentConfig

# In Generator
def generate_trajectory(self, task, context):
    prompt = f"{context}\n\nTask: {task}"
    response = llm_call(
        prompt,
        provider=self.provider,
        model=self.model,
        temperature=self.temperature
    )
    return self._parse_trajectory(response.text)

# In Reflector
def reflect(self, trajectory):
    prompt = self._build_reflection_prompt(trajectory)
    response = llm_call(
        prompt,
        provider=self.provider,
        model=self.model
    )
    return self._parse_insights(response.text)
```

### Experiment Tracking

Always use experiment tracking:

```python
from harness import get_tracker

tracker = get_tracker()
tracker.start_experiment(config)

# Log each iteration
tracker.log_metric("iteration", i)
tracker.log_metric("performance", score)
tracker.log_artifact("context", context)

tracker.finish_experiment()
```

### Configuration Management

Use YAML configs for reproducibility:

```yaml
# configs/ace_config.yaml
ace:
  generator:
    temperature: 0.7
    max_tokens: 2048

  reflector:
    num_refinement_iterations: 2
    critique_depth: "detailed"

  curator:
    max_strategies_per_category: 5
    merge_strategy: "deterministic"

experiment:
  offline:
    num_iterations: 5
    tasks_per_iteration: 20
    evaluation_tasks: 50

  online:
    buffer_size: 10
    update_frequency: 5
```

---

## Testing Strategy

### Unit Tests

Test each component independently:

```python
# Test Generator
def test_generator_produces_trajectory():
    gen = Generator(provider="ollama", model="llama3.2")
    trajectory = gen.generate_trajectory(task, context)
    assert trajectory.steps is not None
    assert len(trajectory.steps) > 0

# Test Reflector
def test_reflector_extracts_insights():
    ref = Reflector(provider="ollama", model="llama3.2")
    insights = ref.reflect(trajectory)
    assert len(insights) > 0
    assert all(i.actionable for i in insights)

# Test Curator
def test_curator_merges_delta():
    cur = Curator()
    new_context = cur.merge_delta(context, delta)
    assert new_context.version > context.version
```

### Integration Tests

Test full ACE workflow:

```python
def test_offline_ace_improves_performance():
    ace = ACEFramework(...)
    initial_score = evaluate(initial_context, test_tasks)

    optimized_context = ace.run_offline(train_tasks, initial_context)

    final_score = evaluate(optimized_context, test_tasks)
    assert final_score > initial_score

def test_online_ace_adapts_continuously():
    ace = ACEFramework(...)
    scores = []

    for batch in task_stream:
        score = ace.run_online_batch(batch)
        scores.append(score)

    # Should improve over time
    assert scores[-1] > scores[0]
```

### Benchmark Tests

Test on standard benchmarks:

```python
def test_appworld_benchmark():
    ace = ACEFramework(...)
    results = run_appworld_benchmark(ace)

    # Log results
    tracker.log_metrics({
        "accuracy": results.accuracy,
        "success_rate": results.success_rate,
        "efficiency": results.efficiency
    })
```

---

## Key Research Questions

### Implementation-Focused

1. **Deterministic Merging**: How to merge deltas without LLM involvement?
2. **Context Organization**: What structure best prevents collapse?
3. **Insight Quality**: How to ensure reflections are actionable?
4. **Efficiency**: How to minimize LLM calls while maintaining quality?

### Evaluation-Focused

5. **Transfer**: Do optimized contexts transfer across models?
6. **Robustness**: How do contexts handle distribution shift?
7. **Scaling**: How do contexts evolve with more iterations?
8. **Interpretability**: Can we understand discovered strategies?

### Comparison-Focused

9. **vs Fine-tuning**: When is ACE better than fine-tuning?
10. **vs RAG**: How does ACE compare to retrieval-augmented generation?
11. **vs Prompting**: What's gained over manual prompt engineering?
12. **vs Agent Memory**: How does ACE compare to existing memory systems?

---

## Development Workflow

### Phase 1: Core Implementation
- [ ] Implement Generator, Reflector, Curator
- [ ] Implement Context management
- [ ] Implement deterministic delta merging
- [ ] Unit tests for all components

### Phase 2: Offline ACE
- [ ] Implement offline optimization loop
- [ ] Add evaluation metrics
- [ ] Test on simple tasks
- [ ] Experiment tracking integration

### Phase 3: Online ACE
- [ ] Implement online adaptation
- [ ] Add buffer management
- [ ] Test on continuous task streams
- [ ] Compare online vs offline

### Phase 4: Benchmarks
- [ ] Integrate AppWorld benchmark
- [ ] Integrate domain-specific tasks
- [ ] Run comparative experiments
- [ ] Reproduce paper results

### Phase 5: Analysis
- [ ] Context evolution analysis
- [ ] Strategy interpretability
- [ ] Ablation studies
- [ ] Transfer experiments

---

## Debugging Tips

### Context Collapse Detection

Monitor context evolution:

```python
def check_context_health(context, previous_context):
    # Check total information content
    current_tokens = count_tokens(context)
    previous_tokens = count_tokens(previous_context)

    if current_tokens < previous_tokens * 0.8:
        warnings.warn("Possible context collapse detected")

    # Check strategy diversity
    categories = set(s.category for s in context.strategies)
    if len(categories) < 3:
        warnings.warn("Low strategy diversity")
```

### Reflection Quality

Ensure insights are actionable:

```python
def validate_insight(insight):
    # Must have concrete description
    assert len(insight.description) > 10

    # Must have when-to-use guidance
    assert insight.when_to_use is not None

    # Should have examples
    if not insight.examples:
        warnings.warn(f"Insight {insight.id} lacks examples")
```

### Performance Monitoring

Track key metrics:

```python
metrics = {
    "performance": task_success_rate,
    "efficiency": avg_tokens_per_task,
    "context_size": len(context),
    "num_strategies": len(context.strategies),
    "update_frequency": deltas_per_iteration
}
```

---

## Integration with Hidden Layer

### Harness
- Use unified LLM provider abstraction
- Experiment tracking for all runs
- Model configuration management

### Shared Resources
- Store evolved contexts in `shared/contexts/ace/`
- Share benchmark datasets
- Common evaluation utilities

### Cross-Project Connections

**With Steerability**:
- Compare context-based vs vector-based steering
- Hybrid approaches: steering + ACE
- Adherence metrics for context-following

**With Theory of Mind**:
- Use SELPHI tasks as ACE benchmark
- Study how ToM emerges in evolved contexts
- Introspection about learned strategies

**With Multi-Agent**:
- Shared evolving playbooks across agents
- Collective learning through ACE
- Coordination strategies in contexts

---

## Paper-Specific Details

### Key Claims to Verify

1. **Performance**: +10.6% on agents, +8.6% on finance
2. **Efficiency**: -82.3% latency, -75.1% rollouts (offline)
3. **Scalability**: Matches production systems with smaller models
4. **Prevention**: No brevity bias or context collapse

### Baselines to Compare

- **GEPA**: Prior context optimization approach
- **Dynamic Cheatsheet**: Online agent memory
- **Manual Prompting**: Human-engineered prompts
- **Fine-tuning**: Traditional adaptation

### Critical Experiments

1. **Offline optimization curve**: Performance vs iterations
2. **Online adaptation**: Performance over continuous tasks
3. **Context evolution**: Qualitative analysis of changes
4. **Ablation**: Remove Generator/Reflector/Curator individually

---

## Next Steps

1. **Immediate**: Implement core components (Generator, Reflector, Curator)
2. **Short-term**: Build offline ACE and test on simple tasks
3. **Medium-term**: Integrate benchmarks and reproduce results
4. **Long-term**: Extensions and novel research questions

---

## Questions?

- See main `README.md` for project overview
- See Hidden Layer `CLAUDE.md` for lab-wide guidelines
- See `harness/README.md` for infrastructure details
