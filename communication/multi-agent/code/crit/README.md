# CRIT - Collective Reasoning for Iterative Testing

A research toolkit for testing collective design critique reasoning agents on challenging design problems.

## Overview

**CRIT** explores whether multi-agent collective critique can produce better design solutions than single-agent approaches. It provides a structured framework for:

- Running design problems through multiple critique strategies
- Comparing single vs. multi-perspective critique
- Evaluating critique quality, coverage, and actionability
- Testing iterative refinement and adversarial critique

### Research Question

**Can collective design critique from multiple expert perspectives produce better solutions than a single generalist critic?**

We're interested in understanding:
1. **When** collective critique helps (problem types, complexity)
2. **Why** it helps (coverage, diversity, synthesis)
3. **How much** it helps (cost-benefit analysis)
4. **What** makes effective critique (actionability, specificity)

## Features

- **8 Pre-defined Design Problems** across multiple domains
- **4 Critique Strategies** (single, multi-perspective, iterative, adversarial)
- **9 Critique Perspectives** (usability, security, performance, etc.)
- **Comprehensive Evaluation** (coverage, quality, depth metrics)
- **Flexible Framework** for custom problems and perspectives
- **Integration** with hidden-layer harness for experiment tracking

## Design Domains

### UI/UX Design
- Mobile checkout flow optimization
- Analytics dashboard layout

### API Design
- REST API versioning strategy
- GraphQL schema design

### System Architecture
- Microservices decomposition
- Multi-layer caching strategy

### Data Modeling
- Flexible permission system

### Workflow Design
- Multi-stage approval workflow

## Critique Strategies

### 1. Single Critic (Baseline)
One expert provides comprehensive feedback.

```python
from crit import run_critique_strategy, MOBILE_CHECKOUT

result = run_critique_strategy("single", MOBILE_CHECKOUT, provider="ollama")
```

### 2. Multi-Perspective Critique
Multiple experts review from different viewpoints (usability, security, accessibility, etc.) with optional synthesis.

```python
result = run_critique_strategy(
    "multi_perspective",
    MOBILE_CHECKOUT,
    synthesize=True,
    provider="ollama"
)
```

### 3. Iterative Critique
Cycles of critique → propose improvements → critique improvements.

```python
result = run_critique_strategy(
    "iterative",
    MOBILE_CHECKOUT,
    iterations=2,
    provider="ollama"
)
```

### 4. Adversarial Critique
Two-agent debate: proposer suggests improvements, challenger finds flaws, proposer responds.

```python
result = run_critique_strategy(
    "adversarial",
    MOBILE_CHECKOUT,
    provider="ollama"
)
```

## Quick Start

```python
from crit import (
    run_critique_strategy,
    MOBILE_CHECKOUT,
    evaluate_critique,
    compare_strategies
)

# Run multi-perspective critique
result = run_critique_strategy(
    "multi_perspective",
    MOBILE_CHECKOUT,
    provider="ollama",
    synthesize=True
)

# Evaluate the critique
evaluation = evaluate_critique(
    MOBILE_CHECKOUT,
    result,
    method="combined"
)

print(f"Quality Score: {evaluation['quality']['overall_quality']:.2f}")
print(f"Coverage Score: {evaluation['coverage']['overall_coverage']:.2f}")
print(f"Recommendations: {len(result.recommendations)}")
```

## Installation

CRIT is part of the hidden-layer project. From the project root:

```bash
# Activate environment
source venv/bin/activate

# The crit package is already available
python -c "from crit import MOBILE_CHECKOUT; print('Ready!')"
```

## Usage

### Working with Design Problems

```python
from crit import (
    ALL_PROBLEMS,
    get_problems_by_domain,
    get_problems_by_difficulty,
    DesignDomain
)

# Get all UI/UX problems
ui_problems = get_problems_by_domain(DesignDomain.UI_UX)

# Get all hard problems
hard_problems = get_problems_by_difficulty("hard")

# View a problem
print(problem.to_critique_prompt())
```

### Custom Perspectives

```python
from crit import multi_perspective_critique, CritiquePerspective

# Specify custom perspectives
custom_perspectives = [
    CritiquePerspective.USABILITY,
    CritiquePerspective.SECURITY,
    CritiquePerspective.PERFORMANCE,
    CritiquePerspective.ACCESSIBILITY
]

result = multi_perspective_critique(
    problem,
    perspectives=custom_perspectives,
    synthesize=True,
    provider="ollama"
)
```

### Evaluating Critiques

```python
from crit import evaluate_critique

# Evaluate with all metrics
evaluation = evaluate_critique(
    problem,
    critique_result,
    method="combined",  # or "coverage", "quality", "depth"
    judge_provider="ollama"
)

# Access metrics
print(f"Coverage: {evaluation['coverage']['overall_coverage']:.2f}")
print(f"Quality: {evaluation['quality']['overall_quality']:.2f}")
print(f"Depth: {evaluation['depth']['depth_score']:.2f}")
print(f"Combined: {evaluation['combined_score']:.2f}")
```

### Comparing Strategies

```python
from crit import compare_strategies

# Run multiple strategies
results = {
    "single": run_critique_strategy("single", problem),
    "multi": run_critique_strategy("multi_perspective", problem),
    "iterative": run_critique_strategy("iterative", problem),
}

# Compare
comparison = compare_strategies(problem, results, judge_provider="ollama")

# View rankings
for rank in comparison["rankings"]["combined"]:
    print(f"{rank['strategy']}: {rank['score']:.3f}")
```

## Available Problems

### UI/UX
- **MOBILE_CHECKOUT**: E-commerce checkout flow (medium)
- **DASHBOARD_LAYOUT**: Analytics dashboard layout (medium)

### API
- **REST_API_VERSIONING**: API versioning strategy (hard)
- **GRAPHQL_SCHEMA**: GraphQL schema for social media (hard)

### System
- **MICROSERVICES_SPLIT**: Monolith decomposition (hard)
- **CACHING_STRATEGY**: Multi-layer caching (medium)

### Data
- **PERMISSION_MODEL**: Flexible permission system (hard)

### Workflow
- **APPROVAL_WORKFLOW**: Multi-stage approval process (hard)

## Critique Perspectives

```python
from crit import CritiquePerspective

CritiquePerspective.USABILITY          # User experience and ease of use
CritiquePerspective.ACCESSIBILITY      # Inclusive design
CritiquePerspective.PERFORMANCE        # Efficiency and speed
CritiquePerspective.SECURITY           # Safety and privacy
CritiquePerspective.MAINTAINABILITY    # Long-term sustainability
CritiquePerspective.SCALABILITY        # Growth and expansion
CritiquePerspective.AESTHETICS         # Visual and conceptual appeal
CritiquePerspective.CONSISTENCY        # Internal coherence
CritiquePerspective.USER_ADVOCACY      # User needs and goals
```

## Creating Custom Problems

```python
from crit import DesignProblem, DesignDomain

my_problem = DesignProblem(
    name="my_custom_problem",
    domain=DesignDomain.UI_UX,
    description="Design a notification system for a mobile app",
    current_design="""
Current approach:
- Toast notifications at bottom of screen
- Disappear after 3 seconds
- No notification history
- No action buttons
""",
    context="""
- Mobile app for task management
- Users receive 5-20 notifications per day
- Need to support different priority levels
- Some notifications require action
- Users complain they miss important notifications
""",
    success_criteria=[
        "Users don't miss important notifications",
        "Notifications don't disrupt workflow",
        "Easy to take action on notifications",
        "Clear visual hierarchy by priority"
    ],
    known_issues=[
        "Short display time causes users to miss notifications",
        "No way to act on notification after it disappears",
        "All notifications look the same regardless of importance"
    ],
    difficulty="medium"
)

# Run critique on it
result = run_critique_strategy("multi_perspective", my_problem)
```

## Evaluation Metrics

### Coverage
- How many known issues were identified?
- Were success criteria addressed?
- How many perspectives were considered?

### Quality (LLM-as-Judge)
- **Specificity**: Concrete vs vague recommendations
- **Actionability**: Clear implementation path
- **Relevance**: Addresses the actual problem
- **Feasibility**: Practical to implement

### Depth
- Length and thoroughness of critique
- Number of recommendations
- Presence of synthesis
- Multiple rounds of analysis

### Combined Score
Weighted combination: `0.3 * coverage + 0.5 * quality + 0.2 * depth`

## Integration with Experiment Tracking

```python
from harness import get_tracker, ExperimentConfig, ExperimentResult
from crit import run_critique_strategy, evaluate_critique

# Start experiment
tracker = get_tracker()
experiment_dir = tracker.start_experiment(ExperimentConfig(
    experiment_name="design_critique_comparison",
    strategy="multi_perspective",
    provider="ollama",
    metadata={"project": "crit", "problem": "mobile_checkout"}
))

# Run critique
result = run_critique_strategy("multi_perspective", problem, provider="ollama")

# Evaluate
evaluation = evaluate_critique(problem, result, method="combined")

# Log
tracker.log_result(ExperimentResult(
    task_input=problem.to_critique_prompt(),
    output=result.synthesis or "",
    strategy_name=result.strategy_name,
    latency_s=result.latency_s,
    tokens_in=result.total_tokens_in,
    tokens_out=result.total_tokens_out,
    cost_usd=result.total_cost_usd,
    metadata={
        "problem": problem.name,
        "domain": problem.domain.value,
        "coverage_score": evaluation["coverage"]["overall_coverage"],
        "quality_score": evaluation["quality"]["overall_quality"],
        "combined_score": evaluation["combined_score"],
        "recommendations": result.recommendations
    }
))

summary = tracker.finish_experiment()
```

## Notebook Examples

See `notebooks/crit/01_basic_critique_experiments.ipynb` for comprehensive examples including:
- Single vs multi-perspective comparison
- Iterative refinement experiments
- Adversarial critique examples
- Cross-domain testing
- Strategy comparison
- Custom perspective configuration

## Research Directions

### Suggested Experiments

1. **Strategy Effectiveness**
   - Which strategy finds the most issues?
   - Which produces the most actionable recommendations?
   - How does cost scale vs. quality improvement?

2. **Perspective Value**
   - Which perspectives are most valuable per domain?
   - How much overlap exists between perspectives?
   - Can we identify essential vs. redundant perspectives?

3. **Domain Specialization**
   - Do certain strategies work better for certain domains?
   - Are different perspective combinations needed per domain?

4. **Model Comparison**
   - How do different LLMs compare on design critique?
   - Do larger models give qualitatively better critiques?
   - Can smaller models be fine-tuned for critique?

5. **Iteration Dynamics**
   - How much does each iteration improve the design?
   - What's the point of diminishing returns?
   - Can we predict optimal iteration count?

6. **Synthesis Quality**
   - Does synthesis improve upon individual critiques?
   - What's lost in synthesis?
   - Can we measure synthesis effectiveness?

### Measurement Questions

- What metrics best capture "good design critique"?
- How do we measure actionability quantitatively?
- Can we validate critiques against real-world outcomes?
- How to benchmark against human expert critique?

## Existing Benchmarks & Datasets

**Note**: This is an emerging research area. While there are some related resources, standardized design critique benchmarks are limited. Consider:

- **HCI Research**: Papers on design critique and expert review methods
- **Software Engineering**: Code review and design review best practices
- **Design Thinking**: Frameworks for design critique (Norman, IDEO, etc.)
- **Domain-Specific**: UI/UX patterns, API design principles, architecture patterns

**Opportunity**: The CRIT framework can help create validated benchmark datasets for design critique evaluation.

## Project Structure

```
code/crit/
├── __init__.py         # Package exports
├── problems.py         # Design problem definitions
├── strategies.py       # Critique strategies
├── evals.py           # Evaluation functions
└── README.md          # This file

notebooks/crit/
└── 01_basic_critique_experiments.ipynb  # Tutorial
```

## API Reference

### Core Functions

#### `run_critique_strategy(strategy_name, problem, **kwargs)`
Run a critique strategy by name.

**Parameters:**
- `strategy_name`: "single", "multi_perspective", "iterative", or "adversarial"
- `problem`: DesignProblem object
- `**kwargs`: Strategy-specific and llm_call arguments

**Returns:** CritiqueResult

#### `evaluate_critique(problem, critique_result, method, **kwargs)`
Evaluate a critique result.

**Parameters:**
- `problem`: DesignProblem object
- `critique_result`: CritiqueResult to evaluate
- `method`: "coverage", "quality", "depth", or "combined"
- `**kwargs`: Evaluation arguments (judge_provider, judge_model, etc.)

**Returns:** Evaluation dictionary

### Strategy Functions

- `single_critic_strategy()`: Single expert critique
- `multi_perspective_critique()`: Multiple expert perspectives
- `iterative_critique()`: Iterative refinement
- `adversarial_critique()`: Adversarial debate

### Evaluation Functions

- `evaluate_critique_coverage()`: Coverage metrics
- `evaluate_recommendation_quality()`: LLM-as-judge quality
- `evaluate_critique_depth()`: Depth and thoroughness
- `compare_strategies()`: Compare multiple strategies
- `batch_evaluate()`: Evaluate multiple results

## Future Enhancements

- [ ] Additional design domains (security, networking, ML systems)
- [ ] More sophisticated synthesis strategies
- [ ] Human expert baselines for validation
- [ ] Fine-tuned models for specific critique types
- [ ] Interactive critique refinement
- [ ] Real-time collaboration between agents
- [ ] Visual design critique (images, mockups)
- [ ] Integration with design tools (Figma, etc.)
- [ ] Validated benchmark dataset creation

## Contributing

To add new design problems:
1. Create a DesignProblem with all required fields
2. Add to the appropriate registry in `problems.py`
3. Test with multiple strategies
4. Document expected critique aspects

To add new strategies:
1. Implement strategy function in `strategies.py`
2. Add to STRATEGIES registry
3. Document strategy behavior and use cases
4. Test on multiple problem types

## References

### Design Critique Methods
- Norman, D. A. (2013). "The Design of Everyday Things"
- Schön, D. A. (1983). "The Reflective Practitioner"
- IDEO Design Thinking methodology

### Multi-Agent Systems
- Hong, S., et al. (2023). "MetaGPT: Meta Programming for Multi-Agent Systems"
- Qian, C., et al. (2023). "Communicative Agents for Software Development"

### Collective Intelligence
- Woolley, A. W., et al. (2010). "Evidence for a Collective Intelligence Factor"
- Malone, T. W., et al. (2009). "Harnessing Crowds: Mapping the Genome of Collective Intelligence"

### Design Patterns & Best Practices
- Gang of Four (1994). "Design Patterns: Elements of Reusable Object-Oriented Software"
- Fowler, M. (2002). "Patterns of Enterprise Application Architecture"
- Nielsen, J. (1994). "Heuristic Evaluation"

## License

Part of the hidden-layer research project. See main project README for details.

## Citation

If you use CRIT in your research, please cite:

```
@software{crit2024,
  title={CRIT: Collective Reasoning for Iterative Testing},
  author={Hidden Layer Research Lab},
  year={2024},
  url={https://github.com/yourusername/hidden-layer}
}
```

---

**CRIT** - Because great design emerges from diverse perspectives.
