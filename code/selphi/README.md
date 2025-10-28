# SELPHI - Study of Epistemic and Logical Processing in Heuristic Intelligence

A research toolkit for testing theory of mind (ToM) and epistemology in language models.

## Overview

**SELPHI** provides a structured framework for evaluating how well language models understand mental states, beliefs, knowledge attribution, and perspective-taking. It's part of the hidden-layer research lab and integrates seamlessly with the existing harness for experiment tracking.

### What is Theory of Mind?

Theory of Mind (ToM) is the ability to attribute mental states—beliefs, intents, desires, knowledge—to oneself and others, and to understand that others have beliefs and knowledge that may differ from one's own.

In language models, we're interested in questions like:
- Can models track false beliefs (beliefs that differ from reality)?
- Do models understand that different agents have different knowledge?
- Can models reason from multiple perspectives?
- Do models understand how beliefs update with new information?
- Can models handle second-order beliefs (beliefs about beliefs)?

### Why Study ToM in LLMs?

Understanding ToM capabilities in language models is crucial for:
1. **Safety & Alignment**: Models need to understand human mental states to be truly helpful
2. **Multi-Agent Systems**: ToM is essential for effective agent collaboration
3. **Human-AI Interaction**: Better ToM leads to more natural communication
4. **Cognitive Science**: LLMs as models of cognitive processes
5. **Capability Assessment**: Understanding the boundaries of current models

## Features

- **Pre-defined Scenarios**: 9+ carefully designed ToM tests covering different aspects
- **Multiple ToM Types**: False belief, knowledge attribution, perspective taking, belief updating, second-order beliefs, epistemic states, pragmatic reasoning
- **Flexible Evaluation**: Both semantic matching and LLM-as-judge evaluation methods
- **Model Comparison**: Easy comparison of multiple models on same scenarios
- **Difficulty Levels**: Scenarios categorized as easy, medium, or hard
- **Integration**: Works seamlessly with the hidden-layer harness

## Quick Start

```python
from selphi import run_scenario, SALLY_ANNE, evaluate_scenario

# Run a classic false belief test
result = run_scenario(SALLY_ANNE, provider="ollama")

# Evaluate the response
eval_result = evaluate_scenario(SALLY_ANNE, result.model_response)
print(f"Score: {eval_result['average_score']:.2f}")
```

## Installation

SELPHI is part of the hidden-layer project. From the project root:

```bash
# Activate environment
source venv/bin/activate

# The selphi package is already available
python -c "from selphi import SALLY_ANNE; print('Ready!')"
```

## Usage

### Running a Single Scenario

```python
from selphi import run_scenario, CHOCOLATE_BAR

result = run_scenario(
    CHOCOLATE_BAR,
    provider="ollama",
    model="llama3.2:latest",
    temperature=0.1
)

print(result.model_response)
print(f"Latency: {result.latency_s:.2f}s")
```

### Running Multiple Scenarios

```python
from selphi import run_multiple_scenarios, get_scenarios_by_difficulty

# Get all easy scenarios
easy_scenarios = get_scenarios_by_difficulty("easy")

# Run them
results = run_multiple_scenarios(
    easy_scenarios,
    provider="ollama",
    verbose=True
)
```

### Evaluating Results

```python
from selphi import evaluate_scenario, evaluate_batch, results_to_dict_list

# Single scenario evaluation
eval_result = evaluate_scenario(
    scenario,
    response_text,
    method="semantic"  # or "llm_judge"
)

# Batch evaluation
batch_results = results_to_dict_list(results)
batch_eval = evaluate_batch(batch_results, method="semantic")
print(f"Overall score: {batch_eval['overall_average']:.2f}")
```

### Comparing Models

```python
from selphi import compare_models_on_scenarios, ToMType, get_scenarios_by_type

# Define models
models = [
    {'name': 'ollama-default', 'provider': 'ollama', 'model': None},
    {'name': 'gpt-4o-mini', 'provider': 'openai', 'model': 'gpt-4o-mini'},
]

# Run comparison
scenarios = get_scenarios_by_type(ToMType.FALSE_BELIEF)
results = compare_models_on_scenarios(scenarios, models, verbose=True)

# Evaluate
from selphi import compare_models
comparison = compare_models(
    {name: results_to_dict_list(res) for name, res in results.items()}
)
```

## Available Scenarios

### False Belief
- **SALLY_ANNE**: Classic Sally-Anne marble test
- **CHOCOLATE_BAR**: Chocolate moved while protagonist is away

### Knowledge Attribution
- **SURPRISE_PARTY**: Track who knows what about a surprise party
- **BROKEN_VASE**: Temporal tracking of knowledge acquisition

### Perspective Taking
- **MOVIE_OPINIONS**: Understanding different preferences and emotions

### Belief Updating
- **WEATHER_UPDATE**: How beliefs change with new information

### Second-Order Belief
- **GIFT_SURPRISE**: Beliefs about beliefs (Mark knows Lisa thinks he doesn't know)

### Epistemic State
- **COIN_FLIP**: Distinguishing knowing from believing from guessing

### Pragmatic Reasoning
- **DOOR_LOCKED**: Understanding implied knowledge in communication

## ToM Types

```python
from selphi import ToMType

ToMType.FALSE_BELIEF          # Classic Sally-Anne style tests
ToMType.KNOWLEDGE_ATTRIBUTION # Who knows what based on observation
ToMType.PERSPECTIVE_TAKING    # Different viewpoints and preferences
ToMType.BELIEF_UPDATING       # How beliefs change with new info
ToMType.SECOND_ORDER_BELIEF   # Beliefs about beliefs
ToMType.EPISTEMIC_STATE       # Knowing vs believing vs guessing
ToMType.PRAGMATIC_REASONING   # Implied knowledge in communication
```

## Creating Custom Scenarios

```python
from selphi import ToMScenario, ToMType, run_scenario

my_scenario = ToMScenario(
    name="my_custom_test",
    tom_type=ToMType.FALSE_BELIEF,
    setup="Alice puts her keys in the drawer...",
    events=[
        "Bob moves the keys to the table while Alice is away",
        "Alice returns"
    ],
    test_questions=[
        "Where will Alice look for her keys?",
        "Where are the keys actually located?"
    ],
    correct_answers=[
        "Alice will look in the drawer",
        "The keys are on the table"
    ],
    reasoning="Tests if model understands Alice's false belief",
    difficulty="easy"
)

# Run it
result = run_scenario(my_scenario, provider="ollama")
```

## Evaluation Methods

### Semantic Matching
Fast, keyword-based evaluation. Good for initial screening.

```python
eval_result = evaluate_scenario(scenario, response, method="semantic")
```

### LLM-as-Judge
More sophisticated evaluation using another LLM. Better for nuanced responses.

```python
eval_result = evaluate_scenario(
    scenario,
    response,
    method="llm_judge",
    judge_provider="ollama",
    judge_model="llama3.2:latest"
)
```

## Integration with Harness

SELPHI works with the hidden-layer experiment tracker:

```python
from harness import get_tracker, ExperimentConfig, ExperimentResult
from selphi import run_all_scenarios, evaluate_batch, results_to_dict_list

# Start experiment
tracker = get_tracker()
experiment_dir = tracker.start_experiment(ExperimentConfig(
    experiment_name="tom_baseline",
    strategy="single",
    provider="ollama",
    model="llama3.2:latest",
    metadata={"project": "selphi", "eval_type": "tom"}
))

# Run scenarios
results = run_all_scenarios(provider="ollama", verbose=True)

# Evaluate
eval_results = evaluate_batch(results_to_dict_list(results))

# Log each result
for result, eval_result in zip(results, eval_results['evaluations']):
    tracker.log_result(ExperimentResult(
        task_input=result.metadata['prompt'],
        output=result.model_response,
        strategy_name="single",
        latency_s=result.latency_s,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
        metadata={
            "scenario": result.scenario_name,
            "tom_type": result.scenario_type,
            "difficulty": result.difficulty,
            "score": eval_result['average_score']
        }
    ))

# Finish
summary = tracker.finish_experiment()
print(f"Results saved to: {experiment_dir}")
```

## Notebook Examples

See `notebooks/selphi/01_basic_tom_tests.ipynb` for comprehensive examples including:
- Running individual scenarios
- Batch processing
- Model comparison
- LLM-as-judge evaluation
- Creating custom scenarios
- Research question templates

## API Reference

### Core Functions

#### `run_scenario(scenario, provider, model, **kwargs)`
Run a single ToM scenario with a model.

**Parameters:**
- `scenario`: ToMScenario object
- `provider`: LLM provider ("ollama", "mlx", "openai", "anthropic")
- `model`: Model name (optional, uses default if None)
- `**kwargs`: Additional arguments for llm_call (temperature, max_tokens, etc.)

**Returns:** ToMTaskResult

#### `evaluate_scenario(scenario, response, method, **kwargs)`
Evaluate a model's response to a scenario.

**Parameters:**
- `scenario`: ToMScenario object
- `response`: Model's response text
- `method`: "semantic" or "llm_judge"
- `**kwargs`: Additional arguments for evaluation

**Returns:** Dictionary with scores and analysis

### Helper Functions

- `run_multiple_scenarios()`: Run multiple scenarios
- `run_all_scenarios()`: Run all available scenarios
- `run_scenarios_by_type()`: Run scenarios of a specific ToM type
- `run_scenarios_by_difficulty()`: Run scenarios of a specific difficulty
- `compare_models_on_scenarios()`: Compare multiple models
- `evaluate_batch()`: Evaluate multiple results
- `compare_models()`: Compare evaluation results across models

## Research Directions

### Suggested Experiments

1. **Model Size vs ToM**: Does model size correlate with ToM performance?
2. **ToM Type Analysis**: Which types are hardest for current models?
3. **Multi-Agent ToM**: Do multi-agent strategies improve ToM reasoning?
4. **Fine-Tuning Impact**: Does ToM-specific fine-tuning help?
5. **Prompt Engineering**: What prompt formats maximize ToM performance?
6. **Cross-Linguistic ToM**: Do ToM capabilities transfer across languages?
7. **Failure Mode Analysis**: Why do models fail specific ToM types?

### Measurement Questions

- What metrics best capture ToM understanding?
- How do we measure second-order belief understanding?
- Can we develop automated ToM benchmarks?
- What's the relationship between ToM and other cognitive capabilities?

### Architectural Questions

- Which layers encode mental state information?
- Are there "ToM neurons" or circuits?
- How does attention relate to perspective-taking?
- Can we identify when models "switch perspectives"?

## Project Structure

```
code/selphi/
├── __init__.py         # Package exports
├── scenarios.py        # Pre-defined ToM scenarios
├── tasks.py           # Task execution functions
├── evals.py           # Evaluation functions
└── README.md          # This file

notebooks/selphi/
└── 01_basic_tom_tests.ipynb  # Tutorial notebook
```

## Future Enhancements

- [ ] More scenario types (deception, cooperation, competition)
- [ ] Automated scenario generation
- [ ] Visual ToM tests (image-based scenarios)
- [ ] Longitudinal tracking (memory of mental states)
- [ ] Multi-turn dialogue ToM
- [ ] Cross-cultural ToM variations
- [ ] Integration with interpretability tools
- [ ] Benchmark dataset creation

## Contributing

To add new scenarios:

1. Create a ToMScenario object with all required fields
2. Add to the appropriate registry in `scenarios.py`
3. Test with multiple models
4. Document expected behavior

To add new ToM types:

1. Add to ToMType enum in `scenarios.py`
2. Create representative scenarios
3. Update documentation

## References

### Classic ToM Literature
- Baron-Cohen, S., et al. (1985). "Does the autistic child have a theory of mind?"
- Wimmer, H., & Perner, J. (1983). "Beliefs about beliefs"
- Wellman, H. M. (2014). "Making Minds: How Theory of Mind Develops"

### ToM in AI/LLMs
- Kosinski, M. (2023). "Theory of Mind May Have Spontaneously Emerged in Large Language Models"
- Shapira, N., et al. (2023). "Clever Hans or Neural Theory of Mind?"
- Sap, M., et al. (2022). "Neural Theory-of-Mind?"

### Related Projects
- BigToM: https://github.com/ying-hui-he/bigToM-Benchmark
- ToMi Dataset: https://github.com/facebookresearch/ToMi
- SocialIQA: https://maartensap.com/social-iqa/

## License

Part of the hidden-layer research project. See main project README for details.

## Citation

If you use SELPHI in your research, please cite:

```
@software{selphi2024,
  title={SELPHI: Study of Epistemic and Logical Processing in Heuristic Intelligence},
  author={Hidden Layer Research Lab},
  year={2024},
  url={https://github.com/yourusername/hidden-layer}
}
```

---

**SELPHI** - Because understanding minds is the first step to building better ones.
