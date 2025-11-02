# Transformer Introspection - Implementation Guide

## Overview

This implementation replicates the methodology from Anthropic's paper ["Emergent Introspective Awareness in Large Language Models"](https://transformer-circuits.pub/2025/introspection/index.html) within your existing multi-agent research harness.

**Core Question**: Can language models accurately report on their internal states when concepts are injected into their activations?

## What Was Built

### 1. Activation Steering (`code/harness/activation_steering.py`)

Enables direct manipulation of model activations during inference.

**Key Features**:
- Extract activation vectors from specific layers
- Inject concept vectors during generation
- Multiple steering strategies (add, replace, scale)
- Configurable strength and position

**Usage**:
```python
from harness import ActivationSteerer, SteeringConfig
from mlx_lm import load

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Extract a concept
happiness_vec = steerer.extract_contrastive_concept(
    positive_prompt="I feel very happy and joyful!",
    negative_prompt="I feel neutral.",
    layer_idx=15,
    position="last"
)

# Generate with steering
steered_text, metadata = steerer.generate_with_steering(
    prompt="Tell me a story",
    concept_vector=happiness_vec,
    config=SteeringConfig(layer_idx=15, strength=1.5)
)
```

### 2. Concept Vector Library (`code/harness/concept_vectors.py`)

Store and manage extracted concept representations.

**Key Features**:
- Persistent storage (pickle format)
- Similarity search
- Built-in concept builders (emotions, topics)
- Metadata tracking

**Usage**:
```python
from harness import ConceptLibrary, build_emotion_library

# Build a library of emotion concepts
library = build_emotion_library(
    steerer=steerer,
    layer=15,
    model_name="llama-3.2-3b"
)

# Save for later
library.save("concepts/emotions.pkl")

# Use concepts
happiness = library.get("happiness")
similar = library.find_similar("happiness", top_k=3)
```

### 3. Introspection Tasks (`code/harness/introspection_tasks.py`)

Generate and evaluate introspection test cases.

**Task Types**:
1. **Detection**: Can model notice something was injected?
2. **Identification**: Can model identify what was injected?
3. **Recall**: Can model recall prior internal states?
4. **Discrimination**: Can model distinguish its outputs from prefills?

**Usage**:
```python
from harness import (
    IntrospectionTaskGenerator,
    IntrospectionEvaluator,
    IntrospectionTaskType
)

# Generate tasks
generator = IntrospectionTaskGenerator()
task = generator.detection_task(
    concept="happiness",
    base_prompt="Tell me about your day",
    layer=15,
    strength=1.5
)

# Evaluate responses
evaluator = IntrospectionEvaluator()
result = evaluator.evaluate(
    task=task,
    model_response=steered_response,
    baseline_response=baseline_response
)

print(f"Correct: {result.is_correct}, Confidence: {result.confidence}")
```

### 4. Introspection Strategy (`code/harness/strategies.py`)

Integrated into existing strategy framework - run like any other multi-agent strategy.

**Usage**:
```python
from harness import run_strategy

# Detection task
result = run_strategy(
    "introspection",
    task_input="Describe your current feelings",
    concept="happiness",
    layer=15,
    strength=1.5,
    task_type="detection",
    provider="mlx",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit"
)

print(f"Introspection Correct: {result.metadata['introspection_correct']}")
print(f"Confidence: {result.metadata['introspection_confidence']:.2f}")

# Identification task
result = run_strategy(
    "introspection",
    task_input="Think about emotions",
    concept="anger",
    distractors=["happiness", "sadness", "fear"],
    layer=15,
    strength=1.5,
    task_type="identification",
    provider="mlx",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit"
)
```

### 5. Experiment Notebook (`notebooks/03_introspection_experiments.ipynb`)

Complete workflow for systematic introspection experiments.

**Includes**:
- Concept extraction and library building
- Basic steering validation
- Detection and identification tests
- Layer sensitivity analysis
- Steering strength analysis
- Concept specificity analysis
- False positive rate testing
- Visualization and summary statistics

## Quick Start

### 1. Run a Simple Test

```python
from harness import run_strategy

result = run_strategy(
    "introspection",
    task_input="Tell me a story",
    concept="happiness",
    layer=15,
    strength=2.0,
    task_type="detection",
    provider="mlx",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    verbose=True
)
```

### 2. Build a Concept Library

```python
from mlx_lm import load
from harness import ActivationSteerer, build_emotion_library

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Extract emotions
library = build_emotion_library(steerer, layer=15, model_name="llama-3.2-3b")
library.save("concepts/emotions.pkl")

# Or build custom concepts
library.add_concept(
    name="creativity",
    vector=steerer.extract_contrastive_concept(
        "I feel highly creative and innovative",
        "I feel uncreative and conventional",
        layer_idx=15
    ),
    layer=15,
    extraction_prompt="I feel highly creative",
    model_name="llama-3.2-3b"
)
```

### 3. Run Systematic Experiments

```bash
cd notebooks
jupyter notebook 03_introspection_experiments.ipynb
```

## Key Findings to Validate

Based on the paper, you should expect:

### ✓ Middle/Late Layers Work Better
Layers 15-25 typically show higher introspection accuracy than early layers (1-10).

### ✓ Steering Strength Matters
- Weak steering (0.5-1.0): Harder to detect
- Moderate (1.5-2.0): Good balance
- Strong (3.0+): Easy to detect but may distort generation

### ✓ Some Concepts Are Easier
Distinctive concepts (emotions, topics) are easier to detect than subtle ones.

### ✓ Model Size Matters
Larger models (7B, 13B, 70B) show better introspection than tiny models (1B, 3B).

### ✓ Low False Positives
Models should NOT detect concepts when nothing is injected (FP rate < 20%).

## Architecture Details

### How Activation Steering Works

```
Input → Tokenizer → Embeddings
                       ↓
                    Layer 0
                       ↓
                    Layer 1
                       ↓
                      ...
                       ↓
                    Layer 15  ← INJECT CONCEPT HERE
                       ↓
                    Layer 16
                       ↓
                      ...
                       ↓
                    Final Layer → Generation
```

**Injection Methods**:
1. **Add**: `activation[layer] = activation[layer] + (concept_vec * strength)`
2. **Replace**: `activation[layer] = concept_vec * strength`
3. **Scale**: `activation[layer] = activation[layer] * (1 + strength)`

### Concept Extraction: Contrastive Method

```python
# Get activations for positive and negative prompts
pos_act = extract_activation("I feel happy")
neg_act = extract_activation("I feel neutral")

# Difference captures the "happiness" direction
concept_vec = pos_act - neg_act

# Optional: normalize
concept_vec = concept_vec / ||concept_vec||
```

This gives you a direction in activation space that represents the concept.

## Limitations

### 1. MLX Only
**Problem**: Only works with MLX models (local Apple Silicon)
**Reason**: Ollama and API providers don't expose activations
**Workaround**: Use llama.cpp Python bindings for non-MLX local models

### 2. Model-Specific Concepts
**Problem**: Concept vectors extracted from one model may not transfer to others
**Reason**: Different models have different internal representations
**Workaround**: Extract separate concept libraries per model

### 3. Layer Sensitivity
**Problem**: Optimal layer varies by model and concept
**Reason**: Different information is represented at different depths
**Workaround**: Run layer sweeps to find optimal layers for your use case

### 4. Evaluation Ambiguity
**Problem**: Model responses are freeform text, hard to parse
**Reason**: No structured output format
**Workaround**: Use multiple evaluation methods (keywords + LLM judge + embeddings)

## Advanced Usage

### Multi-Concept Steering

```python
# Inject multiple concepts simultaneously
happiness_vec = library.get("happiness").vector
excitement_vec = library.get("surprise").vector

combined_vec = happiness_vec + excitement_vec

steerer.generate_with_steering(
    prompt="Describe the party",
    concept_vector=combined_vec,
    config=SteeringConfig(layer_idx=15, strength=1.0)
)
```

### Layer-Specific Steering

```python
# Different concepts at different layers
from harness import ActivationCache

cache = ActivationCache(model, tokenizer, layer_indices=[10, 15, 20])

# Inject emotion at layer 10, topic at layer 20
# (Would require extending ActivationSteerer to support multi-layer)
```

### Custom Introspection Prompts

```python
from harness import IntrospectionTaskGenerator

generator = IntrospectionTaskGenerator()

# Custom detection prompt
task = generator.detection_task(
    concept="creativity",
    base_prompt="Write a poem",
    prompt_template="On a scale of 1-10, how creative are you feeling right now? Explain."
)
```

### Integration with Experiment Tracker

```python
from harness import get_tracker, ExperimentConfig

tracker = get_tracker()
config = ExperimentConfig(
    experiment_name="introspection_layer_sweep",
    strategy="introspection",
    provider="mlx",
    model="llama-3.2-3b"
)

run_dir = tracker.start_experiment(config)

# Run introspection experiments
for layer in [10, 15, 20, 25]:
    result = run_strategy(
        "introspection",
        task_input="Describe your state",
        concept="happiness",
        layer=layer,
        strength=1.5,
        task_type="detection",
        provider="mlx",
        model="mlx-community/Llama-3.2-3B-Instruct-4bit"
    )

    # Log to tracker
    # (Would need to extend ExperimentResult for introspection metadata)

summary = tracker.finish_experiment()
```

## Troubleshooting

### Issue: MLX Import Error

**Error**: `ImportError: No module named 'mlx'`

**Solution**:
```bash
pip install mlx mlx-lm
```

### Issue: Model Loading Fails

**Error**: `RuntimeError: Failed to load model`

**Solution**:
- Check model name is correct
- Ensure you have enough RAM (3B models need ~4GB)
- Verify MLX is properly installed for your hardware

### Issue: Activations Are All Zeros

**Error**: Extracted concept vectors are all zeros

**Solution**:
- Check layer index is valid (0 to num_layers-1)
- Verify model is in eval mode (not training)
- Try different token positions ("last" vs "mean")

### Issue: Low Introspection Accuracy

**Error**: Model scores < 30% accuracy

**Solution**:
- Increase steering strength (try 2.0-3.0)
- Use middle/late layers (15-25)
- Test with more distinctive concepts
- Try larger models (7B+ instead of 3B)

### Issue: High False Positive Rate

**Error**: Model detects concepts even when nothing injected

**Solution**:
- Refine introspection prompts (be more specific)
- Use LLM-as-judge evaluation instead of keywords
- Test with more trials to reduce noise

## Performance Considerations

### Memory Usage

**Model Size → RAM Needed** (4-bit quantized):
- 3B model: ~4GB
- 7B model: ~8GB
- 13B model: ~14GB
- 70B model: ~70GB

**Tip**: Use smaller models (3B) for development, scale up for final evaluation.

### Speed

**Typical Times** (M4 Max):
- Concept extraction: ~2-5 seconds per concept
- Steered generation: ~1-3 seconds for 100 tokens
- Full layer sweep (5 layers × 3 concepts): ~2-3 minutes

**Optimization**:
- Cache concept libraries (don't re-extract every time)
- Use lower max_tokens for faster iteration
- Parallelize independent experiments

## Next Steps

### Short-Term (Week 1-2)
1. Run `03_introspection_experiments.ipynb` end-to-end
2. Validate paper findings on your hardware
3. Build comprehensive concept library (emotions + topics + styles)
4. Tune layer selection and steering strength for your model

### Medium-Term (Month 1)
1. Compare introspection across model sizes (3B vs 7B vs 13B)
2. Test with fine-tuned models (does fine-tuning improve introspection?)
3. Explore concept composition (multiple concepts injected)
4. Build standardized introspection benchmark suite

### Long-Term (Research Direction)
1. Cross-model concept transfer
2. Causal analysis: which layers represent which concepts?
3. Safety applications: detecting and steering harmful concepts
4. Meta-learning: can models learn to improve their introspection?

## References

- **Original Paper**: [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html)
- **Activation Steering**: [Steering GPT-2-XL by adding an activation vector](https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/)
- **MLX Documentation**: [MLX Modules and Hooks](https://ml-explore.github.io/mlx/build/html/usage/modules.html)
- **Mechanistic Interpretability**: [Transformer Circuits Thread](https://transformer-circuits.pub/)

## Questions?

If you encounter issues or have questions:

1. Check this README first
2. Look at `INTROSPECTION_PLAN.md` for technical details
3. Review `03_introspection_experiments.ipynb` for examples
4. Open an issue on GitHub

---

**Built with**: MLX, Apple Silicon, and curiosity about what models know about themselves.
