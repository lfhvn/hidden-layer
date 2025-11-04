# Introspection - Development Guide

## Project Overview

Model introspection experiments inspired by Anthropic's recent findings on whether models can accurately report their internal states.

**Research Question**: Can models accurately report their internal states, and how does this relate to alignment?

**Core Components**:
- Concept vectors and activation steering
- Introspection tasks
- API-based introspection (for frontier models)
- Connection to deception detection

**Uses**: `harness/` for LLM provider abstraction and experiment tracking

---

## Architecture

### Core Components

**Activation Steering** (`code/activation_steering.py`):
- Capture activations from model layers
- Apply steering vectors
- Test behavioral changes

**Concept Vectors** (`code/concept_vectors.py`):
- Extract concept representations
- Build concept libraries (emotions, topics, etc.)
- Stored in `/shared/concepts/`

**Introspection Tasks** (`code/introspection_tasks.py`):
- Task types: emotion detection, topic classification, etc.
- Evaluate introspection accuracy
- Compare self-report vs. ground truth

**API Introspection** (`code/introspection_api.py`):
- Prompt-based steering (for API models)
- Natural introspection prompts
- Frontier model evaluation

---

## Development Workflows

### Building a Concept Library

```python
from mlx_lm import load
from code.concept_vectors import ConceptLibrary, build_emotion_library
from code.activation_steering import ActivationSteerer

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Build emotion library
library = build_emotion_library(
    steerer=steerer,
    layer=15,
    model_name="llama-3.2-3b"
)

# Save to shared concepts
library.save("/home/user/hidden-layer/shared/concepts/emotions_layer15.pkl")
library.export_json("/home/user/hidden-layer/shared/concepts/emotions_layer15.json")
```

### Running Introspection Tasks

```python
from code.introspection_tasks import IntrospectionTaskGenerator, IntrospectionEvaluator
from harness import get_tracker

# Generate tasks
generator = IntrospectionTaskGenerator(concept_library=library)
tasks = generator.generate_tasks(n=100, task_type="emotion")

# Run evaluation
evaluator = IntrospectionEvaluator(steerer=steerer, library=library)
results = evaluator.evaluate_tasks(tasks)

print(f"Accuracy: {results['accuracy']:.2f}")
print(f"Calibration: {results['calibration']:.2f}")
```

### API-Based Introspection

```python
from code.introspection_api import APIIntrospectionTester, PromptSteerer

# Test frontier model
tester = APIIntrospectionTester(provider="anthropic")

result = tester.test_introspection(
    text="I am feeling very happy today!",
    target_concept="happiness",
    model="claude-3-5-sonnet-20241022"
)

print(f"Self-report: {result['self_report']}")
print(f"Confidence: {result['confidence']:.2f}")
```

---

## Research Questions

1. **Accuracy**: How well can models report their internal states?
   - Varies by layer?
   - Varies by concept type?

2. **Calibration**: Are confidence levels accurate?
   - Do models know when they don't know?

3. **Steering**: Can we steer models to be more honest?
   - Connection to alignment

4. **Deception**: Can models deceive about their states?
   - Detection strategies

5. **Generalization**: Does introspection ability transfer?
   - Across tasks?
   - Across model scales?

---

## Integration Points

**With SELPHI**:
- Understanding self (introspection) vs. others (ToM)
- Is there a unified mechanism?

**With Latent Space**:
- What SAE features correspond to introspection?
- Can we navigate latent space to understand self-knowledge?

**With Steerability**:
- Can we steer models toward honest reporting?
- Is introspection a reliable alignment signal?

---

## Key Files

- `code/activation_steering.py` - Activation capture and steering
- `code/concept_vectors.py` - Concept representation
- `code/introspection_tasks.py` - Introspection evaluation
- `code/introspection_api.py` - API-based introspection
- `/shared/concepts/` - Shared concept vectors

---

## Concept Vector Storage

See `/shared/concepts/README.md` for:
- Building custom concept libraries
- Layer selection guidelines
- Prompt engineering best practices
- Sharing concepts across projects

---

## Testing

```bash
# Run tests
cd projects/introspection
pytest tests/ -v

# Quick test (requires MLX)
python -c "from code.concept_vectors import ConceptLibrary; print('Introspection ready')"
```

---

## See Also

- Concept vector guide: `/shared/concepts/README.md`
- Infrastructure: `/docs/infrastructure/`
- Research connections: `/RESEARCH.md`
