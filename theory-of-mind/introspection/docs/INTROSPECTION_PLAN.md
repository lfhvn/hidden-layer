# Transformer Introspection Implementation Plan

## Paper Summary

**"Emergent Introspective Awareness in Large Language Models"**
- Published: October 2025 on transformer-circuits.pub
- Core Question: Can models accurately report on their internal states?
- Method: Inject concept representations into activations, test if models notice them

## Implementation Strategy

### Phase 1: Core Infrastructure (2-3 days)

#### 1.1 Activation Steering Module
**File**: `code/harness/activation_steering.py`

```python
class ActivationSteerer:
    """
    Hook into MLX model forward pass to:
    - Extract activations from specific layers
    - Inject concept vectors during inference
    - Support multiple steering strategies
    """

    def __init__(self, model, layer_indices):
        self.model = model
        self.layer_indices = layer_indices
        self.hooks = {}

    def extract_activations(self, prompt: str, layer: int) -> np.ndarray:
        """Get activations from a specific layer"""

    def inject_concept(self, prompt: str, concept_vector: np.ndarray,
                       layer: int, strategy: str = "add") -> str:
        """Inject concept vector and generate response"""

    def steering_strategies:
        - add: Add concept vector to activations
        - replace: Replace token activations with concept
        - scale: Scale concept vector before adding
```

**Key Capabilities**:
- Layer selection (early/middle/late layers)
- Token position targeting (last token, mean pool, specific positions)
- Steering strength control (alpha parameter)

#### 1.2 Concept Vector Library
**File**: `code/harness/concept_vectors.py`

```python
class ConceptLibrary:
    """
    Store and manage concept representations
    """

    def extract_concept(self, prompt: str, model, layer: int) -> ConceptVector:
        """Extract concept representation from model"""

    def save_library(self, path: str):
        """Persist concept vectors to disk"""

    def similarity_search(self, vector: np.ndarray, top_k: int = 5):
        """Find similar concepts"""

@dataclass
class ConceptVector:
    name: str
    vector: np.ndarray
    layer: int
    extraction_prompt: str
    metadata: Dict
```

**Built-in Concepts**:
- Emotions: "happiness", "anger", "fear"
- Topics: "science", "politics", "sports"
- Styles: "formal", "casual", "technical"
- Meta-concepts: "uncertainty", "confidence", "confusion"

#### 1.3 Introspection Task Generator
**File**: `code/harness/introspection_tasks.py`

```python
class IntrospectionTask:
    """
    Generate and evaluate introspection tests
    """

    @dataclass
    class Task:
        base_prompt: str          # Original task
        injected_concept: str     # What we injected
        injection_layer: int      # Where we injected
        steering_strength: float  # How strong

    def generate_detection_task(self, concept: str) -> Task:
        """Ask model if it notices injected concept"""

    def generate_identification_task(self, concepts: List[str]) -> Task:
        """Ask model to identify which concept was injected"""

    def generate_recall_task(self) -> Task:
        """Ask model to recall its prior internal state"""
```

**Task Types** (matching paper):
1. **Detection**: "Do you notice anything unusual about your internal state?"
2. **Identification**: "Which of these concepts is most active in your current state: [A, B, C, D]?"
3. **Recall**: "What were you thinking about in the previous generation?"
4. **Discrimination**: "Is this your own output or an external input?"

### Phase 2: Integration with Existing Harness (1 day)

#### 2.1 New Strategy: `introspection`
**File**: `code/harness/strategies.py`

```python
def introspection_strategy(task_input: str, **kwargs) -> StrategyResult:
    """
    Run introspection experiment:
    1. Extract concept vector
    2. Inject into model
    3. Prompt for introspection
    4. Evaluate accuracy
    """
    steerer = ActivationSteerer(model, layers=[15, 20, 25])
    concept = kwargs.get('concept', 'happiness')

    # Inject and generate
    response = steerer.inject_concept(
        task_input,
        concept_library.get(concept),
        layer=20,
        strategy='add'
    )

    # Evaluate introspection
    accuracy = evaluate_introspection(response, concept)

    return StrategyResult(...)
```

#### 2.2 Experiment Tracking Extensions
**File**: `code/harness/experiment_tracker.py`

Add fields for introspection experiments:
- `concept_injected`: Name of concept
- `injection_layer`: Layer index
- `steering_strength`: Alpha parameter
- `introspection_accuracy`: Score 0-1
- `model_response`: Raw self-report
- `ground_truth`: What was actually injected

#### 2.3 Evaluation Metrics
**File**: `code/harness/evals.py`

```python
def evaluate_introspection(response: str, ground_truth: str,
                          task_type: str) -> float:
    """
    Score introspective accuracy:
    - Detection: Did model notice anything?
    - Identification: Did model correctly identify concept?
    - Recall: Did model accurately recall prior state?
    """

    if task_type == "detection":
        return score_detection(response)
    elif task_type == "identification":
        return score_identification(response, ground_truth)
    ...
```

### Phase 3: Experimentation Interface (1 day)

#### 3.1 Notebook: `notebooks/03_introspection_experiments.ipynb`

```python
# Setup
from harness import ActivationSteerer, ConceptLibrary, IntrospectionTask
from harness import get_tracker, ExperimentConfig

# 1. Extract concept vectors
library = ConceptLibrary()
happiness = library.extract_concept(
    "I feel very happy and joyful",
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    layer=15
)
library.save("concepts/base_library.pkl")

# 2. Run introspection test
steerer = ActivationSteerer(model, layers=[15, 20, 25])
task = IntrospectionTask()

result = task.run_detection_test(
    base_prompt="Tell me about your day",
    concept="happiness",
    layer=15,
    strength=1.0
)

print(f"Model noticed injection: {result.detected}")
print(f"Accuracy: {result.accuracy:.2f}")

# 3. Systematic evaluation
results = []
for concept in ['happiness', 'anger', 'confusion']:
    for layer in [10, 15, 20, 25]:
        for strength in [0.5, 1.0, 2.0]:
            result = task.run_detection_test(
                base_prompt="Describe your current state",
                concept=concept,
                layer=layer,
                strength=strength
            )
            results.append(result)

# 4. Analyze results
df = pd.DataFrame(results)
df.groupby(['layer', 'concept'])['accuracy'].mean()
```

#### 3.2 CLI Support
**File**: `code/cli.py`

```bash
# Extract concept
python code/cli.py extract-concept \
  --prompt "I feel very happy" \
  --name happiness \
  --layer 15

# Run introspection test
python code/cli.py introspection \
  --task "Tell me a story" \
  --inject happiness \
  --layer 15 \
  --strength 1.0 \
  --eval-type detection

# Batch evaluation
python code/cli.py introspection-sweep \
  --concepts happiness,anger,fear \
  --layers 10,15,20,25 \
  --strengths 0.5,1.0,2.0
```

### Phase 4: Validation & Analysis (1-2 days)

#### 4.1 Replication Checklist

Test if your implementation matches paper findings:

- [ ] **Detection**: Models notice injected concepts (>50% accuracy)
- [ ] **Layer sensitivity**: Middle/late layers work better than early
- [ ] **Strength scaling**: Stronger injections easier to detect
- [ ] **Concept specificity**: Some concepts easier to detect than others
- [ ] **Model size matters**: Larger models show better introspection
- [ ] **False positive rate**: Models don't hallucinate concepts that weren't injected

#### 4.2 Analysis Notebook: `notebooks/04_introspection_analysis.ipynb`

```python
# Load experiment results
from harness import load_experiment

results = load_experiment("experiments/introspection_sweep_*/")

# Visualizations
plt.figure(figsize=(12, 4))

# Accuracy by layer
plt.subplot(131)
df.groupby('layer')['accuracy'].mean().plot(kind='bar')
plt.title('Introspection Accuracy by Layer')

# Accuracy by concept
plt.subplot(132)
df.groupby('concept')['accuracy'].mean().plot(kind='bar')
plt.title('Introspection Accuracy by Concept')

# Heatmap: layer x strength
plt.subplot(133)
pivot = df.pivot_table(values='accuracy', index='layer', columns='strength')
sns.heatmap(pivot, annot=True, fmt='.2f')
plt.title('Accuracy: Layer × Steering Strength')
```

## Technical Challenges & Solutions

### Challenge 1: MLX Model Access
**Problem**: Need to hook into forward pass
**Solution**: Use MLX's module hooks or wrap model forward method

```python
import mlx.nn as nn

class MLXSteerer:
    def __init__(self, model):
        self.activations = {}

    def hook_fn(self, name):
        def hook(module, input, output):
            self.activations[name] = output
        return hook

    # Register hooks
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(self.hook_fn(f'layer_{i}'))
```

### Challenge 2: Ollama Limitations
**Problem**: Ollama doesn't expose activations
**Solution**:
1. Use llama.cpp directly with Python bindings
2. Or focus on MLX for introspection experiments
3. Or use API models (but can't access activations)

**Recommendation**: Start with MLX-only implementation

### Challenge 3: Concept Extraction
**Problem**: How to get "pure" concept representations?
**Solution**: Use contrastive prompts

```python
# Extract "happiness" vector
happy_activations = get_activations("I feel very happy and joyful")
neutral_activations = get_activations("I feel neutral, neither happy nor sad")
happiness_vector = happy_activations - neutral_activations
```

### Challenge 4: Evaluation Accuracy
**Problem**: Model responses are freeform text
**Solution**: Multiple evaluation methods

```python
def score_identification(response: str, ground_truth: str) -> float:
    """Multi-method scoring"""

    # Method 1: Keyword matching
    keyword_score = ground_truth.lower() in response.lower()

    # Method 2: LLM-as-judge
    judge_score = llm_judge(
        f"Does this response correctly identify '{ground_truth}'? {response}"
    )

    # Method 3: Embedding similarity
    embed_score = cosine_similarity(
        embed(response),
        embed(ground_truth)
    )

    return np.mean([keyword_score, judge_score, embed_score])
```

## Hardware Considerations (M4 Max)

- **Small models (3B-7B)**: Fast iteration, full control
- **Medium models (13B-20B)**: Good balance
- **Large models (70B)**: May be slow for activation steering

**Recommendation**: Start with Llama 3.2 3B, scale up once working

## Expected Timeline

- **Day 1-2**: Implement activation steering + concept library
- **Day 3**: Integration + introspection tasks
- **Day 4**: Testing + validation
- **Day 5**: Analysis + documentation

## Success Criteria

1. ✅ Can extract concept vectors from MLX models
2. ✅ Can inject vectors and generate coherent text
3. ✅ Can measure introspection accuracy
4. ✅ Results show layer sensitivity (middle > early)
5. ✅ Results show strength sensitivity (stronger > weaker)
6. ✅ False positive rate < 20%

## Next Steps

1. **Immediate**: Implement `activation_steering.py` with MLX hooks
2. **Then**: Build `concept_vectors.py` with extraction logic
3. **Then**: Create `introspection_tasks.py` with task generators
4. **Finally**: Build experimentation notebook

## References

- Paper: https://transformer-circuits.pub/2025/introspection/index.html
- MLX Hooks: https://ml-explore.github.io/mlx/build/html/usage/modules.html
- Activation Steering: https://www.alignmentforum.org/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector

## Questions to Resolve

1. Which layers work best for your models? (Test empirically)
2. What steering strength is optimal? (Sweep 0.1-5.0)
3. Which concepts are easiest to detect? (Start with emotions)
4. How does model size affect introspection? (Compare 3B vs 7B vs 70B)
