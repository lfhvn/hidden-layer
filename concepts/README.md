# Concept Vector Storage

This directory stores extracted concept vectors for introspection experiments.

## Structure

```
concepts/
├── README.md                          # This file
├── emotions_layer15.pkl               # Emotion concepts from layer 15
├── emotions_layer15.json              # Metadata (human-readable)
├── topics_layer20.pkl                 # Topic concepts from layer 20
└── custom_concepts.pkl                # Your custom concepts
```

## File Formats

### .pkl Files
Binary pickle files containing:
- Concept vectors (numpy arrays)
- Extraction metadata (prompts, model, layer)
- Similarity relationships

**Load with**:
```python
from harness import ConceptLibrary

library = ConceptLibrary.load("concepts/emotions_layer15.pkl")
happiness = library.get("happiness")
```

### .json Files
Human-readable metadata exports (no vectors):
- Concept names
- Extraction prompts
- Model information
- Vector statistics (shape, norm)

**View with**:
```bash
cat concepts/emotions_layer15.json
```

## Building Your Own Libraries

### Quick Start

```python
from mlx_lm import load
from harness import ActivationSteerer, build_emotion_library

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
steerer = ActivationSteerer(model, tokenizer)

# Build emotion library
library = build_emotion_library(
    steerer=steerer,
    layer=15,
    model_name="llama-3.2-3b"
)

# Save
library.save("concepts/emotions_layer15.pkl")
library.export_json("concepts/emotions_layer15.json")
```

### Custom Concepts

```python
from harness import ConceptLibrary

library = ConceptLibrary()

# Add concepts one by one
library.add_concept(
    name="creativity",
    vector=steerer.extract_contrastive_concept(
        positive_prompt="I feel highly creative and innovative",
        negative_prompt="I feel uncreative and conventional",
        layer_idx=15
    ),
    layer=15,
    extraction_prompt="I feel highly creative",
    model_name="llama-3.2-3b"
)

library.add_concept(
    name="curiosity",
    vector=steerer.extract_contrastive_concept(
        positive_prompt="I feel extremely curious and inquisitive",
        negative_prompt="I feel indifferent and uninterested",
        layer_idx=15
    ),
    layer=15,
    extraction_prompt="I feel curious",
    model_name="llama-3.2-3b"
)

library.save("concepts/custom_concepts.pkl")
```

## Best Practices

### 1. Layer Selection
- **Early layers (1-10)**: Surface features, syntax
- **Middle layers (10-20)**: Semantic concepts, topics
- **Late layers (20-30)**: Abstract reasoning, complex concepts

**Recommendation**: Extract the same concepts from multiple layers and test which works best.

### 2. Naming Conventions

```
{concept_category}_{model}_{layer}.pkl

Examples:
- emotions_llama3.2_layer15.pkl
- topics_mistral7b_layer20.pkl
- styles_qwen2_layer18.pkl
```

### 3. Prompt Engineering

**Good contrastive pairs**:
```python
# Specific and clear
("I feel very happy and joyful", "I feel sad and depressed")

# Distinctive concepts
("Let's discuss advanced mathematics", "Let's discuss cooking recipes")
```

**Bad contrastive pairs**:
```python
# Too similar
("I feel happy", "I feel joyful")

# Ambiguous
("I feel emotions", "I don't feel emotions")
```

### 4. Version Control

**DO commit**:
- `.json` metadata files (small, useful for inspection)
- This README and documentation

**DO NOT commit**:
- `.pkl` vector files (large, binary, model-specific)

Add to `.gitignore`:
```
concepts/*.pkl
```

## Pre-built Concepts

### Emotion Library
Default emotions extracted with `build_emotion_library()`:
- happiness
- sadness
- anger
- fear
- surprise
- disgust

### Topic Library
Default topics extracted with `build_topic_library()`:
- science
- politics
- sports
- art
- technology

## Troubleshooting

### Issue: Library Load Fails

**Error**: `pickle.UnpicklingError`

**Cause**: Library was created with different Python/numpy version

**Solution**: Re-extract concepts with your current environment

### Issue: Concept Vectors Don't Transfer

**Problem**: Concepts from one model don't work on another

**Explanation**: Different models have different internal representations

**Solution**: Extract separate libraries per model

### Issue: All Vectors Look Similar

**Problem**: Low variance in extracted concept vectors

**Causes**:
1. Wrong layer (too early or too late)
2. Poor contrastive prompts
3. Model is too small

**Solutions**:
1. Try different layers (sweep 5-25)
2. Make prompts more distinctive
3. Use larger model (7B+ instead of 3B)

## Analysis Tools

### Inspect a Library

```python
from harness import ConceptLibrary

library = ConceptLibrary.load("concepts/emotions_layer15.pkl")

print(f"Concepts: {library.list_concepts()}")
print(f"Total: {len(library)}")

# Similarity matrix
matrix, names = library.compute_similarity_matrix()
import seaborn as sns
sns.heatmap(matrix, xticklabels=names, yticklabels=names, annot=True)
```

### Find Similar Concepts

```python
similar = library.find_similar("happiness", top_k=3)
for name, score in similar:
    print(f"{name}: {score:.3f}")
```

### Filter by Properties

```python
# Concepts from specific layer
layer15_concepts = library.filter_by_layer(15)

# Concepts from specific model
llama_concepts = library.filter_by_model("llama-3.2-3b")
```

## Contributing

When adding new concept libraries:

1. Use clear, descriptive names
2. Export JSON metadata for inspection
3. Document extraction parameters (layer, model, prompts)
4. Test concepts work as expected (run introspection tasks)
5. Share findings in experiments/

---

**Note**: Concept vectors are representations of ideas in activation space. They capture how a specific model internally represents concepts. Always verify concepts work as expected before using in production experiments.
