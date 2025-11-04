# Shared Resources

Resources used across multiple Hidden Layer projects.

## Structure

```
shared/
├── concepts/         # Concept vectors (emotions, topics, custom)
├── datasets/         # Benchmark datasets
└── utils/            # Common Python utilities
```

## Concept Vectors (`concepts/`)

Extracted concept representations used for:
- **Introspection**: Testing model self-knowledge
- **Latent Space**: Understanding representations
- **Steerability**: Steering model behavior

**See**: [concepts/README.md](concepts/README.md) for detailed usage.

**Format**:
- `.pkl` files - Binary concept vectors
- `.json` files - Human-readable metadata

**Usage**:
```python
from harness import ConceptLibrary

library = ConceptLibrary.load("shared/concepts/emotions_layer15.pkl")
happiness = library.get("happiness")
```

## Datasets (`datasets/`)

Benchmark datasets shared across projects:
- ToMBench (theory of mind)
- OpenToM (theory of mind)
- SocialIQA (social reasoning)
- UICrit (design critique)
- Custom datasets

**Usage**:
```python
from harness import load_benchmark

dataset = load_benchmark('tombench', split='test')
```

## Utilities (`utils/`)

Common Python code shared across projects:
- Data processing
- Visualization helpers
- Analysis tools

**Usage**:
```python
from shared.utils import my_utility_function
```

---

## Best Practices

### Concept Vectors

1. **Name clearly**: `{category}_{model}_{layer}.pkl`
2. **Document**: Include JSON metadata
3. **Version**: Note model and extraction parameters

### Datasets

1. **Original source**: Keep reference to original
2. **Preprocessing**: Document any transformations
3. **Splits**: Use standard train/val/test splits

### Code

1. **Test**: Add tests for shared utilities
2. **Document**: Clear docstrings
3. **Generalize**: Useful across multiple projects

---

## See Also

- Concepts guide: [concepts/README.md](concepts/README.md)
- Benchmark usage: [/docs/workflows/benchmarking.md](../docs/workflows/benchmarking.md)
