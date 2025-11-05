# AI-to-AI Communication - Development Guide

## Project Overview

**Status**: Early Research

**Research Question**: Can LLMs communicate more efficiently through latent representations instead of natural language?

**Motivation**: Based on recent research showing that AI-to-AI communication through internal states can be more efficient and expressive than natural language.

**Uses**: `harness/` for LLM providers, `shared/concepts/` for latent representations

---

## Research Questions

1. **Efficiency**: Is latent communication faster/cheaper than natural language?
   - Token count comparison
   - Latency comparison
   - Information density

2. **Expressiveness**: Can latent communication convey more information?
   - Nuanced concepts hard to express in language
   - Gradients of meaning
   - Simultaneous multi-concept transmission

3. **Protocols**: What communication protocols emerge?
   - Shared vocabulary in latent space
   - Error correction
   - Compression strategies

4. **Generalization**: Does it work across models?
   - Same architecture, different sizes
   - Different architectures
   - Need for translation layers

---

## Potential Approaches

### 1. Latent Vector Messaging

Agents share activation vectors instead of text:
```python
# Sender
sender_activations = extract_activations(sender_model, concept)

# Receiver
receiver_interpretation = apply_activations(receiver_model, sender_activations)
```

### 2. Concept Vector Communication

Use shared concept libraries:
```python
from shared.concepts import ConceptLibrary

library = ConceptLibrary.load("shared/concepts/emotions.pkl")

# Sender: "I want to communicate 'happiness' + 'excitement'"
message = library.get("happiness") + 0.5 * library.get("excitement")

# Receiver: decode the concept blend
interpretation = library.find_similar(message, top_k=3)
```

### 3. Embedding Space Coordinates

Direct coordinate sharing in embedding space:
```python
# Sender provides coordinates
coords = [0.23, -0.45, 0.89, ...]  # High-dimensional point

# Receiver finds nearest concepts
nearest = find_nearest_in_vocab(coords, top_k=5)
```

---

## Integration Points

**With Multi-Agent**:
- Can debate/coordination happen in latent space?
- Does it improve multi-agent performance?

**With Latent Space**:
- Use latent-lens to visualize AI-to-AI messages
- Use latent-topologies to experience communication

**With Introspection**:
- What activations correspond to "sending" vs. "receiving"?
- Can models introspect on their communication?

---

## Implementation Plan (TBD)

### Phase 1: Proof of Concept
- [ ] Simple latent vector passing between models
- [ ] Measure information retention
- [ ] Compare to natural language baseline

### Phase 2: Protocol Development
- [ ] Develop compression strategies
- [ ] Error correction mechanisms
- [ ] Shared vocabulary construction

### Phase 3: Multi-Agent Integration
- [ ] Integrate with multi-agent strategies
- [ ] Measure coordination improvements
- [ ] Analyze emergent protocols

### Phase 4: Cross-Model Communication
- [ ] Test across different architectures
- [ ] Translation layers
- [ ] Standardization

---

## Key Files (To Be Created)

- `code/latent_messaging.py` - Core messaging implementation
- `code/compression.py` - Message compression strategies
- `code/protocols.py` - Communication protocols
- `code/translation.py` - Cross-model translation
- `notebooks/` - Experiments

---

## Related Research

- Recent papers on AI-to-AI communication (add specific citations)
- Activation space geometry
- Multi-agent coordination

---

## Next Steps

1. Literature review on AI-to-AI communication
2. Simple proof-of-concept implementation
3. Benchmark against natural language
4. Define research protocol

---

## See Also

- Research themes: `/RESEARCH.md`
- Shared concepts: `/shared/concepts/README.md`
- Multi-agent coordination: `/projects/multi-agent/`
- Latent space projects: `/projects/latent-space/`
