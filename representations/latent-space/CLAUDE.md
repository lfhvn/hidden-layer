# Latent Space - Development Guide

## Project Overview

Understanding latent space representations through SAE interpretability and generative modeling:

1. **Lens**: SAE interpretability API (training Sparse Autoencoders, discovering features)
2. **CALM**: Continuous autoregressive language modeling experiments (autoencoder + energy transformer)

**Research Question**: How can we understand and make interpretable the high-dimensional latent representations that models use?

**Uses**: `harness/` for experiment tracking, `shared/concepts/` for concept vectors

---

## Sub-Projects

### Latent Lens (`lens/`)

**Purpose**: Interactive SAE training and feature discovery

**Stack**: FastAPI + PyTorch + HuggingFace transformers

**Features**:
- Train Sparse Autoencoders on model activations
- Capture activations via forward hooks on any HF model
- Discover interpretable features
- Feature extraction pipeline with labeling support

**Quick Start**:
```bash
cd representations/latent-space/lens
make dev  # Starts Docker services
# Backend API: http://localhost:8000
```

**See**: `lens/README.md` for detailed setup

### CALM (`calm/`)

**Purpose**: Continuous autoregressive language modeling experiments

**Stack**: PyTorch (variational autoencoder + energy transformer)

**See**: `calm/README.md` and `calm/CLAUDE.md`

---

## Research Questions

1. **What representations** do models learn?
   - What features emerge in different layers?
   - How do representations differ across models?

2. **Geometry of meaning**:
   - What is the topology of latent space?
   - How do concepts cluster?
   - What are the boundaries between concepts?

3. **Lens-specific**:
   - What SAE features are most interpretable?
   - How do features compose?
   - Can we steer via feature activation?

---

## Integration Points

**With Introspection**:
- What features activate during introspection tasks?
- Can we use SAE features to understand concept vectors?

**With SELPHI**:
- What features activate during ToM reasoning?
- How does latent geometry relate to perspective-taking?

**With AI-to-AI Communication**:
- Can agents communicate via latent space coordinates?
- Is the geometry shared across models?

**With Multi-Agent**:
- Do multi-agent systems develop shared representations?
- Can we visualize agent coordination in latent space?

---

## Development Workflows

### Adding Features to Lens

**Backend** (Python/FastAPI):
```python
# In lens/backend/app/api/routes/

@router.post("/my-endpoint")
async def my_endpoint(data: MyModel):
    # Implementation
    return result
```

---

## Key Files

### Lens
- `lens/backend/` - FastAPI backend (SAE models, activation capture, pipelines)
- `lens/docker-compose.yml` - Services setup
- `lens/openapi.yaml` - API specification

### CALM
- `calm/src/autoencoder.py` - Variational autoencoder
- `calm/src/energy_transformer.py` - Energy transformer

---

## Testing

### Lens
```bash
cd representations/latent-space/lens
make test
```

---

## See Also

- Shared concepts: `/shared/concepts/README.md`
- Research connections: `/RESEARCH.md`
- Infrastructure: `/docs/infrastructure/`
