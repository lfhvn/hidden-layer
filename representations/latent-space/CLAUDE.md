# Latent Space - Development Guide

## Project Overview

Understanding and experiencing latent space representations through two complementary approaches:

1. **Lens**: SAE interpretability web app (training Sparse Autoencoders, discovering features)
2. **Topologies**: Mobile latent space exploration (visual, audio, haptic experience)

**Research Question**: How can we understand and make interpretable the high-dimensional latent representations that models use?

**Uses**: `harness/` for experiment tracking, `shared/concepts/` for concept vectors

---

## Sub-Projects

### Latent Lens (`lens/`)

**Purpose**: Interactive SAE training and feature discovery

**Stack**: FastAPI (backend) + Next.js (frontend) + PyTorch

**Features**:
- Train Sparse Autoencoders on model activations
- Discover interpretable features
- Browse feature gallery
- Analyze text through activation lens
- Label and annotate features

**Quick Start**:
```bash
cd projects/latent-space/lens
make dev  # Starts Docker services
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

**See**: `lens/README.md` for detailed setup

### Latent Topologies (`topologies/`)

**Purpose**: Mobile app for experiencing latent spaces

**Stack**: React Native + Expo

**Features**:
- Visual constellation navigation
- Audio representation of meaning gradients
- Haptic feedback for boundary transitions
- On-device embedding model
- Annotation and reshaping

**Status**: Concept/early development

**Quick Start**:
```bash
cd projects/latent-space/topologies
npx expo start
```

**See**: `topologies/README.md` and `topologies/PRD.md`

---

## Research Questions

### Shared Questions

1. **What representations** do models learn?
   - What features emerge in different layers?
   - How do representations differ across models?

2. **How can humans understand** high-dimensional spaces?
   - Visual metaphors (constellations)
   - Audio mappings (sound)
   - Haptic feedback (touch)
   - Interactive exploration

3. **Geometry of meaning**:
   - What is the topology of latent space?
   - How do concepts cluster?
   - What are the boundaries between concepts?

### Lens-Specific

- What SAE features are most interpretable?
- How do features compose?
- Can we steer via feature activation?

### Topologies-Specific

- Can people navigate latent space intuitively?
- Does multimodal experience (visual + audio + haptic) aid understanding?
- Can annotation reshape the space?

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

**Frontend** (TypeScript/Next.js):
```typescript
// In lens/frontend/src/app/my-page/page.tsx

export default function MyPage() {
  return <div>My New Feature</div>
}
```

### Adding to Topologies

See `topologies/TECH_PLAN.md` for architecture and implementation guide.

---

## Key Files

### Lens
- `lens/backend/` - FastAPI backend
- `lens/frontend/` - Next.js frontend
- `lens/docker-compose.yml` - Services setup
- `lens/openapi.yaml` - API specification

### Topologies
- `topologies/src/` - React Native code
- `topologies/research/` - Research notes
- `topologies/PRD.md` - Product requirements
- `topologies/TECH_PLAN.md` - Technical architecture

---

## Testing

### Lens
```bash
cd projects/latent-space/lens
make test
```

### Topologies
```bash
cd projects/latent-space/topologies
# Testing approach TBD
```

---

## See Also

- Shared concepts: `/shared/concepts/README.md`
- Research connections: `/RESEARCH.md`
- Infrastructure: `/docs/infrastructure/`
