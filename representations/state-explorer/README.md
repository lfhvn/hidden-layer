# LLM State Explorer

**Real-time visualization of LLM internal states**

A highly visual, shareable web app for exploring what's happening inside language models as they process text.

## What is This?

Point the tool at a model, type some text, and watch internal activations light up across layers in real-time.

**Core Features:**
- 🔍 Real-time activation visualization
- 🎨 Interactive heatmaps showing layer activations
- 🤖 Multi-model support (local and API)
- 📊 Concept mapping (activations → human concepts)
- 🎬 Token-by-token playback
- 🔗 Shareable visualization links

## Quick Start

**Prerequisites:**
- Python 3.10+
- Node.js 18+
- (Optional) GPU for faster inference

**1. Setup Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**2. Setup Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**3. Open Browser:**
```
http://localhost:3000
```

Type some text and watch activations appear!

## Current Status

**MVP Phase** - Basic functionality working:
- ✅ GPT-2 activation capture
- ✅ Real-time heatmap visualization
- ✅ Token-level view
- ✅ All-layer simultaneous display

**In Progress:**
- 🔨 Multi-model support
- 🔨 Layer detail view
- 🔨 Concept mapping
- 🔨 Token playback

**Planned:**
- 📋 Shareable links
- 📋 SAE feature integration
- 📋 Export capabilities

## Documentation

- **[Full Plan](../LLM_STATE_VISUALIZATION_PLAN.md)** - Comprehensive vision and roadmap
- **[Quick Start Guide](../STATE_EXPLORER_QUICKSTART.md)** - Build MVP in 60 minutes
- **[API Reference](./docs/api.md)** - Backend API documentation (coming soon)

## Architecture

```
Frontend (Next.js)  ←→  Backend (FastAPI)  ←→  Models (Local/API)
    Port 3000              Port 8000              GPT-2, Ollama, etc.
```

**Backend:**
- FastAPI for REST API
- PyTorch + Transformers for model inference
- Forward hooks for activation capture
- WebSocket for real-time updates

**Frontend:**
- Next.js 14 with App Router
- TypeScript for type safety
- Canvas API for performant heatmaps
- Zustand for state management

## Key Concepts

**Activation**: The internal state (vector) at a specific layer for a specific token

**Heatmap**: Visualization of activation intensity across (tokens × layers)

**Aggregation**: Converting high-dimensional activation vectors to single intensity values
- L2 Norm: √(Σ x²) - magnitude of vector
- Max: Maximum value in vector
- Mean: Average value in vector

**Layer**: A transformer block in the model (e.g., `transformer.h.6`)

**Concept Vector**: Pre-computed direction in activation space corresponding to human concept (e.g., "joy", "animals")

## Relationship to Other Projects

**State Explorer vs. Lens:**
- **Lens** = SAE research tool (train autoencoders, discover features)
- **State Explorer** = Interactive visualization tool (explore any model)

**Shared Infrastructure:**
- Both use activation capture from Lens
- Can load SAE checkpoints from Lens
- Complementary workflows

**Integration:**
```
Train SAE in Lens → Load features in State Explorer → Visualize alongside raw activations
```

## Examples

**Simple Input:**
```
Input: "The cat sat on the mat"
Tokens: ["The", "Ġcat", "Ġsat", "Ġon", "Ġthe", "Ġmat"]
Layers: 12 (GPT-2)
Result: 6×12 heatmap
```

**Complex Input:**
```
Input: "In the depths of the ocean, a mysterious creature lurked"
Tokens: 14
Layers: 12
Observation: Early layers show syntax, late layers show semantics
```

## Development

**Backend Development:**
```bash
cd backend
pip install -e .
pytest  # Run tests (coming soon)
```

**Frontend Development:**
```bash
cd frontend
npm run dev  # Dev server with hot reload
npm run build  # Production build
npm run lint  # ESLint
```

**Docker:**
```bash
docker-compose up  # Start both services
```

## Contributing

This is part of the Hidden Layer Lab research initiative.

**Key Areas for Contribution:**
1. Additional model providers (Ollama, MLX, etc.)
2. New visualization modes (attention, gradients, etc.)
3. Concept vector libraries
4. Performance optimizations
5. Documentation and examples

## License

MIT License - See LICENSE file

## Contact

Part of Hidden Layer Lab - https://github.com/lfhvn/hidden-layer

Questions? Open an issue or see main repo documentation.

---

**Status**: Early MVP - Expect breaking changes

**Last Updated**: 2025-11-09
