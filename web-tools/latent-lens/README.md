# 🔬 Latent Lens - SAE Feature Explorer

Explore sparse autoencoder (SAE) features discovered in language models. Browse interpretable features, see activating examples, and analyze your own text.

## Features

- **Feature Gallery**: Browse discovered SAE features with descriptions
- **Activation Examples**: See real text that activates each feature
- **Text Analyzer**: Paste text and see which features activate
- **Feature Search**: Find features by concept or keyword
- **Model Comparison**: Compare features across different models

## Quick Start

```bash
# Development
cd web-tools/latent-lens
cp .env.example .env
make dev

# Frontend: http://localhost:3002
# Backend:  http://localhost:8002
```

## What are SAEs?

Sparse Autoencoders (SAEs) are neural networks that learn interpretable features from model activations. Each "feature" represents a pattern or concept the model uses.

For example, a feature might activate for:
- Names of cities
- Positive sentiment
- Technical jargon
- First-person pronouns

## Architecture

### Backend (FastAPI)

- **Read-only API**: No training, just serving pre-computed features
- **Feature Library**: Pre-trained SAE features stored as JSON/pickle
- **Text Analysis**: Compute activations for user-provided text
- **Fast**: Features are preloaded in memory

### Frontend (Next.js)

- **Feature Gallery**: Grid view of all features
- **Feature Detail**: Click to see full examples and metadata
- **Text Analyzer**: Interactive text analysis
- **Search & Filter**: Find specific features

## Data

Pre-trained features are stored in `/shared/sae-features/`:
- `features_gpt2.json` - Features from GPT-2
- `features_llama.json` - Features from Llama models
- Each feature has: ID, description, activation examples, statistics

## Usage

### Browse Features

Navigate to the gallery and click on any feature to see:
- Description of what it detects
- Example texts that activate it
- Activation strength statistics
- Related features

### Analyze Text

1. Go to "Analyze" tab
2. Paste your text
3. See which features activate and how strongly
4. Click features to learn more

## Development

### Adding New Features

1. Train SAE using research code in `/projects/latent-space/lens/`
2. Export features to JSON:
   ```python
   features.save_json("web-tools/latent-lens/features.json")
   ```
3. Features will be automatically loaded on backend restart

### Backend API

```bash
GET  /api/features           # List all features
GET  /api/features/{id}      # Get feature detail
POST /api/analyze            # Analyze text
GET  /api/search?q=emotion   # Search features
```

## Deployment

Same as other web tools:

```bash
./deploy.sh
```

Backend: Railway
Frontend: Vercel

## Research Connection

This tool showcases research from `/projects/latent-space/lens/`.

**Research Questions**:
- What interpretable features emerge in language models?
- How do features compose?
- Can we use features to understand model behavior?

## Cost

**Free to run**: All features are pre-computed. Only costs are hosting:
- Railway: $5-10/month (backend)
- Vercel: Free (frontend, static files)
- No API costs (no model inference)

## Future Ideas

- [ ] Feature clustering visualization
- [ ] Compare features across model layers
- [ ] Community feature annotations
- [ ] Export feature analysis as report
- [ ] Feature-based text generation steering

---

Powered by research from [Hidden Layer Lab](https://github.com/lfhvn/hidden-layer)
