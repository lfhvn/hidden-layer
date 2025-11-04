# 🚀 Latent Lens Setup Guide

## Complete Project Structure

```
latent-lens/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI application
│   │   ├── config.py                    # Configuration management
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── sae.py                   # Sparse Autoencoder implementation
│   │   │   ├── activation_capture.py    # Layer hook utilities
│   │   │   └── feature_extractor.py     # Feature analysis
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── database.py              # Database connection
│   │   │   └── schemas.py               # SQLModel schemas
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_loader.py        # WikiText & data loading
│   │   │   └── feature_extraction.py    # Extraction pipeline
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── sae_service.py           # SAE training service
│   │   │   └── feature_service.py       # Feature management
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── dependencies.py          # API key verification
│   │       ├── routes/
│   │       │   ├── __init__.py
│   │       │   ├── experiments.py       # Experiment endpoints
│   │       │   ├── features.py          # Feature endpoints
│   │       │   └── activations.py       # Analysis endpoints
│   │       └── websocket.py             # Real-time updates
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py                  # Pytest fixtures
│   │   ├── test_sae.py                  # SAE tests
│   │   ├── test_activation_capture.py   # Activation tests
│   │   ├── test_features.py             # Feature service tests
│   │   └── fixtures/
│   │       ├── __init__.py
│   │       └── sample_data.py           # Test data
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                 # Home page
│   │   │   ├── layout.tsx               # Root layout
│   │   │   ├── globals.css              # Global styles
│   │   │   ├── layer-explorer/
│   │   │   │   └── page.tsx             # Layer selection view
│   │   │   ├── feature-gallery/
│   │   │   │   └── page.tsx             # Feature browsing
│   │   │   ├── activation-lens/
│   │   │   │   └── page.tsx             # Text analysis view
│   │   │   └── labeling/
│   │   │       └── page.tsx             # Feature labeling
│   │   ├── components/
│   │   │   ├── ui/                      # shadcn/ui components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   └── badge.tsx
│   │   │   ├── FeatureCard.tsx          # Feature display
│   │   │   ├── ActivationHeatmap.tsx    # Visualization
│   │   │   ├── LayerSelector.tsx        # Layer picker
│   │   │   ├── LabelEditor.tsx          # Label form
│   │   │   └── Navigation.tsx           # Nav bar
│   │   ├── lib/
│   │   │   ├── api.ts                   # API client
│   │   │   ├── utils.ts                 # Utilities
│   │   │   └── websocket.ts             # WebSocket client
│   │   └── types/
│   │       └── index.ts                 # TypeScript types
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── components.json
│   └── Dockerfile
│
├── docker-compose.yml
├── Makefile
├── README.md
├── SETUP.md                            # This file
├── LICENSE
├── .env.example
├── .gitignore
└── openapi.yaml                        # API specification
```

## Step-by-Step Setup

### Option 1: Docker (Recommended)

1. **Install Docker & Docker Compose**
   - Docker Desktop (Mac/Windows): https://www.docker.com/products/docker-desktop
   - Docker Engine (Linux): https://docs.docker.com/engine/install/

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env and set API_KEY to a secure value
   ```

3. **Start Services**
   ```bash
   make dev
   ```

4. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - OpenAPI Spec: http://localhost:8000/openapi.json

### Option 2: Local Development

#### Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Run backend
uvicorn app.main:app --reload
```

Backend runs on: http://localhost:8000

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend runs on: http://localhost:3000

## Usage Workflow

### 1. Create an Experiment

**Via UI:**
1. Navigate to **Layer Explorer** (http://localhost:3000/layer-explorer)
2. Enter experiment details:
   - Name: `gpt2_layer6_experiment`
   - Model: `gpt2`
   - Layer index: `6`
   - Hidden dim: `4096`
   - Sparsity: `0.01`
3. Click "Create Experiment"

**Via API:**
```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-in-production" \
  -d '{
    "name": "gpt2_layer6_experiment",
    "model_name": "gpt2",
    "layer_name": "transformer.h.6.mlp",
    "layer_index": 6,
    "input_dim": 768,
    "hidden_dim": 4096,
    "sparsity_coef": 0.01
  }'
```

### 2. Train SAE (Python)

```python
from app.models.sae import SparseAutoencoder, SAETrainingConfig
from app.services import SAEService
import torch

# Create SAE
config = SAETrainingConfig(
    input_dim=768,
    hidden_dim=4096,
    sparsity_coef=0.01,
    num_epochs=10,
    batch_size=32,
    device="cpu"
)

sae = SparseAutoencoder(config)

# Generate sample training data (replace with real activations)
activations = torch.randn(1000, 768)

# Train
service = SAEService()
history = service.train(sae, activations, experiment_id=1)

print(f"Final loss: {history['train_loss'][-1]}")
```

### 3. Browse Features

Navigate to **Feature Gallery** (http://localhost:3000/feature-gallery) to:
- View all discovered features
- Filter by experiment ID
- See activation statistics
- View top-activating tokens

### 4. Analyze Text

Go to **Activation Lens** (http://localhost:3000/activation-lens):
1. Enter text: "The quick brown fox jumps over the lazy dog"
2. Select experiment ID
3. Click "Analyze"
4. View heatmap of token-level activations

### 5. Label Features

In **Labeling** view (http://localhost:3000/labeling):
1. Select experiment
2. Click on features to label
3. Add labels with descriptions and tags
4. Export labeled features

## Testing

```bash
# Run all tests
make test

# Backend tests with coverage
cd backend
pytest -v --cov=app --cov-report=html

# Frontend type checking
cd frontend
npm run type-check

# Lint code
make lint

# Format code
make format
```

## Common Commands

```bash
# Start development environment
make dev

# View logs
make logs

# Stop services
make down

# Clean everything
make clean

# Run backend only
make backend

# Run frontend only
make frontend
```

## Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Check what's using port 3000
lsof -i :3000

# Kill processes or change ports in docker-compose.yml
```

### Database Issues

```bash
# Reset database
rm backend/latent_lens.db

# Restart services
make restart
```

### Model Download Issues

```bash
# Set HuggingFace cache directory
export HF_CACHE_DIR=./model_cache

# Or in .env file:
HF_CACHE_DIR=./model_cache
```

### CORS Errors

Ensure `ALLOWED_ORIGINS` in `.env` includes your frontend URL:
```
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

## Production Deployment

### Environment Variables

```env
# Production settings
API_KEY=<generate-strong-random-key>
DATABASE_URL=postgresql://user:pass@postgres-host/latent_lens
DEVICE=cuda  # If GPU available
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://your-domain.com
```

### Build for Production

```bash
# Build images
docker-compose build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

## Next Steps

1. **Train your first SAE**: Follow the usage workflow above
2. **Explore the API**: Visit http://localhost:8000/docs
3. **Customize**: Edit `.env` for your use case
4. **Integrate**: Use the API client in your own tools
5. **Contribute**: See README.md for contribution guidelines

## Resources

- **API Documentation**: http://localhost:8000/docs
- **OpenAPI Spec**: `openapi.yaml`
- **Architecture Docs**: See README.md
- **Example Notebooks**: Coming soon in `examples/`

## Support

- **Issues**: Report bugs on GitHub
- **Questions**: Open a discussion
- **Documentation**: Check README.md and inline docs

---

**Happy Exploring! 🔬**
