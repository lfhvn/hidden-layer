# 🔬 Latent Lens

**Interactive LLM Interpretability with Sparse Autoencoders**

Latent Lens is a production-grade web application for training Sparse Autoencoders (SAEs) on language model activations, discovering interpretable features, and analyzing model behavior through an intuitive interface.

## 🌟 Features

- **Layer Explorer**: Select model layers and configure SAE training
- **Feature Gallery**: Browse and filter discovered features
- **Activation Lens**: Analyze which features activate for specific text inputs
- **Feature Labeling**: Annotate features with human-readable labels
- **Real-time Training**: WebSocket updates during SAE training
- **Comprehensive API**: RESTful API with OpenAPI 3.1 specification
- **Production Ready**: Docker support, tests, logging, and error handling

## 🏗️ Architecture

### Backend (Python + FastAPI)
- **Models**: SAE implementation with PyTorch, activation capture hooks
- **Pipelines**: Dataset loading (WikiText), feature extraction
- **Storage**: SQLite/PostgreSQL with SQLModel ORM
- **API**: REST endpoints + WebSocket for real-time updates

### Frontend (Next.js + TypeScript)
- **Views**: Layer Explorer, Feature Gallery, Activation Lens, Labeling
- **UI Components**: shadcn/ui components with Tailwind CSS
- **Charts**: Recharts for visualizations
- **State Management**: React hooks with API client

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- (Optional) Python 3.11+ and Node.js 18+ for local development

### Using Docker (Recommended)

1. **Clone the repository**
```bash
git clone <repository-url>
cd latent-lens
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your settings (API_KEY, etc.)
```

3. **Start the application**
```bash
make dev
```

This will:
- Build and start both backend and frontend containers
- Backend available at http://localhost:8000
- Frontend available at http://localhost:3000
- API docs at http://localhost:8000/docs

### Local Development

#### Backend

```bash
cd backend
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## 📋 Usage

### 1. Create an Experiment

Navigate to **Layer Explorer** and:
- Select a model (e.g., `gpt2`)
- Choose a layer index (e.g., `6`)
- Configure SAE parameters:
  - Hidden dimension (4096 default)
  - Sparsity coefficient (0.01 default)
- Click "Create Experiment"

### 2. Browse Features

Go to **Feature Gallery** to:
- View all discovered features
- Filter by experiment, sparsity range
- See top-activating tokens for each feature

### 3. Analyze Text

Use **Activation Lens** to:
- Input text to analyze
- Select an experiment
- View token-level feature activations
- See heatmap visualization

### 4. Label Features

In the **Labeling** view:
- Select features to annotate
- Add human-readable labels
- Tag features for categorization
- Export labeled features

## 🧪 Testing

```bash
# Run all tests
make test

# Backend tests only
cd backend && pytest -v

# Frontend type checking
cd frontend && npm run type-check
```

## 📚 API Documentation

Once running, access:
- Interactive API docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

### Key Endpoints

```
POST /api/experiments          - Create experiment
GET  /api/experiments          - List experiments
GET  /api/experiments/{id}     - Get experiment details

GET  /api/features             - List features (with filters)
POST /api/features/{id}/labels - Add label to feature

POST /api/activations/analyze  - Analyze text

WS   /ws/experiments/{id}      - Real-time training updates
```

## 🔧 Configuration

Edit `.env` file:

```env
# Security
API_KEY=your-secret-key

# Database
DATABASE_URL=sqlite:///./latent_lens.db
# Or PostgreSQL:
# DATABASE_URL=postgresql://user:pass@localhost/latent_lens

# Model
DEFAULT_MODEL_NAME=gpt2
DEVICE=cpu  # or cuda
HF_CACHE_DIR=./model_cache

# SAE
MAX_FEATURES=1024
SAE_HIDDEN_DIM=4096
SPARSITY_COEFFICIENT=0.01
```

## 🛠️ Development

### Project Structure

```
latent-lens/
├── backend/
│   ├── app/
│   │   ├── models/          # SAE, activation capture
│   │   ├── storage/         # Database models
│   │   ├── pipelines/       # Data loading
│   │   ├── services/        # Business logic
│   │   └── api/             # FastAPI routes
│   └── tests/               # Pytest tests
├── frontend/
│   └── src/
│       ├── app/             # Next.js pages
│       ├── components/      # React components
│       ├── lib/             # API client, utils
│       └── types/           # TypeScript types
├── docker-compose.yml
├── Makefile
└── README.md
```

### Adding New Features

**Backend:**
```python
# Add a new API endpoint
@router.post("/my-endpoint")
async def my_endpoint(data: MyModel, api_key: str = Depends(verify_api_key)):
    # Implementation
    return result
```

**Frontend:**
```typescript
// Add a new page
// frontend/src/app/my-page/page.tsx
export default function MyPage() {
  return <div>My New Page</div>
}
```

## 📦 Deployment

### Production Build

```bash
# Build optimized images
docker-compose -f docker-compose.prod.yml build

# Run in production
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables for Production

```env
API_KEY=<strong-random-key>
DATABASE_URL=postgresql://user:pass@db-host/latent_lens
DEVICE=cuda
LOG_LEVEL=WARNING
ALLOWED_ORIGINS=https://your-domain.com
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Sparse Autoencoder research by Anthropic
- HuggingFace Transformers library
- FastAPI and Next.js frameworks
- shadcn/ui component library

## 📞 Support

- Issues: GitHub Issues
- Documentation: `/docs` endpoint
- Examples: See `examples/` directory

## 🗺️ Roadmap

- [ ] Fine-tuning SAEs on custom datasets
- [ ] Multi-model comparison view
- [ ] Feature steering/intervention
- [ ] Advanced visualizations (t-SNE, UMAP)
- [ ] Export to various formats (JSON, CSV, HDF5)
- [ ] Integration with additional interpretability tools

---

**Built with ❤️ for interpretability research**
