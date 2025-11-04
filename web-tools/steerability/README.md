# 🎛️ Steerability Dashboard

**Live LLM Steering with Adherence Metrics**

Real-time control panel for steering language model behavior using activation vectors, with continuous monitoring of adherence to constraints.

## 🌟 Features

- **Live Steering Controls**: Apply steering vectors in real-time
- **Adherence Metrics**: Monitor how well outputs follow constraints
- **Vector Library**: Manage and create steering vectors
- **A/B Comparison**: Side-by-side steered vs unsteered outputs
- **Constraint Enforcement**: Define and track behavioral constraints
- **Real-time Monitoring**: WebSocket updates for metrics
- **Experiment Tracking**: Track multiple steering experiments

## 🏗️ Architecture

### Backend (FastAPI)
- **Steering Engine**: Vector injection, activation modification
- **Metrics System**: Adherence tracking, time-series monitoring
- **Constraints**: Keyword, pattern, sentiment enforcement
- **Vector Library**: Storage and management

### Frontend (Next.js)
- **Control Panel**: Live steering interface
- **Metrics View**: Real-time adherence charts
- **Experiments Dashboard**: Track multiple experiments
- **Vector Manager**: Create and manage vectors

## 🚀 Quick Start

```bash
# Clone and setup
cd steerability-dashboard
cp .env.example .env

# Start services
make dev

# Access
# Frontend: http://localhost:3001
# Backend:  http://localhost:8001
# API Docs: http://localhost:8001/docs
```

## 📋 Usage

### 1. Steer Text Generation

```bash
curl -X POST http://localhost:8001/api/steering/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Write about the weather",
    "vector_name": "positive_sentiment",
    "strength": 1.5
  }'
```

### 2. Monitor Adherence

Navigate to the dashboard to see:
- Real-time adherence scores
- Constraint satisfaction rates
- Comparative analysis

### 3. Create Steering Vectors

```python
from app.steering.vector_library import create_steering_vector

vector = create_steering_vector(
    name="formal_tone",
    positive_examples=["Good day, sir...", "I would like to..."],
    negative_examples=["Hey!", "Sup..."],
    model=model,
    tokenizer=tokenizer,
    layer_index=6
)
```

## 🧪 Testing

```bash
make test
```

## 📦 Key Components

- `steering/engine.py` - Core steering logic
- `steering/vector_library.py` - Vector management
- `steering/constraints.py` - Constraint enforcement
- `metrics/adherence.py` - Adherence tracking
- `metrics/monitoring.py` - Real-time monitoring

## 🔧 Configuration

Edit `.env`:

```env
API_KEY=your-secret-key
DEFAULT_MODEL_NAME=gpt2
MAX_STEERING_STRENGTH=5.0
ADHERENCE_WINDOW_SIZE=100
```

## 📊 Metrics

- **Adherence Score**: 0-1 score of constraint satisfaction
- **Success Rate**: Fraction of generations meeting threshold
- **Constraint Violations**: Per-constraint tracking
- **Time Series**: Historical adherence trends

## 📝 License

MIT License

## 🙏 Acknowledgments

- Steering vectors research
- Activation engineering techniques
- FastAPI and Next.js frameworks
