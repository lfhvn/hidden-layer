# AgentMesh - Product Platform

**Multi-agent workflow orchestration built on Hidden Layer research**

---

## Overview

AgentMesh is a commercial product that wraps Hidden Layer's research-backed multi-agent strategies in a production-ready platform with:

- Visual workflow design
- Persistent state management
- Multi-tenant architecture
- Rich observability
- Human-in-the-loop capabilities

**Built on**: Hidden Layer Harness (research infrastructure)

---

## Architecture

```
AgentMesh Product (this repo)
    ↓ imports
Hidden Layer Harness (/harness, /communication/multi-agent)
    ↓ calls
LLM Providers (OpenAI, Anthropic, Ollama, MLX)
```

**Key principle**: AgentMesh doesn't reimplement strategies. It imports and orchestrates them.

---

## Directory Structure

```
agentmesh/
├── api/                    # FastAPI REST server
│   ├── routes/            # API endpoints
│   ├── schemas/           # Pydantic models (API layer)
│   └── main.py            # FastAPI app
│
├── core/                   # Core orchestration logic
│   ├── models/            # Domain models (Workflow, Run, Step)
│   ├── nodes/             # Workflow node types (wrap harness)
│   ├── orchestrator/      # Workflow execution engine
│   └── __init__.py
│
├── db/                     # Database layer
│   ├── migrations/        # Alembic migrations
│   ├── session.py         # DB session management
│   └── models.py          # SQLAlchemy models
│
├── web/                    # Next.js frontend (separate)
│   └── (to be created)
│
├── docker-compose.yml     # Local development stack
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for Postgres, Redis)
- Hidden Layer harness installed (from `/harness`)

### Installation

```bash
# 1. Install harness (if not already)
cd ../harness
pip install -e .

# 2. Install AgentMesh
cd ../agentmesh
pip install -r requirements.txt

# 3. Start infrastructure (Postgres, Redis)
docker-compose up -d

# 4. Run migrations
alembic upgrade head

# 5. Start API server
python -m agentmesh.api.main
```

API available at `http://localhost:8000`

---

## Usage

### Create a Workflow

```bash
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Debate Analysis",
    "graph": {
      "nodes": [
        {
          "id": "debate1",
          "type": "strategy",
          "strategy_id": "debate",
          "config": {"n_debaters": 3}
        }
      ],
      "edges": []
    }
  }'
```

### Execute a Run

```bash
curl -X POST http://localhost:8000/api/workflows/{workflow_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "Should we invest in renewable energy?"
    },
    "context": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022"
    }
  }'
```

### Get Run Status

```bash
curl http://localhost:8000/api/runs/{run_id}
```

---

## Development

### Running Tests

```bash
pytest agentmesh/tests/
```

### Adding a New Strategy Node

When Hidden Layer research adds a new strategy:

```python
# /communication/multi-agent/strategies.py (research)
def new_strategy(task_input, **kwargs):
    # Research implementation
    pass

STRATEGIES['new_strategy'] = new_strategy
```

Wrap it in AgentMesh (1-2 hours):

```python
# agentmesh/core/nodes/new_strategy_node.py
from agentmesh.core.nodes.base import StrategyNode

class NewStrategyNode(StrategyNode):
    """Wraps Hidden Layer new_strategy"""
    strategy_id = "new_strategy"
```

Register it:

```python
# agentmesh/core/nodes/__init__.py
from .new_strategy_node import NewStrategyNode

NODE_REGISTRY['new_strategy'] = NewStrategyNode
```

Done! Available in workflows immediately.

---

## API Reference

### Workflows

- `POST /api/workflows` - Create workflow
- `GET /api/workflows` - List workflows
- `GET /api/workflows/:id` - Get workflow
- `PUT /api/workflows/:id` - Update workflow
- `DELETE /api/workflows/:id` - Delete workflow

### Runs

- `POST /api/workflows/:id/runs` - Execute workflow
- `GET /api/runs` - List runs
- `GET /api/runs/:id` - Get run details
- `POST /api/runs/:id/cancel` - Cancel run

### Steps

- `GET /api/runs/:id/steps` - List steps for run
- `GET /api/steps/:id` - Get step details

---

## Tech Stack

**Backend**:
- FastAPI (Python) - API server
- SQLAlchemy - ORM
- Alembic - Migrations
- Celery - Async task execution
- Redis - Queue & caching
- Postgres - Primary database

**Frontend** (coming):
- Next.js (React + TypeScript)
- TailwindCSS
- React Flow - Graph editor
- Recharts - Analytics

**Infrastructure**:
- Docker Compose (local dev)
- Kubernetes (production - future)

---

## Relationship to Hidden Layer

**Hidden Layer** = Research lab
- Multi-agent strategies (debate, CRIT, consensus, etc.)
- LLM provider abstraction (harness)
- Theory of mind, introspection, alignment research

**AgentMesh** = Product spinoff
- Wraps HL strategies in production platform
- Adds UI, persistence, multi-tenancy
- Commercial SaaS offering

**Data flow**:
1. HL research develops strategy
2. AgentMesh wraps as node type
3. Customers use via product
4. Feedback informs HL research

---

## MVP Scope (v0.1)

- [x] Core models (Workflow, Run, Step)
- [x] Orchestrator (executes workflows using harness)
- [x] Strategy nodes (debate, CRIT, consensus)
- [ ] FastAPI REST endpoints
- [ ] Database schema & migrations
- [ ] Basic web UI (run list, detail view)
- [ ] Authentication (JWT)
- [ ] Deploy to staging

**Goal**: Validate product-market fit with 10 design partners

---

## Roadmap

### v0.1 (MVP - 3 months)
- Core workflows + strategy execution
- Simple form-based UI
- Single-tenant only
- Manual deployment

### v0.5 (Growth - 6 months)
- Visual graph editor
- Multi-user support
- Human-in-the-loop gates
- Webhook integrations

### v1.0 (Scale - 12 months)
- Advanced observability
- Template marketplace
- Enterprise features (SSO, audit logs)
- On-prem deployment option

---

## Contributing

**For Hidden Layer researchers**:
- Add strategies in `/communication/multi-agent/`
- AgentMesh team will wrap them

**For AgentMesh product team**:
- Build on harness, don't reimplement
- Maintain backward compatibility
- Sync with research team weekly

---

## License

**Proprietary** - AgentMesh product code

Uses **Hidden Layer Harness** (Apache 2.0)

---

## Links

- Product strategy: `/docs/AGENTMESH_PRODUCT_STRATEGY.md`
- Architecture: `/docs/AGENTMESH_ARCHITECTURE.md`
- Hidden Layer research: `/README.md`
