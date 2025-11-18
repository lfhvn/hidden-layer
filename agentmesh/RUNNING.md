# Running AgentMesh - Complete Guide

Get the full AgentMesh stack running (backend + web UI).

---

## Quick Start (5 minutes)

### Terminal 1: Start Infrastructure

```bash
cd agentmesh

# Start Postgres & Redis
docker-compose up -d

# Verify running
docker-compose ps
```

### Terminal 2: Start Backend API

```bash
cd agentmesh

# Install harness first (if not already)
cd ../harness && pip install -e . && cd ../agentmesh

# Install AgentMesh dependencies
pip install -r requirements.txt

# Set API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Start FastAPI server
python -m agentmesh.api.main
```

Server starts at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

### Terminal 3: Start Web UI

```bash
cd agentmesh/web

# Install dependencies (first time only)
npm install

# Start Next.js dev server
npm run dev
```

Web UI starts at **http://localhost:3000**

---

## What You Can Do Now

### 1. Create a Workflow (Web UI)

1. Open http://localhost:3000
2. Click "Create Workflow"
3. Enter name: "Renewable Energy Analysis"
4. Click a strategy to add (e.g., "Debate")
5. Connect nodes: start → debate1 → end
6. Click "Save Workflow"

### 2. Execute via API (Example Script)

```bash
cd agentmesh
python example_workflow.py
```

This creates and executes a debate workflow.

### 3. View Results (Web UI)

1. Click on a run ID
2. See execution timeline
3. View metrics (cost, tokens, latency)
4. Expand step outputs

---

## Available Strategies

All from Hidden Layer research:

- **debate** - Multi-agent debate with judge
- **crit** - Design critique from multiple perspectives
- **consensus** - Agents find agreement
- **manager_worker** - Task decomposition
- **self_consistency** - Multiple samples, majority vote
- **single** - Baseline single agent

---

## Architecture

```
┌──────────────────────────────────────┐
│  Web UI (Next.js)                    │
│  http://localhost:3000               │
└──────────────────────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  API Server (FastAPI)                │
│  http://localhost:8000               │
└──────────────────────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  Orchestrator + Nodes                │
│  (imports Hidden Layer harness)      │
└──────────────────────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  Hidden Layer Harness                │
│  (multi-agent strategies)            │
└──────────────────────────────────────┘
                ↓
┌──────────────────────────────────────┐
│  LLM Providers                       │
│  (Anthropic, OpenAI, Ollama, MLX)    │
└──────────────────────────────────────┘
```

---

## Troubleshooting

### "Connection refused" to Postgres

```bash
docker-compose restart postgres
docker-compose ps  # Should show "Up"
```

### "Module harness not found"

```bash
cd harness
pip install -e .
cd ../agentmesh
```

### "ANTHROPIC_API_KEY not set"

```bash
# Edit .env file
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Or use Ollama (no API key needed):

```json
{
  "context": {
    "provider": "ollama",
    "model": "llama3.2:latest"
  }
}
```

### "npm install fails"

Make sure you have Node.js 18+ installed:
```bash
node --version  # Should be 18+
npm --version
```

### API server crashes with DB errors

```bash
# Reset database
docker-compose down -v
docker-compose up -d
```

---

## Development Workflow

### Adding a New Strategy

**Step 1**: Researcher adds to Hidden Layer harness
```python
# /communication/multi-agent/strategies.py
def my_new_strategy(task_input, **kwargs):
    # Implementation
    pass

STRATEGIES['my_new'] = my_new_strategy
```

**Step 2**: Wrap in AgentMesh (10 minutes)
```python
# agentmesh/core/nodes/strategy_nodes.py
class MyNewStrategyNode(StrategyNode):
    strategy_id = "my_new"

NODE_REGISTRY['my_new'] = MyNewStrategyNode
```

**Step 3**: Update web UI palette (optional)
```typescript
// agentmesh/web/app/workflows/new/page.tsx
const STRATEGIES = [
  // ...
  { id: 'my_new', name: 'My New Strategy', emoji: '🎯', config: {} },
]
```

**Done!** Strategy is now available in workflows.

---

## Testing

### Test Backend

```bash
cd agentmesh
pytest tests/  # (when tests are added)
```

### Test Web UI

```bash
cd agentmesh/web
npm run build  # Check for TypeScript errors
npm run lint   # Check for linting issues
```

### End-to-End Test

1. Create workflow via web UI
2. Execute via API
3. View results in web UI
4. Verify metrics are tracked

---

## Production Deployment (Future)

### Backend

- Deploy FastAPI on AWS/GCP/Fly.io
- Use managed Postgres (RDS/Cloud SQL)
- Use managed Redis (ElastiCache/MemoryStore)
- Add Celery workers for async execution

### Frontend

- Deploy Next.js on Vercel/Netlify
- Set `NEXT_PUBLIC_API_URL` to production API
- Enable caching and optimization

---

## Stopping

```bash
# Stop web UI: Ctrl+C in terminal
# Stop API: Ctrl+C in terminal

# Stop infrastructure
cd agentmesh
docker-compose down

# Remove all data (fresh start)
docker-compose down -v
```

---

## Next Steps

Now that you have the full stack running:

1. **Create workflows** - Try different Hidden Layer strategies
2. **Compare results** - Debate vs. Consensus vs. Single
3. **Track costs** - See how multi-agent affects cost/latency
4. **Collect feedback** - Show to design partners
5. **Iterate** - Add features based on feedback

---

**You now have a complete multi-agent workflow platform!** 🎉

- ✅ Backend API with 6 research-backed strategies
- ✅ Web UI for creating and managing workflows
- ✅ Real-time execution monitoring
- ✅ Cost and performance tracking
- ✅ Built on Hidden Layer research

Questions? Check the docs:
- `/docs/AGENTMESH_PRODUCT_STRATEGY.md` - Business strategy
- `/docs/AGENTMESH_ARCHITECTURE.md` - Technical architecture
- `agentmesh/README.md` - Backend docs
- `agentmesh/web/README.md` - Frontend docs
