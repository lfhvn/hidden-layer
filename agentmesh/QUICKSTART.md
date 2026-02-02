# AgentMesh Quick Start

Get AgentMesh running in 5 minutes.

---

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Hidden Layer harness (from `/harness`)

---

## Step 1: Start Infrastructure

```bash
cd agentmesh

# Start Postgres & Redis
docker-compose up -d

# Wait for databases to be ready
docker-compose ps
```

---

## Step 2: Install Dependencies

```bash
# Install Hidden Layer harness first
cd ../harness
pip install -e .

# Install AgentMesh
cd ../agentmesh
pip install -r requirements.txt
```

---

## Step 3: Set Environment Variables

```bash
# Copy example env
cp .env.example .env

# Edit .env and add your LLM provider API keys
# At minimum, set ANTHROPIC_API_KEY or OPENAI_API_KEY
nano .env
```

---

## Step 4: Start API Server

```bash
# Run server (auto-creates database tables)
python -m agentmesh.api.main
```

You should see:
```
✅ Database initialized
✅ AgentMesh API server ready
📚 API docs: http://localhost:8000/docs
```

---

## Step 5: Run Example Workflow

In a new terminal:

```bash
cd agentmesh

# Run example (creates and executes debate workflow)
python example_workflow.py
```

This will:
1. Create a workflow with Hidden Layer "debate" strategy
2. Execute it with a task about renewable energy
3. Show you the results (from Hidden Layer research code!)

---

## Step 6: Explore API

Open http://localhost:8000/docs in your browser.

Try the interactive API:

### Create a Workflow

```bash
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Workflow",
    "org_id": "123e4567-e89b-12d3-a456-426614174000",
    "graph": {
      "nodes": [
        {"id": "start", "type": "start", "label": "Start"},
        {
          "id": "debate1",
          "type": "strategy",
          "label": "Debate",
          "strategy_id": "debate",
          "config": {"n_debaters": 3}
        },
        {"id": "end", "type": "end", "label": "End"}
      ],
      "edges": [
        {"id": "e1", "from_node_id": "start", "to_node_id": "debate1"},
        {"id": "e2", "from_node_id": "debate1", "to_node_id": "end"}
      ]
    }
  }'
```

### Execute Workflow

```bash
curl -X POST http://localhost:8000/api/workflows/{workflow_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "What is the capital of France?"
    },
    "context": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022"
    }
  }'
```

### Get Results

```bash
curl http://localhost:8000/api/runs/{run_id}
curl http://localhost:8000/api/runs/{run_id}/steps
```

---

## Available Strategies

All Hidden Layer multi-agent strategies are available:

- **debate** - n-agent debate with judge
- **crit** - Multi-perspective design critique
- **consensus** - Multiple agents find agreement
- **manager_worker** - Decompose → execute → synthesize
- **self_consistency** - Sample multiple times, aggregate
- **single** - Baseline single agent

**When Hidden Layer research adds new strategies, they're automatically available in AgentMesh!**

---

## Architecture

```
AgentMesh (you're here)
    ↓ imports & wraps
Hidden Layer Harness (/harness)
    ↓ contains
Multi-agent strategies (/communication/multi-agent)
    ↓ calls
LLM Providers (Anthropic, OpenAI, Ollama, MLX)
```

**Key insight**: AgentMesh doesn't reimplement strategies. It imports them from Hidden Layer research code.

---

## Troubleshooting

### "Connection refused" to Postgres

```bash
# Check Docker Compose is running
docker-compose ps

# Restart if needed
docker-compose restart postgres
```

### "Module not found: harness"

```bash
# Install harness first
cd ../harness
pip install -e .
```

### "ANTHROPIC_API_KEY not found"

```bash
# Set in .env file
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

Or use Ollama (local, no API key):

```json
{
  "context": {
    "provider": "ollama",
    "model": "llama3.2:latest"
  }
}
```

---

## Next Steps

- **Read**: `/docs/AGENTMESH_ARCHITECTURE.md` - How AgentMesh wraps harness
- **Read**: `/docs/AGENTMESH_PRODUCT_STRATEGY.md` - Product vision
- **Try**: Create custom workflows with multiple strategy nodes
- **Build**: Add a web UI (Next.js frontend)

---

## Stopping

```bash
# Stop API server: Ctrl+C

# Stop infrastructure
docker-compose down

# Remove data (if you want to start fresh)
docker-compose down -v
```

---

**You're running AgentMesh! 🎉**

Create workflows using research-backed multi-agent strategies from Hidden Layer.
