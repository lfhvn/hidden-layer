# AgentMesh Quick Start - macOS

Get AgentMesh running on your Mac in 10 minutes.

---

## Prerequisites

### 1. Install Docker Desktop

**Download**: https://www.docker.com/products/docker-desktop/

After installing, make sure Docker Desktop is running (you'll see the whale icon in your menu bar).

**Verify installation**:
```bash
docker --version
docker compose version  # Note: two words, not docker-compose
```

### 2. Install Python 3.11+

**Check your Python version**:
```bash
python3 --version
```

If you need to install/upgrade Python:
```bash
# Using Homebrew
brew install python@3.11
```

### 3. Install Node.js 18+ (for Web UI)

```bash
# Using Homebrew
brew install node

# Verify
node --version  # Should be 18+
npm --version
```

---

## Quick Start (Full Stack)

### Step 1: Navigate to AgentMesh directory

```bash
cd /path/to/hidden-layer/agentmesh
```

### Step 2: Start Infrastructure (Postgres + Redis)

```bash
# IMPORTANT: Use 'docker compose' (two words) on modern Docker Desktop
docker compose up -d

# Verify services are running
docker compose ps

# You should see:
# postgres  - Up
# redis     - Up
```

**Troubleshooting**:
- If you get `command not found: docker-compose`, you're using the old syntax
- Use `docker compose` (two words) instead
- If Docker Desktop isn't running, start it from Applications

### Step 3: Install Python Dependencies

```bash
# Install Hidden Layer harness first (REQUIRED)
cd ../harness
pip3 install -e .

# Return to agentmesh
cd ../agentmesh

# Install AgentMesh dependencies
pip3 install -r requirements.txt
```

### Step 4: Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit and add your API key
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" >> .env

# Or edit manually
nano .env
```

**Don't have an API key?** You can use Ollama (local, free):
```bash
# Install Ollama
brew install ollama

# Start Ollama server
ollama serve &

# Pull a model
ollama pull llama3.2

# Use in workflows with:
# "context": {"provider": "ollama", "model": "llama3.2:latest"}
```

### Step 5: Start Backend API Server

**Terminal 1 (API Server):**
```bash
cd agentmesh
python3 -m agentmesh.api.main
```

You should see:
```
✅ Database initialized
✅ AgentMesh API server ready
📚 API docs: http://localhost:8000/docs
```

**Keep this terminal running!**

### Step 6: Start Web UI

**Terminal 2 (Web UI):**
```bash
cd agentmesh/web

# First time only: install dependencies
npm install

# Start Next.js dev server
npm run dev
```

You should see:
```
✓ Ready in 2.5s
○ Local:        http://localhost:3000
```

**Keep this terminal running!**

### Step 7 (Optional): Start Celery Worker for Async Execution

**Terminal 3 (Celery Worker):**
```bash
cd agentmesh

# Make script executable (first time only)
chmod +x worker/start_worker.sh

# Start worker
bash worker/start_worker.sh
```

Worker handles background workflow execution.

---

## Using AgentMesh

### Option 1: Web UI (Easiest)

1. Open http://localhost:3000
2. Click "Create Workflow"
3. Name it "My First Workflow"
4. Click a strategy card (e.g., "Debate")
5. Connect: start → debate1 → end
6. Click "Save Workflow"
7. Click "Execute" and fill in the form
8. Watch real-time execution!

### Option 2: Example Script

**Terminal 4:**
```bash
cd agentmesh
python3 example_workflow.py
```

This creates and executes a debate workflow.

### Option 3: API (curl)

```bash
# Create workflow
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "name": "Debate Workflow",
  "org_id": "00000000-0000-0000-0000-000000000001",
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
}
EOF

# Execute workflow (replace {workflow_id} with the ID from response)
curl -X POST http://localhost:8000/api/workflows/{workflow_id}/runs \
  -H "Content-Type: application/json" \
  -d '{
    "input": {"task": "Should we invest in renewable energy?"},
    "context": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"}
  }'
```

---

## What You Get

### 6 Research-Backed Strategies

All from Hidden Layer research:

- **debate** - Multi-agent debate with judge
- **crit** - Design critique from multiple perspectives
- **consensus** - Agents find agreement
- **manager_worker** - Task decomposition & synthesis
- **self_consistency** - Multiple samples, majority vote
- **single** - Baseline single agent

### Features

- ✅ Visual workflow editor (drag & drop)
- ✅ Real-time execution monitoring
- ✅ Cost & performance tracking
- ✅ Sync & async execution
- ✅ Human-in-the-loop nodes
- ✅ Multi-provider support (Anthropic, OpenAI, Ollama, MLX)

---

## Troubleshooting

### "docker-compose: command not found"

**Fix**: Use `docker compose` (two words) instead:
```bash
docker compose up -d      # ✅ Correct
docker-compose up -d      # ❌ Old syntax
```

### "Connection refused" to Postgres

```bash
# Check if Docker Desktop is running (see whale icon in menu bar)
# Check services
docker compose ps

# Restart if needed
docker compose restart postgres

# Or restart all services
docker compose down
docker compose up -d
```

### "Module not found: harness"

```bash
# Install harness first
cd ../harness
pip3 install -e .
cd ../agentmesh
```

### "ANTHROPIC_API_KEY not found"

**Option 1**: Set API key
```bash
echo "ANTHROPIC_API_KEY=sk-ant-your-key" >> .env
```

**Option 2**: Use Ollama (no API key needed)
```bash
brew install ollama
ollama serve &
ollama pull llama3.2

# Use in workflows:
# "context": {"provider": "ollama", "model": "llama3.2:latest"}
```

### "Port already in use"

If ports 8000, 3000, or 5432 are already in use:

```bash
# Find what's using the port
lsof -i :8000
lsof -i :3000
lsof -i :5432

# Kill the process
kill -9 <PID>
```

### Web UI shows "API connection error"

1. Make sure API server is running (Terminal 1)
2. Check http://localhost:8000/health in browser
3. Check browser console for CORS errors

---

## Architecture

```
┌──────────────────────────────┐
│  Web UI (Next.js)            │  Terminal 2
│  http://localhost:3000       │
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│  API Server (FastAPI)        │  Terminal 1
│  http://localhost:8000       │
└──────────────────────────────┘
         ↓           ↓
   (sync)      (async via Celery)
         ↓           ↓
┌──────────────────────────────┐
│  Worker (Celery)             │  Terminal 3 (optional)
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│  Hidden Layer Harness        │
│  (multi-agent strategies)    │
└──────────────────────────────┘
              ↓
┌──────────────────────────────┐
│  LLM Providers               │
│  (Anthropic, OpenAI, Ollama) │
└──────────────────────────────┘

Infrastructure (Docker):
├─ Postgres (localhost:5432)
└─ Redis (localhost:6379)
```

---

## Stopping Everything

```bash
# Terminal 1 (API): Press Ctrl+C
# Terminal 2 (Web): Press Ctrl+C
# Terminal 3 (Worker): Press Ctrl+C

# Stop Docker services
cd agentmesh
docker compose down

# Remove all data (fresh start)
docker compose down -v
```

---

## Next Steps

- **Explore API docs**: http://localhost:8000/docs
- **Try different strategies**: Create workflows with CRIT, Consensus, etc.
- **Read architecture**: `/docs/AGENTMESH_ARCHITECTURE.md`
- **Read product strategy**: `/docs/AGENTMESH_PRODUCT_STRATEGY.md`
- **Complete running guide**: `agentmesh/RUNNING.md`

---

## Mac-Specific Tips

### Use Python 3 explicitly
```bash
python3 -m agentmesh.api.main  # ✅ Correct
python -m agentmesh.api.main   # ❌ Might use system Python 2
```

### Virtual Environment (Recommended)

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Now you can use 'python' instead of 'python3'
python -m agentmesh.api.main
```

### Performance on Apple Silicon (M1/M2/M3/M4)

AgentMesh runs great on Apple Silicon! For even better performance:

```bash
# Use MLX for local inference (Apple Silicon optimized)
pip install mlx-lm

# Use in workflows:
# "context": {"provider": "mlx", "model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}
```

---

**You're all set! 🎉**

Open http://localhost:3000 and start building multi-agent workflows!
