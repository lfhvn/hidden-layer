# AgentMesh Technical Architecture
## Building the Product Layer on Hidden Layer Harness

**Date**: 2025-11-18
**Context**: How AgentMesh (product) is built on top of Hidden Layer Harness (research infrastructure)

---

## Architecture Overview

### The Stack

```
┌─────────────────────────────────────────────────────┐
│  AgentMesh Product (Commercial Platform)            │
│  - Web UI (workflow editor, observability)          │
│  - REST API (CRUD, orchestration endpoints)         │
│  - Multi-tenant DB (users, workflows, runs)         │
│  - Async orchestration (queue, workers)             │
└─────────────────────────────────────────────────────┘
                        ↓ uses
┌─────────────────────────────────────────────────────┐
│  Hidden Layer Harness (Research Infrastructure)     │
│  - LLM provider abstraction (Ollama/MLX/Claude/GPT) │
│  - Multi-agent strategies (debate, CRIT, etc.)      │
│  - Experiment tracking & metrics                    │
│  - Benchmark integration                            │
└─────────────────────────────────────────────────────┘
                        ↓ calls
┌─────────────────────────────────────────────────────┐
│  LLM Providers                                       │
│  - Ollama (local)                                   │
│  - MLX (Apple Silicon)                              │
│  - Anthropic (Claude API)                           │
│  - OpenAI (GPT API)                                 │
└─────────────────────────────────────────────────────┘
```

### Key Insight

**AgentMesh does NOT reimplement the harness.**

Instead:
- ✅ Harness remains the execution engine
- ✅ AgentMesh adds product features (UI, multi-tenancy, persistence)
- ✅ Research strategies flow directly into product
- ✅ Single source of truth for LLM abstractions

---

## Layer Responsibilities

### Layer 1: Hidden Layer Harness (Research)

**What it does**:
- Execute multi-agent strategies
- Abstract over LLM providers
- Track experiment metrics
- Load benchmarks

**What it DOESN'T do**:
- Multi-user support
- Persistent workflow state
- Visual workflow design
- Production SLAs

**Code location**: `/harness/`

**Ownership**: Research team

**Changes**: Based on research needs, backward compatible

---

### Layer 2: AgentMesh Core (Product Runtime)

**What it does**:
- Wrap harness strategies as workflow nodes
- Manage workflow state (runs, steps)
- Async orchestration (queue, retries)
- Cost tracking & rate limiting

**What it DOESN'T do**:
- Implement strategies directly (imports from harness)
- Talk to LLMs directly (uses harness)

**Code location**: `/agentmesh/core/`

**Ownership**: Product team (with research input)

**Changes**: Product requirements, maintain harness compatibility

---

### Layer 3: AgentMesh Platform (Product Features)

**What it does**:
- Multi-tenant web app
- Visual workflow editor
- User authentication & permissions
- Billing & subscriptions
- Observability dashboards

**What it DOESN'T do**:
- Execute strategies (delegates to core)
- Abstract LLMs (uses harness)

**Code location**: `/agentmesh/platform/`

**Ownership**: Product team

**Changes**: Product iterations, customer feedback

---

## Code Integration

### How AgentMesh Uses Harness

```python
# agentmesh/core/nodes/strategy_node.py

from harness import llm_call, run_strategy, STRATEGIES
from agentmesh.core import WorkflowNode

class DebateNode(WorkflowNode):
    """
    AgentMesh node that wraps Hidden Layer debate strategy.

    Adds:
    - State persistence
    - Async execution
    - Error handling & retries
    - Cost tracking

    Delegates:
    - Strategy logic → harness
    - LLM calls → harness
    - Provider abstraction → harness
    """

    def __init__(self, n_debaters=3, n_rounds=2):
        self.n_debaters = n_debaters
        self.n_rounds = n_rounds

    async def execute(self, step_id, input_data, context):
        # AgentMesh product features
        await self.update_step_status(step_id, 'running')
        start_time = time.time()

        try:
            # Delegate to Hidden Layer harness (research code)
            result = run_strategy(
                'debate',
                task_input=input_data,
                n_debaters=self.n_debaters,
                n_rounds=self.n_rounds,
                provider=context.provider,
                model=context.model
            )

            # AgentMesh product features
            latency = time.time() - start_time
            await self.track_cost(step_id, result.cost_usd)
            await self.update_step_status(step_id, 'succeeded')

            return StepResult(
                output=result.output,
                metadata={
                    'strategy': result.strategy_name,
                    'latency_s': latency,
                    'tokens_in': result.tokens_in,
                    'tokens_out': result.tokens_out,
                }
            )

        except Exception as e:
            # AgentMesh product features
            await self.update_step_status(step_id, 'failed')
            await self.log_error(step_id, str(e))
            raise
```

### Key Principles

1. **Harness is imported, not copied**
   ```python
   # Good
   from harness import run_strategy

   # Bad
   def our_own_debate_implementation():
       # Reimplementing strategy
   ```

2. **Strategies live in harness**
   ```python
   # Research adds new strategy here
   # /communication/multi-agent/multi_agent/strategies.py

   def new_strategy(task_input, **kwargs):
       # Research implementation
       pass

   STRATEGIES['new_strategy'] = new_strategy
   ```

   ```python
   # Product automatically gets it
   # /agentmesh/core/nodes/

   class NewStrategyNode(WorkflowNode):
       def execute(self, ...):
           return run_strategy('new_strategy', ...)  # Just works!
   ```

3. **Provider abstraction stays in harness**
   ```python
   # Product doesn't know about provider details
   result = run_strategy(
       'debate',
       task_input=input,
       provider='anthropic',  # Harness handles
       model='claude-3-5-sonnet-20241022'  # Harness handles
   )
   ```

---

## Research → Product Flow

### Scenario: New Strategy Development

**Step 1**: Research discovers new coordination pattern
```python
# /communication/multi-agent/multi_agent/strategies.py

def adversarial_debate(task_input, n_advocates=2, n_critics=2, **kwargs):
    """
    New strategy: Advocates argue for solution, critics find flaws.
    Research question: Does adversarial framing improve reasoning?
    """
    # Research implementation
    # Benchmarked on ToMBench, reasoning tasks
    # Published in paper
    pass

STRATEGIES['adversarial_debate'] = adversarial_debate
```

**Step 2**: Product wraps as node type (1-2 hours of work)
```python
# /agentmesh/core/nodes/adversarial_debate_node.py

class AdversarialDebateNode(WorkflowNode):
    """Wrap Hidden Layer adversarial debate strategy"""

    def execute(self, step_id, input_data, context):
        return run_strategy(
            'adversarial_debate',
            task_input=input_data,
            n_advocates=self.n_advocates,
            n_critics=self.n_critics,
            **context.to_kwargs()
        )
```

**Step 3**: Product exposes in UI
```typescript
// /agentmesh/web/src/components/NodePalette.tsx

const nodeTypes = [
  { id: 'debate', label: 'Debate', icon: '💬' },
  { id: 'crit', label: 'CRIT', icon: '🎨' },
  { id: 'adversarial_debate', label: 'Adversarial Debate', icon: '⚔️' },  // New!
  // ...
]
```

**Step 4**: Customers use immediately
- No reimplementation needed
- Research-validated from day 1
- Product marketing: "Based on NeurIPS 2026 paper"

---

## Shared vs. Product-Specific Code

### Shared (Lives in Harness)

✅ **Multi-agent strategies**
- debate, CRIT, consensus, manager-worker, etc.
- Source of truth: `/communication/multi-agent/`

✅ **LLM provider abstraction**
- Ollama, MLX, Anthropic, OpenAI wrappers
- Source of truth: `/harness/llm_provider.py`

✅ **Experiment tracking**
- Metrics, logging, reproducibility
- Source of truth: `/harness/experiment_tracker.py`

✅ **Evaluation functions**
- exact_match, llm_judge, etc.
- Source of truth: `/harness/evals.py`

### Product-Specific (Lives in AgentMesh)

⚙️ **Web UI**
- Workflow graph editor
- Run list, detail views
- User dashboards

⚙️ **Multi-tenancy**
- User accounts, teams
- Row-level security
- Billing & subscriptions

⚙️ **Workflow state persistence**
- Run/step database models
- Postgres schemas
- State recovery

⚙️ **Async orchestration**
- Celery workers
- Redis queue
- Retry logic

⚙️ **Observability platform**
- Timeline visualization
- Cost analytics
- Error tracking

---

## Database Schema (AgentMesh Product Layer)

### Core Entities

```sql
-- User/tenant management (product-specific)
CREATE TABLE users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE,
  created_at TIMESTAMP
);

CREATE TABLE organizations (
  id UUID PRIMARY KEY,
  name TEXT,
  plan TEXT  -- free, pro, team, enterprise
);

-- Workflows (references harness strategies)
CREATE TABLE workflows (
  id UUID PRIMARY KEY,
  org_id UUID REFERENCES organizations,
  name TEXT,
  graph JSONB,  -- nodes/edges, includes strategy IDs from harness
  created_at TIMESTAMP
);

-- Runs & steps (execution state)
CREATE TABLE workflow_runs (
  id UUID PRIMARY KEY,
  workflow_id UUID REFERENCES workflows,
  status TEXT,  -- pending, running, succeeded, failed
  input JSONB,
  output JSONB,
  started_at TIMESTAMP,
  finished_at TIMESTAMP
);

CREATE TABLE workflow_steps (
  id UUID PRIMARY KEY,
  run_id UUID REFERENCES workflow_runs,
  node_id TEXT,
  node_type TEXT,  -- 'agent', 'debate', 'crit', etc. (maps to harness)
  status TEXT,
  input JSONB,
  output JSONB,

  -- Metrics from harness
  latency_s FLOAT,
  tokens_in INT,
  tokens_out INT,
  cost_usd DECIMAL,

  started_at TIMESTAMP,
  finished_at TIMESTAMP
);
```

### Key Design Principles

1. **Workflows reference harness strategies by ID**
   ```json
   {
     "nodes": [
       {
         "id": "node1",
         "type": "strategy",
         "strategy_id": "debate",  // References harness STRATEGIES['debate']
         "config": {
           "n_debaters": 3
         }
       }
     ]
   }
   ```

2. **Steps store harness metrics**
   - `latency_s`, `tokens_in`, `tokens_out`, `cost_usd` come from harness `StrategyResult`
   - Product doesn't recompute, just stores

3. **Provider config uses harness format**
   ```json
   {
     "provider": "anthropic",
     "model": "claude-3-5-sonnet-20241022",
     "temperature": 0.7
   }
   ```
   - Same keys as harness `llm_call()`
   - Zero impedance mismatch

---

## API Design (AgentMesh Product Layer)

### RESTful Endpoints

```python
# POST /api/workflows/:workflow_id/runs
# Create and execute a workflow run

from fastapi import FastAPI
from agentmesh.core import Orchestrator
from harness import run_strategy, STRATEGIES

app = FastAPI()

@app.post("/api/workflows/{workflow_id}/runs")
async def create_run(workflow_id: str, input: dict):
    # Product layer: Fetch workflow, create run record
    workflow = await db.get_workflow(workflow_id)
    run = await db.create_run(workflow_id, input)

    # Delegate to orchestrator (which uses harness)
    orchestrator = Orchestrator(workflow, run)
    await orchestrator.execute()  # This calls harness strategies

    return {"run_id": run.id}
```

### Orchestrator Implementation

```python
# agentmesh/core/orchestrator.py

from harness import run_strategy

class Orchestrator:
    """
    Product orchestrator that wraps harness execution.

    Adds:
    - Async workflow execution
    - State persistence (runs, steps)
    - Error handling & retries
    - Cost tracking

    Delegates:
    - Strategy execution → harness
    - LLM calls → harness
    """

    def __init__(self, workflow, run):
        self.workflow = workflow
        self.run = run

    async def execute(self):
        """Execute workflow graph using harness strategies"""

        for node in self.workflow.graph['nodes']:
            if node['type'] == 'strategy':
                await self.execute_strategy_node(node)
            elif node['type'] == 'tool':
                await self.execute_tool_node(node)
            # ...

    async def execute_strategy_node(self, node):
        """Execute a harness strategy as a workflow step"""

        # Product: Create step record
        step = await db.create_step(
            run_id=self.run.id,
            node_id=node['id'],
            node_type='strategy'
        )

        # Product: Update status
        await db.update_step(step.id, status='running')

        try:
            # Harness: Execute strategy
            result = run_strategy(
                node['strategy_id'],  # 'debate', 'crit', etc.
                task_input=step.input,
                **node['config']
            )

            # Product: Store results
            await db.update_step(step.id,
                status='succeeded',
                output=result.output,
                latency_s=result.latency_s,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cost_usd=result.cost_usd
            )

        except Exception as e:
            # Product: Handle errors
            await db.update_step(step.id,
                status='failed',
                error=str(e)
            )
            raise
```

---

## Deployment Architecture

### Development (Local)

```bash
# Researcher working on strategies
cd communication/multi-agent/
python -m pytest tests/  # Test new strategy
# Strategy code stays in research repo

# Product engineer testing integration
cd agentmesh/
docker-compose up  # Starts web, API, workers, postgres, redis
# Open http://localhost:3000
# Create workflow using new strategy (automatically available)
```

### Production (Multi-tenant SaaS)

```
┌─────────────────────────────────────────────────────┐
│  CloudFlare (CDN, DDoS protection)                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Load Balancer (AWS ALB)                            │
└─────────────────────────────────────────────────────┘
                        ↓
         ┌──────────────┴──────────────┐
         ↓                              ↓
┌──────────────────┐          ┌──────────────────┐
│  Web App (Next)  │          │  API (FastAPI)   │
│  K8s pods x 3    │          │  K8s pods x 5    │
│  Stateless       │          │  Stateless       │
└──────────────────┘          └──────────────────┘
                                       ↓
                        ┌──────────────┴──────────────┐
                        ↓                              ↓
                ┌──────────────┐            ┌──────────────┐
                │ Redis Queue  │            │ Postgres DB  │
                └──────────────┘            └──────────────┘
                        ↓
                ┌──────────────────┐
                │ Celery Workers   │
                │ K8s pods x 10    │
                │ (execute steps)  │
                │                  │
                │ imports harness  │ ← Uses Hidden Layer research code
                └──────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                                ↓
┌─────────────┐                  ┌─────────────┐
│ Anthropic   │                  │ OpenAI      │
│ Claude API  │                  │ GPT API     │
└─────────────┘                  └─────────────┘
```

### Key Infrastructure Notes

1. **Harness is deployed as Python package**
   ```dockerfile
   # agentmesh/workers/Dockerfile

   FROM python:3.11

   # Install Hidden Layer harness
   COPY ../harness /app/harness
   RUN pip install -e /app/harness

   # Install AgentMesh worker
   COPY . /app/agentmesh-worker
   RUN pip install -e /app/agentmesh-worker

   CMD ["celery", "worker", "-A", "agentmesh.worker"]
   ```

2. **Workers import harness directly**
   ```python
   # In worker pods
   from harness import run_strategy  # Works!
   ```

3. **No separate harness service**
   - Harness is a library, not a microservice
   - Workers have harness code bundled
   - Zero network overhead

---

## Testing Strategy

### Unit Tests (Harness - Research)

```python
# /communication/multi-agent/tests/test_strategies.py

def test_debate_strategy():
    """Test debate strategy in isolation (research code)"""
    from multi_agent.strategies import debate

    result = debate(
        task_input="What is 2+2?",
        n_debaters=3,
        provider="ollama",
        model="llama3.2:latest"
    )

    assert result.strategy_name == "debate"
    assert isinstance(result.output, str)
    assert result.latency_s > 0
```

### Integration Tests (AgentMesh Product)

```python
# /agentmesh/tests/test_orchestrator.py

async def test_debate_node_execution():
    """Test AgentMesh wrapping of harness debate strategy"""
    from agentmesh.core import Orchestrator

    workflow = {
        'nodes': [
            {
                'id': 'node1',
                'type': 'strategy',
                'strategy_id': 'debate',
                'config': {'n_debaters': 3}
            }
        ],
        'edges': []
    }

    run = await db.create_run(workflow_id, input={'task': 'Test'})
    orchestrator = Orchestrator(workflow, run)

    await orchestrator.execute()  # Uses harness

    run = await db.get_run(run.id)
    assert run.status == 'succeeded'
    assert run.output is not None
```

### E2E Tests (AgentMesh Platform)

```typescript
// /agentmesh/web/tests/e2e/workflow-execution.test.ts

test('create and execute debate workflow', async () => {
  // Create workflow via UI
  await page.goto('/workflows/new');
  await page.click('[data-node-type="debate"]');
  await page.fill('[name="n_debaters"]', '3');
  await page.click('[data-action="save"]');

  // Execute workflow
  await page.click('[data-action="run"]');
  await page.fill('[name="input"]', 'What is 2+2?');
  await page.click('[data-action="submit"]');

  // Verify execution (which used harness internally)
  await page.waitForSelector('[data-status="succeeded"]');
  const output = await page.textContent('[data-output]');
  expect(output).toContain('4');
});
```

---

## Versioning & Compatibility

### Harness Versioning

```python
# /harness/__init__.py

__version__ = "0.3.0"

# Semantic versioning
# - 0.3.0 → 0.3.1: Backward compatible (bug fixes)
# - 0.3.0 → 0.4.0: Backward compatible (new features)
# - 0.3.0 → 1.0.0: Breaking changes (rare)
```

### AgentMesh Depends on Harness

```python
# /agentmesh/requirements.txt

hidden-layer-harness>=0.3.0,<0.4.0  # Lock minor version
```

### Strategy Stability Contract

**Guarantee**: Once a strategy is in `STRATEGIES`, it won't break.

```python
# Research adds new parameter (backward compatible)
def debate(task_input, n_debaters=3, n_rounds=2, enable_reflection=False, **kwargs):
    # New parameter with default
    pass

# Product code continues working
result = run_strategy('debate', task, n_debaters=3)  # Still works!
```

**Breaking changes** (rare):
- Announce 1 version ahead
- Deprecation warnings
- Migration guide

---

## Research Independence

### Governance Model

**Research Team Controls**:
- ✅ Harness API design
- ✅ Strategy implementations
- ✅ When to publish findings
- ✅ Benchmark selection
- ✅ Provider support

**Product Team Controls**:
- ✅ AgentMesh UI/UX
- ✅ Workflow persistence
- ✅ Multi-tenancy
- ✅ Billing & subscriptions
- ✅ Customer support

**Shared Decisions**:
- 🤝 New strategy productization
- 🤝 Harness API changes (if breaking)
- 🤝 Research → product timeline

### Weekly Sync

**Agenda**:
1. Research updates (new strategies, papers)
2. Product updates (customer feedback, features)
3. Integration needs (new strategies to wrap)
4. Blockers (harness changes needed, etc.)

**Output**: Shared roadmap (research + product)

---

## Migration Path

### Phase 1: Extract Harness (Week 1-2)

**Current State**: Harness is part of Hidden Layer monorepo

**Target State**: Harness is standalone package

```bash
# Create standalone harness package
cd hidden-layer/harness/
python setup.py bdist_wheel

# Install in AgentMesh
cd ../agentmesh/
pip install ../harness/dist/hidden_layer_harness-0.3.0-py3-none-any.whl
```

### Phase 2: Build AgentMesh Core (Week 3-6)

**Implement**:
- Orchestrator (uses harness)
- Node types (wrap strategies)
- DB models (runs, steps)
- Worker execution (Celery)

**Verify**: All strategies work through AgentMesh layer

### Phase 3: Build AgentMesh Platform (Week 7-10)

**Implement**:
- Web UI (React/Next)
- REST API (FastAPI)
- User auth (JWT)
- Basic observability

**Verify**: End-to-end workflow execution through UI

### Phase 4: Launch (Week 11-12)

**Activities**:
- Deploy to production
- Open free tier
- Product Hunt launch
- Monitor & iterate

---

## FAQ

### Q: Does AgentMesh reimplement any harness functionality?

**A**: No. AgentMesh is a thin wrapper that adds product features (UI, persistence, multi-tenancy) on top of harness execution.

### Q: What if harness needs to change for product needs?

**A**: Product team proposes changes, research team reviews. If it benefits research too, merge. If product-specific, add to AgentMesh layer only.

### Q: Can researchers use AgentMesh?

**A**: Yes! Researchers can use AgentMesh UI to visualize their experiments. But they can also use harness directly (faster for prototyping).

### Q: What if a strategy needs product-specific features?

**A**: Keep strategy logic in harness (provider-agnostic). Add product wrappers in AgentMesh layer (e.g., rate limiting, caching).

### Q: How do we ensure harness stays research-focused?

**A**: Research team has final say on harness design. Product requirements are input, not mandates.

---

## Summary

### Key Architectural Decisions

1. ✅ **Harness is the execution engine** (not reimplemented)
2. ✅ **AgentMesh is product layer** (UI, persistence, multi-tenancy)
3. ✅ **Strategies live in harness** (single source of truth)
4. ✅ **Research team controls harness** (product team is user)
5. ✅ **Clear boundaries** (research vs. product concerns)

### Benefits of This Architecture

**For Research**:
- ✅ Research code is used directly (not ported)
- ✅ Product validates research (real-world use)
- ✅ Faster research → product cycle

**For Product**:
- ✅ Leverage research innovations automatically
- ✅ No need to reimplement strategies
- ✅ Differentiation through research-backed patterns

**For Both**:
- ✅ Single codebase for LLM abstractions
- ✅ Shared provider support
- ✅ Research findings improve product; product data informs research

---

**Architecture Assessment Complete**

See also:
- Product strategy: `docs/AGENTMESH_PRODUCT_STRATEGY.md`
- Core comparison: `docs/AGENTMESH_ASSESSMENT.md`
- Roles/capabilities: `docs/AGENTMESH_ROLES_ASSESSMENT.md`
