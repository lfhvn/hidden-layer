# AgentMesh Architecture

## 1. Product Scope for v0

**Goal:** Build an MVP of AgentMesh: a platform to design, run, and observe multi-agent workflows.

### Core v0 capabilities
1. Define agents and tools.
2. Define workflows as directed graphs (nodes = agents/tools, edges = message passing).
3. Execute a workflow run with a given input payload.
4. Visualize runs (timeline, node-level inputs/outputs).
5. Basic human-in-the-loop step (manual approval / edit gate).

### Non-goals for v0
- No multi-tenant billing.
- No complex RBAC.
- No fully pluggable model routing (keep it simple).

---

## 2. High-Level Architecture

### 2.1 Components

#### Web App (Frontend)
- **Tech:** React + TypeScript (Next.js or Vite).
- **Responsibilities:**
  - Workflow graph editor (nodes/edges).
  - Agent config editor (prompts, model, tools).
  - Runs explorer (list + details).
  - Real-time run status via WebSocket/SSE.

#### API / Control Plane
- **Tech:** TypeScript (NestJS / Express) or Python (FastAPI).
- **Responsibilities:**
  - REST/JSON APIs for CRUD on agents, workflows, runs.
  - Authentication (simple JWT or session).
  - Orchestration API surface (start workflow, cancel run).
  - Persistence layer abstraction (Postgres).

#### Orchestrator / Runtime
- **Tech:** Could live inside the API initially as a module; later separated as its own service.
- **Responsibilities:**
  - Execute workflow DAGs.
  - Manage node scheduling, retries, backoff.
  - Persist state for each run/step.
  - Emit events for observability.

#### Workers (Agent Runners)
- **Tech:** Same language as orchestrator for simplicity.
- **Responsibilities:**
  - Execute one "step" (agent call, tool call, human task).
  - Call external LLM providers (OpenAI, Anthropic, local, etc.).
  - Enforce timeouts, basic error handling.

#### Message Bus / Queue
- **Tech (v0):** Redis streams or a simple in-process task queue. Later: NATS, Kafka, or Celery/RQ depending on stack.
- **Responsibilities:**
  - Dispatch work units (step executions).
  - Decouple orchestrator from worker execution.

#### Storage
- **Postgres:** Agents, tools, workflows, runs, steps.
- **Redis:** Short-lived state, locks, rate limiting.
- **Optional later:** ClickHouse/BigQuery for long-term logs, Vector DB for agent memories.

---

## 3. Domain Model (Core Entities)

Design these as DB tables + TypeScript/Python models.

### 3.1 Agent
Represents a logical agent (LLM + behavior + tools).

```ts
type Provider = 'openai' | 'anthropic' | 'local';

interface Agent {
  id: string;
  projectId: string;
  name: string;
  description?: string;
  modelProvider: Provider;
  modelName: string;
  systemPrompt: string;
  temperature: number;
  tools: string[];           // tool ids
  metadata: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}
```

### 3.2 Tool
Represents an external tool or function that an agent can call.

```ts
interface Tool {
  id: string;
  projectId: string;
  name: string;
  description?: string;
  type: 'http' | 'function' | 'builtin';
  // For type === 'http'
  config: {
    method?: 'GET' | 'POST';
    url?: string;
    headers?: Record<string, string>;
    // For type === 'function', reference internal code path or plugin
    functionName?: string;
  };
  inputSchema: Record<string, any>;   // JSON Schema
  outputSchema: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}
```

### 3.3 Workflow
High-level workflow metadata + graph.

```ts
interface Workflow {
  id: string;
  projectId: string;
  name: string;
  description?: string;
  // List of nodes and edges (graph structure)
  graph: WorkflowGraph;
  createdAt: string;
  updatedAt: string;
}

interface WorkflowGraph {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

type NodeType =
  | 'agent'
  | 'tool'
  | 'start'
  | 'end'
  | 'human_approval'
  | 'branch';

interface WorkflowNode {
  id: string;
  type: NodeType;
  label: string;
  agentId?: string;         // for type 'agent'
  toolId?: string;          // for type 'tool'
  config: Record<string, any>; // e.g., branch condition, human step instructions
}

interface WorkflowEdge {
  id: string;
  fromNodeId: string;
  toNodeId: string;
  condition?: string;      // expression against context, for branches
}
```

### 3.4 Run & Steps
Each execution of a workflow produces a Run with many Steps.

```ts
type RunStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'canceled';

interface WorkflowRun {
  id: string;
  workflowId: string;
  projectId: string;
  status: RunStatus;
  input: any;               // initial payload
  output?: any;             // final result
  startedAt?: string;
  finishedAt?: string;
  error?: string;
}

type StepStatus = 'pending' | 'running' | 'succeeded' | 'failed' | 'waiting_human';

interface WorkflowStep {
  id: string;
  runId: string;
  workflowId: string;
  nodeId: string;
  nodeType: NodeType;
  status: StepStatus;
  input: any;
  output?: any;
  error?: string;
  startedAt?: string;
  finishedAt?: string;
  // pointer to previous steps for traceability
  parentStepIds: string[];
}
```

---

## 4. Orchestrator Design

The orchestrator executes a DAG of nodes for each run.

### 4.1 Execution Semantics
1. Start from the start node.
2. For each node:
   - Compute node input from:
     - initial run input
     - outputs of parent nodes
     - workflow context (accumulated state).
   - Enqueue a step to be executed by a worker.
3. Worker executes step:
   - If agent node:
     - Build LLM prompt (system + context + user).
     - Call provider.
     - Parse response, persist output.
   - If tool node:
     - Call HTTP/function with input.
   - If human_approval node:
     - Mark step waiting_human and notify.
   - If branch node:
     - Evaluate condition and choose outgoing edge.
4. Orchestrator listens for step completion events:
   - Mark step status.
   - Enqueue next steps whose dependencies are satisfied.
5. Run completes when:
   - end node(s) completed → mark run succeeded.
   - Any fatal error without recovery → mark failed.

### 4.2 Internal APIs (Pseudo-code)

```http
// Start a run
POST /api/workflows/:workflowId/runs
body: { input: any }
returns: { runId: string }
```

```ts
// Orchestrator loop (simplified)
async function processRun(runId: string) {
  const run = await db.getRun(runId);
  const wf = await db.getWorkflow(run.workflowId);
  const readyNodes = findReadyNodes(wf.graph, run);
  for (const node of readyNodes) {
    const step = await createStepForNode(run, node);
    await queue.enqueue('step.execute', { stepId: step.id });
  }
}
```

---

## 5. Worker / Step Execution

Worker consumes `step.execute` jobs.

```ts
async function executeStep(stepId: string) {
  const step = await db.getStep(stepId);
  const node = await getNode(step.workflowId, step.nodeId);

  await db.updateStep(stepId, { status: 'running', startedAt: now() });

  try {
    let output;
    if (node.type === 'agent') {
      output = await runAgentNode(step, node);
    } else if (node.type === 'tool') {
      output = await runToolNode(step, node);
    } else if (node.type === 'human_approval') {
      await db.updateStep(stepId, {
        status: 'waiting_human',
      });
      emitEvent('step.waiting_human', { stepId });
      return;
    } else if (node.type === 'branch') {
      output = await evaluateBranch(step, node);
    }

    await db.updateStep(stepId, {
      status: 'succeeded',
      output,
      finishedAt: now(),
    });

    emitEvent('step.completed', { stepId });
  } catch (err) {
    await db.updateStep(stepId, {
      status: 'failed',
      error: String(err),
      finishedAt: now(),
    });
    emitEvent('step.failed', { stepId });
  }
}
```

### Agent Execution

```ts
async function runAgentNode(step: WorkflowStep, node: WorkflowNode) {
  const agent = await db.getAgent(node.agentId!);
  const messages = buildMessagesFromContext(step, agent);

  const response = await callLLM({
    provider: agent.modelProvider,
    model: agent.modelName,
    messages,
    temperature: agent.temperature,
  });

  return {
    raw: response,
    text: extractText(response),
  };
}
```

---

## 6. Observability & Logging

### 6.1 Event Model

Emit structured events from orchestrator + workers to an event sink (could be Postgres JSONB, Redis, or a simple log table).

**Event types:**
- `run.started`, `run.completed`, `run.failed`
- `step.started`, `step.completed`, `step.failed`, `step.waiting_human`
- `agent.request`, `agent.response`
- `tool.request`, `tool.response`

### 6.2 Log Record

```ts
interface EventLog {
  id: string;
  timestamp: string;
  projectId: string;
  runId?: string;
  stepId?: string;
  type: string;
  payload: any;   // structured JSON
}
```

This powers:
- Gantt-like run timelines in the UI.
- Node detail panels with input/output.
- Cost/latency dashboards later.

---

## 7. Frontend (MVP Design)

### 7.1 Pages
1. **Projects Dashboard**
   - List projects, link to workflows/runs.
2. **Workflow Editor**
   - Canvas + left sidebar of node types (start, agent, tool, branch, human).
   - Node inspector panel (right) for config:
     - For agent node: select agent, override prompt, temperature.
     - For branch node: condition expression (e.g. `context.last_output.score > 0.7`).
3. **Runs List**
   - Table of runs (status, start time, duration, workflow, input summary).
4. **Run Detail View**
   - Graph with nodes colored by step status.
   - Timeline of steps (vertical list with expand for I/O).
   - JSON viewer for any step input/output.
   - “Resume after human approval” button for human steps.

### 7.2 Data Fetching
- Use REST endpoints:
  - `GET /api/workflows`, `GET /api/workflows/:id`
  - `POST /api/workflows`
  - `GET /api/workflows/:id/runs`
  - `GET /api/runs/:runId`
- Use WebSocket/SSE:
  - Subscribe to `runId` to stream events and update UI in real time.

---

## 8. API Sketch (REST)

```
POST /api/projects
POST /api/agents
GET  /api/agents?projectId=...
POST /api/tools
POST /api/workflows
GET  /api/workflows/:workflowId
PUT  /api/workflows/:workflowId

POST /api/workflows/:workflowId/runs
GET  /api/workflows/:workflowId/runs
GET  /api/runs/:runId
POST /api/runs/:runId/cancel

POST /api/steps/:stepId/human-complete
body: { output: any }
```

---

## 9. v0 Implementation Plan (Milestones)

### Milestone 1 – Skeleton & Models (1–2 weeks)
- Set up repo (monorepo or separate frontend/backend).
- Implement:
  - Project, Agent, Tool, Workflow, Run, Step models & DB migrations.
  - Basic CRUD APIs for Agents and Workflows (no runtime yet).

### Milestone 2 – Workflow Runtime (2–3 weeks)
- Implement graph traversal logic.
- Implement orchestrator loop for a run.
- Implement worker that executes:
  - Simple “agent” node calling a single LLM provider.
  - “tool” node with HTTP calls.
- Implement event logging and run/step status updates.

### Milestone 3 – Frontend Core (2–3 weeks)
- Workflow editor with minimal graph UI (React Diagram library).
- Run list and run detail views.
- Live updates via polling (then upgrade to WebSocket/SSE).

### Milestone 4 – Human-in-the-Loop + Basic Observability (2 weeks)
- Human approval step node.
- UI to show “waiting on human” and submit resolution.
- Timeline visualization of steps with logs.

---

## 10. How to Use This with a Code Model

You can:
- Paste this into `/docs/ARCHITECTURE.md`.
- Then prompt a code model with:
  - “Implement the data models from section 3 using Prisma + Postgres.”
  - “Generate NestJS controllers for the API sketch in section 8.”
  - “Generate React components for the workflow editor described in section 7.”

If you tell me your preferred stack (TypeScript/Nest/Prisma, Python/FastAPI/SQLAlchemy, etc.), I can generate concrete schema definitions, initial migrations, and starter service/controller code next.
