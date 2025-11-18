# AgentMesh Roles & Capabilities - Extended Assessment

**Date**: 2025-11-18
**Branch**: `claude/assess-agentmesh-01YZQpGBabjiNxb5qwi6rUMe`
**Context**: Extension to core AgentMesh assessment with roles/permissions/skills architecture

---

## Overview

This document assesses the **Roles & Capabilities** extension to AgentMesh, which adds:
- **Role Templates** - Behavioral archetypes (planner, researcher, reviewer, etc.)
- **Skills** - Abstract capabilities (tools, workflows, MCP integrations)
- **Permission Policies** - Fine-grained access control (RBAC/ABAC)

---

## Architecture Analysis

### Core Innovation: Three Orthogonal Axes

```
Agent = RoleTemplate + Skills + Permissions + ModelConfig + PromptOverrides
```

This cleanly separates:
1. **What kind of agent** (role: planner, researcher, coder)
2. **What it can do** (skills: tools, workflows, MCP)
3. **What it's allowed to do** (permissions: read/write/execute)

**This is excellent design** - modular, composable, extensible.

---

## Comparison to Hidden Layer's Approach

### Hidden Layer Multi-Agent (Current)

**Agent Definition**: Implicit in strategy code
```python
def debate(task_input, n_debaters=3, **kwargs):
    # All agents are identical
    # No roles, no differentiated capabilities
    # No permission system
    for i in range(n_debaters):
        response = llm_call(task_input, **kwargs)
    # ...
```

**Characteristics**:
- ❌ No role differentiation
- ❌ No permission system
- ❌ All agents have same capabilities
- ✅ Simple and fast to implement
- ✅ No overhead for research prototypes

### AgentMesh Roles & Capabilities

**Agent Definition**: Explicit configuration
```typescript
const agent = {
  roleTemplateId: "researcher",
  skillIds: ["web_search", "knowledge_base"],
  permissionPolicyId: "read_only_web",
  modelProvider: "anthropic",
  // ...
}
```

**Characteristics**:
- ✅ Explicit role differentiation
- ✅ Fine-grained permissions
- ✅ Skills as first-class abstraction
- ❌ Complex to set up initially
- ❌ Higher overhead for simple cases

---

## Role Templates - Research Value

### Proposed Roles in Spec

1. **Planner** - Break down tasks, orchestrate
2. **Researcher** - Information retrieval
3. **Coder** - Code editing, testing
4. **Reviewer** - Quality control, critique
5. **Router** - Skill/agent selection
6. **Safety Guard** - Safety/compliance
7. **Memory Manager** - State management

### Hidden Layer Research Implications

**Question**: Do role-specialized agents outperform generalist agents?

This is actually a **testable hypothesis** that aligns with Hidden Layer's research:

**Hypothesis 1**: Role-specialized prompts improve coordination
- Test: Compare "debaters" vs. "researcher + reviewer + synthesizer"
- Metric: Answer quality, coverage, diversity

**Hypothesis 2**: Explicit roles reduce prompt engineering burden
- Test: Role template vs. hand-crafted prompts per task
- Metric: Performance consistency across tasks

**Hypothesis 3**: Permissions prevent harmful agent behavior
- Test: Safety guard effectiveness, unauthorized action attempts
- Metric: Safety violations, alignment metrics

**This could be a new research area for Hidden Layer!**

---

## Skills Abstraction - Integration Points

### Skill Types
1. **tool** - HTTP calls, functions
2. **workflow** - Nested AgentMesh workflows
3. **mcp** - Model Context Protocol servers

### Hidden Layer Integration Opportunities

#### 1. SELPHI as a Skill
```typescript
{
  id: "selphi_tom_eval",
  name: "Theory of Mind Evaluation",
  type: "workflow",
  binding: {
    workflowId: "selphi_false_belief_test"
  },
  description: "Evaluate agent's theory of mind capabilities"
}
```

**Use Case**: Multi-agent system where one agent evaluates another's ToM

#### 2. Introspection as a Skill
```typescript
{
  id: "introspection_steer",
  name: "Activation Steering",
  type: "tool",
  binding: {
    toolId: "concept_vector_steering"
  },
  description: "Steer agent toward concept (happiness, analytical, etc.)"
}
```

**Use Case**: Dynamic agent personality adjustment mid-workflow

#### 3. Latent Communication as a Skill
```typescript
{
  id: "latent_message",
  name: "Latent Space Messaging",
  type: "mcp",
  binding: {
    mcpServerId: "ai_to_ai_comm",
    mcpToolName: "encode_latent_message"
  },
  description: "Send non-linguistic message via embeddings"
}
```

**Use Case**: Test whether agents coordinate better via latent vs. language

#### 4. Harness Provider as Skill
```typescript
{
  id: "switch_provider",
  name: "Change LLM Provider",
  type: "tool",
  binding: {
    toolId: "harness_provider_switch"
  },
  description: "Switch between Ollama/MLX/Claude/GPT mid-workflow"
}
```

**Use Case**: Adaptive workflows that choose provider based on task

---

## Permission Policies - Research & Safety

### Permission System Design

```typescript
interface PermissionPolicy {
  rules: PermissionRule[];
}

interface PermissionRule {
  resourceType: 'workflow' | 'file' | 'external_http' | ...;
  resourceScope: string;  // e.g., "project:123:files:/workspace/**"
  actions: ['read', 'write', 'execute', ...];
  constraints?: {
    maxDollarCost?: number;
    allowedDomains?: string[];
  };
}
```

### Research Applications

#### 1. Alignment Research
**Question**: Can permission systems prevent misaligned behavior?

**Test Setup**:
- Agent with goal to "maximize output quality by any means"
- Permission policy that restricts certain actions
- Measure: circumvention attempts, goal achievement vs. safety

**Connects to**: Hidden Layer's alignment research area

#### 2. Deception Detection
**Question**: Do agents attempt to bypass permissions deceptively?

**Test Setup**:
- Safety guard agent monitors other agents
- Restricted agent attempts to achieve goal
- Measure: deceptive requests, permission violations

**Connects to**: SELPHI deception detection research

#### 3. Cost Control
**Constraint**: `maxDollarCost` per agent

**Research Value**:
- Optimize multi-agent workflows under budget constraints
- Study agent behavior when cost-limited
- Tradeoff analysis: cost vs. quality

---

## MCP Integration - Ecosystem Alignment

### What is MCP?
**Model Context Protocol** - Anthropic's standard for connecting LLMs to tools/data

### Why This Matters for Hidden Layer

1. **Ecosystem Compatibility**
   - Hidden Layer harness is provider-agnostic
   - MCP is becoming industry standard
   - Integration future-proofs infrastructure

2. **Tool Ecosystem**
   - Access to MCP server library (filesystem, git, databases, etc.)
   - Don't reinvent tools
   - Focus on research, not infrastructure

3. **Potential Research Direction**
   - "MCP for Research" - tools optimized for AI research
   - Benchmarking toolkit as MCP server
   - Introspection tools as MCP server

### Integration with Harness

**Option**: Hidden Layer Harness as MCP Server

```python
# harness/mcp_server.py

from mcp import MCPServer

class HarnessTools(MCPServer):
    @tool
    def run_strategy(self, strategy: str, task: str, **kwargs):
        """Run Hidden Layer multi-agent strategy"""
        from communication.multi_agent import run_strategy
        return run_strategy(strategy, task, **kwargs)

    @tool
    def evaluate_tom(self, prompt: str, scenario: str):
        """Run SELPHI theory of mind evaluation"""
        from theory_of_mind.selphi import run_scenario
        return run_scenario(scenario, prompt)

    @tool
    def apply_steering(self, prompt: str, concept: str, strength: float):
        """Apply concept vector steering"""
        from theory_of_mind.introspection import steer
        return steer(prompt, concept, strength)

# Expose Hidden Layer research tools to any MCP-compatible agent
server = HarnessTools()
server.serve()
```

**This would make Hidden Layer tools available to any system using MCP!**

---

## Prompt Templates - Research Reproducibility

### System Prompt Resolution

```
Final Prompt =
  RoleTemplate.systemPromptTemplate (with {{placeholders}})
  + Organization guardrails
  + Project context
  + Agent overrides
```

### Hidden Layer Benefit: Reproducibility

**Current State**: System prompts in `config/system_prompts/`
- Manually managed
- No templating
- Hard to vary systematically

**With Prompt Templates**:
- Systematic variation of prompts
- A/B testing role definitions
- Reproducible experiments across role variations

**Research Value**:
- "How sensitive are results to role prompt variations?"
- "What role definitions optimize coordination?"
- "Can we discover better roles via optimization?"

---

## Complexity Cost Analysis

### Current Hidden Layer
```python
# Define a new "role" in existing system
def researcher_agent(task, **kwargs):
    prompt = f"You are a researcher. {task}"
    return llm_call(prompt, **kwargs)
```
**Lines of code**: 2
**Complexity**: Minimal

### AgentMesh Roles & Capabilities
```typescript
// 1. Define role template
const roleTemplate = {
  id: "researcher",
  systemPromptTemplate: "...",
  defaultSkillIds: ["web_search"],
  defaultPermissionPolicyId: "read_only",
};

// 2. Define skills
const skill = {
  id: "web_search",
  type: "mcp",
  binding: { mcpServerId: "web", mcpToolName: "search" },
};

// 3. Define permission policy
const policy = {
  id: "read_only",
  rules: [{ resourceType: "external_http", actions: ["read"] }],
};

// 4. Create agent
const agent = {
  roleTemplateId: "researcher",
  skillIds: ["web_search"],
  permissionPolicyId: "read_only",
};

// 5. Use in workflow
const workflow = { nodes: [{ type: "agent", agentId: agent.id }] };
```
**Lines of code**: 40+
**Complexity**: High
**Setup time**: 30+ minutes for first agent

---

## When Roles & Capabilities Pay Off

### Pays Off When:
1. ✅ **Deploying to production** - Need security and permissions
2. ✅ **Multi-role workflows** - 5+ different agent types
3. ✅ **Reusing configurations** - Same roles across projects
4. ✅ **Safety-critical** - Need to restrict agent actions
5. ✅ **Team collaboration** - Multiple people designing agents

### Doesn't Pay Off When:
1. ❌ **Rapid prototyping** - Want to test hypothesis in 5 minutes
2. ❌ **Simple workflows** - 1-2 agents doing similar things
3. ❌ **One-off experiments** - Won't reuse configuration
4. ❌ **Research focus** - Testing coordination, not deployment
5. ❌ **Solo researcher** - No need for permission boundaries

**Hidden Layer is currently in the "doesn't pay off" category.**

---

## Research Opportunities Unlocked

If Hidden Layer implemented AgentMesh with Roles & Capabilities:

### New Research Directions

1. **Role Specialization vs. Generalist Agents**
   - Compare performance of role-specialized vs. general agents
   - Measure: quality, latency, cost, error rates

2. **Permission Constraints & Alignment**
   - Study agent behavior under permission constraints
   - Measure: goal achievement, circumvention attempts, safety

3. **Deception & Role-Playing**
   - Do agents "pretend" to be their role or internalize it?
   - Connect to SELPHI theory of mind research

4. **Emergent Role Specialization**
   - Start with general agents, let them specialize
   - Measure: task allocation patterns, efficiency gains

5. **Cross-Functional Teams (XFN)**
   - Implement XFN teams with explicit roles
   - Compare to current ad-hoc multi-agent strategies

6. **Skill Discovery**
   - Can agents learn what skills exist and when to use them?
   - Meta-learning over skill library

### Extensions to Existing Research

1. **SELPHI + Roles**
   - Does role specialization affect theory of mind performance?
   - Do "reviewer" agents show better perspective-taking?

2. **Introspection + Permissions**
   - Can agents introspect on their permission boundaries?
   - Do permission constraints affect internal representations?

3. **AI-to-AI Comm + Skills**
   - Latent communication as a skill
   - Compare language vs. latent coordination with explicit roles

---

## Integration Recommendation

### Phased Approach

#### Phase 0: Current State (Keep)
```python
# Simple strategy functions
from communication.multi_agent import run_strategy
result = run_strategy('debate', task)
```
**Use for**: Core research, prototyping, papers

#### Phase 1: Lightweight Roles (2 weeks)
```python
# Add role parameter to strategies
from communication.multi_agent import run_strategy

result = run_strategy(
    'debate',
    task,
    agent_roles=['researcher', 'reviewer', 'synthesizer']
)
```
**Implementation**: Just prompt templates, no infrastructure
**Use for**: Role specialization research

#### Phase 2: Skills Library (2-3 weeks)
```python
# Harness exposes tools as "skills"
from harness import Skill

skill = Skill.from_tool('web_search', provider='mcp', ...)
agent.add_skill(skill)
```
**Implementation**: Skills abstraction in harness
**Use for**: Testing skill composition, MCP integration

#### Phase 3: Permission Policies (2 weeks)
```python
# Add safety constraints to agents
from harness import PermissionPolicy

policy = PermissionPolicy(
    allow=['read'],
    deny=['external_http'],
    max_cost_usd=1.0
)
agent.set_policy(policy)
```
**Implementation**: Enforcement in harness
**Use for**: Alignment research, safety studies

#### Phase 4: Full AgentMesh (8+ weeks)
**Only if phases 1-3 prove valuable**
- Complete roles/skills/permissions system
- Visual workflow editor
- Production deployment features

---

## Specific Recommendations

### 1. Start with Role Templates Only

**Implement**: Simple role prompt templates in harness

```python
# harness/roles.py

ROLE_TEMPLATES = {
    'researcher': {
        'system_prompt': 'You are a researcher. Your job is to...',
        'temperature': 0.1,
    },
    'reviewer': {
        'system_prompt': 'You are a reviewer. Your job is to...',
        'temperature': 0.3,
    },
    # ...
}

# Usage
from harness import llm_call, ROLE_TEMPLATES

response = llm_call(
    task,
    role='researcher',
    **ROLE_TEMPLATES['researcher']
)
```

**Cost**: 1-2 days
**Benefit**: Test if role specialization helps
**Risk**: Minimal

---

### 2. Add MCP Integration to Harness

**Implement**: MCP provider in harness

```python
# harness/llm_provider.py

class MCPProvider(LLMProvider):
    def call_tool(self, server_id, tool_name, input):
        # Call MCP server
        return mcp_client.call(server_id, tool_name, input)

# Usage
from harness import llm_call

response = llm_call(
    task,
    provider='mcp',
    tools=['web_search', 'code_interpreter']
)
```

**Cost**: 1 week
**Benefit**: Access to MCP ecosystem
**Risk**: Low - additive, doesn't break existing code

---

### 3. Expose Hidden Layer as MCP Server

**Implement**: MCP server wrapper for HL tools

```python
# harness/mcp_server.py

@mcp_tool
def run_multi_agent_strategy(strategy: str, task: str):
    """Run Hidden Layer multi-agent strategy"""
    return run_strategy(strategy, task)

@mcp_tool
def evaluate_theory_of_mind(scenario: str, prompt: str):
    """Run SELPHI ToM evaluation"""
    return run_scenario(scenario, prompt)
```

**Cost**: 1 week
**Benefit**: HL tools available to any MCP client
**Risk**: Low - exposes existing functionality

---

### 4. Don't Build Full Permission System Yet

**Reason**: Unclear research value for current work

**Wait until**:
- You need to deploy agents to untrusted environments
- You're studying alignment under constraints
- You have safety-critical workflows

**Instead**: Add simple cost limits to harness

```python
# harness/experiment_tracker.py

class CostGuard:
    def __init__(self, max_cost_usd):
        self.max_cost = max_cost_usd
        self.spent = 0.0

    def check(self, estimated_cost):
        if self.spent + estimated_cost > self.max_cost:
            raise CostLimitExceeded()
        self.spent += estimated_cost
```

**Cost**: 1 day
**Benefit**: 80% of permission value for research
**Risk**: None

---

## Research Experiment Design

If you want to test roles/skills/permissions as research questions:

### Experiment 1: Role Specialization

**Hypothesis**: Role-specialized agents outperform generalists in multi-agent coordination

**Setup**:
- Task: Complex reasoning problems (MMLU, BigBench)
- Condition A: 3 general agents (current debate strategy)
- Condition B: 3 role-specialized agents (researcher, reviewer, synthesizer)
- Measure: accuracy, coverage, cost

**Implementation**: Phase 1 only (role templates)

---

### Experiment 2: Skill Composition

**Hypothesis**: Agents with explicit skills perform better than agents with implicit capabilities

**Setup**:
- Task: Multi-step problems requiring tools (web search, calculation, code)
- Condition A: Agent with access to all tools (current)
- Condition B: Agent with explicit skill library + skill selection prompt
- Measure: task success rate, tool usage efficiency

**Implementation**: Phase 2 (skills library)

---

### Experiment 3: Permission Constraints

**Hypothesis**: Agents achieve goals differently under permission constraints

**Setup**:
- Task: Open-ended task (e.g., "research and summarize topic")
- Condition A: No constraints
- Condition B: Read-only permissions (no external writes)
- Condition C: Cost-limited (max $0.10)
- Measure: goal achievement, constraint violations, strategy adaptation

**Implementation**: Phase 3 (permission policies)

---

## Conclusion

### AgentMesh Roles & Capabilities: High-Quality Design

The proposed architecture is **well-designed, modular, and production-ready**. The separation of roles, skills, and permissions is clean and extensible.

### But: Over-Engineered for Current Research Needs

Hidden Layer's current research does not require:
- Production-grade permission systems
- Complex role hierarchies
- MCP integration (yet)
- Visual workflow editors

### Recommended Path

1. **Keep existing multi-agent infrastructure** for core research
2. **Add lightweight role templates** to harness (1-2 days)
3. **Add MCP support** to harness as provider (1 week)
4. **Expose HL tools as MCP server** for ecosystem integration (1 week)
5. **Run experiments** to test if roles/skills improve coordination
6. **Only then** consider full AgentMesh if experiments validate need

### Research Opportunities

The roles/skills/permissions model enables interesting research:
- Role specialization in multi-agent systems
- Permission constraints and alignment
- Skill composition and tool use
- Deception under role-playing

These align with Hidden Layer's research areas and could warrant investigation.

### Critical Question

**Before building AgentMesh**: Can you run these experiments with lightweight extensions to existing harness?

If yes → extend harness incrementally
If no → need AgentMesh

**My assessment: Yes, you can extend harness for these experiments.**

---

## Next Steps

1. **Decide**: Is full AgentMesh needed or extend harness?
2. **If extend harness**:
   - Implement role templates (1-2 days)
   - Test role specialization hypothesis
   - Add MCP support if needed
3. **If build AgentMesh**:
   - Start with Phase 1 POC (4 weeks)
   - Validate with user studies
   - Reassess before full build

---

**Assessment Complete: Roles & Capabilities Extension**

See also:
- Core AgentMesh assessment: `docs/AGENTMESH_ASSESSMENT.md`
- Multi-agent research: `communication/multi-agent/CLAUDE.md`
- Research philosophy: `CLAUDE.md`
