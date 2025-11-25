# Claude Development Guide - Hidden Layer Lab

## Lab Identity

**Hidden Layer** is an independent research lab investigating:
- Agent communication & coordination
- Theory of mind & self-knowledge
- Internal representations & interpretability
- Alignment, steerability & deception

## Research Philosophy

**Orientation**: foundations → theory → implementation → experiment → synthesis

**Guiding Principles**:
- **Radical Curiosity**: Question everything, even the question
- **Theoretical Discipline**: Every claim connects to measurable evidence
- **Paradigm Awareness**: Understand frameworks, then leap beyond them
- **Architectural Creativity**: Design systems that could discover new science
- **Empirical Elegance**: Simple mechanisms → emergent complexity

---

## Research Areas & Projects

Research is organized by thematic areas, with projects grouped by their primary focus:

### 1. Communication (`communication/`)
How do multiple AI agents communicate and coordinate?

**Projects**:
- **Multi-Agent** → `communication/multi-agent/CLAUDE.md`
  - Coordination strategies: debate, CRIT, XFN teams, consensus
  - Research Q: When and why do multi-agent systems outperform single agents?

- **AI-to-AI Communication** → `communication/ai-to-ai-comm/CLAUDE.md`
  - Non-linguistic LLM communication via latent representations
  - Research Q: Can agents communicate more efficiently than through language?

### 2. Theory of Mind (`theory-of-mind/`)
How do AI systems understand mental states (self and others)?

**Projects**:
- **SELPHI** → `theory-of-mind/selphi/CLAUDE.md`
  - Theory of mind evaluation and benchmarking
  - Research Q: How do LLMs understand mental states and perspective-taking?

- **Introspection** → `theory-of-mind/introspection/CLAUDE.md`
  - Model introspection experiments (Anthropic-style)
  - Concept vectors, activation steering
  - Research Q: Can models accurately report their internal states?

### 3. Representations (`representations/`)
What are internal representations and how can we make them interpretable?

**Projects**:
- **Latent Space** → `representations/latent-space/CLAUDE.md`
  - **Lens**: SAE interpretability (web app)
  - **Topologies**: Mobile latent space exploration (visual/audio/haptic)
  - Research Q: How can we understand and experience latent representations?

- **State Explorer** → `representations/state-explorer/`
  - LLM internal state visualization and exploration
  - Research Q: How can we visualize and navigate model states?

### 4. Alignment (`alignment/`)
How can we reliably steer AI systems and detect deception?

**Projects**:
- **Steerability** → `alignment/steerability/CLAUDE.md`
  - Steering vectors, adherence metrics, alignment
  - Research Q: How can we reliably control model behavior?

### 5. Memory (`memory/`)
How can AI systems maintain long-term memory and personalization?

**Projects**:
- **Lifelog Personalization** → `memory/lifelog-personalization/README.md`
  - Long-term memory evaluation and lifelog retrieval
  - Preference-aware personalization and adapter promotion gates
  - Research Q: How can systems maintain coherent long-term context?

---

## Platform & Tools

### AgentMesh (`agentmesh/`)
**Multi-agent workflow orchestration platform built on Hidden Layer research**

- Visual workflow design and execution
- Wraps research strategies (debate, CRIT, consensus) in production-ready API
- Persistent state management with Postgres/Redis
- REST API with FastAPI

**Status**: Product spinoff - commercial SaaS offering
**Documentation**: `agentmesh/README.md`, `agentmesh/QUICKSTART.md`

### MLX Lab (`mlx_lab/`)
**CLI tool for local MLX model management and research**

- Model management and downloading
- Performance benchmarking
- Concept browser for SAE features
- Integrated with Hidden Layer workflows

**Usage**: `mlx-lab` CLI command (after `pip install -e .`)

### Academic Papers (`papers/`)
**LaTeX sources for research publications**

Papers documenting the research:
- `multi-agent-coordination.tex` - Multi-agent strategies
- `ai-to-ai-communication.tex` - Non-linguistic communication
- `selphi-theory-of-mind.tex` - Theory of mind evaluation
- `model-introspection.tex` - Introspection experiments
- `latent-lens-sae.tex` - SAE interpretability
- `latent-topologies-multimodal.tex` - Multimodal latent exploration
- `steerability-adherence.tex` - Steering and alignment

---

### Research Theme Connections

Projects are deeply interconnected:

**Communication**:
- Multi-agent + AI-to-AI comm → Agent coordination mechanisms

**Theory of Mind**:
- SELPHI (understanding others) + Introspection (understanding self)

**Representations**:
- Latent Lens + Latent Topologies + Introspection → Making sense of internal states

**Alignment**:
- SELPHI + Introspection + Steerability → Honest, controllable systems

**See** `RESEARCH.md` for detailed research questions and cross-project connections.

---

## Infrastructure

### The Harness (Standalone Library)

**Location**: `/harness/`

**Purpose**: Core infrastructure used by all research projects. Can be open-sourced independently.

**Provides**:
- Unified LLM provider abstraction
- Experiment tracking & reproducibility
- Evaluation utilities
- Benchmark dataset loading
- Model configuration management
- System prompt management

**Philosophy**: Provider-agnostic. Supports:
- **Local**: Ollama, MLX (rapid iteration, full control)
- **API**: Claude, GPT, etc. (frontier capabilities)

Switch providers seamlessly:
```python
from harness import llm_call

# Local
response = llm_call(prompt, provider="ollama", model="llama3.2:latest")

# API
response = llm_call(prompt, provider="anthropic", model="claude-3-5-sonnet-20241022")
```

**Documentation**: See `harness/README.md` and `docs/infrastructure/provider-limitations.md`

### Shared Resources

**Location**: `/shared/`

**Includes**:
- `concepts/` - Concept vectors (used by introspection, latent-space)
- `datasets/` - Benchmark datasets
- `utils/` - Common utilities

---

## Development Workflows

### Working on a Project

1. Navigate to project: `cd {area}/{project-name}/` (e.g., `cd communication/multi-agent/`)
2. Read project CLAUDE.md: `cat CLAUDE.md`
3. Follow project-specific setup and instructions

### Adding Cross-Project Features

If a feature benefits multiple projects:
1. Add to `harness/` (if core infrastructure) or `shared/utils/` (if utility)
2. Update `docs/infrastructure/`
3. Update relevant project guides

### Research Methodology

1. **Frame the Problem**
   - Restate in first principles
   - What paradigm does this challenge?
   - What hidden assumptions exist?

2. **Decompose & Theorize**
   - Identify constraints and untested assumptions
   - Generate multiple approaches
   - Could this be done fundamentally differently?

3. **Design & Implement**
   - Simple, interpretable mechanisms
   - Easy to probe and inspect
   - Design for extensibility

4. **Experiment**
   - Log everything (use harness experiment tracker)
   - Reproducible experiments
   - Compare across conditions

5. **Synthesize & Reflect**
   - What did this reveal?
   - Does this generalize?
   - What new questions does this enable?

See the Research Methodology section above for detailed guidance.

---

## Key Documentation

### Lab-Wide Documentation (`/docs/`)

**Infrastructure**:
- `harness/README.md` - Harness usage and API reference
- `docs/infrastructure/provider-limitations.md` - Provider capabilities and constraints

**Hardware** (optional - for local models):
- `docs/hardware/local-setup.md` - M4 Max setup
- `docs/hardware/mlx-models.md` - MLX model selection

**Workflows**:
- `docs/workflows/benchmarking.md` - Benchmark usage

**Architecture & Planning**:
- `docs/ARCHITECTURE.md` - System architecture
- `docs/BENCHMARKS.md` - Benchmark information

### Project-Specific Documentation

Each project has:
- `README.md` - Project overview and quick start
- `CLAUDE.md` - Development guide for that project
- Additional docs as needed

---

## Development Principles

1. **Maintain Backward Compatibility**: Projects depend on harness APIs
2. **Log Everything**: Use experiment tracker for all runs
3. **Document Decisions**: Update relevant .md files
4. **Test with Small Models First**: Rapid iteration with local models
5. **Version Control Configs**: Commit model configs and prompts
6. **Theoretical Discipline**: Every feature should:
   - Enable a new research hypothesis, OR
   - Make existing research faster/easier/more reproducible
7. **Architectural Creativity**: Question existing patterns

---

## Questions to Keep in Mind

While developing, constantly ask:

### Technical Level
1. Does this maintain the flexible infrastructure (local + API)?
2. Is this reproducible and logged?
3. Does this generalize across projects?
4. Is this interpretable and inspectable?

### Paradigm Level
5. What hidden assumptions am I encoding?
6. Could this work fundamentally differently?
7. Does this enable testing new hypotheses?
8. What would falsify this approach?

### Research Impact
9. Will this help understand *why*, not just *that*?
10. Does this make internal states more visible?
11. Could this generalize to biological/social intelligence?
12. What new questions does this unlock?

---

## Integration Points

When working across projects, consider:

**Communication**:
- Can multi-agent strategies use latent messaging? (multi-agent + ai-to-ai-comm)
- What coordination mechanisms emerge? (multi-agent)

**Theory of Mind**:
- Can SELPHI tasks measure introspection honesty? (selphi + introspection)
- How does ToM relate to deception? (selphi + alignment)

**Representations**:
- What features activate during ToM tasks? (latent-lens + selphi)
- Can we navigate latent space to steer behavior? (representations/latent-space/topologies + steerability)

**Alignment**:
- Can we steer ToM behavior? (steerability + selphi)
- Is introspection a reliable alignment signal? (introspection + alignment)

---

## File Organization

```
hidden-layer/
├── harness/                    # Core infrastructure (standalone library)
├── shared/                     # Shared resources (concepts, datasets, utils)
├── web-tools/                  # Deployment versions of web applications
│
├── communication/              # Research Area: Agent Communication
│   ├── multi-agent/           # Multi-agent coordination (project files)
│   ├── multi_agent/           # Python package (importable)
│   ├── ai-to-ai-comm/         # AI-to-AI communication (project files)
│   └── ai_to_ai_comm/         # Python package (importable)
│
├── theory-of-mind/             # Research Area: Theory of Mind & Self-Knowledge
│   ├── selphi/                # Theory of mind evaluation
│   └── introspection/         # Model introspection
│
├── theory_of_mind/             # Python package wrapper for theory-of-mind
│
├── representations/            # Research Area: Internal Representations
│   ├── latent-space/
│   │   ├── lens/              # SAE interpretability
│   │   └── topologies/        # Mobile latent exploration
│   └── state-explorer/        # LLM state visualization
│
├── alignment/                  # Research Area: Alignment & Steerability
│   └── steerability/          # Steering vectors & metrics
│
├── memory/                     # Research Area: Long-term Memory
│   └── lifelog-personalization/  # Lifelog retrieval & preference
│
├── agentmesh/                  # Platform: Workflow orchestration
├── mlx_lab/                    # Tool: MLX model management CLI
├── papers/                     # Academic paper sources (.tex)
│
├── docs/                       # Lab-wide documentation
├── tests/                      # Lab-wide tests
├── config/                     # Lab-wide configuration
├── scripts/                    # Utility scripts
│
├── README.md                   # Lab overview
├── RESEARCH.md                 # Research themes & connections
├── CLAUDE.md                   # This file (development guide)
├── QUICKSTART.md               # Quick start guide
└── SETUP.md                    # Setup instructions
```

---

## Directory Naming Convention

**Important**: The codebase uses a dual-directory pattern for Python compatibility:

| Purpose | Naming | Example |
|---------|--------|---------|
| Project files (README, config, notebooks) | Dash-separated | `multi-agent/` |
| Python packages (importable code) | Underscore-separated | `multi_agent/` |

**Why?** Python identifiers cannot contain dashes. Research projects use dash-naming for readability, but Python imports require underscores.

**How to use**:
```python
# Import from underscore-named packages
from communication.multi_agent import strategies
from theory_of_mind.selphi import scenarios

# Navigate to dash-named directories for project context
# cd communication/multi-agent/  # README, CLAUDE.md, notebooks here
```

**For new projects**:
1. Create dash-named directory for project files: `myarea/my-project/`
2. Create underscore-named directory for Python code: `myarea/my_project/`
3. Actual code can live in either location; the underscore package loads it

---

### Key Organizational Principles

1. **Research Areas at Top Level**: Projects grouped by thematic focus
2. **Infrastructure at Root**: `harness/`, `shared/`, `web-tools/` for easy imports
3. **Scalable Structure**: Easy to add new areas or projects within areas
4. **No Breaking Changes**: Import paths unchanged (`from harness import ...`)
5. **Dual-Directory Pattern**: Dash for docs, underscore for imports

### Working with Research Areas

**Navigate to an area**:
```bash
cd communication/     # or theory-of-mind/, representations/, alignment/, memory/
```

**Each area contains**:
- `README.md` - Area overview, research questions, cross-connections
- Project subdirectories with their own CLAUDE.md and README.md

**For new research directions**: Create a new top-level directory (e.g., `emergent-behavior/`)

**For new projects within an area**: Add subdirectory (e.g., `communication/swarm-intelligence/`)

---

**For project-specific guidance**: See `{area}/{project}/CLAUDE.md`

**For research area overview**: See `{area}/README.md`

**For research context**: See `RESEARCH.md`

**For infrastructure details**: See `docs/infrastructure/`
