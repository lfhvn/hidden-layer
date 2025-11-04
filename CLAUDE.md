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

## Projects

### Active Research Initiatives

1. **Multi-Agent Architecture** → `projects/multi-agent/CLAUDE.md`
   - Coordination strategies: debate, CRIT, XFN teams, consensus
   - Research Q: When and why do multi-agent systems outperform single agents?

2. **SELPHI** → `projects/selphi/CLAUDE.md`
   - Theory of mind evaluation and benchmarking
   - Research Q: How do LLMs understand mental states and perspective-taking?

3. **Latent Space** → `projects/latent-space/CLAUDE.md`
   - **Lens**: SAE interpretability (web app)
   - **Topologies**: Mobile latent space exploration (visual/audio/haptic)
   - Research Q: How can we understand and experience latent representations?

4. **Introspection** → `projects/introspection/CLAUDE.md`
   - Model introspection experiments (Anthropic-style)
   - Concept vectors, activation steering
   - Research Q: Can models accurately report their internal states?

5. **AI-to-AI Communication** → `projects/ai-to-ai-comm/CLAUDE.md`
   - Non-linguistic LLM communication via latent representations
   - Research Q: Can agents communicate more efficiently than through language?

6. **Steerability** → `projects/steerability/CLAUDE.md`
   - Steering vectors, adherence metrics, alignment
   - Research Q: How can we reliably control model behavior?

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

**Documentation**: `docs/infrastructure/`

### Shared Resources

**Location**: `/shared/`

**Includes**:
- `concepts/` - Concept vectors (used by introspection, latent-space)
- `datasets/` - Benchmark datasets
- `utils/` - Common utilities

---

## Development Workflows

### Working on a Project

1. Navigate to project: `cd projects/{project-name}/`
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

**See** `docs/workflows/research-methodology.md` for detailed guidance.

---

## Key Documentation

### Lab-Wide Documentation (`/docs/`)

**Infrastructure**:
- `docs/infrastructure/llm-providers.md` - Provider setup and usage
- `docs/infrastructure/experiment-tracking.md` - Reproducibility
- `docs/infrastructure/provider-limitations.md` - Known issues

**Hardware** (optional):
- `docs/hardware/local-setup.md` - M4 Max setup
- `docs/hardware/mlx-models.md` - MLX model selection

**Workflows**:
- `docs/workflows/research-methodology.md` - Research process
- `docs/workflows/benchmarking.md` - Benchmark usage
- `docs/workflows/reproducibility.md` - Experiment logging

**Conventions**:
- `docs/conventions/coding-standards.md` - Code style
- `docs/conventions/naming-conventions.md` - Naming patterns

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
- Can we navigate latent space to steer behavior? (latent-topologies + steerability)

**Alignment**:
- Can we steer ToM behavior? (steerability + selphi)
- Is introspection a reliable alignment signal? (introspection + alignment)

---

## File Organization

```
hidden-layer/
├── harness/              # Core infrastructure (standalone library)
├── shared/               # Shared resources (concepts, datasets, utils)
├── projects/             # Research projects
│   ├── multi-agent/      # Multi-agent coordination
│   ├── selphi/           # Theory of mind
│   ├── latent-space/     # Latent representations
│   ├── introspection/    # Model introspection
│   ├── ai-to-ai-comm/    # Non-linguistic communication
│   └── steerability/     # Steering & alignment
├── docs/                 # Lab-wide documentation
├── README.md             # Lab overview
├── RESEARCH.md           # Research themes & connections
└── CLAUDE.md             # This file
```

---

**For project-specific guidance**: See `projects/{project}/CLAUDE.md`

**For research context**: See `RESEARCH.md`

**For infrastructure details**: See `docs/infrastructure/`
