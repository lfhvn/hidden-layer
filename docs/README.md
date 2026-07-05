# Hidden Layer Documentation

Lab-wide documentation for infrastructure, workflows, and conventions.

## Structure

```
docs/
├── FAQ.md              # Common questions and troubleshooting
├── PROJECT_GUIDE.md    # Overview of all projects, how to get started
├── SETUP.md            # Advanced / manual setup
├── ARCHITECTURE.md     # System architecture
├── BENCHMARKS.md       # Benchmark information
├── infrastructure/     # Core systems (LLM providers, model management)
├── hardware/           # Hardware setup (local models - optional)
└── workflows/          # Research processes (benchmarking, notebooks)
```

## Quick Navigation

### Getting Started

- **[Project Guide](PROJECT_GUIDE.md)** - All projects, organized by research area
- **[FAQ](FAQ.md)** - Common questions and troubleshooting
- **[Advanced Setup](SETUP.md)** - Manual environment configuration
- **[Quickstart](../QUICKSTART.md)** - Zero-to-notebook setup (repo root)

### Infrastructure

Core systems used across all projects:

- **[Provider Limitations](infrastructure/provider-limitations.md)** - Provider capabilities and constraints
- **[Model Management](infrastructure/model-management.md)** - Centralized model storage

### Hardware (Optional - Local Models)

- **[Local Setup](hardware/local-setup.md)** - M4 Max configuration
- **[MLX Models](hardware/mlx-models.md)** - Model selection guide

### Workflows

Research processes and best practices:

- **[Benchmarking](workflows/benchmarking.md)** - Using standard benchmarks
- **[Notebook Setup](workflows/notebook-setup.md)** - Provider configuration for notebooks

### Research

- **Research Questions** - See [/RESEARCH.md](../RESEARCH.md)
- **Cross-Project Connections** - See [/RESEARCH.md](../RESEARCH.md)
- **Research Methodology** - See [/CLAUDE.md](../CLAUDE.md) Research Methodology section

---

## Project-Specific Docs

Each project has its own documentation:

**Communication**:
- [Multi-Agent](../communication/multi-agent/CLAUDE.md)
- [AI-to-AI Communication](../communication/ai-to-ai-comm/CLAUDE.md)

**Theory of Mind**:
- [SELPHI](../theory-of-mind/selphi/CLAUDE.md)
- [Introspection](../theory-of-mind/introspection/CLAUDE.md)

**Representations**:
- [Latent Space](../representations/latent-space/CLAUDE.md)

**Alignment**:
- [Steerability](../alignment/steerability/CLAUDE.md)

**Memory**:
- [Lifelog Personalization](../memory/lifelog-personalization/README.md)

**Platform & Tools**:
- [AgentMesh](../agentmesh/README.md) (product docs in `agentmesh/docs/`)
- [MLX Lab](../mlx_lab/README.md)
- [AI Research Aggregator](../ai_research_aggregator/README.md)

---

## For New Developers

1. Start with [RESEARCH.md](../RESEARCH.md) for research overview
2. Read [CLAUDE.md](../CLAUDE.md) for development guide and methodology
3. Optional: Configure hardware: [Local Setup](hardware/local-setup.md)
4. Pick a research area: communication, theory-of-mind, representations, alignment, or memory
5. Read project CLAUDE.md for specific guidance

---

## Contributing to Docs

When adding documentation:

1. **Infrastructure** → Affects all projects
2. **Workflows** → Research processes
3. **Project-specific** → Put in project's own docs

Keep documentation:
- **Clear**: Easy to understand
- **Concise**: No unnecessary detail
- **Current**: Update when things change
- **Connected**: Link to related docs
