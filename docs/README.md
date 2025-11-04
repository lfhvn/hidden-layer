# Hidden Layer Documentation

Lab-wide documentation for infrastructure, workflows, and conventions.

## Structure

```
docs/
├── infrastructure/     # Core systems (LLM providers, tracking, etc.)
├── hardware/          # Hardware setup (local models - optional)
├── workflows/         # Research processes
├── conventions/       # Coding standards
└── research/          # Research methodology
```

## Quick Navigation

### Infrastructure

Core systems used across all projects:

- **[LLM Providers](infrastructure/llm-providers.md)** - Setup and usage (Ollama, MLX, Claude, GPT)
- **[Experiment Tracking](infrastructure/experiment-tracking.md)** - Reproducibility
- **[Provider Limitations](infrastructure/provider-limitations.md)** - Known issues

### Hardware (Optional - Local Models)

- **[Local Setup](hardware/local-setup.md)** - M4 Max configuration
- **[MLX Models](hardware/mlx-models.md)** - Model selection guide

### Workflows

Research processes and best practices:

- **[Research Methodology](workflows/research-methodology.md)** - Frame → Theory → Implement → Experiment → Synthesize
- **[Benchmarking](workflows/benchmarking.md)** - Using standard benchmarks
- **[Reproducibility](workflows/reproducibility.md)** - Experiment logging

### Conventions

- **[Coding Standards](conventions/coding-standards.md)** - Python style guide
- **[Naming Conventions](conventions/naming-conventions.md)** - Files, variables, etc.

### Research

- **Research Questions** - See [/RESEARCH.md](../RESEARCH.md)
- **Cross-Project Connections** - See [/RESEARCH.md](../RESEARCH.md)

---

## Project-Specific Docs

Each project has its own documentation:

- [Multi-Agent](../projects/multi-agent/CLAUDE.md)
- [SELPHI](../projects/selphi/CLAUDE.md)
- [Latent Space](../projects/latent-space/CLAUDE.md)
- [Introspection](../projects/introspection/CLAUDE.md)
- [AI-to-AI Communication](../projects/ai-to-ai-comm/CLAUDE.md)
- [Steerability](../projects/steerability/CLAUDE.md)

---

## For New Developers

1. Start with [Research Methodology](workflows/research-methodology.md)
2. Set up infrastructure: [LLM Providers](infrastructure/llm-providers.md)
3. Optional: Configure hardware: [Local Setup](hardware/local-setup.md)
4. Pick a project: [/projects/](../projects/)
5. Read project CLAUDE.md for specific guidance

---

## Contributing to Docs

When adding documentation:

1. **Infrastructure** → Affects all projects
2. **Workflows** → Research processes
3. **Conventions** → Coding standards
4. **Project-specific** → Put in project's own docs

Keep documentation:
- **Clear**: Easy to understand
- **Concise**: No unnecessary detail
- **Current**: Update when things change
- **Connected**: Link to related docs
