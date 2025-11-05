# Hidden Layer Documentation

Lab-wide documentation for infrastructure, workflows, and conventions.

## Structure

```
docs/
├── infrastructure/     # Core systems (LLM providers, tracking, etc.)
├── hardware/          # Hardware setup (local models - optional)
├── workflows/         # Research processes
└── archive/           # Historical documentation
```

## Quick Navigation

### Infrastructure

Core systems used across all projects:

- **[Provider Limitations](infrastructure/provider-limitations.md)** - Provider capabilities and constraints

### Hardware (Optional - Local Models)

- **[Local Setup](hardware/local-setup.md)** - M4 Max configuration
- **[MLX Models](hardware/mlx-models.md)** - Model selection guide

### Workflows

Research processes and best practices:

- **[Benchmarking](workflows/benchmarking.md)** - Using standard benchmarks

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

---

## For New Developers

1. Start with [RESEARCH.md](../RESEARCH.md) for research overview
2. Read [CLAUDE.md](../CLAUDE.md) for development guide and methodology
3. Optional: Configure hardware: [Local Setup](hardware/local-setup.md)
4. Pick a research area: communication, theory-of-mind, representations, or alignment
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
