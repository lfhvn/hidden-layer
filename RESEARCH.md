# Hidden Layer - Research Overview

## Mission

Hidden Layer is an independent research lab investigating fundamental questions about AI systems, focusing on understanding how AI agents communicate, represent knowledge internally, and can be aligned with human values.

## Research Themes

### 1. Agent Communication & Coordination

**Core Question**: How do multiple AI agents communicate and coordinate to solve problems more effectively than single agents?

**Projects**:
- **Multi-Agent Architecture** (`projects/multi-agent/`)
  - Debate, CRIT, XFN team strategies
  - When and why do multi-agent systems outperform single agents?

- **AI-to-AI Communication** (`projects/ai-to-ai-comm/`)
  - Non-linguistic communication via latent representations
  - Can LLMs communicate more efficiently through internal states?

**Open Questions**:
- What coordination mechanisms emerge from multi-agent interaction?
- How can agents leverage latent messaging instead of natural language?
- What are the efficiency vs. quality tradeoffs?

---

### 2. Theory of Mind & Self-Knowledge

**Core Question**: How do AI systems understand mental states - both their own (introspection) and others' (theory of mind)?

**Projects**:
- **SELPHI** (`projects/selphi/`)
  - Theory of mind evaluation (ToMBench, OpenToM, SocialIQA)
  - False belief, perspective-taking, epistemic reasoning

- **Introspection** (`projects/introspection/`)
  - Reproducing Anthropic's introspection findings
  - Can models accurately report their internal states?

**Open Questions**:
- Which ToM capabilities transfer across model scales?
- How does introspection relate to deception detection?
- Can models develop genuine self-knowledge or just learned patterns?

**Connections**:
- ToM + Introspection → Understanding self vs. understanding others
- ToM + Alignment → Detecting deceptive behavior
- Introspection + Latent Space → What representations enable self-knowledge?

---

### 3. Internal Representations & Interpretability

**Core Question**: What are the internal representations models use, and how can we make them interpretable to humans?

**Projects**:
- **Latent Space** (`projects/latent-space/`)
  - **Lens**: SAE interpretability (web app)
  - **Topologies**: Mobile latent space exploration (visual/audio/haptic)

- **Introspection** (`projects/introspection/`)
  - Concept vectors and activation steering
  - What features activate during introspection tasks?

**Open Questions**:
- What features do models learn (via SAEs)?
- How can humans experience high-dimensional embeddings?
- What is the geometry of latent space for different concepts?
- Can we navigate latent space to understand model behavior?

**Connections**:
- Latent Space + Introspection → What activations correspond to self-knowledge?
- Latent Space + AI-to-AI Comm → Can agents communicate via latent coordinates?
- Latent Space + Multi-Agent → Do agents develop shared representations?

---

### 4. Alignment, Steerability & Deception

**Core Question**: How can we reliably steer AI systems toward desired behaviors and detect when they're being deceptive?

**Projects**:
- **Steerability** (`projects/steerability/`)
  - Steering vectors and adherence metrics
  - Real-time control and monitoring

- **SELPHI** (`projects/selphi/`)
  - Deception detection via ToM

- **Introspection** (`projects/introspection/`)
  - Honest reporting vs. learned responses

**Open Questions**:
- What steering methods are most reliable?
- How do we detect deceptive ToM reasoning?
- Can introspection be used for alignment verification?

**Connections**:
- Steerability + Introspection → Can we steer models to be more honest?
- ToM + Deception → Models understanding human mental states to deceive
- Introspection + Alignment → Truthful reporting as alignment signal

---

## Cross-Project Research Connections

### Understanding Communication

```
Multi-Agent ──┐
              ├──→ How do agents communicate effectively?
AI-to-AI ─────┘
```

Can multi-agent systems benefit from latent messaging instead of natural language?

### Theory of Mind Spectrum

```
SELPHI (others) ──┐
                  ├──→ Understanding mental states (self vs. others)
Introspection ────┘
```

Is there a unified mechanism for ToM (both self and other)?

### Making Latent Space Interpretable

```
Latent Lens ──────┐
Latent Topologies ├──→ Different ways to understand internal representations
Introspection ────┘
```

Can we combine SAE features, embedding geometry, and activation steering for complete interpretability?

### Alignment through Understanding

```
SELPHI ────────┐
Introspection ├──→ Honest, aligned AI systems
Steerability ──┘
```

Can ToM + introspection + steering create reliably aligned systems?

---

## Infrastructure

All projects use the **Hidden Layer Harness** - a standalone library providing:
- Unified LLM provider abstraction (local + API)
- Experiment tracking and reproducibility
- Evaluation utilities
- Benchmark datasets

**Philosophy**: Flexible infrastructure supporting both local models (Ollama, MLX) for rapid iteration and frontier models (Claude, GPT) via API for capability testing.

---

## Research Methodology

1. **Frame**: Identify the paradigm being challenged
2. **Theorize**: Generate multiple plausible approaches
3. **Implement**: Build simple, interpretable mechanisms
4. **Experiment**: Run reproducible experiments with logging
5. **Synthesize**: Connect findings across projects

**Guiding Principles**:
- Radical curiosity - question everything
- Theoretical discipline - connect to measurable evidence
- Paradigm awareness - understand frameworks, then transcend them
- Architectural creativity - design systems that could discover new science
- Empirical elegance - simple mechanisms, emergent complexity

---

## Publications & Findings

_To be added as research progresses_

### Papers & Presentations

_Placeholder for publications_

### Key Findings

_Placeholder for research findings_

### Open Datasets

_Placeholder for released datasets_

---

## Collaboration

Interested in collaborating? Open an issue or reach out to discuss specific research questions.

## License

- Code: MIT
- Documentation: CC-BY
- Research outputs: TBD per project
