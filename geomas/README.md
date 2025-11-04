# GeoMAS - Geometric Memory Analysis for Multi-Agent Systems

**Understanding when and why multi-agent LLM strategies work by analyzing their internal geometric representations**

---

## Quick Links

- [Research Proposal](../GEOMETRIC_MEMORY_PROPOSAL.md) - Full research plan and motivation
- [Module Design](./docs/MODULE_DESIGN.md) - Technical architecture (coming next)
- [Experimental Protocol](./docs/EXPERIMENTAL_PROTOCOL.md) - Detailed experiments (coming next)
- [Notebooks](./notebooks/) - Interactive experiments

---

## Overview

GeoMAS investigates how multi-agent LLM systems construct and refine **geometric memory structures**. Inspired by findings that Transformers spontaneously develop global geometric (rather than purely associative) representations, we ask:

**Do multi-agent strategies create fundamentally different geometric structures, and can we predict when they'll help based on single-model geometry?**

### Core Hypothesis

Multi-agent reasoning improves performance when single-model geometric memory is ambiguous or insufficient, through:
- **Triangulation**: Combining multiple imperfect geometric representations
- **Decomposition**: Breaking into cleaner geometric subspaces
- **Refinement**: Iteratively improving geometric structure through debate

---

## Project Status

**Phase**: Planning → Implementation
**Started**: November 2025
**Current Milestone**: Module design and validation experiments

### Roadmap

- [ ] **Phase 1: Validation** (Weeks 1-3)
  - [ ] Geometric probes module
  - [ ] Hidden state extraction (MLX/Ollama)
  - [ ] Reproduce Noroozizadeh path-star task
  - [ ] Validate geometric vs associative memory measurement

- [ ] **Phase 2: Single-Model Baseline** (Weeks 4-6)
  - [ ] Task suite definition and geometric benchmarks
  - [ ] Model comparison across geometric quality
  - [ ] Performance correlation analysis

- [ ] **Phase 3: Multi-Agent Analysis** (Weeks 7-10)
  - [ ] Strategy comparison (single, debate, manager-worker)
  - [ ] Geometric evolution across debate rounds
  - [ ] Decomposition effects on geometric structure

- [ ] **Phase 4: Predictive Framework** (Weeks 11-14)
  - [ ] Train regression: geometry → performance gain
  - [ ] Validate on held-out tasks
  - [ ] Build recommendation system

- [ ] **Phase 5: Deep Interpretability** (Weeks 15-16)
  - [ ] Layer-by-layer analysis
  - [ ] Attention pattern correlations
  - [ ] Visualization tools

---

## Quick Start

### Installation

```bash
# From hidden-layer root
source venv/bin/activate

# Install GeoMAS dependencies
pip install scipy scikit-learn umap-learn plotly networkx

# Verify setup
python -c "from geomas.code.geometric_probes import GeometricProbe; print('Ready!')"
```

### Basic Usage

```python
from geomas.code.geometric_probes import GeometricProbe, geometric_quality_score
from harness import run_strategy

# Create probe
probe = GeometricProbe(model="llama3.2:latest", provider="ollama")

# Analyze task with single model
single_result = run_strategy("single", task_input="Your task here")
geometry = probe.analyze(single_result)

print(f"Geometric quality: {geometry.quality_score:.3f}")
print(f"Spectral gap: {geometry.spectral_gap:.3f}")
print(f"Cluster coherence: {geometry.cluster_coherence:.3f}")

# Predict if multi-agent would help
if geometry.quality_score < 0.5:
    print("→ Recommendation: Try multi-agent (debate or manager-worker)")
else:
    print("→ Recommendation: Single model sufficient")
```

### Example: Compare Strategies

```python
from geomas.code.multi_agent_analyzer import MultiAgentGeometricAnalyzer

analyzer = MultiAgentGeometricAnalyzer(model="llama3.2:latest")

# Run and analyze multiple strategies
comparison = analyzer.compare_strategies(
    task="Complex multi-hop reasoning task...",
    strategies=["single", "debate", "manager_worker"]
)

# Visualize geometric differences
analyzer.visualize_comparison(comparison, save_path="comparison.html")
```

---

## Architecture

```
geomas/
├── code/
│   ├── geometric_probes.py        # Core geometric analysis
│   ├── multi_agent_analyzer.py    # Strategy comparison
│   ├── hidden_state_extraction.py # Model-specific extraction
│   ├── tasks.py                   # Task generators
│   ├── evals.py                   # Evaluation metrics
│   └── visualizations.py          # Plotting and viz
│
├── notebooks/
│   ├── 01_geometric_validation.ipynb       # Reproduce paper
│   ├── 02_single_model_baseline.ipynb     # Baseline metrics
│   ├── 03_multi_agent_comparison.ipynb    # Strategy analysis
│   ├── 04_predictive_framework.ipynb      # Build predictor
│   └── 05_deep_interpretability.ipynb     # Layer analysis
│
├── experiments/                    # Logged results
├── docs/                          # Technical documentation
└── README.md                      # This file
```

---

## Key Concepts

### Geometric vs Associative Memory

**Associative Memory** (traditional view):
- Facts stored in weight matrices as arbitrary lookup tables
- Embeddings don't encode structure themselves
- Multi-hop reasoning requires multiple matrix operations

**Geometric Memory** (what actually emerges):
- Embeddings themselves encode global relationships
- Related concepts clustered in structured manifold
- Multi-hop reasoning becomes 1-step geometric lookup
- Related to graph Laplacian eigenvectors (spectral structure)

### Geometric Quality Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Spectral Gap** | Strength of primary geometric axis | λ₂ - λ₁ >> 0 |
| **Cluster Coherence** | Separation of concept clusters | High |
| **Fiedler Alignment** | Consistency with task structure | Close to 1 |
| **Global Structure Score** | Overall geometric organization | High |

### Research Questions

1. **Structural**: Do multi-agent systems create different geometric structures?
2. **Predictive**: Can we predict multi-agent benefit from single-model geometry?
3. **Mechanistic**: How does debate/iteration refine geometry?
4. **Practical**: When should we use multi-agent vs single-model?

---

## Integration with Existing Projects

GeoMAS builds on and enhances other Hidden Layer projects:

| Project | How GeoMAS Helps |
|---------|------------------|
| **Hidden Layer (harness)** | Adds interpretability layer to strategy comparison |
| **CRIT** | Explains *why* multi-perspective critique helps (geometric diversity) |
| **SELPHI** | Analyzes geometric requirements for Theory of Mind |
| **Latent Topologies** | Informs visualization of dynamic reasoning processes |

---

## Foundation Paper

This research is inspired by:

**Noroozizadeh, S., Nagarajan, V., Rosenfeld, E., & Kumar, S. (2025)**
*"Deep sequence models tend to memorize geometrically; it is unclear why"*
arXiv:2510.26745
[Paper Link](https://arxiv.org/abs/2510.26745)

**Key Finding**: Transformers spontaneously develop global geometric representations that encode relationships between all entities, even those that never co-occurred in training.

**Our Extension**: How do multi-agent systems construct, share, and refine these geometric representations?

---

## Status: What's Built

- [x] Research proposal
- [x] Project structure
- [ ] Core modules (in progress)
- [ ] Validation experiments
- [ ] Analysis tools
- [ ] Visualization toolkit

---

## Contributing

This is an active research project. To contribute:

1. Read the [full proposal](../GEOMETRIC_MEMORY_PROPOSAL.md)
2. Check current issues and roadmap
3. Run experiments and share findings
4. Improve tools and documentation

---

## Citation

If you use GeoMAS in your research:

```bibtex
@software{geomas2025,
  title={GeoMAS: Geometric Memory Analysis for Multi-Agent Systems},
  author={Martinson, Leif Haven},
  year={2025},
  url={https://github.com/lfhvn/hidden-layer/geomas}
}
```

---

## License

Part of the Hidden Layer research project
See main project for license details

---

**GeoMAS** - Understanding the geometry of collective thought
