# Research Proposal: Geometric Memory in Multi-Agent LLM Systems

**Project Name**: GeoMAS (Geometric Memory Analysis for Multi-Agent Systems)
**Principal Investigator**: Leif Haven Martinson
**Date**: November 4, 2025
**Status**: Proposal / Planning Phase

---

## Executive Summary

This proposal outlines a research program to investigate **geometric vs. associative memory structures in multi-agent LLM systems**. Inspired by recent findings that Transformers spontaneously develop global geometric representations rather than purely local associative memory ([Noroozizadeh et al., 2025, arXiv:2510.26745](https://arxiv.org/abs/2510.26745)), we propose to extend this analysis to multi-agent architectures to answer: **When and why does multi-agent reasoning benefit from or alter geometric memory structures?**

### Core Hypothesis

Multi-agent strategies (debate, manager-worker, self-consistency) improve performance when single-model geometric memory is ambiguous or insufficient, by either:
1. **Triangulating** better geometric structure from multiple imperfect representations
2. **Decomposing** complex problems into cleaner geometric subspaces
3. **Refining** noisy geometric structures through iterative reasoning

### Key Innovation

While existing work characterizes geometric memory in *single models*, **no research has examined how multi-agent systems construct, share, or refine geometric representations**. This project provides:
- **Diagnostic framework**: Predict multi-agent benefit from single-model geometric analysis
- **Interpretability tools**: Visualize and measure geometric structures across agents
- **Practical guidelines**: When to use multi-agent vs. single-model approaches

---

## 1. Background & Motivation

### 1.1 The Geometric Memory Discovery

Recent work by Noroozizadeh et al. (2025) demonstrates that sequence models (Transformers, Mamba) learn **geometric representations** that encode global relationships between concepts, rather than purely associative (local co-occurrence) memory. Key findings:

- **Emergence**: Geometry arises spontaneously, despite no obvious architectural/optimization pressure
- **Global Structure**: Models synthesize relationships between entities that never co-occurred
- **Computational Advantage**: Transforms complex multi-hop reasoning into simple 1-step geometric lookup
- **Spectral Bias**: Related to graph Laplacian eigenvectors (Fiedler vectors)
- **Headroom**: Node2Vec shows stronger geometry than Transformers → room for improvement

**Why this matters**: Understanding *what* representations models use internally is crucial for predicting *when* and *why* strategies succeed or fail.

### 1.2 Multi-Agent Systems Gap

Current multi-agent research focuses on:
- **Task performance**: When does debate/decomposition help?
- **Emergent behaviors**: Communication, coordination, specialization
- **Scaling laws**: Cost-benefit analysis of multiple agents

**Missing**: Analysis of *internal representations* — how do multiple agents' geometric structures combine, conflict, or refine each other?

### 1.3 Relationship to Existing Projects

This research sits at the intersection of several ongoing efforts in the Hidden Layer lab:

| Project | Focus | Overlap with GeoMAS | Distinction |
|---------|-------|---------------------|-------------|
| **Hidden Layer (main)** | Multi-agent strategy comparison | Uses same harness, same core question | GeoMAS adds interpretability layer |
| **Latent Topologies** | Human experience of embedding space | Both analyze geometric structure | LT: pre-computed, human-facing, UX<br>GeoMAS: dynamic, model-internal, predictive |
| **CRIT** | Design critique via multi-agent | Could use geometric analysis | CRIT: domain-specific (design)<br>GeoMAS: general reasoning framework |
| **SELPHI** | Theory of mind in LLMs | Could analyze ToM geometric structure | SELPHI: specific capability<br>GeoMAS: general memory mechanism |

**GeoMAS is distinct because**:
- Focus on *internal representations* not just task performance
- Provides *predictive framework* not just analysis
- Applies to *any reasoning task* not domain-specific
- Studies *dynamic geometric construction* not static embeddings

**GeoMAS complements existing projects**:
- Could explain *why* CRIT's multi-perspective critique helps
- Could analyze *how* SELPHI tasks require geometric reasoning
- Could inform Latent Topologies' visualization of reasoning processes
- Enhances Hidden Layer harness with interpretability tools

---

## 2. Research Questions

### 2.1 Primary Questions

1. **Structural Differences**
   *Do multi-agent systems create fundamentally different geometric structures than single models?*
   - Hypothesis: Multi-agent creates more separable, less ambiguous clusters
   - Measurement: Spectral gap, cluster coherence, Fiedler vector alignment

2. **Performance Prediction**
   *Can we predict multi-agent benefit from single-model geometric quality?*
   - Hypothesis: Poor single-model geometry → high multi-agent benefit
   - Measurement: Correlation between geometric metrics and accuracy gain

3. **Refinement Mechanisms**
   *How does debate/iteration change geometric structure?*
   - Hypothesis: Successive rounds improve spectral alignment and cluster separation
   - Measurement: Geometric quality across debate rounds

4. **Decomposition Effects**
   *Does task decomposition (manager-worker) create cleaner geometric subspaces?*
   - Hypothesis: Subtasks have higher geometric quality than full task
   - Measurement: Per-subtask geometric metrics vs. monolithic

### 2.2 Secondary Questions

5. **Model Architecture**: Do different architectures (Llama, Mistral, Qwen) show different geometric biases?
6. **Task Type**: Which task types require strong geometric structure? (reasoning, math, planning, etc.)
7. **Fine-Tuning**: Does task-specific fine-tuning improve geometric memory?
8. **Prompt Engineering**: Do different prompts elicit different geometric structures?
9. **Hidden Layer Analysis**: Which layers develop the strongest geometry? Where does it emerge?

---

## 3. Methodology

### 3.1 Experimental Design

#### Phase 1: Validation (Weeks 1-3)
**Goal**: Reproduce Noroozizadeh findings on local models

- Implement path-star graph task
- Extract hidden states from MLX/Ollama models
- Validate geometric vs. associative memory measurement
- Establish baseline geometric metrics

**Deliverables**:
- `code/geomas/geometric_probes.py` — extraction & analysis tools
- `notebooks/geomas/01_geometric_validation.ipynb` — reproduction notebook
- Validation report confirming local models exhibit geometric memory

#### Phase 2: Single-Model Baseline (Weeks 4-6)
**Goal**: Characterize geometric quality across tasks and models

- Define task suite: reasoning, math, planning, QA (leverage existing harness tasks)
- Measure geometric quality for each (task, model) pair
- Correlate geometric metrics with task performance
- Identify which tasks/models have poor geometry

**Deliverables**:
- Task-specific geometric benchmarks
- Model comparison: geometric quality profiles
- "Geometric difficulty" ratings for tasks

#### Phase 3: Multi-Agent Analysis (Weeks 7-10)
**Goal**: Compare geometric structures across strategies

**Experiments**:

| Strategy | Geometric Questions | Analysis Method |
|----------|---------------------|-----------------|
| **Single** | Baseline geometry | Extract hidden states, compute spectral structure |
| **Debate** | Does iteration refine geometry? | Compare geometry across rounds, measure convergence |
| **Self-consistency** | Do samples share geometry? | Compute geometric distance between samples |
| **Manager-worker** | Are subtask geometries cleaner? | Compare per-subtask vs. full-task metrics |

**Key Comparisons**:
- Spectral gap (before vs. after multi-agent)
- Cluster coherence (single vs. multi-agent)
- Fiedler vector alignment (single vs. aggregated)

**Deliverables**:
- Strategy-specific geometric profiles
- Geometric refinement curves (iteration → quality)
- Multi-agent geometric advantage scores

#### Phase 4: Predictive Framework (Weeks 11-14)
**Goal**: Build predictive model for multi-agent benefit

- Train regression model: `geometric_metrics → performance_gain`
- Test on held-out tasks
- Develop decision heuristic: "Use multi-agent when geometric_quality < threshold"
- Validate on new tasks

**Deliverables**:
- Predictive model with validation metrics
- Practical guidelines for strategy selection
- Integration into Hidden Layer harness (`recommend_strategy()` function)

#### Phase 5: Deep Interpretability (Weeks 15-16)
**Goal**: Understand *how* geometry emerges and changes

- Layer-by-layer analysis: where does geometry emerge?
- Attention pattern analysis: does attention structure correlate with geometry?
- Ablation studies: which components are necessary for geometric memory?
- Visualization: interactive geometric structure browser

**Deliverables**:
- Mechanistic understanding of geometric memory formation
- Visualization tools for exploring geometric structures
- Hypotheses for improving geometric bias in models

### 3.2 Technical Approach

#### Hidden State Extraction

```python
# For MLX models
def extract_mlx_hidden_states(model, input_ids, layer_indices):
    """Extract activations from specified layers during forward pass"""
    # Use model hooks to capture intermediate activations

# For Ollama models
def extract_ollama_hidden_states(model, input_text, layer_indices):
    """Use Ollama's embedding endpoint or model inspection"""
    # May require model-specific APIs
```

#### Geometric Analysis

```python
def compute_geometric_structure(hidden_states):
    """
    Compute spectral properties of hidden state manifold

    Returns:
        - spectral_gap: λ₂ - λ₁ (larger = stronger geometric structure)
        - fiedler_vector: 2nd eigenvector (encodes primary geometric axis)
        - cluster_coherence: intra-cluster distance / inter-cluster distance
        - global_structure_score: composite metric
    """
    # 1. Compute pairwise similarity matrix S = H @ H^T
    # 2. Construct graph Laplacian L = D - S
    # 3. Eigendecomposition: L = Q Λ Q^T
    # 4. Analyze eigenspectrum and eigenvectors
```

#### Visualization

```python
def visualize_geometric_evolution(hidden_states_over_time):
    """Show how geometry changes across debate rounds or layers"""
    # Use UMAP for 2D projection
    # Animate transitions
    # Highlight cluster formation/refinement
```

### 3.3 Evaluation Metrics

| Metric | Purpose | Range | Interpretation |
|--------|---------|-------|----------------|
| **Spectral Gap** | Strength of geometric structure | [0, ∞) | Higher = stronger primary axis |
| **Cluster Coherence** | Separation of concepts | [0, 1] | Higher = clearer boundaries |
| **Fiedler Alignment** | Consistency with graph structure | [-1, 1] | Closer to 1 = aligned with task structure |
| **Geometric Quality Score** | Overall composite | [0, 1] | Weighted combination of above |
| **Performance Gain** | Multi-agent benefit | [−∞, ∞) | (multi_acc - single_acc) |
| **Cost-Adjusted Gain** | Efficiency metric | [−∞, ∞) | gain / (latency_ratio × cost_ratio) |

### 3.4 Task Suite

Leverage existing Hidden Layer tasks, categorized by expected geometric complexity:

**High Geometric Demand** (prediction: multi-agent helps most):
- Multi-hop reasoning (path-finding, logical chains)
- Analogical reasoning (A:B :: C:?)
- Planning with constraints

**Medium Geometric Demand**:
- Mathematical problem solving
- Argument synthesis
- Causal reasoning

**Low Geometric Demand** (prediction: minimal multi-agent benefit):
- Simple factual QA
- Classification tasks
- Direct retrieval

---

## 4. Technical Requirements

### 4.1 Hardware
- ✅ **Available**: M4 Max with 128GB RAM
- ✅ Can run 70B models (4-bit quantized)
- ✅ Can run 3-4 × 7B models simultaneously
- ✅ MLX optimized for unified memory

### 4.2 Software Stack

**Core Dependencies**:
```python
# Existing
mlx, mlx-lm          # Apple Silicon models
ollama               # Local model serving
anthropic, openai    # API providers

# New for GeoMAS
scipy                # Eigendecomposition
scikit-learn         # Clustering, dimensionality reduction
umap-learn           # Visualization
networkx             # Graph analysis (optional)
plotly               # Interactive visualizations
```

**Integration Points**:
- Extend `code/harness/llm_provider.py` with hidden state extraction
- New module: `code/geomas/` for geometric analysis
- New notebooks: `notebooks/geomas/` for experiments

### 4.3 Data & Logging

**Experiment Outputs** (extends existing structure):
```
experiments/geomas_{name}_{timestamp}/
├── config.json                    # Standard experiment config
├── results.jsonl                  # Task results
├── summary.json                   # Aggregated metrics
├── geometric_analysis/
│   ├── hidden_states_{task_id}.npz     # Raw activations
│   ├── spectral_properties_{task_id}.json
│   ├── visualizations_{task_id}.html   # Interactive plots
│   └── comparison_across_strategies.json
└── README.md
```

---

## 5. Expected Outcomes

### 5.1 Scientific Contributions

1. **First characterization** of geometric memory in multi-agent LLM systems
2. **Predictive framework** for when multi-agent strategies provide benefit
3. **Mechanistic understanding** of how debate/decomposition refines representations
4. **Interpretability tools** for analyzing internal model structures

### 5.2 Practical Deliverables

1. **Diagnostic tool**: `recommend_strategy(task) → {"single" | "debate" | "manager_worker"}`
   - Based on geometric analysis of task
   - Provides confidence and rationale

2. **Visualization toolkit**: Interactive browser for geometric structures
   - Explore hidden states across layers
   - Compare strategies visually
   - Identify problematic regions

3. **Integration with harness**: Seamless addition to existing workflows
   ```python
   from harness import run_strategy
   from geomas import GeometricProbe, predict_strategy_benefit

   # Analyze task first
   probe = GeometricProbe(model)
   recommendation = predict_strategy_benefit(probe, task)

   # Run recommended strategy
   result = run_strategy(recommendation.strategy, task, **kwargs)
   ```

4. **Research paper**: "Geometric Memory in Multi-Agent LLM Systems"
   - Submit to: NeurIPS, ICLR, ICML, or ACL
   - Focus: Novel interpretability angle + practical implications

### 5.3 Broader Impact

**For Research**:
- New lens on multi-agent systems (internal representations, not just behavior)
- Bridge interpretability and multi-agent architecture design
- Potential to improve model training (enhance geometric bias)

**For Applications**:
- Cost savings: avoid unnecessary multi-agent when single model sufficient
- Quality improvements: use multi-agent when geometric structure poor
- Debugging: identify when models have weak internal representations

**For Related Projects**:
- **CRIT**: Explain why multi-perspective critique helps (geometric diversity)
- **SELPHI**: Analyze geometric requirements for Theory of Mind tasks
- **Latent Topologies**: Inform visualization of reasoning processes

---

## 6. Relationship to Latent Topologies

### 6.1 Conceptual Overlap

Both projects study **geometry of representations**, but from different angles:

| Aspect | Latent Topologies | GeoMAS |
|--------|-------------------|--------|
| **Focus** | Human perception of static embedding space | Model-internal dynamics during reasoning |
| **Modality** | Visual, audio, haptic exploration | Computational analysis |
| **Output** | Mobile app, phenomenological research | Diagnostic tools, scientific paper |
| **Geometry** | Pre-computed embeddings (sentence transformers) | Task-specific hidden states |
| **Goal** | Experience and understand AI's conceptual world | Predict and improve reasoning strategies |
| **User** | General public, designers, researchers | ML researchers, engineers |

### 6.2 Potential Synergies

**Could GeoMAS be a *sub-project* of Latent Topologies?**

**Arguments FOR**:
- Both study geometry of meaning
- Complementary perspectives (human-facing vs. technical)
- Shared visualization techniques (UMAP, spectral analysis)
- Unified research narrative: "Understanding the topology of thought"

**Arguments AGAINST**:
- Different research communities (HCI/phenomenology vs. ML/interpretability)
- Different deliverables (app vs. scientific tooling)
- Different timelines (LT: 12-week product; GeoMAS: 16-week research)
- Different technical stacks (React Native + embeddings vs. MLX + hidden states)

### 6.3 Recommended Structure

**Option 1: Separate Projects Under Shared Umbrella**
```
hidden-layer/
├── README.md                    # Umbrella research lab
├── code/harness/               # Shared multi-agent harness
├── latent-topologies/          # Mobile phenomenology project
├── geomas/                     # Geometric memory analysis (NEW)
│   ├── README.md
│   ├── code/
│   ├── notebooks/
│   └── experiments/
├── code/crit/                  # Design critique
└── code/selphi/                # Theory of mind
```

**Option 2: GeoMAS as Research Track Within Latent Topologies**
```
latent-topologies/
├── README.md                   # Product + research overview
├── app/                        # Mobile app code
├── research/
│   ├── PHENOMENOLOGY_NOTES.md
│   ├── STUDY_PROTOCOL.md
│   └── GEOMETRIC_MEMORY/       # GeoMAS (NEW)
│       ├── PROPOSAL.md (this file)
│       ├── code/
│       └── notebooks/
```

**Recommendation: Option 1 (Separate Projects)**

**Rationale**:
1. **Distinct deliverables**: App vs. scientific tooling
2. **Different audiences**: General public vs. ML researchers
3. **Independent value**: Each stands alone
4. **Clear scope**: Easier to manage and explain
5. **Collaborative potential**: Can still cross-reference and share insights

**However**: Maintain strong **conceptual links**:
- Shared "geometry of thought" narrative
- Cross-reference in papers/documentation
- Potential future integration (visualize GeoMAS results in LT app)

---

## 7. Timeline & Milestones

### 16-Week Research Program

| Weeks | Phase | Key Deliverables | Success Criteria |
|-------|-------|------------------|------------------|
| 1-3 | **Validation** | Geometric probes module, validation notebook | Reproduce Noroozizadeh findings locally |
| 4-6 | **Single-Model Baseline** | Task suite geometric benchmarks | Geometric quality correlates with performance |
| 7-10 | **Multi-Agent Analysis** | Strategy comparison results | Identify geometric differences across strategies |
| 11-14 | **Predictive Framework** | Recommendation tool, validation | Predict multi-agent benefit with >70% accuracy |
| 15-16 | **Deep Interpretability** | Layer analysis, visualization tools | Mechanistic understanding of emergence |
| 17+ | **Paper Writing** | Draft, submission | Accepted to ML/NLP conference |

### Milestones

**M1** (Week 3): ✓ Geometric analysis tools validated
**M2** (Week 6): ✓ Single-model geometric profiles complete
**M3** (Week 10): ✓ Multi-agent geometric comparison complete
**M4** (Week 14): ✓ Predictive framework validated
**M5** (Week 16): ✓ Full analysis and visualization tools ready
**M6** (Week 20): ✓ Paper submitted

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Hidden state extraction difficult** | Medium | High | Start with simpler models; use embeddings as fallback |
| **Geometric metrics don't predict performance** | Medium | High | Explore alternative metrics; qualitative analysis |
| **Multi-agent geometry too complex to interpret** | Medium | Medium | Focus on simpler cases; aggregate metrics |
| **Computational cost too high** | Low | Medium | Use smaller models; sample subset of tasks |
| **Findings don't generalize across models** | Medium | Medium | Test multiple architectures early; document limitations |

---

## 9. Budget & Resources

### Time Investment
- **Week 1-3**: Full-time (40 hrs/week) — critical foundation
- **Week 4-14**: Part-time (20-25 hrs/week) — steady progress
- **Week 15-16**: Full-time (40 hrs/week) — intensive analysis

**Total**: ~400-450 hours over 16 weeks

### Computational Resources
- ✅ **Local**: M4 Max (sufficient for primary experiments)
- **API Costs**: ~$50-100 for comparison with Claude/GPT (optional)
- **Storage**: ~50-100 GB for hidden states (manageable)

### No Additional Costs
- All software is open-source
- Hardware already available
- No human subjects or external data needed

---

## 10. Success Metrics

### Tier 1: Essential (Must-Have)
- [ ] Demonstrate geometric memory in local models (validation)
- [ ] Show measurable differences in geometry across strategies
- [ ] Build working predictive framework (even if accuracy modest)

### Tier 2: Core (Should-Have)
- [ ] Achieve >70% accuracy in predicting multi-agent benefit
- [ ] Identify at least 3 task types with clear geometric patterns
- [ ] Produce visualizations that clearly show geometric differences

### Tier 3: Stretch (Nice-to-Have)
- [ ] Mechanistic understanding of geometric emergence
- [ ] Published paper at top-tier venue
- [ ] Integration with CRIT/SELPHI for cross-project insights

---

## 11. Next Steps

### Immediate (Week 1)
1. **Setup**
   - [x] Review Noroozizadeh paper thoroughly
   - [ ] Create `geomas/` directory structure
   - [ ] Install additional dependencies (scipy, umap, etc.)

2. **Prototype**
   - [ ] Write basic hidden state extraction for one MLX model
   - [ ] Implement spectral analysis for toy example
   - [ ] Test on simple path-finding task

3. **Planning**
   - [ ] Define exact task suite (reuse existing + add 5-10 new tasks)
   - [ ] Finalize evaluation metrics
   - [ ] Create detailed experimental protocols

### Short-Term (Weeks 2-3)
- [ ] Complete validation phase
- [ ] Write validation report
- [ ] Decide on final project structure (separate vs. sub-project)

### Medium-Term (Month 2)
- [ ] Complete single-model baseline
- [ ] Begin multi-agent experiments
- [ ] Draft methodology section of paper

---

## 12. References

### Foundation Paper
- **Noroozizadeh, S., et al. (2025)**. "Deep sequence models tend to memorize geometrically; it is unclear why." arXiv:2510.26745. [Link](https://arxiv.org/abs/2510.26745)

### Related Work: Geometric & Spectral Methods
- Belkin, M., & Niyogi, P. (2003). "Laplacian Eigenmaps for Dimensionality Reduction"
- Grover, A., & Leskovec, J. (2016). "node2vec: Scalable Feature Learning for Networks"
- Von Luxburg, U. (2007). "A tutorial on spectral clustering"

### Related Work: LLM Interpretability
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits"
- Geva, M., et al. (2023). "Dissecting Recall of Factual Associations in Auto-Regressive Models"
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT"

### Related Work: Multi-Agent LLMs
- Du, Y., et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate"
- Liang, T., et al. (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate"
- Hong, S., et al. (2023). "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework"

### Related Work: Reasoning in LLMs
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
- Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with LLMs"
- Wang, X., et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning"

---

## 13. Conclusion

The **GeoMAS project** represents a natural evolution of the Hidden Layer research program: moving from *behavioral* analysis of multi-agent systems to *mechanistic* understanding of their internal representations. By characterizing geometric memory structures, we can:

1. **Predict** when multi-agent strategies provide value
2. **Understand** why they work (or fail)
3. **Improve** both single-model and multi-agent architectures
4. **Bridge** interpretability and practical system design

This research is **feasible** (all resources available), **novel** (unexplored intersection), **impactful** (both scientific and practical contributions), and **aligned** with the lab's broader goals of understanding intelligence through interpretability.

**Recommended Action**: Approve as **separate project** under Hidden Layer umbrella, with strong conceptual links to Latent Topologies, CRIT, and SELPHI.

---

## Appendix A: Quick Start Guide (Post-Approval)

```bash
# 1. Create project structure
mkdir -p geomas/{code,notebooks,experiments,docs}
cd geomas

# 2. Install dependencies
pip install scipy scikit-learn umap-learn plotly networkx

# 3. Create initial notebook
cd notebooks
jupyter notebook 01_geometric_validation.ipynb

# 4. Start with validation
# Reproduce path-star task from Noroozizadeh paper
# Extract hidden states from local MLX model
# Compute spectral properties
# Visualize geometric structure
```

---

## Appendix B: Code Outline

```python
# geomas/code/geometric_probes.py

class GeometricProbe:
    """Analyze geometric vs associative memory in models"""

    def __init__(self, model, provider, layer_indices=None):
        self.model = model
        self.provider = provider
        self.layer_indices = layer_indices or [-1, -2, -3]

    def extract_hidden_states(self, inputs):
        """Extract activations from specified layers"""

    def compute_spectral_structure(self, hidden_states):
        """Compute eigendecomposition of manifold"""

    def geometric_quality_score(self, hidden_states):
        """Composite metric of geometric structure"""

    def visualize_geometry(self, hidden_states, labels=None):
        """2D/3D visualization using UMAP"""

class MultiAgentGeometricAnalyzer:
    """Compare geometric structures across strategies"""

    def compare_strategies(self, task, strategies):
        """Run task with multiple strategies, analyze geometry"""

    def geometric_evolution_across_rounds(self, debate_result):
        """Show how geometry changes during debate"""

    def predict_strategy_benefit(self, single_model_geometry):
        """Predict which strategy will help based on geometry"""

# geomas/code/tasks.py

def path_finding_task(graph_structure, depth):
    """Generate path-finding task for validation"""

def reasoning_task_suite():
    """Suite of reasoning tasks with varying geometric complexity"""

# geomas/code/evals.py

def geometric_performance_correlation(geometry_metrics, task_results):
    """Analyze correlation between geometric quality and accuracy"""
```

---

**END OF PROPOSAL**

---

**Next Step**: Review this proposal, provide feedback, and decide:
1. Approve as-is and proceed to implementation?
2. Request modifications to scope/approach?
3. Integrate differently with Latent Topologies?
4. Prioritize differently among the research questions?

I'm ready to begin implementation immediately upon approval.
