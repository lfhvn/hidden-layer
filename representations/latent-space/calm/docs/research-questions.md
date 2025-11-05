# CALM Research Questions & Methodology

## Overview

This document outlines the research questions, hypotheses, and experimental methodology for investigating Continuous Autoregressive Language Models (CALM).

## Primary Research Questions

### RQ1: Representation Quality & Structure

**Question**: What semantic and syntactic information is encoded in CALM's continuous vectors?

**Sub-questions**:
1. How does information encoding change with semantic bandwidth K?
   - K=1: Single token representation
   - K=4: Phrase-level representation
   - K=8: Sentence-fragment representation

2. What is the relationship between latent dimension l and information capacity?
   - Does increasing l improve representation quality?
   - Is there a sweet spot for l given K?

3. How does the latent manifold structure affect generation quality?
   - Is the manifold smooth and continuous?
   - Are semantically similar chunks nearby in latent space?

**Hypotheses**:
- H1.1: Larger K encodes more semantic (vs. syntactic) information
- H1.2: Optimal l/K ratio exists (paper suggests l = 32K)
- H1.3: VAE regularization creates semantically structured manifolds

**Experiments**:
- **E1.1**: Semantic probing
  - Train linear probes on latent vectors for:
    - Part-of-speech tags
    - Named entity types
    - Sentiment
    - Semantic roles
  - Measure probe accuracy vs. K

- **E1.2**: Reconstruction analysis
  - Vary K ∈ {1, 2, 4, 8, 16}
  - Measure reconstruction accuracy
  - Analyze error patterns (semantic vs. syntactic errors)

- **E1.3**: Manifold visualization
  - Project latent space to 2D/3D (t-SNE, UMAP)
  - Color by semantic properties
  - Measure local smoothness

**Metrics**:
- Reconstruction accuracy (token-level)
- Probing accuracy (by linguistic property)
- Manifold smoothness (local distance variance)

---

### RQ2: Interpretability & Comparison to SAEs

**Question**: How do CALM's continuous vectors compare to Sparse Autoencoder (SAE) features?

**Sub-questions**:
1. Are CALM dimensions interpretable?
   - Can we identify dimensions corresponding to specific features?
   - Are dimensions entangled or disentangled?

2. How does CALM's distributed representation compare to SAE's sparse representation?
   - Information density: CALM (dense, low-dim) vs. SAE (sparse, high-dim)
   - Interpretability: Dense vs. sparse trade-off

3. Can we convert between CALM and SAE representations?
   - Is there a linear mapping?
   - What information is preserved/lost?

**Hypotheses**:
- H2.1: CALM dimensions are entangled (distributed representation)
- H2.2: SAEs are more interpretable per-feature, CALM more efficient per-dimension
- H2.3: Approximate linear mapping exists (both capture similar information)

**Experiments**:
- **E2.1**: Dimension intervention
  - Manipulate individual CALM dimensions
  - Observe effects on decoded text
  - Compare to SAE feature activation

- **E2.2**: Feature extraction
  - Apply PCA/ICA to CALM vectors
  - Compare principal components to SAE features
  - Measure interpretability (human evaluation)

- **E2.3**: Cross-model comparison
  - Extract both CALM and SAE representations
  - Measure correlation (CCA, mutual information)
  - Test linear mappability

**Metrics**:
- Dimension interpretability score (human evaluation)
- Feature overlap (SAE ↔ CALM)
- CCA correlation coefficient
- Linear mapping accuracy

---

### RQ3: Steerability & Control

**Question**: Can we reliably control generation by manipulating CALM's latent vectors?

**Sub-questions**:
1. Can we find "steering directions" in CALM space?
   - Sentiment direction: positive ↔ negative
   - Formality direction: casual ↔ formal
   - Topic directions: science, politics, etc.

2. How does vector-level steering compare to token-level methods?
   - Effectiveness: Success rate of desired behavior
   - Efficiency: Compute cost of steering
   - Granularity: Fine vs. coarse control

3. How robust is steering to vector magnitude and direction?
   - Linear vs. nonlinear effects
   - Optimal steering strength

**Hypotheses**:
- H3.1: Smooth manifold enables linear steering directions
- H3.2: Vector-level steering is more efficient than token-level
- H3.3: Steering effectiveness depends on K (larger K = coarser control)

**Experiments**:
- **E3.1**: Direction discovery
  - Contrastive activation: positive vs. negative examples
  - Extract steering directions via difference
  - Test on held-out data

- **E3.2**: Steering evaluation
  - Apply steering vectors during generation
  - Measure behavioral change (sentiment, formality, etc.)
  - Compare to baseline steering methods

- **E3.3**: Robustness analysis
  - Vary steering magnitude
  - Test on different K values
  - Measure generation quality vs. control strength

**Metrics**:
- Steering success rate (behavioral change)
- Generation quality (fluency, coherence)
- Compute efficiency (FLOPs)

---

### RQ4: Communication Efficiency

**Question**: Can agents communicate more efficiently via CALM vectors than via tokens?

**Sub-questions**:
1. What is the information throughput of vector-based communication?
   - Bits per vector vs. bits per token
   - Compression ratio (K tokens → 1 vector)

2. How robust is vector communication to noise?
   - Reconstruction accuracy under noise
   - Graceful degradation vs. catastrophic failure

3. Can agents develop specialized vector "protocols"?
   - Fine-tune autoencoder for agent communication
   - Emergent vector semantics

**Hypotheses**:
- H4.1: Vector communication achieves K× throughput increase
- H4.2: Smooth manifold provides noise robustness
- H4.3: Specialized autoencoders improve communication efficiency

**Experiments**:
- **E4.1**: Throughput measurement
  - Agents communicate via vectors vs. tokens
  - Measure task completion time
  - Measure information transfer rate

- **E4.2**: Noise robustness
  - Add Gaussian noise to vectors
  - Measure communication success rate
  - Compare to token-based (adversarial perturbations)

- **E4.3**: Protocol learning
  - Fine-tune autoencoder with communication objective
  - Measure compression improvement
  - Analyze emergent vector semantics

**Metrics**:
- Information throughput (bits/second)
- Communication success rate
- Noise tolerance (max σ for >95% success)

---

### RQ5: Efficiency-Quality Trade-offs

**Question**: How does semantic bandwidth K affect the performance-compute frontier?

**Sub-questions**:
1. What is the optimal K for different compute budgets?
   - Small models: K=?
   - Large models: K=?

2. Does optimal K depend on task characteristics?
   - Long-form generation: prefer larger K?
   - Precise control: prefer smaller K?

3. Can we predict performance from autoencoder metrics?
   - Reconstruction accuracy → final BrierLM?
   - KL divergence → generation quality?

**Hypotheses**:
- H5.1: Optimal K increases with model size (capacity hypothesis)
- H5.2: Task-dependent optimal K exists
- H5.3: Autoencoder quality predicts downstream performance

**Experiments**:
- **E5.1**: Scaling study
  - Train models at multiple sizes (S, M, L, XL)
  - Vary K ∈ {1, 2, 4, 8, 16} for each size
  - Plot performance-compute frontiers

- **E5.2**: Task-specific optimization
  - Evaluate K on different tasks:
    - Long-form generation (stories)
    - Precise generation (code)
    - Interactive generation (dialogue)
  - Identify optimal K per task

- **E5.3**: Predictive modeling
  - Collect autoencoder metrics (reconstruction, KL, etc.)
  - Train regression model: AE metrics → BrierLM
  - Test predictive power

**Metrics**:
- BrierLM score
- Training/inference FLOPs
- Wall-clock time
- Task-specific metrics (BLEU, CodeBLEU, etc.)

---

## Cross-Project Research Questions

### CRQ1: CALM ↔ Latent Lens (SAE Interpretability)

**Question**: Can SAE interpretability tools help us understand CALM vectors?

**Integration points**:
- Visualize CALM vectors using Lens interface
- Compare feature sparsity: CALM (dense) vs. SAE (sparse)
- Test if SAE features can be linearly combined to approximate CALM vectors

**Experiments**:
- Port Lens visualization tools to CALM
- Train hybrid model: CALM encoder + SAE decoder
- Measure interpretability-efficiency trade-offs

---

### CRQ2: CALM ↔ Latent Topologies (Mobile Exploration)

**Question**: Can we navigate CALM's latent manifold using mobile interfaces?

**Integration points**:
- Use Topologies' mobile interface for CALM exploration
- Visual/audio/haptic feedback for latent navigation
- Embodied latent space understanding

**Experiments**:
- Adapt Topologies for CALM vectors
- User study: Can humans learn CALM's latent structure?
- Measure navigation efficiency

---

### CRQ3: CALM ↔ AI-to-AI Communication

**Question**: Does CALM enable efficient inter-agent communication?

**Integration points**:
- Replace token-based messages with CALM vectors
- Measure coordination efficiency
- Test on multi-agent tasks

**Experiments**:
- Integrate CALM into multi-agent framework
- Compare coordination time: vectors vs. tokens
- Measure task success rate

---

### CRQ4: CALM ↔ Steerability

**Question**: Does CALM's smooth manifold improve steering?

**Integration points**:
- Compare vector-level steering to token-level
- Test steering vector transferability
- Measure alignment metrics

**Experiments**:
- Extract steering vectors from CALM
- Apply to generation
- Compare to baseline steering methods

---

## Experimental Methodology

### Phase 1: Reproduction & Validation (Weeks 1-2)
**Goal**: Reproduce paper results, validate implementation

1. Implement autoencoder
2. Implement Energy Transformer
3. Train on Pile dataset
4. Evaluate with BrierLM
5. Compare to paper baselines

**Success criteria**:
- Autoencoder: >99.9% reconstruction accuracy
- CALM-M (K=4): BrierLM ≈ 5.7 (within 5% of paper)

---

### Phase 2: Representation Analysis (Weeks 3-4)
**Goal**: Understand what CALM vectors encode

1. Semantic probing (E1.1)
2. Reconstruction analysis (E1.2)
3. Manifold visualization (E1.3)
4. SAE comparison (E2.1, E2.2, E2.3)

**Success criteria**:
- Identify linguistic properties encoded in vectors
- Visualize interpretable manifold structure
- Quantify CALM vs. SAE trade-offs

---

### Phase 3: Control & Steerability (Weeks 5-6)
**Goal**: Test controllability of CALM

1. Discover steering directions (E3.1)
2. Evaluate steering effectiveness (E3.2)
3. Robustness analysis (E3.3)

**Success criteria**:
- Find reliable steering directions (>80% success)
- Demonstrate advantage over token-level steering
- Maintain generation quality while steering

---

### Phase 4: Communication & Integration (Weeks 7-8)
**Goal**: Test communication potential, integrate with other projects

1. Agent communication experiments (E4.1, E4.2, E4.3)
2. Integration with Lens (CRQ1)
3. Integration with Topologies (CRQ2)
4. Integration with AI-to-AI Comm (CRQ3)

**Success criteria**:
- Demonstrate K× communication speedup
- Working cross-project integrations
- Novel insights from integrated systems

---

### Phase 5: Scaling & Optimization (Weeks 9-10)
**Goal**: Optimize performance-compute trade-offs

1. Scaling study (E5.1)
2. Task-specific optimization (E5.2)
3. Predictive modeling (E5.3)

**Success criteria**:
- Identify optimal K for different settings
- Build predictive model for K selection
- Document scaling laws

---

## Open Questions

1. **Context-aware autoencoder**: How much does conditioning on previous vectors improve quality?

2. **Multi-modal CALM**: Can we extend to vision, audio? (Compress patches/frames?)

3. **Hierarchical CALM**: Can we have multiple compression levels? (Meta-vectors?)

4. **Adaptive K**: Can K vary dynamically based on content complexity?

5. **Learned distance metrics**: Can we learn better distance functions for energy loss?

6. **Alignment**: Does CALM make it easier to detect deception (continuous vs. discrete)?

---

## Ethical Considerations

1. **Efficiency gains**: CALM reduces compute → more accessible AI
2. **Interpretability trade-off**: Dense vectors may be harder to interpret than tokens
3. **Dual-use**: Efficient generation could enable scaled misuse
4. **Research transparency**: Open-source implementation, reproducible experiments

---

## Success Metrics

### Technical Success
- Reproduce paper results (±5%)
- Answer all RQ1-RQ5 with empirical evidence
- Publish findings (paper/blog)

### Integration Success
- Working integrations with 2+ other projects
- Demonstrated cross-project insights

### Impact Success
- Novel research directions identified
- Foundation for future work at Hidden Layer
- Contributions to broader research community
