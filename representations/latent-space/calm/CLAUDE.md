# CALM - Continuous Autoregressive Language Models

## Project Overview

**CALM** (Continuous Autoregressive Language Models) explores a paradigm shift from discrete next-token prediction to continuous next-vector prediction. Instead of generating language token-by-token, CALM compresses K tokens into a single continuous vector, then models language as a sequence of these vectors.

**Based on**: [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688) (Shao et al., 2025)

## Research Position

### Primary Area: Representations
CALM is fundamentally about **continuous latent representations** of language:
- How can we compress multiple tokens into a single robust vector?
- What semantic/syntactic information do these vectors encode?
- How does the latent manifold structure affect generation quality?

### Cross-Area Connections

**Communication** (`communication/ai-to-ai-comm/`):
- 4x compression (K=4): proof that efficient non-linguistic communication is possible
- Could enable direct latent-vector communication between agents
- Question: Can agents coordinate via CALM vectors instead of tokens?

**Alignment** (`alignment/steerability/`):
- Smooth latent manifolds may be more amenable to steering
- Vector-level behavioral control (vs. token-level)
- Question: Does continuous representation improve steerability?

**Representations** (sibling projects):
- **Lens** (SAE interpretability): How do CALM vectors compare to SAE features?
- **Topologies** (latent exploration): Can we navigate CALM's smooth manifold?

## Core Research Questions

### 1. Representation Quality
- What information do CALM vectors encode at different K?
- How does compression (K) affect semantic vs. syntactic representation?
- What is the information capacity of different latent dimensions?

### 2. Interpretability
- Can we interpret individual dimensions of CALM vectors?
- How do CALM vectors compare to SAE features?
- Can we visualize the latent manifold structure?

### 3. Steerability
- Can we steer generation by manipulating latent vectors?
- How does vector perturbation affect generated text?
- Can we find "steering directions" in CALM space?

### 4. Efficiency-Quality Trade-offs
- How does semantic bandwidth K affect the performance-compute frontier?
- What is the optimal K for different tasks?
- Can we predict performance from autoencoder metrics?

### 5. Communication Potential
- Can agents communicate via CALM vectors?
- How robust is vector-based communication to noise?
- What's the efficiency gain vs. token-based communication?

## Technical Architecture

### Components

**1. Autoencoder** (discrete ↔ continuous mapping)
- **Encoder**: K tokens → l-dimensional vector
- **Decoder**: l-dimensional vector → K tokens
- **Key innovation**: Robust latent space via VAE + dropout
- **Target**: >99.9% reconstruction accuracy

**2. Energy Transformer** (likelihood-free LM)
- **Backbone**: Standard Transformer
- **Generative Head**: Energy-based, single-step generation
- **Training**: Energy loss (strictly proper scoring rule)
- **Advantage**: No iterative sampling (vs. diffusion/flow matching)

**3. Evaluation** (BrierLM metric)
- **Challenge**: No explicit likelihoods → can't compute perplexity
- **Solution**: Brier score (strictly proper, likelihood-free)
- **Estimation**: Unbiased via sampling
- **BrierLM**: Geometric mean of Brier-n for n=1..4

**4. Sampling** (likelihood-free temperature control)
- **Exact algorithm**: Rejection sampling via Bernoulli factory
- **Approximate algorithm**: Batch-based (asymptotically unbiased)
- **Trade-off**: Batch size N controls accuracy-diversity

## Implementation Plan

### Phase 1: Reproduction (Weeks 1-2)
- [ ] Set up environment and dependencies
- [ ] Implement autoencoder architecture
- [ ] Implement Energy Transformer
- [ ] Reproduce baseline results from paper

### Phase 2: Interpretability (Weeks 3-4)
- [ ] Build vector analysis tools
- [ ] Probe what information vectors encode
- [ ] Compare to SAE features (integrate with Lens)
- [ ] Visualize latent manifold

### Phase 3: Steerability (Weeks 5-6)
- [ ] Implement vector manipulation tools
- [ ] Search for steering directions
- [ ] Test behavioral control via vectors
- [ ] Compare to token-level steering

### Phase 4: Communication (Weeks 7-8)
- [ ] Design agent communication protocols via vectors
- [ ] Test robustness to noise
- [ ] Measure efficiency vs. token-based
- [ ] Integrate with multi-agent framework

## Key Insights from Paper

### 1. Autoencoder Design
**Challenge**: Deterministic autoencoders create brittle representations
**Solution**: Variational + dropout regularization
- VAE: Smooth latent manifold (encoder outputs Gaussian distribution)
- Dropout: Redundant representation (robust to noise)
- KL clipping: Prevent posterior collapse
- Result: Robust to σ ≈ 0.3 Gaussian noise

**Architecture**:
- K=4 tokens → l=128 dimensions
- Lightweight: ~75M parameters
- Hidden size: d=512
- Training: 15B tokens, 30k steps

### 2. Energy Transformer
**Why not diffusion/flow?**: Iterative sampling = inference bottleneck
**Energy Score**: Strictly proper scoring rule
- Diversity term: E[||x' - x''||^α]
- Fidelity term: -2E[||x - y||^α]
- α = 1 (optimal empirically)

**Training**:
- N=8 model samples (multi-sample gradient)
- M=100 target samples (from VAE posterior)
- Energy loss: Unbiased Monte Carlo estimator

### 3. Performance-Compute Trade-offs
**Key finding**: CALM achieves baseline performance at lower compute
- CALM-M (371M params, K=4) ≈ Transformer-S (281M params)
- Training: 44% fewer FLOPs
- Inference: 34% fewer FLOPs

**Scaling K**:
- K=1: Worse than baseline (continuous prediction harder)
- K=2: ~2x speedup, marginal quality drop
- K=4: Surpasses baseline frontier
- K=8: Larger drop (capacity limitation)

### 4. Temperature Sampling
**Challenge**: No explicit distribution → can't do standard temperature sampling
**Solution**: Two algorithms
- **Exact**: Rejection sampling (provably correct, expensive)
- **Approximate**: Batch-based (asymptotically unbiased, efficient)

**Key insight**: Batch size N more effective than temperature T for controlling diversity

## Development Workflow

### 1. Environment Setup

```bash
cd representations/latent-space/calm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Training Autoencoder

```bash
# Train autoencoder on Pile subset
python src/train_autoencoder.py \
  --data pile \
  --tokens 15B \
  --chunk_size 4 \
  --latent_dim 128 \
  --steps 30k \
  --batch_size 512k
```

### 3. Training CALM

```bash
# Train CALM model
python src/train_calm.py \
  --autoencoder checkpoints/autoencoder_k4.pt \
  --data pile \
  --model_size M \
  --steps 250k \
  --batch_size 2M
```

### 4. Evaluation

```bash
# Evaluate with BrierLM
python src/evaluate.py \
  --model checkpoints/calm_m_k4.pt \
  --data wikitext103 \
  --metric brierlm
```

### 5. Interpretability Analysis

```bash
# Analyze vector representations
python experiments/analyze_vectors.py \
  --model checkpoints/calm_m_k4.pt \
  --analysis semantic_probing

# Compare to SAE features
python experiments/compare_sae.py \
  --calm_model checkpoints/calm_m_k4.pt \
  --sae_model ../lens/models/sae.pt
```

## Integration with Harness

CALM uses the harness infrastructure for:
- **Model management**: Provider-agnostic abstractions
- **Experiment tracking**: Reproducible runs
- **Dataset loading**: Pile, WikiText-103
- **Evaluation**: BrierLM metric integration

```python
from harness import ExperimentTracker, load_dataset
from calm import CALMModel, Autoencoder

# Initialize
tracker = ExperimentTracker("calm_k4_experiment")
data = load_dataset("pile", subset="15B")

# Train
autoencoder = Autoencoder(K=4, latent_dim=128)
model = CALMModel(autoencoder=autoencoder, size="M")

# Track everything
tracker.log_hyperparams({"K": 4, "size": "M"})
tracker.log_metrics({"brierlm": 5.72, "train_flops": 3.7e20})
```

## File Organization

```
calm/
├── src/
│   ├── autoencoder.py       # VAE autoencoder implementation
│   ├── energy_transformer.py # Energy Transformer + generative head
│   ├── train_autoencoder.py  # Autoencoder training script
│   ├── train_calm.py          # CALM training script
│   ├── evaluate.py            # BrierLM evaluation
│   └── sampling.py            # Temperature sampling algorithms
│
├── experiments/
│   ├── analyze_vectors.py     # Vector interpretability analysis
│   ├── compare_sae.py         # Compare to SAE features
│   ├── steering.py            # Steerability experiments
│   ├── scaling_k.py           # Study effect of K
│   └── communication.py       # Agent communication via vectors
│
├── docs/
│   ├── architecture.md        # Detailed architecture notes
│   ├── brierlm.md            # BrierLM metric explanation
│   ├── sampling.md           # Temperature sampling algorithms
│   └── results.md            # Experimental results
│
├── tests/
│   ├── test_autoencoder.py
│   ├── test_energy_transformer.py
│   └── test_sampling.py
│
├── CLAUDE.md                  # This file
├── README.md                  # Project overview
└── requirements.txt           # Dependencies
```

## Key Metrics to Track

### Autoencoder Quality
- **Reconstruction accuracy**: Token-level accuracy (target: >99.9%)
- **Robustness**: Accuracy under Gaussian noise (σ=0.3)
- **Posterior collapse**: Number of collapsed dimensions
- **KL divergence**: Per-dimension KL

### CALM Performance
- **BrierLM**: Composite language modeling score
- **Brier-n**: n-gram scores (n=1..4)
- **Training FLOPs**: Total computation
- **Inference FLOPs**: Per-token computation
- **Wall-clock time**: Actual speed

### Interpretability
- **Semantic probing accuracy**: Can linear probes recover linguistic features?
- **SAE similarity**: Correlation with SAE features
- **Steering effectiveness**: Success rate of vector manipulation

## Common Pitfalls

### 1. Posterior Collapse
**Symptom**: Some latent dimensions collapse to prior
**Solution**: Use KL clipping (λ_KL = 0.5)
**Check**: Monitor per-dimension KL divergence

### 2. Brittle Representations
**Symptom**: Small vector perturbations → garbage reconstruction
**Solution**: Increase dropout rates (p=0.15 for both token and latent)
**Check**: Test reconstruction under noise injection

### 3. Energy Loss Instability
**Symptom**: Training diverges or collapses
**Solution**:
- Use N=8 model samples (not too small)
- Use M=100 target samples (reduce variance)
- Check α=1 (not <1 or =2)

### 4. Discrete Input Degradation
**Symptom**: Performance drops significantly
**Solution**: Use discrete token input (not continuous latent input)
**Reason**: Model struggles to unpack compact representations

## Future Directions

### Short-term (Weeks 1-8)
1. Reproduce paper results
2. Build interpretability tools
3. Compare to SAE features
4. Test steerability

### Medium-term (Months 2-3)
1. Context-aware autoencoder
2. Larger scale experiments
3. Multi-agent communication
4. Integration with Topologies (mobile latent exploration)

### Long-term (Months 4+)
1. Semantically-grounded latent space
2. Autoregressive autoencoder
3. Unified scaling laws (parameters, data, K)
4. Real-world applications

## References

**Primary Paper**:
- [Continuous Autoregressive Language Models](https://arxiv.org/abs/2510.27688)
- Code: https://github.com/shaochenze/calm
- Project: https://shaochenze.github.io/blog/2025/CALM

**Related Work**:
- Energy Transformer (Shao et al., 2025)
- Latent Diffusion (Rombach et al., 2022)
- VQ-VAE (van den Oord et al., 2017)
- MegaByte (Yu et al., 2023)
- Large Concept Models (Meta, 2024)

**Hidden Layer Context**:
- Latent Lens: `../lens/`
- Latent Topologies: `../topologies/`
- AI-to-AI Communication: `../../../communication/ai-to-ai-comm/`

## Questions?

- **Lab-wide setup**: See `/docs/infrastructure/`
- **Research methodology**: See `/docs/workflows/research-methodology.md`
- **Latent space projects**: See `../README.md`
