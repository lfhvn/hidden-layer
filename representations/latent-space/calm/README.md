# CALM: Continuous Autoregressive Language Models

> From discrete tokens to continuous vectors: A new paradigm for efficient language modeling

## Overview

CALM (Continuous Autoregressive Language Models) replaces traditional token-by-token generation with **vector-by-vector generation**. Instead of predicting the next token, CALM predicts the next continuous vector, where each vector compresses K tokens with >99.9% reconstruction accuracy.

**Result**: K-fold reduction in autoregressive steps → better performance-compute trade-offs

**Based on**: [Shao et al., 2025 - arXiv:2510.27688](https://arxiv.org/abs/2510.27688)

## Key Innovation

```
Traditional LM:  [The] → [cat] → [sat] → [on] → [the] → [mat]
                  ├────┼────┼────┼────┼────┤
                  6 autoregressive steps

CALM (K=3):      [The|cat|sat] → [on|the|mat]
                  ├────────────┼────────────┤
                  2 autoregressive steps (3x faster)
```

## Architecture Components

### 1. Autoencoder (Discrete ↔ Continuous)
- **Input**: K tokens (e.g., "The cat sat")
- **Output**: Single l-dimensional vector (l=128 for K=4)
- **Reconstruction**: >99.9% token-level accuracy
- **Robustness**: Tolerates σ≈0.3 Gaussian noise

**Key technique**: VAE + dropout → smooth, robust latent manifold

### 2. Energy Transformer (Likelihood-Free LM)
- **Backbone**: Standard Transformer
- **Generative Head**: Energy-based, single-step generation
- **Training**: Energy loss (strictly proper scoring rule)
- **Advantage**: No iterative sampling (vs. diffusion/flow matching)

### 3. BrierLM (Likelihood-Free Evaluation)
- **Challenge**: No explicit likelihoods → can't compute perplexity
- **Solution**: Brier score (strictly proper, sample-based)
- **BrierLM**: Geometric mean of Brier-n for n-grams (n=1..4)

### 4. Temperature Sampling (Likelihood-Free Control)
- **Exact**: Rejection sampling algorithm (provably correct)
- **Approximate**: Batch-based algorithm (efficient, asymptotically unbiased)

## Performance Results

From the paper:

| Model | Params | Train FLOPs | Infer FLOPs | BrierLM |
|-------|--------|-------------|-------------|---------|
| Transformer-S | 281M | 6.6e20 | 4.4e8 | 6.05 |
| **CALM-M (K=4)** | **371M** | **3.7e20** | **2.9e8** | **5.72** |
| Transformer-M | 465M | 11.9e20 | 7.9e8 | 7.07 |

**CALM-M achieves Transformer-S performance at:**
- 44% fewer training FLOPs
- 34% fewer inference FLOPs

## Research Questions

### Representation Quality
- What information do CALM vectors encode at different K?
- How does compression affect semantic vs. syntactic representation?

### Interpretability
- Can we interpret individual dimensions of CALM vectors?
- How do CALM vectors compare to SAE features?

### Steerability
- Can we steer generation by manipulating latent vectors?
- Can we find "steering directions" in CALM space?

### Communication
- Can agents communicate via CALM vectors?
- What's the efficiency gain vs. token-based communication?

### Efficiency-Quality Trade-offs
- How does semantic bandwidth K affect the performance-compute frontier?
- What is the optimal K for different tasks?

## Quick Start

### Installation

```bash
cd representations/latent-space/calm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train Autoencoder

```bash
python src/train_autoencoder.py \
  --data pile \
  --tokens 15B \
  --chunk_size 4 \
  --latent_dim 128
```

### Train CALM Model

```bash
python src/train_calm.py \
  --autoencoder checkpoints/autoencoder_k4.pt \
  --model_size M \
  --steps 250k
```

### Evaluate

```bash
python src/evaluate.py \
  --model checkpoints/calm_m_k4.pt \
  --data wikitext103 \
  --metric brierlm
```

## Project Structure

```
calm/
├── src/                    # Core implementation
│   ├── autoencoder.py      # VAE autoencoder
│   ├── energy_transformer.py # Energy Transformer
│   ├── train_*.py          # Training scripts
│   └── evaluate.py         # BrierLM evaluation
│
├── experiments/            # Research experiments
│   ├── analyze_vectors.py  # Interpretability
│   ├── compare_sae.py      # Compare to SAEs
│   ├── steering.py         # Steerability tests
│   └── communication.py    # Agent communication
│
├── docs/                   # Detailed documentation
├── tests/                  # Unit tests
└── CLAUDE.md              # Development guide
```

## Connection to Other Projects

### Latent Space (siblings)
- **Lens** (SAE interpretability): Compare CALM vectors to SAE features
- **Topologies** (mobile exploration): Navigate CALM's smooth manifold

### Communication
- **AI-to-AI Comm**: Use CALM vectors for efficient agent communication
- **Multi-Agent**: Coordination via latent vectors

### Alignment
- **Steerability**: Vector-level behavioral control

## Key Insights from Paper

### Why CALM Works

1. **Scalable Information Density**
   - Discrete tokens: 15-18 bits (log₂(vocab_size))
   - Continuous vectors: Gracefully scalable by increasing dimension

2. **Smooth Latent Manifold**
   - VAE regularization → smooth space
   - Dropout → redundant, robust representation
   - Result: Small errors don't break generation

3. **Single-Step Generation**
   - Energy Transformer: No iterative sampling
   - Diffusion/Flow: 100s of steps needed
   - Trade-off: Speed vs. expressiveness

4. **Semantic Bandwidth as Scaling Axis**
   - Traditional scaling: Parameters, data
   - CALM adds: Semantic bandwidth (K)
   - New design space for efficiency

### Performance-Compute Frontier

```
            Performance (BrierLM ↑)
                   ↑
    CALM-L (K=4)  ●
                 /│
               /  │
    CALM-M   ●   │  ← CALM frontier
           /     │
         /       ●  Transformer-L
       /       /
     ●       /      Transformer-M
   /       ●
 /       /
      ●  Transformer-S
     └────────────────→ Compute (FLOPs)
```

## Implementation Status

- [ ] Phase 1: Reproduction
  - [ ] Autoencoder implementation
  - [ ] Energy Transformer implementation
  - [ ] Training pipeline
  - [ ] BrierLM evaluation

- [ ] Phase 2: Interpretability
  - [ ] Vector analysis tools
  - [ ] Semantic probing
  - [ ] SAE comparison

- [ ] Phase 3: Steerability
  - [ ] Vector manipulation
  - [ ] Steering direction search
  - [ ] Behavioral control tests

- [ ] Phase 4: Communication
  - [ ] Agent communication protocols
  - [ ] Efficiency measurements
  - [ ] Multi-agent integration

## Resources

**Paper & Code**:
- Paper: https://arxiv.org/abs/2510.27688
- Official Code: https://github.com/shaochenze/calm
- Project Page: https://shaochenze.github.io/blog/2025/CALM

**Hidden Layer Context**:
- Development Guide: `CLAUDE.md`
- Latent Space Overview: `../README.md`
- Lab Documentation: `/docs/`

## Citation

```bibtex
@article{shao2025calm,
  title={Continuous Autoregressive Language Models},
  author={Shao, Chenze and Li, Darren and Meng, Fandong and Zhou, Jie},
  journal={arXiv preprint arXiv:2510.27688},
  year={2025}
}
```
