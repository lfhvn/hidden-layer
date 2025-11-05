# Continuous Autoregressive Language Models (CALM)

**arXiv ID:** 2510.27688
**Authors:** Chenze Shao, Darren Li, Fandong Meng, Jie Zhou
**Submitted:** October 31, 2025
**GitHub:** https://github.com/shaochenze/calm
**Contact:** chenzeshao@tencent.com

## Overview

CALM introduces a paradigm shift in language modeling from **discrete next-token prediction** to **continuous next-vector prediction**.

## Core Innovation

### Traditional Approach
- Autoregressive LLMs predict one token at a time
- Sequential generation is a bottleneck for speed
- Limited semantic bandwidth per generation step

### CALM Approach
1. **High-Fidelity Autoencoder**: Compresses K tokens into a single continuous vector
   - Reconstruction accuracy: >99.9%
   - Autoencoder size: 75M parameters

2. **Continuous Autoregressive Model**: Operates in vector space rather than token space
   - Predicts entire token chunks simultaneously
   - Reduces generation steps by factor of K

### Key Benefit
Increases the **semantic bandwidth** of each generative step, overcoming the sequential bottleneck.

## Methodology

### Two-Stage Pipeline

#### Stage 1: Autoencoder Training
- Compresses chunks of K consecutive tokens → single continuous vector
- Near-perfect reconstruction (99.9%+ accuracy)
- Training data: ~15B tokens from Pile-uncopyrighted dataset

#### Stage 2: Continuous Language Model
- Autoregressive model in vector space
- Energy-based training methodology
- Likelihood-free approach

### Technical Components

**Training Framework:**
- Energy-based training for generative modeling
- Likelihood-free evaluation using BrierLM metric
- Temperature sampling for controlled generation

**Data Requirements:**
- Dataset: Pile-uncopyrighted (~2.5TB)
- Autoencoder: ~15B tokens
- CALM model: Remaining Pile data

## Available Models

Pre-trained checkpoints on HuggingFace:

| Model | Parameters | BrierLM Score |
|-------|-----------|---------------|
| CALM-M | 371M | 5.72 |
| CALM-L | 735M | 6.58 |
| CALM-XL | 1.82B | 8.53 |
| Autoencoder | 75M | - |

## Evaluation

**BrierLM**: A likelihood-free evaluation metric developed for CALM
- Enables assessment without computing explicit probabilities
- Higher scores indicate better performance

## Research Implications

### For Language Modeling
- Challenges token-by-token generation paradigm
- Demonstrates viability of continuous intermediate representations
- Opens new scaling dimension beyond just parameter count

### For Efficiency
- Reduces number of generation steps by factor K
- Potential for faster inference
- More efficient semantic representation

### For AI Research (Hidden Layer Lab Context)

**Relevant to Multiple Projects:**

1. **Latent Space Projects**
   - Direct application to understanding continuous representations
   - Could inform Latent Lens (SAE interpretability)
   - Relevant to Latent Topologies (exploring representation spaces)

2. **AI-to-AI Communication**
   - CALM's vector-based approach aligns with non-linguistic communication research
   - Demonstrates language can be represented and transmitted as continuous vectors
   - Potential for more efficient agent-to-agent messaging

3. **Introspection & Interpretability**
   - Continuous representations may be easier to analyze than discrete tokens
   - Autoencoder's high fidelity suggests strong structure in representation space
   - Could enable new introspection techniques

4. **Multi-Agent Systems**
   - Vector-based communication between agents
   - Potential for more efficient coordination protocols
   - Shared continuous representation space

## Technical Details

### Architecture Components
- **Autoencoder**: Token chunk ↔ continuous vector mapping
- **Energy-based model**: Scores vector sequences
- **Sampling mechanism**: Temperature-based generation from vector predictions

### Training Approach
- Likelihood-free: Avoids explicit probability computation
- Energy-based: Learns energy function over vector sequences
- Two-stage: Separate autoencoder and language model training

## Links & Resources

**Paper:**
- arXiv abstract: https://arxiv.org/abs/2510.27688
- PDF: https://arxiv.org/pdf/2510.27688.pdf
- HTML: https://arxiv.org/html/2510.27688

**Code & Models:**
- GitHub: https://github.com/shaochenze/calm
- HuggingFace: https://huggingface.co/papers/2510.27688
- Blog: https://shaochenze.github.io/blog/2025/CALM

**Alternatives:**
- EmergentMind: https://www.emergentmind.com/papers/2510.27688
- alphaXiv: https://www.alphaxiv.org/abs/2510.27688
- DeepLearn: https://deeplearn.org/arxiv/647198/continuous-autoregressive-language-models

## Questions for Further Investigation

1. **Compression-Generation Tradeoff**: How does chunk size K affect reconstruction vs. generation quality?
2. **Representation Structure**: What structure emerges in the continuous vector space?
3. **Interpretability**: Are continuous vectors more interpretable than tokens?
4. **Scaling Laws**: How do continuous models scale compared to token-based models?
5. **Transfer Learning**: Can the autoencoder generalize to domains beyond training data?
6. **Multi-Agent Applications**: Can CALM representations enable better agent communication?

## Potential Experiments (Hidden Layer Lab)

### Latent Space Exploration
- Visualize CALM's continuous vector space
- Compare to SAE representations
- Explore topology and structure

### Communication Experiments
- Use CALM vectors for agent-to-agent messaging
- Compare efficiency vs. token-based communication
- Test on multi-agent coordination tasks

### Introspection Studies
- Can models report on their continuous representations?
- Do continuous states enable better self-knowledge?
- Compare to token-based introspection

### Interpretability Analysis
- Apply SAE techniques to CALM's vector space
- Identify semantic dimensions
- Test steerability via vector manipulation

---

*Summary compiled from arXiv search results and GitHub repository (https://github.com/shaochenze/calm)*
*Full paper available at: https://arxiv.org/abs/2510.27688*
