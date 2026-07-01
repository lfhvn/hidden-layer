# Representations Research Area

**Research Focus**: What are the internal representations models use, and how can we make them interpretable to humans?

---

## Projects

### latent-space/

Research into understanding latent representations:

#### lens/
SAE (Sparse Autoencoder) interpretability backend for exploring learned features.

**Capabilities**:
- SAE training on captured activations
- Activation capture via forward hooks on HF models
- Feature extraction pipeline
- Feature labeling and annotation

**See**: `latent-space/lens/README.md` for details

#### calm/
Continuous autoregressive language modeling experiments (variational autoencoder + energy transformer).

**See**: `latent-space/calm/README.md` for details

---

## Key Research Questions

1. **Features**: What features do models learn? (via SAEs)
2. **Geometry**: What is the structure of latent space?
3. **Navigation**: Can we navigate latent space to understand/steer behavior?
4. **Interpretability**: Can we make internal representations human-understandable?

---

## Cross-Area Connections

### With Theory of Mind & Introspection
What features/activations correspond to:
- Theory of mind reasoning?
- Self-knowledge and introspection?
- Can we identify "ToM circuits"?

### With Communication
- Can agents communicate via latent coordinates instead of language?
- Do agents in multi-agent settings develop shared representations?
- What is the geometry of "communication subspace"?

### With Alignment
- Can we navigate latent space to steer behavior toward alignment?
- What steering vectors produce reliable behavioral changes?
- How do we monitor for deceptive representations?

---

## Research Methodology

### Lens (SAE Interpretability)
1. **Extract**: Use SAEs to find interpretable features
2. **Visualize**: Heatmaps, activation patterns
3. **Label**: Human annotation of features
4. **Analyze**: What concepts cluster together?

---

## Recent Findings

_To be updated as research progresses_

---

## Future Directions

- Multi-modal representation analysis
- Temporal dynamics of representations
- Cross-model representation comparisons
- Representation drift during fine-tuning
- Causal representation analysis
