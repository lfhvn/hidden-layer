# Representations Research Area

**Research Focus**: What are the internal representations models use, and how can we make them interpretable to humans?

---

## Projects

### latent-space/

Research into understanding and experiencing latent representations through two complementary approaches:

#### lens/
SAE (Sparse Autoencoder) interpretability web application for exploring learned features.

**Capabilities**:
- Feature extraction and visualization
- Activation heatmaps
- Layer-wise exploration
- Feature labeling and annotation

**See**: `latent-space/lens/README.md` for details

> **Moved to the incubator.** Two earlier representations efforts —
> **topologies/** (multi-sensory mobile latent exploration) and
> **state-explorer/** (real-time activation visualization) — have been moved to
> `incubator/` because they are design/stub stage and not yet runnable. See
> `incubator/README.md` for status and what each needs to graduate back here.

---

## Key Research Questions

1. **Features**: What features do models learn? (via SAEs)
2. **Experience**: How can humans experience high-dimensional embeddings?
3. **Geometry**: What is the structure of latent space?
4. **Navigation**: Can we navigate latent space to understand/steer behavior?
5. **Interpretability**: Can we make internal representations human-understandable?

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

### Lens (Digital Exploration)
1. **Extract**: Use SAEs to find interpretable features
2. **Visualize**: Heatmaps, activation patterns
3. **Label**: Human annotation of features
4. **Analyze**: What concepts cluster together?

_(Topologies' embodied-exploration methodology now lives with the project in
`incubator/topologies/`.)_

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
