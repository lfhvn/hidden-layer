# Theory of Mind Research Area

**Research Focus**: How do AI systems understand mental states - both their own (introspection) and others' (theory of mind)?

---

## Projects

### selphi/
Theory of mind evaluation using ToMBench, OpenToM, and SocialIQA benchmarks.

**Capabilities Tested**:
- False belief understanding
- Perspective-taking
- Epistemic reasoning
- Social dynamics

**Key Questions**:
- Which ToM capabilities transfer across model scales?
- How does ToM relate to deception detection?
- Can models develop genuine understanding or just learned patterns?

**See**: `selphi/README.md` for details

### introspection/
Reproducing Anthropic's introspection findings - can models accurately report their internal states?

**Focus Areas**:
- Concept vectors and activation steering
- Honest reporting vs. learned responses
- What activations correspond to introspection?

**Key Questions**:
- Can models accurately report internal states?
- Is introspection genuine self-knowledge or pattern matching?
- How can we use introspection for alignment?

**See**: `introspection/README.md` for details

---

## The ToM Spectrum

```
Understanding Others (SELPHI) ←→ Understanding Self (Introspection)
```

**Central Question**: Is there a unified mechanism for theory of mind, whether directed at others or at oneself?

---

## Cross-Area Connections

### With Representations
What latent features activate during ToM tasks? Can we identify "ToM neurons" or circuits?

### With Alignment
- ToM enables deception detection (understanding when others are lying)
- Introspection provides alignment signals (honest reporting)
- Together: Can we create reliably aligned systems?

### With Communication
Do agents with better ToM coordinate more effectively in multi-agent settings?

---

## Research Methodology

1. **Evaluation**: Benchmark performance on ToM tasks
2. **Mechanistic**: What features/circuits enable ToM?
3. **Intervention**: Can we steer ToM behavior?
4. **Application**: Use ToM for alignment and coordination

---

## Recent Findings

_To be updated as research progresses_

---

## Future Directions

- Cross-linguistic ToM
- ToM in multi-modal models
- Developmental trajectories (how ToM emerges with scale/training)
- ToM and deception in adversarial settings
