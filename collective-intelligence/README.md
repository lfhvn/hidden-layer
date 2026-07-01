# Collective Intelligence Research Area

**Research Focus**: How does intelligence emerge from interactions among many heterogeneous entities — and what environment could continuously discover better forms of it?

This area treats collective intelligence as a discipline missing its central instrument: existing research either analyzes natural collectives or engineers fixed multi-agent architectures. We pursue a third methodology — persistent interactive environments that generate, evaluate, and evolve hypotheses about intelligent organization through joint human–AI participation.

---

## Projects

### guild/
The flagship project: a research program and (eventually) multiplayer strategy environment where the object of play is designing systems of interacting intelligences. Currently comprises the position paper, literature survey, candidate interaction primitives, and a minimal simulation kernel demonstrating emergent division of labor and stable partnerships from interaction-level rules alone.

**Key Questions**:
- What are the primitive interactions from which organizations emerge?
- Which interaction patterns produce robust collective intelligence, and which fail?
- Can strategies discovered inside an environment transfer to real multi-agent systems?

**See**: `guild/README.md` for the document map and quick start.

---

## Cross-Area Connections

### With Communication
`communication/multi-agent/` strategies (debate, CRIT, consensus) are hand-designed points in the organizational design space Guild searches. They serve as baselines for Deliverable 5 (transfer validation), and Guild's event logs give them a common measurement language.

### With Theory of Mind
Trust, reputation, and deception are emergent phenomena in Guild once signaling exists. SELPHI-style ToM measures apply to participants; introspection connects to what loci can report about their own state.

### With Alignment
Which incentive structures keep emergent organizations aligned with their designers' intent? Guild makes organizational misalignment (e.g., lock-in without competence — see the kernel's "structure without competence" result) an observable, reproducible phenomenon.

---

## Research Methodology

1. **Kernel**: minimal interaction-first simulations; emergence claims encoded as tests
2. **Enrichment**: add one primitive at a time (SIGNAL, TRANSFER); measure its marginal contribution
3. **Population**: LLM-backed and human participants over the same event interface
4. **Transfer**: port discovered structures into real multi-agent systems and measure the delta

---

## Recent Findings

- Division of labor and persistent partnerships emerge from joint action + plasticity + forgetting alone — no communication or economics required (see `guild/docs/minimal-simulation.md`).
- Specialization and stable grouping are *separable*: practice produces the former without the latter; bond reinforcement produces the latter without improving the former.
- Bond reinforcement without capability growth yields lock-in around weak teams — an emergent organizational failure mode ("structure without competence").

---

## Future Directions

- Phase diagram of the "organization phase" over plasticity parameters
- Communication and resource primitives; emergent trust and markets
- Environmental drift and organizational adaptation vs. death
- Human-playable interface over the kernel event stream
