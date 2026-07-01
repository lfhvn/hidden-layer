# Project Codename: Guild

**Product Requirements Document v0.1**

---

## Vision

Build the first interactive environment for discovering the principles of collective intelligence.

Rather than optimizing individual AI systems, Guild explores how intelligence emerges from interactions among many heterogeneous entities. The platform serves simultaneously as:

- A compelling strategy game.
- A research environment.
- A benchmark.
- A training environment.
- A hypothesis generator for future AI architectures.

The long-term ambition is not to discover a single optimal architecture, but to create an environment that continually discovers better forms of intelligence.

---

## Problem

Modern AI research is increasingly moving beyond individual models toward systems of interacting components:

- Multiple agents
- Memory
- Tools
- Humans
- Planning systems
- Verification systems
- External environments

While individual model capability continues to improve, there is no accepted theory describing how collections of intelligent entities should interact.

Current research explores this design space manually through papers, experiments, and benchmarks. This approach is inherently low bandwidth.

We believe a sufficiently compelling game could motivate millions of participants to collectively explore this design space.

---

## Product Thesis

Games are powerful search processes.

- Players naturally optimize.
- They discover unexpected strategies.
- They develop new metas.
- They continuously explore enormous combinatorial spaces.

Guild transforms this exploration into a scientific process. Every successful strategy becomes a hypothesis about collective intelligence.

---

## Design Principles

**Principle 1 — Do not assume the correct architecture.**
The platform exists to discover architectures.

**Principle 2 — Do not hardcode organizations.**
Organizations should emerge.

**Principle 3 — Prioritize interactions over entities.**
The fundamental objects of the platform are interactions. Entities are temporary structures emerging from repeated interaction.

**Principle 4 — Optimize for emergence.**
Interesting behavior should arise from simple rules rather than scripted mechanics.

**Principle 5 — Research and gameplay must reinforce one another.**
Winning the game should naturally generate valuable scientific data.

---

## Research Questions

The platform should help answer questions such as:

- How should intelligent systems divide work?
- How should they communicate?
- When should they cooperate? When should they compete?
- How should trust develop?
- What kinds of memory are useful?
- How should specialization emerge?
- What interaction patterns produce robust intelligence?
- Which organizational structures fail?
- What new organizational structures exist that humans have never intentionally designed?

---

## Core Product

A multiplayer strategy RPG centered on designing systems of interacting intelligences.

Players do not directly control characters. Instead they shape relationships, interactions, constraints, and incentives. The game rewards the emergence of successful organizations rather than tactical execution alone.

---

## Research Platform

Every session records:

- Interaction topology
- Communication patterns
- Information flow
- Coordination strategy
- Resource allocation
- Decision history
- Emergent organizational structures
- Performance
- Adaptation over time

These become structured research data.

---

## Long-Term Platform

The platform eventually becomes:

- A benchmark for collective intelligence.
- A reinforcement learning environment.
- A simulation platform.
- A research environment.
- A source of novel organizational data.
- Potentially a foundation for future AI systems capable of designing intelligent organizations.

---

## Success Metrics

| Dimension | Metric |
|-----------|--------|
| Consumer | Players return because the game is intrinsically rewarding. |
| Research | The platform produces reproducible hypotheses about collective intelligence. |
| Scientific | Strategies discovered inside the environment transfer to real multi-agent systems. |
| Platform | The environment continuously evolves rather than converging on a single dominant solution. |

---

## Unknowns

These represent the primary research agenda rather than implementation work.

**Unknown 1 — What are the primitive interactions?**
We intentionally avoid assuming concepts such as "agent," "class," or "organization." Instead we seek the smallest useful interaction primitives from which larger structures emerge.
→ Current candidates: [docs/interaction-primitives.md](docs/interaction-primitives.md)

**Unknown 2 — What should players actually manipulate?**
Possibilities include: individuals, relationships, communication rules, resource flows, incentives, institutions, entire civilizations. This remains open.

**Unknown 3 — What constitutes progress?**
Players may optimize for survival, discovery, adaptation, scientific advancement, civilization growth, or unknown objectives. The objective itself is a design variable.

**Unknown 4 — What data is genuinely valuable?**
The product should not assume that all gameplay produces useful AI research. One of the platform's core responsibilities is identifying which interaction patterns transfer beyond the game.

---

## Immediate Next Deliverables

1. ✅ Write a research manifesto: "Toward a Science of Collective Intelligence." → [MANIFESTO.md](MANIFESTO.md)
2. ✅ Survey literature spanning multi-agent AI, complex systems, game theory, organizational science, artificial life, evolutionary computation, and human-computer interaction. → [docs/literature-survey.md](docs/literature-survey.md)
3. ✅ Define candidate interaction primitives without assuming existing organizational metaphors. → [docs/interaction-primitives.md](docs/interaction-primitives.md)
4. ✅ Design the smallest possible simulation capable of producing emergent coordination. → [docs/minimal-simulation.md](docs/minimal-simulation.md), implemented in [`guild/`](guild/)
5. ⬜ Validate whether player-discovered strategies improve real-world multi-agent systems. → Planned: port kernel-discovered team structures into `communication/multi-agent/` strategies and measure the delta.

---

## Working Mission Statement

We are building an environment where new forms of collective intelligence can be discovered, tested, and evolved.

The game is the interface. The research platform is the product.

The long-term objective is a science of intelligence that extends beyond individual minds.
