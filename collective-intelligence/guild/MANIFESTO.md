# Toward a Science of Collective Intelligence

**A Position Paper**

*Hidden Layer Lab — Project Guild*

---

## Abstract

Artificial intelligence research has historically focused on improving the capabilities of individual models. Recent advances increasingly rely on collections of interacting entities: foundation models, specialized agents, external tools, memory systems, humans, and institutional processes. While individual intelligence has improved dramatically, there remains no general science describing how heterogeneous intelligences should organize, communicate, adapt, and evolve.

We argue that this represents a fundamental gap in AI research.

Existing work approaches collective intelligence in two ways: by analyzing natural collective systems (markets, organizations, swarms, crowds), or by engineering specific multi-agent architectures and evaluating them on fixed benchmarks. We propose a third methodology: **designing persistent interactive worlds that continuously generate, evaluate, and evolve hypotheses about collective intelligence through the joint participation of humans and AI.** Unlike static benchmarks or handcrafted agent architectures, such environments constitute an ongoing search process over interaction patterns, communication protocols, organizational structures, and emergent behaviors.

Rather than searching for a single optimal architecture, we seek to build environments that continually discover better forms of collective intelligence.

---

## 1. Introduction

For decades, AI has largely optimized individual minds.

Scaling laws, larger models, improved training objectives, and reinforcement learning have all focused on increasing the capability of individual systems. This program has succeeded beyond most expectations. But its very success has exposed a boundary: the frontier systems of the present are no longer individuals.

Modern AI systems increasingly consist of interacting components:

- multi-agent reasoning systems
- retrieval-augmented generation
- tool-using agents
- planning and decomposition systems
- verifier and critic models
- human–AI teams embedded in institutional processes

In practice, state-of-the-art performance increasingly depends on how these components interact rather than on the capability of any single component. A weaker model in a well-structured verification loop can outperform a stronger model reasoning alone; a team of specialized agents can fail catastrophically under a communication topology that a trivial reorganization would fix. Research on multi-agent systems, hybrid human–AI intelligence, and agent orchestration reflects this transition — but there is still no unified theory describing how such systems should be designed.

The situation resembles chemistry before the periodic table: an abundance of recipes, a shortage of laws.

We argue that the next frontier of AI is **organizational rather than individual**, and that treating it as such requires new scientific instruments — not just new architectures.

## 2. The Missing Science

Existing research asks:

- How should we train models?
- Which architecture performs best?
- Which prompting strategy works?

These are questions about individuals. We believe the more fundamental questions are questions about collectives:

- How should intelligence divide labor?
- What forms of communication maximize collective performance?
- What kinds of specialization emerge, and under what pressures?
- How should memory be distributed across a collective?
- How should trust develop, and how should it decay?
- How should computation be allocated across many minds?
- Which interaction patterns produce resilience, and which produce fragility?
- What entirely novel organizational forms exist that no human institution has ever instantiated?

Fragments of answers exist, scattered across organizational science, economics, distributed systems, evolutionary biology, and multi-agent reinforcement learning. But each field studies its own substrate with its own methods, and none of them can run controlled, repeatable, large-scale experiments on *heterogeneous collectives of humans and machine intelligences* — the very systems now being deployed.

Today these questions are explored through relatively small-scale manual experimentation: a paper proposes an agent topology, evaluates it on a handful of benchmarks, and reports a delta. This is real progress, but it is inherently low-bandwidth. The design space of interacting intelligences is combinatorially enormous, and we are sampling it by hand.

## 3. Beyond Multi-Agent Systems

This paper does not propose another multi-agent framework.

Current frameworks typically assume, as fixed primitives:

- agents, as bounded persistent individuals
- roles, assigned in advance
- communication protocols, designed by the engineer
- hierarchies, mirrored from human organizations

These assumptions may themselves be incorrect. They import the folk ontology of human institutions into a substrate with radically different properties: machine intelligences can be copied, merged, forked, paused, and specialized in ways no human employee can. There is no reason to expect that the org chart — a technology for coordinating creatures with one body and one lifetime — is the right abstraction for coordinating software minds.

Instead we propose studying **interaction itself**.

Organizations, hierarchies, roles, markets, memories, and institutions should be treated as *emergent hypotheses* rather than fixed design primitives — structures that arise, persist, and dissolve within a field of repeated interactions, and that earn their place in the ontology only by recurring.

This shifts the research objective from optimizing existing architectures to **discovering new ontologies for collective intelligence**.

## 4. Why Games?

Games represent one of humanity's most effective mechanisms for exploring large strategy spaces.

They naturally produce:

- optimization under pressure
- adaptation to shifting conditions
- emergence of unplanned strategies
- cooperation and competition in the same arena
- metagame formation — strategy about strategy

Players routinely discover strategies unforeseen by designers. Every long-lived competitive game develops a meta: a live, community-scale search process over the space of viable strategies, complete with hypothesis generation, empirical testing, and publication. This is science in everything but name — and it is conducted, for free, at a scale no laboratory can fund.

Citizen-science games such as Foldit, EteRNA, and Eyewire have shown that well-designed interactive systems can contribute to genuine scientific discovery, improve on expert algorithms, and generate datasets of lasting value. But these games aimed players at *fixed* problems: fold this protein, design this RNA.

We argue that games should evolve from solving fixed scientific problems to **exploring open-ended spaces of collective intelligence** — where the object of play is the design of intelligent organization itself, and where every successful strategy is, simultaneously, a scientific hypothesis.

## 5. The Collective Intelligence Environment

Rather than proposing another benchmark, we propose a continuously evolving environment.

Its defining properties:

- **persistent** — the world outlives any session; consequences accumulate
- **multiplayer** — many humans and many AI systems, simultaneously
- **partially observable** — no participant sees the whole; information is a resource
- **cooperative and competitive** — both pressures present, neither dominant
- **adaptive** — the environment itself responds to the strategies discovered in it
- **open-ended** — no terminal state, no final leaderboard

Humans and AI participate together, on symmetric footing where possible. Players do not micromanage characters; they shape relationships, interactions, constraints, and incentives, and are rewarded for the *organizations* that emerge.

Every interaction is simultaneously gameplay and experiment. Each session records interaction topology, communication patterns, information flow, coordination strategy, resource allocation, decision history, emergent structure, performance, and adaptation over time. The environment thereby functions, at once, as:

- a research platform
- a benchmark
- a reinforcement learning environment
- a simulation testbed
- a hypothesis generator for future AI architectures

## 6. Research Questions

An environment of this kind makes previously untestable questions empirical:

- Can entirely novel organizational structures emerge — forms with no human precedent?
- How does communication topology affect collective reasoning quality?
- Which interaction primitives consistently produce better outcomes across task distributions?
- Which organizations are robust to failure, defection, and deception — and which are brittle?
- How does diversity of capability, information, and objective influence collective performance?
- Can organizations themselves be *learned* — optimized as first-class objects?
- Can AI invent organizations humans never discovered?
- Can humans invent organizations AI fails to discover?

The last two questions are symmetric on purpose. A joint human–AI search process is interesting precisely where the two populations explore differently.

## 7. A New Kind of Benchmark

Benchmarks today are largely static. They are constructed once, saturate, and are retired — often within a few years, now sometimes within months. A static benchmark measures a fixed capability; it cannot measure the capacity to *keep organizing well as the world changes*, which is precisely the capability that matters for collectives.

An evolving environment changes continuously. Its population of strategies is itself the difficulty: every solution published into the world becomes a condition that future solutions must survive. The benchmark becomes:

- **self-renewing** — new strategies create new problems
- **adversarial** — competing collectives probe each other's weaknesses
- **adaptive** — the environment redistributes pressure toward unsolved regimes
- **increasingly difficult** — a ratchet driven by the participants themselves

This resembles natural evolution and immune dynamics more than traditional evaluation. Saturation is not failure; it is a signal consumed by the environment to generate the next regime.

## 8. From Architectures to Ecologies

Most AI research assumes intelligence is a property of individual entities: an agent *has* intelligence, and the research question is how to give it more.

We propose a different working hypothesis:

> **Intelligence may be better understood as a property of interactions within adaptive systems.**

On this view, an "agent" is a slow-moving pattern in a field of interactions — the way a whirlpool is a pattern in water — and organizational forms are the phases and textures this field can take. The research object is not the entity but the ecology: which interaction patterns are stable, which are fertile, which laws govern their transitions.

This connects naturally to work on complex adaptive systems, network science, and hybrid human–AI collective intelligence. It extends them by proposing an experimental platform *specifically designed to discover new interaction laws rather than evaluate existing ones* — a laboratory in which the ontology itself is under test.

## 9. Open Problems

We intentionally leave fundamental questions unresolved. They constitute the research agenda, not the implementation backlog.

**What is the correct primitive?** Not "agent." Not "role." Not "organization." Perhaps *interaction*. Perhaps *information flow*. Perhaps *commitment*. Perhaps something not yet named. The purpose of the environment is to discover this, and our first candidate decomposition (signal, transfer, bind, act, reinforce) is offered to be falsified.

**What should participants manipulate?** Individuals, relationships, communication rules, resource flows, incentives, institutions, entire civilizations — the right lever is unknown, and is itself an experimental variable.

**What constitutes progress?** Survival, discovery, adaptation, growth — the objective function is a design variable, and the sensitivity of emergent organization to the objective is one of the central phenomena to measure.

**What data transfers?** Not all gameplay is science. A core responsibility of the platform is identifying which interaction patterns generalize beyond the game — validated by porting discovered strategies into real multi-agent systems and measuring the delta.

## 10. Conclusion

The central claim of this paper is simple.

The next generation of AI progress may depend less on building increasingly capable individual models and more on understanding the principles governing systems of interacting intelligences.

If that is true, then the scientific community lacks its most important tool: **an environment capable of continuously discovering, testing, and evolving those principles.**

Analyzing natural collectives gives us observation without control. Engineering fixed architectures gives us control without search. The third methodology — persistent, evolving worlds explored jointly by humans and AI — gives us both, at a bandwidth no manual research program can match.

We propose that building such environments should become a central research agenda for the coming decade. Guild is our first instrument.

---

*Companion documents: [PRD.md](PRD.md) (product requirements), [docs/literature-survey.md](docs/literature-survey.md) (what exists and where the gap is), [docs/interaction-primitives.md](docs/interaction-primitives.md) (candidate primitives), [docs/minimal-simulation.md](docs/minimal-simulation.md) (the smallest simulation that produces emergent coordination, with results).*
