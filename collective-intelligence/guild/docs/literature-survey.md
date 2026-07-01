# Literature Survey: The Space Around Guild

**Purpose**: Map the fields Guild draws on, identify what each contributes, and show precisely where the gap is — no existing line of work provides a *continuously evolving laboratory for discovering the interaction laws of intelligence itself, explored jointly by humans and AI*.

This survey is organized to support the manifesto's central claim: existing research studies collective intelligence either by **analyzing natural systems** or by **engineering specific architectures**; Guild proposes a third methodology.

Status: v0.1 — breadth-first pass from memory of the literature; citations should be verified and extended before external publication.

---

## 1. Collective Intelligence (the established field)

**What it studies**: How groups — human, animal, or mixed — solve problems beyond the capability of their members.

**Key work**:
- Malone & Bernstein (eds.), *Handbook of Collective Intelligence* (2015) — the field's synthesis; Malone's "collective intelligence genome" (who/what/why/how) is an early attempt at organizational primitives.
- Woolley et al., "Evidence for a Collective Intelligence Factor in the Performance of Human Groups" (*Science*, 2010) — a measurable *c* factor for groups, predicted by social perceptiveness and conversational turn-taking more than by member IQ. Directly relevant: collective capability is a property of *interaction patterns*, not member capability.
- Surowiecki, *The Wisdom of Crowds* (2004) — aggregation conditions: diversity, independence, decentralization.
- Hong & Page (2004) — diversity can trump ability in problem-solving groups, under specified conditions.
- Centola, *Change* and experimental work on network topology and behavior spread.

**What it gives Guild**: Evidence that interaction structure causally determines collective capability; candidate measures (c-factor analogues) for scoring emergent organizations.

**What it lacks**: Its methods are observational or small-N experimental, on human groups only. It cannot search the design space, and it cannot include machine participants at scale.

---

## 2. Multi-Agent AI Systems

**What it studies**: Engineered systems of interacting artificial agents.

**Key work — classical**:
- Stone & Veloso, "Multiagent Systems: A Survey from a Machine Learning Perspective" (2000); RoboCup as a long-running coordination testbed.
- Wooldridge, *An Introduction to MultiAgent Systems* — the agent/role/protocol ontology Guild deliberately suspends.
- Contract nets (Smith, 1980), blackboard architectures — early interaction primitives, worth revisiting.

**Key work — LLM era**:
- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (2023); Irving et al., "AI Safety via Debate" (2018).
- Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (2023) — emergent social behavior from memory + reflection + planning; the closest existing thing to an emergence-first LLM world, but single-designer and non-competitive.
- AutoGen (Wu et al., 2023), MetaGPT (Hong et al., 2023), CAMEL (Li et al., 2023), ChatDev — orchestration frameworks; all hardcode roles and org charts (explicitly: MetaGPT encodes SOPs of a software company).
- Emergent communication: Foerster et al. (2016), Lazaridou et al. (2017–2020) — protocols can be learned, but in narrow referential games.
- Our own `communication/multi-agent/` results: debate, CRIT, consensus strategies — hand-designed topologies whose relative performance is task-dependent, which is itself evidence that the topology→performance law is unknown.

**What it gives Guild**: The component technology (capable agents, orchestration plumbing) and a library of baseline organizational forms to compare against emergent ones.

**What it lacks**: Every system fixes its ontology in advance. Evaluation is static-benchmark. The search over organizational forms is performed by graduate students, one paper at a time.

---

## 3. Complex Adaptive Systems & Emergence

**What it studies**: How simple local rules produce global structure.

**Key work**:
- Holland, *Hidden Order* (1995) and *Emergence* (1998) — aggregation, tagging, nonlinearity, flows as CAS primitives; the closest existing candidate-primitive vocabulary to ours.
- Reynolds, "Boids" (1987) — three local rules → flocking; the canonical minimal-emergence result and the stylistic target for the Guild kernel.
- Schelling, segregation model (1971) — macro-structure nobody chose.
- Axelrod, *The Evolution of Cooperation* (1984) — tournaments as strategy-discovery engines; note the method: *open submission of strategies into a shared arena*, a direct ancestor of Guild's methodology.
- Dorigo — ant colony optimization; stigmergy (Grassé) as coordination through the environment rather than through messages.
- Ostrom, *Governing the Commons* (1990) — institutions as evolved solutions to collective-action problems; her design principles are testable predictions for Guild.

**What it gives Guild**: The emergence-first methodology; the standard of "simple mechanisms → complex structure"; stigmergy as a non-message interaction primitive.

**What it lacks**: CAS models use fixed, simple agents. No participant is intelligent, no human is in the loop, and the environments do not evolve in response to discovered strategies.

---

## 4. Game Theory & Mechanism Design

**What it studies**: Strategic interaction under fixed rules; and (mechanism design) choosing rules to induce desired equilibria.

**Key work**:
- Von Neumann & Morgenstern; Nash equilibria; Aumann on correlated equilibrium — relevant because a *correlation device* is an organizational primitive.
- Hurwicz, Maskin, Myerson — mechanism design as "inverse game theory."
- Evolutionary game theory (Maynard Smith, *Evolution and the Theory of Games*, 1982) — strategies as populations, fitness as payoff; the formal backbone for a self-renewing meta.
- Algorithmic game theory and empirical game-theoretic analysis (Wellman) — analyzing games too large for closed-form solution by simulating strategy populations.

**What it gives Guild**: The formal language for incentives, and the crucial design insight that *the rules of the environment are themselves the experimental instrument* (mechanism design applied to research).

**What it lacks**: Assumes fixed player ontology and typically fixed rationality. Cannot generate novel organizational forms — only evaluate given ones.

---

## 5. Organizational Science & Economics of Organization

**What it studies**: Why human organizations take the forms they do.

**Key work**:
- Coase, "The Nature of the Firm" (1937); Williamson, transaction cost economics — organizations exist where interaction costs make markets inefficient. Directly transplantable question: what are transaction costs *between AI agents*, and what organizational forms do they predict?
- March & Simon, *Organizations* (1958); Simon, bounded rationality — organizations as prosthetics for cognitive limits. If AI limits differ, optimal forms should differ. Testable in Guild.
- Burns & Stalker, mechanistic vs. organic structures; contingency theory (Lawrence & Lorsch) — no universally best structure; fit to environment is what matters. This *is* Guild's thesis, stated in 1960s vocabulary.
- Conway's law (1968) — designs mirror communication structures; in Guild, directly measurable as a correlation between interaction topology and artifact structure.
- Baldwin & Clark, *Design Rules* (2000) — modularity as an evolved response to complexity.

**What it gives Guild**: A century of hypotheses about which organizational variables matter (span of control, centralization, formalization, modularity) — all of which become *measurable observables* in the platform.

**What it lacks**: Human-only substrate, observational methods, no counterfactuals, decade-long feedback loops. Guild is, in one sense, experimental organizational science at machine speed.

---

## 6. Artificial Life & Evolutionary Computation

**What it studies**: Open-ended evolution of structure in synthetic substrates.

**Key work**:
- Ray, Tierra (1991); Ofria & Wilke, Avida — digital organisms; emergent parasitism and ecosystems from replication + mutation + competition.
- Lenia (Chan, 2019), Conway's Life — pattern-level "entities" emerging from field-level rules; the strongest existing demonstration of Principle 3 (entities as temporary structures).
- Stanley & Lehman, novelty search and *Why Greatness Cannot Be Planned* (2015) — objective-free search finds what objective-driven search cannot; core argument for open-endedness over benchmarks.
- Wang et al., POET (2019) — paired open-ended trailblazer: environments and agents co-evolve; the closest algorithmic precedent for a self-renewing benchmark.
- Jaderberg et al., population-based training; OpenAI hide-and-seek (Baker et al., 2020) — emergent tool use and counter-strategies across six distinct strategy epochs — the clearest existing demonstration that competitive multi-agent environments generate autocurricula.

**What it gives Guild**: The theory of open-endedness; co-evolution of environment and population as the mechanism for a benchmark that never saturates.

**What it lacks**: Participants are simple programs or RL policies, not language-capable intelligences; no humans; and the emergent structures are behavioral, not organizational (no one has evolved an *institution*).

---

## 7. Citizen Science Games & Human Computation

**What it studies**: Harnessing player effort and ingenuity for scientific problems.

**Key work**:
- Cooper et al., "Predicting protein structures with a multiplayer online game" (*Nature*, 2010) — Foldit; players beat state-of-the-art solvers on specific folds, and player strategies were codified into algorithms (Khatib et al., 2011). This is the existence proof for "gameplay → transferable scientific knowledge."
- Lee et al., EteRNA (2014) — players designed RNA that outperformed algorithms; notably, the game *promoted player-derived design rules into the scientific literature*.
- Eyewire, Galaxy Zoo, Sea Hero Quest, Borderlands Science — scale demonstrations (millions of participants).
- von Ahn, human computation / GWAP — the general framework for aligning play with useful work.

**What it gives Guild**: The recruitment and motivation playbook; the validation pattern (player strategy → codified rule → benchmarked against algorithms) that Deliverable 5 will reuse.

**What it lacks**: All existing citizen-science games target *fixed, externally defined* problems. None makes the object of play the discovery of organizational principles, and none has AI participants as first-class players.

---

## 8. RL Environments & Game-Based Benchmarks

**What it studies**: Environments as drivers and measures of agent capability.

**Key work**:
- ALE (Bellemare et al., 2013), StarCraft II / AlphaStar (Vinyals et al., 2019), Dota 2 / OpenAI Five, Diplomacy / Cicero (Meta FAIR, 2022 — mixed-motive negotiation with humans in natural language), XLand (DeepMind, 2021 — procedurally generated task distributions).
- Melting Pot (Leibo et al., 2021) — evaluation specifically of *social* generalization in multi-agent populations; the closest existing benchmark in spirit.
- Neural MMO (Suarez et al., 2019) — persistent, many-agent, open-ended-ish world.
- Werewolf/Avalon/Hanabi (Bard et al., 2020) — coordination and theory-of-mind under partial observability (connects to `theory-of-mind/selphi`).

**What it gives Guild**: Engineering patterns for persistent multi-agent worlds; evidence that environments generate capabilities (autocurricula) rather than merely measuring them.

**What it lacks**: These environments evaluate *policies*, not *organizations*. Their rules are fixed; their metas are discovered by labs, not captured as research data; humans appear only as opponents or data sources, not as co-investigators.

---

## 9. Hybrid Human–AI Collective Intelligence & HCI

**Key work**:
- Dellermann et al., "Hybrid Intelligence" (2019); Peeters et al. (2021) — frameworks for human–AI complementarity.
- Malone, *Superminds* (2018) — explicitly forecasts human–computer collectives as the next form of CI.
- Bansal et al., human-AI team performance (2019–2021) — team performance is not monotone in model accuracy; interaction design dominates. Another instance of the manifesto's core claim.
- Kittur et al., crowdsourcing complex work (2011–2013) — decomposition and workflow as the bottleneck.

**What it gives Guild**: Design knowledge for mixed-initiative interaction; evidence that human–AI collectives are a distinct regime, not an average of the two.

**What it lacks**: Studies dyads and small teams in fixed workflows; no persistent world, no open-ended search.

---

## 10. The Gap, Stated Precisely

Assemble the requirements Guild needs and check each field against them:

| Requirement | CI field | MAS | CAS | Game theory | Org science | ALife | Citizen sci. | RL envs | Hybrid CI |
|---|---|---|---|---|---|---|---|---|---|
| Interaction-first ontology (no fixed agents/roles) | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Intelligent (language-capable) participants | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Humans and AI as joint participants | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | partial | ✓ |
| Persistent, evolving environment (autocurriculum) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | partial | ✗ |
| Organizations as measured output, not input | ✗ | ✗ | partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Strategies captured as structured research data | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Validated transfer out of the environment | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |

No row of the design space is empty — every requirement is satisfied *somewhere*. No column is complete — no existing paradigm satisfies them jointly. That conjunction is Guild's contribution: not a new field, but a new instrument for an existing one.

---

## 11. Reading Priorities for the Team

1. Baker et al., "Emergent Tool Use from Multi-Agent Autocurricula" (2020) — the mechanism Guild scales up.
2. Wang et al., POET (2019) + Stanley & Lehman (2015) — open-endedness theory.
3. Woolley et al. (2010) — how to measure a collective's intelligence.
4. Cooper et al. (2010) + Khatib et al. (2011) — the transfer-validation pattern.
5. Leibo et al., Melting Pot (2021) — the nearest benchmark to differentiate from.
6. Holland (1995) — candidate primitive vocabulary to steal from and improve on.
7. Coase (1937) + Ostrom (1990) — organizational hypotheses to encode as testable predictions.
