# Candidate Interaction Primitives

**Deliverable 3**: Define candidate interaction primitives *without assuming existing organizational metaphors*.

Status: v0.1 — a proposed basis set, offered to be falsified by the kernel and its successors.

---

## 1. Method: How to Propose Primitives Without Smuggling In an Ontology

The trap in this exercise is vocabulary. Words like *agent*, *role*, *team*, *message*, *task assignment*, and *organization* each import a theory. To avoid this we adopt three rules:

1. **Events, not entities.** A primitive is a *kind of event* that can occur between interaction endpoints. Endpoints ("loci") carry the minimum state needed for events to have consequences — nothing else. Anything persistent must be built out of event history.

2. **Observability.** A primitive must be recordable as a log entry with no interpretation. "Locus 3 transferred 2 units to locus 7" is a primitive; "locus 3 delegated the subtask" is not — delegation is a *pattern* we may later detect in logs.

3. **Consequence.** A primitive must change some state (of a locus, a bond, or the world). Events without consequences cannot participate in emergence.

A useful test for whether a concept is primitive: **can it fail to exist?** Organizations can fail to emerge in a world with these primitives; therefore organizations are not primitive. A signal cannot "fail to exist" if the event type is in the rule set; therefore it is a candidate primitive.

## 2. The Candidate Basis Set

Five event families, ordered by what they conserve or create.

### P1. SIGNAL — information moves, nothing is conserved

One locus's state becomes partially observable to another. Signals are cheap (or costed), copyable, and unverifiable by default — a signal is *not* guaranteed truthful, which is what makes trust an emergent phenomenon rather than an assumption.

Sub-parameters: bandwidth, noise, cost, broadcast vs. addressed, verifiable vs. cheap talk.

*What can emerge from it*: language, reputation, deception, advertising, protocols.

### P2. TRANSFER — a conserved quantity moves

Resources (energy, materials, compute budget, attention slots) move between loci or between a locus and the world. Conservation is the point: transfer creates scarcity, and scarcity creates economics.

*What can emerge from it*: trade, markets, wages, investment, parasitism, gift economies.

### P3. BIND / RELEASE — a persistent correlation is created or destroyed

Two or more loci enter a state where future events between them are (dis)favored or their outcomes are coupled. A bond is the minimal unit of *commitment* — it constrains the future, which is what distinguishes an organization from a crowd.

Sub-parameters: strength, symmetry, decay rate, transferability, whether binding requires consent of both endpoints.

*What can emerge from it*: partnerships, membership, hierarchy (asymmetric bonds), contracts, institutions (bonds to a rule rather than to a locus).

### P4. ACT / SENSE — coupling to the world

One or more loci jointly attempt to change the environment (ACT), or sample its state (SENSE). Joint action with complementarity — where the outcome depends on the *composition* of participants, not just their number — is the selection pressure that makes coordination worth anything.

*What can emerge from it*: division of labor, specialization, tools, territory, stigmergic coordination (acting on the world as a way of signaling).

### P5. REINFORCE / DECAY — plasticity

The update rules that make history matter: outcomes modify locus state (skills, dispositions) and bond state (strength). Decay is as important as reinforcement — without forgetting, every structure is permanent and the environment cannot keep discovering.

*What can emerge from it*: learning, habit, trust dynamics, institutional memory, cultural drift.

## 3. What the Familiar Concepts Become

Under this basis, the standard vocabulary is *reconstructed as observable patterns*, each with a concrete log-level detector:

| Folk concept | Reconstruction | Detector (on event logs) |
|---|---|---|
| Agent | A locus with sufficiently rich adaptive state | — (deliberately deflationary) |
| Team / organization | Persistently recurring interaction set | High `team_recurrence`; components in the strong-bond graph (`emergent_groups`) |
| Role / specialization | Stable pattern in which skills a locus exercises | Low entropy of a locus's coverage history (`specialization_index`) |
| Hierarchy | Asymmetry in bond strengths and event initiation | In/out imbalance in recruitment or signal initiation |
| Market | Transfers whose rates equilibrate across pairs | Price-like invariants in TRANSFER logs |
| Trust | Bond strength conditioned on signal accuracy history | Correlation of bond updates with verified signals |
| Institution | A constraint on event admissibility that outlives any locus | Rule-shaped regularities in which events *don't* occur |
| Communication protocol | Recurring signal sequences with predictive consequences | Mutual information between signal patterns and subsequent events |

This table is the research program in miniature: every row is a hypothesis that the folk concept *will* emerge under some parameter regime and *fail to* emerge under others — and that the boundary between regimes is a discoverable law.

## 4. Selection Criteria for the Basis

We will judge any proposed primitive set on:

1. **Minimality** — remove any primitive and some known organizational form becomes unreachable.
2. **Expressivity** — the reconstructions in §3 are all reachable in principle.
3. **Observability** — every event is loggable without interpretation (research requirement).
4. **Learnability** — both humans and RL/LLM policies can act over the primitives (platform requirement).
5. **Transferability** — patterns detected over the primitives map onto real multi-agent systems (validation requirement, Deliverable 5).

Open question we are explicitly *not* resolving now: whether P1–P5 is minimal. Plausible reductions exist (SIGNAL as TRANSFER of zero-conservation tokens; BIND as reified expectation of future events). The kernel treats reductions as empirical questions: if a reduced basis still yields the same emergent structures, the basis was not minimal.

## 5. What the Minimal Kernel Uses

The first kernel (see [minimal-simulation.md](minimal-simulation.md)) deliberately uses the *smallest sub-basis we believed could produce emergent coordination*:

- **ACT** (joint, with complementarity): teams of loci attempt skill-demanding tasks.
- **BIND via REINFORCE/DECAY**: shared success strengthens pairwise bonds; bonds decay; bond weights bias future recruitment.
- **REINFORCE/DECAY on skills**: exercised skills improve (learning by doing); unexercised competence decays (forgetting).

Notably absent: SIGNAL and TRANSFER. This is intentional — it tests the hypothesis that *persistent organization does not require communication or economics*, only jointness, plasticity, and forgetting. The result (division of labor and stable partnerships emerge; see the results table) confirms the sufficiency of this sub-basis for the weakest form of organization, and sets up the next experiments: add SIGNAL and ask what communication buys; add TRANSFER and ask what economics buys.

## 6. Next Experiments This Document Implies

1. **Reduction test**: implement BIND as expected-future-ACT and check whether emergent structure changes (tests minimality).
2. **SIGNAL ablation-in-reverse**: add cheap talk to the kernel; measure whether communication accelerates or reshapes partnership formation.
3. **TRANSFER + scarcity**: add a conserved resource consumed by ACT; test whether markets or hierarchies emerge first, and under what cost structure.
4. **Institution detector**: implement the §3 detectors as first-class metrics so emergent structure is measured, never asserted.
