# Guild UX Vision: Making Emergence Playable

**Status**: v0.1 — the first UX iteration. The presentation layer described here (Terrarium, guild-fantasy skin, Influence, scenarios) is superseded by [game-design.md](game-design.md) ("Understory", ecological/sociological theme); the pillars, precedent analysis, staging, and research constraints in this document still stand. The v0.1 prototype is preserved at [`../prototype/terrarium-v0.1.html`](../prototype/terrarium-v0.1.html).

---

## 1. The Engagement Problem, Stated Honestly

The PRD's core mechanic — *players shape relationships, constraints, and incentives rather than controlling characters* — is the scientific thesis expressed as a game rule. It is also, historically, how ambitious games die. Indirect-control designs fail in three specific ways:

1. **Illegibility.** Emergence the player can't *see* is indistinguishable from randomness. Dwarf Fortress produced the richest emergent stories in gaming and stayed niche for twenty years because reading the simulation required expertise the game never taught.
2. **The spreadsheet trap.** When the levers are parameters (decay rates, incentive weights), the game feels like configuring software. Nobody's fantasy is "tuning hyperparameters."
3. **No attachment.** Players bond with *characters* and *stories*, not with distributions. If nothing in the world has a name, a face, and a history, there is nothing to come back for.

The UX program is therefore not "add a UI to the kernel." It is: **translate emergence into objects a human can see, love, brag about, and argue over** — without violating Principle 2 (never hardcode the organizations themselves).

## 2. The Player Fantasy

> **You are not the hero. You are the founder.**

The fantasy is building something that *thrives without you*: a guild with a culture, veterans, rivalries, and a reputation — that keeps completing contracts while you sleep and surprises you when you return. Direct control is not just absent; its absence **is the power fantasy**. Anyone can micromanage. You built a thing that doesn't need you.

This fantasy has proven mainstream carriers: manager modes in sports games (you never kick the ball), autobattlers (you never cast a spell), colony sims (you suggest, dwarves decide). Guild's version is distinct in one way: in those games the *units* are the content; in Guild the **organization itself is the character** you are raising.

## 3. What Working Precedents Teach Us

| Precedent | What proved mainstream | What we take |
|---|---|---|
| Teamfight Tactics / autobattlers | "Compose the team, then watch it fight" is a top-10 genre | The **plan → hands-off resolution → read the result** rhythm; combat is a *report* you learn to read |
| RimWorld | Sells itself as a "story generator" | Emergence needs a **narrator**: the game must caption what just happened ("Ana had a breakdown") — we caption organizational events ("The Weavers dissolved after two failed contracts") |
| Opus Magnum / Zachtronics | Players share solution GIFs compulsively | Strategy must compile to a **shareable artifact** — the org design and its replay, not a save file |
| Foldit / EteRNA | Players out-designed algorithms and got cited | The **"my strategy became science" credit loop** is a real motivator; make contribution visible and attributed |
| Screeps / Gladiabots | Program-your-units sustains a small hardcore | Full indirection appeals to a niche — so the **onboarding ramp must start more direct** than the endgame |
| Idle / incubation games | "It progressed while I was away" retains casually | The persistent world's downtime is a feature: **return to find out what your guild did overnight** |
| Fantasy sports | Managing entities you don't control, at 60M+ players | Attachment via **stats, history, and drafting** — biography is enough; you don't need control to care |

## 4. Design Pillars

**P1 — Emergence must be visible before it is understood.**
Every invisible quantity gets a direct visual encoding: specialization = *color* (a locus's hue is its dominant skill — division of labor is literally the world becoming colorful), bonds = *drawn links* that thicken with weight, organizations = *spatial clusters* (layout is bond-driven, so groups physically congeal), success = *light*. A first-time player should see structure forming in under two minutes without reading anything.

**P2 — The simulation narrates itself.**
A ticker converts log patterns into story beats: teams that recur get **auto-generated names**; first successes, win streaks, dissolutions, and lock-ins become captioned events. This is the RimWorld move applied to organizations — and it is "free," because the narrator is just the research detectors (recurrence, groups, specialization) verbalized. *The research instrumentation and the storytelling layer are the same code.* That is Principle 5 (research and gameplay reinforce each other) landing in the UX.

**P3 — Interventions are diegetic acts, not parameters.**
The player never sees "bond_decay = 0.01." They see **Charter** (encourage two loci to work together), **Sever** (break up a clique), **Post a contract** (bias which tasks arrive), **Fund training** (raise plasticity), **Recruit** (add a fresh locus). Same kernel variables underneath — costumed as verbs a guildmaster would use. Each intervention costs a budgeted resource (Influence), which converts "tuning" into "spending," the difference between configuring and playing.

**P4 — Minutes, sessions, seasons.**
Three nested loops, each with its own payoff:
- *Minute*: intervene → watch the ecology respond → read the ticker. (Prototype scope.)
- *Session*: take a scenario/contract, shape the guild to beat it, bank the org design.
- *Season*: the environment drifts (task meta shifts), org designs that dominated get stress-tested, leaderboards reset — the autocurriculum from manifesto §7 *is* the content cadence, exactly like a card-game balance patch, except the players generate it.

**P5 — Strategies are artifacts with lineage.**
An org design (the intervention history + resulting structure) can be saved, shared, forked, and cited. When someone forks your design and wins a league, you're in the credits. Lineage = status = retention, and the lineage graph is *itself research data on which organizational forms propagate*. This is the Foldit credit loop generalized.

## 5. The Core Loop (Session-Level)

```
  ┌─> OBSERVE   the guild runs continuously; color, clusters, ticker
  │             tell you what it's becoming
  │
  ├─> DIAGNOSE  read the dashboard like a coach reads stats:
  │             "we're locked into one aging trio" / "no one covers Lore"
  │
  ├─> INTERVENE spend Influence on Charters, Contracts, Recruiting,
  │             Severances — a few deliberate moves, not micromanagement
  │
  └─< RESOLVE   consequences emerge over the next hundreds of rounds;
                the ticker reports; you were right or you weren't
```

The skill being trained is **organizational literacy** — reading a collective and knowing which small push changes its trajectory. That skill is the game, and it is also precisely the data the research platform wants: every intervention is a human hypothesis about collective dynamics, timestamped against its outcome.

## 6. Onboarding: Teach the Thesis Through Frustration → Delight

The tutorial arc *is* the manifesto's argument, played rather than read:

1. **Act 1 — Meddle.** The player gets cheap per-interaction control (hand-pick every team). It works for a 6-locus guild. The guild grows to 24; hand-picking becomes obviously miserable. *(Lesson: micromanagement doesn't scale — felt, not told.)*
2. **Act 2 — Delegate to structure.** The player gets Charters and Contracts; watches teams self-form and colors differentiate. First auto-named team appears; success rate beats their hand-picked era. *(Lesson: incentives out-perform commands.)*
3. **Act 3 — Meet failure.** Scripted-free but reliable: lock-in ("structure without competence" — the cronyism regime we measured in the kernel) appears; the player must diagnose and break it. *(Lesson: organizations fail in patterned ways; you can learn the patterns.)*

Act 3 matters most: the first time a player *recognizes a failure mode from its visual signature* and fixes it, they have the "I can read this world" moment that converts curiosity into competence-based retention.

## 7. Product Staging

**Stage 0 — The Terrarium (prototype, now).** Single-player, one screen, kernel-backed. Sandbox + scenario cards with goals ("Break the Cartel", "Cold Start"). Success metric: do playtesters intervene *more than once*, unprompted, and can they narrate back what happened? This validates P1–P3 before any multiplayer spend.

**Stage 1 — Contracts (async competition).** Your persistent guild bids on a shared contract board against other players' guilds, autobattler-style (asynchronous — their guilds are live simulations, but no realtime coordination needed). Daily "puzzle ecologies" (same seed for everyone, compare fixes) give a Wordle-shaped shareable. Org designs become postable artifacts with lineage. This is the smallest thing that has a meta.

**Stage 2 — The World (the PRD's product).** Persistent shared economy; AI participants as first-class players; institutions as player-craftable objects; seasons driven by environment drift. Research telemetry (with consent) flows from everything above; the benchmark/RL-environment products are exports of this world.

Each stage is a real product with its own retention test; no stage bets on the next one existing.

## 8. The Research Flywheel, as UX

Players should *feel* the science, not just power it:

- **Lab notebook**: every guild automatically keeps one — interventions, hypotheses ("severing the cartel will raise success"), outcomes. Reviewable, publishable.
- **Hypothesis cards**: seasonal community questions ("does diversity beat specialization at difficulty 0.8?") that players answer by playing; results aggregated and published back with player credit.
- **Discovery feed**: when a player's org design transfers — beats a baseline in `communication/multi-agent/`, say — that's announced like a world-first raid clear. "Your guild design is now a benchmark entry" is a status reward no other game can offer.

## 9. What We Deliberately Do NOT Build

- **No scripted organizations** — no team builder UI, no org-chart editor. The moment players can *specify* structure, Principle 2 dies and the data is worthless. Players get pressures, never blueprints.
- **No direct unit control past Act 1** — it would be easier to ship and would hollow out the fantasy.
- **No dark-pattern retention** (energy timers, streak guilt). The persistent world creates natural return intent ("what did they do overnight?"); leaning on compulsion would poison both the game's reputation and the research population.

## 10. Open UX Questions (Next Iterations)

1. **Attachment ceiling**: are auto-named teams + histories enough, or do loci need faces/portraits? (Test: naming-recall in playtests — do players refer to teams by name unprompted?)
2. **Time-scale feel**: is watching plasticity too slow at 1×? The prototype ships a speed control; find the default that keeps "watching" from becoming "waiting."
3. **Influence economy**: what's the right intervention budget cadence — per-round trickle (idle-like) or per-contract lump (roguelike draft-like)?
4. **Multiplayer texture**: competitive contract-sniping vs. shared-world cohabitation first? (Stage 1 assumes competition; playtests may say cooperation is the hook.)
5. **AI participants in the UX**: when AI guildmasters join leagues, are they labeled? (Research says symmetric footing; product intuition says transparency wins trust. Proposal: labeled, and beatable.)

## 11. Prototype Scope (What Exists Today)

[`prototype/index.html`](../prototype/index.html) — one self-contained file, zero dependencies, faithful JS port of the Python kernel (same rules, same defaults, seeded):

- Force layout driven by bonds (P1: organizations congeal spatially); locus hue = dominant skill; edge width = bond strength; success flashes.
- Live dashboard: success rate, specialization, team recurrence, group count — the research metrics, restyled as a coach's stat line.
- Narrator ticker with auto-named recurring teams (P2).
- Diegetic interventions costing Influence: Charter, Sever, Contracts (task bias), Fund Training, Recruit (P3).
- Scenario cards: **Sandbox**, **The Ossified Guild** (start in the measured lock-in regime — practice off, crews bonded; funding apprenticeships is the scarce move; goal: 15% success within 1,500 rounds), **Cold Start** (difficulty 0.8; narrow the contract board so practice concentrates; goal: 12% within 2,500 rounds).

What it's for: putting the minute-loop in front of humans this week and answering one question — *is watching this thing organize itself, and nudging it, intrinsically fun for at least ten minutes?* Everything else in this document is contingent on that answer.

### A design finding from building the scenarios

Our first puzzle draft ("break a cartel of over-bonded veterans") **solved itself**: because loci practice on every attempt, the ecology adapts around any bond structure within a few hundred rounds — conveners simply skill up and route around the dead weight. Kernel-verified: pre-seeding a fame-heavy cartel changed late success by roughly nothing. Two consequences for game design:

1. **The kernel is self-healing, so pure "fix the structure" puzzles need a learning-limited regime** (hence The Ossified Guild: practice off, training as a scarce purchasable window — kernel-verified at 2–5% success unfunded vs. 18–23% funded, so the goal genuinely requires the player).
2. This is the UX surfacing a research result, not fighting it: *plasticity dominates topology* in this world. Where that holds, organizational interventions matter less than learning investments — itself a testable transfer hypothesis for real multi-agent systems, and it fell out of playtesting a prototype. The research↔gameplay flywheel (§8) is already turning at n=1.
