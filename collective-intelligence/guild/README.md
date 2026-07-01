# Guild

**An interactive environment for discovering the principles of collective intelligence.**

Guild treats collective intelligence as a missing scientific discipline and builds the instrument it lacks: a persistent, evolving environment in which humans and AI jointly explore the design space of intelligent organizations. The game is the interface; the research platform is the product.

**Research question**: What are the interaction laws from which intelligent organization emerges?

---

## Documents

| Document | What it is |
|---|---|
| [MANIFESTO.md](MANIFESTO.md) | Position paper: *Toward a Science of Collective Intelligence* |
| [PRD.md](PRD.md) | Product requirements (v0.1) — vision, principles, unknowns |
| [docs/literature-survey.md](docs/literature-survey.md) | Survey of the 9 surrounding fields and the precise gap |
| [docs/interaction-primitives.md](docs/interaction-primitives.md) | Candidate interaction primitives (SIGNAL / TRANSFER / BIND / ACT / REINFORCE) |
| [docs/minimal-simulation.md](docs/minimal-simulation.md) | Design + results of the minimal emergence kernel |
| [docs/ux-vision.md](docs/ux-vision.md) | UX vision v0.1: making emergence playable (pillars, core loop, staging) |
| [docs/game-design.md](docs/game-design.md) | Game design v0.3: "Understory" — consumer layer, run structure, win/loss conditions |

A LaTeX version of the manifesto lives at `papers/toward-collective-intelligence.tex`.

## Understory (playable prototype)

[`prototype/index.html`](prototype/index.html) — open it in any browser (no build, no server). The kernel, skinned as a valley community observed by its steward: named villagers with faces and quirks, one need per day ("Mend the weir at Reedmarsh 🪚🧭🏺"), 28-day seasons that rotate demand, surplus earned by successes and spent on steward verbs (Share a hearth, Give space, Teaching season, Welcome a newcomer, Raise a call), companionship hearts, self-naming "circles," and field notes written as diary prose.

The full run structure is playable: **traditions** (the cultural ratchet — crafts the valley itself comes to know), **the Closing** (win: a year in which they needed your hand at most twice), and **the Exodus** (loss: hard seasons pull the unattached away until the clearing empties). Lo-fi solarpunk presentation with glitch reserved for moments the data layer shows through. Research metrics live in a collapsed ledger; `window.__game` is exposed as a scripting hook. The earlier research-skinned prototype is preserved at `prototype/terrarium-v0.1.html`.

## The Kernel

`guild/` contains the first implementation: the smallest simulation we could design that produces emergent coordination — division of labor and stable partnerships — without encoding agents, roles, teams, or organizations anywhere.

Pure standard-library Python, fully seeded, every event logged.

```bash
# From the repo root — run the ablation experiment (three conditions, five seeds)
python -m collective_intelligence.guild.experiment --rounds 3000 --seeds 5

# Or use it as a library
python -c "
from collective_intelligence.guild import Kernel, KernelConfig, specialization_index, team_recurrence
kernel = Kernel(KernelConfig(seed=0))
events = kernel.run(3000)
print('late success:', sum(e.success for e in events[-300:]) / 300)
print('specialization:', round(specialization_index(kernel.skills), 3))
print('team recurrence:', round(team_recurrence(events, window=300), 3))
"
```

Headline result (5 seeds × 3000 rounds): with both plasticity rules on, collective success doubles over a run and the same teams re-form ~8× more often than chance; ablate either rule and you lose performance, structure, or both. Full table and interpretation in [docs/minimal-simulation.md](docs/minimal-simulation.md).

## Tests

```bash
python -m pytest tests/test_guild.py
```

The emergence hypotheses (H1–H3) are encoded as tests, so the scientific claims are checked in CI.

## Status

- ✅ Deliverables 1–4 (manifesto, survey, primitives, minimal simulation)
- ⬜ Deliverable 5: transfer validation into `communication/multi-agent/`
- Roadmap: [docs/minimal-simulation.md §7](docs/minimal-simulation.md)
