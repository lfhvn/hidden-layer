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
| [docs/ux-vision.md](docs/ux-vision.md) | UX vision: making emergence playable (pillars, core loop, staging) |

A LaTeX version of the manifesto lives at `papers/toward-collective-intelligence.tex`.

## The Terrarium (playable prototype)

[`prototype/index.html`](prototype/index.html) — open it in any browser (no build, no server). A faithful JS port of the kernel wrapped in the game layer from the UX vision: force layout where organizations congeal spatially, hue = dominant craft, a narrator chronicle that auto-names recurring teams, diegetic interventions (Charter, Sever, Contract board, Fund training, Recruit) costing Influence, and three scenario cards (Sandbox, The Ossified Guild, Cold Start — the latter two with kernel-verified win/fail conditions).

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
