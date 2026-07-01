# Claude Development Guide - Guild

## Project Identity

**Guild** is Hidden Layer's collective-intelligence project: a research program (and eventual game/platform) for discovering the principles by which many interacting intelligences organize. See [README.md](README.md) for the document map and [PRD.md](PRD.md) for vision and unknowns.

## Non-Negotiable Design Principles

These come from the PRD and override convenience:

1. **Interactions are primitive; entities are emergent.** Never add a `Team`, `Role`, `Organization`, or `Manager` class to the kernel. If you need one of these concepts, write a *detector* over event logs instead (see interaction-primitives.md §3).
2. **Organizations must emerge.** Any mechanism that assigns structure (fixed teams, designated leaders, routing tables) is a bug, even if it improves metrics.
3. **Everything observable.** Every event must be loggable without interpretation. New mechanisms must extend the `Event` record.
4. **Everything reproducible.** All randomness flows through the seeded `random.Random` in `Kernel`. No global RNG, no wall-clock.
5. **Claims are tests.** Scientific claims about emergence live in `tests/test_guild.py` as assertions on fixed seeds. If you change dynamics, re-verify the hypotheses hold across seeds before updating test constants — do not weaken assertions to make them pass.

## Code Layout

```
collective-intelligence/guild/     # project files (this directory)
├── guild/                         # the actual Python package
│   ├── kernel.py                  # Kernel, KernelConfig, Task, Event — the world
│   ├── metrics.py                 # emergence detectors (spec., recurrence, groups…)
│   └── experiment.py              # ablation experiment (full / no_reinforcement / no_practice)
├── docs/                          # research documents
└── MANIFESTO.md, PRD.md, README.md

collective_intelligence/guild/     # import wrapper (repo dual-directory convention)
tests/test_guild.py                # repo-level tests
papers/toward-collective-intelligence.tex
```

Import as `from collective_intelligence.guild import Kernel, KernelConfig`.

## Working on the Kernel

- Keep it standard-library only for as long as possible; cheapness of runs is a research feature.
- When adding a primitive (SIGNAL, TRANSFER — see interaction-primitives.md §6), add: a config switch defaulting to *off*, an `Event` extension, at least one metric that can detect its consequences, and an ablation condition in `experiment.py`.
- Parameter changes: re-run `python -m collective_intelligence.guild.experiment --rounds 3000 --seeds 5` and update the results table in `docs/minimal-simulation.md`. Never leave stale numbers in docs.
- The known-good regime and why it works (forgetting makes specialization a commitment) is documented in minimal-simulation.md §2 — read it before retuning.

## Connections to Other Projects

- `communication/multi-agent/` — hand-designed coordination strategies (debate, CRIT, consensus) are the baselines that kernel-discovered structures must eventually beat (Deliverable 5).
- `theory-of-mind/` — trust and deception become measurable once SIGNAL is added.
- `harness/` — use for LLM-backed loci when the kernel graduates from scalar skills; keep the event interface identical.
