# Agent Instructions — Hidden Layer

This file exists for coding agents that read `AGENTS.md` (Codex, Cursor,
aider, etc.). The canonical, complete guide is **[CLAUDE.md](CLAUDE.md)** —
read it in full. The rules below are the subset that must never be violated,
duplicated here so no agent can miss them.

## Hard Rules

1. **Never fabricate results.** No numbers, tables, citations, or "findings"
   not produced by an actual run. Illustrative data must be labeled
   `SYNTHETIC`. This repo once contained 7 papers of fabricated results;
   they were deleted. Do not recreate that problem.
2. **No claim without a results file.** Cited numbers trace to a committed
   run directory under `results/` (see `results/README.md`).
3. **Verify before reporting done**: `python -m pytest tests/ -q` must pass;
   if you claim something works, you ran it.
4. **One source of truth** — never create duplicate tests/docs/configs.
5. **Fix references when you move or delete files**
   (`grep -rn "old-name" --include="*.md" --include="*.py" .`).
6. **Imports go through the underscore packages**
   (`from communication.multi_agent import run_strategy`) — the dash-named
   directories are not importable; the underscore `__init__.py` files are
   path-sensitive loader shims. Details in CLAUDE.md.

## Orientation

- Repo map with per-component ground-truth status: CLAUDE.md § Repo Map
- Harness API cheatsheet: CLAUDE.md § Harness Quick Reference
- Working commands (setup, tests, lint, smoke experiment): CLAUDE.md
  § Commands That Work
- Research plan and methodology: `ROADMAP.md`
