# Incubator

Early-stage and paused projects that are **not yet part of the active research loop**.
Code or design lives here so it isn't lost, but these are explicitly *not* maintained,
not tested in CI, and not expected to run end-to-end. Nothing in the active tree imports
from `incubator/`.

This directory exists to keep the top-level surface area **honest**: the research areas
(`communication/`, `theory-of-mind/`, `representations/`, `alignment/`, `memory/`) should
contain only things a researcher can actually run today.

## What's here

| Project | Was at | Status | To graduate, it needs |
|---|---|---|---|
| `state-explorer/` | `representations/state-explorer/` | **Stub** — ~8 LOC FastAPI boilerplate; extensive design docs (`LLM_STATE_VISUALIZATION_PLAN.md`, `STATE_EXPLORER_QUICKSTART.md`, `ASSESSMENT_SUMMARY.md`) but no working activation-capture or visualization pipeline. | A runnable backend (activation hooks + routes) and at least one smoke test. |
| `topologies/` | `representations/latent-space/topologies/` | **Design + data-prep only** — PRD, UX, study protocol, and Python corpus/embedding scripts, but no committed React Native app. | The mobile app committed and buildable, or a decision to retarget as a web demo. |

## Graduating a project

When a project here can actually run an experiment (or serve a usable app) and has a
smoke test:

1. `git mv incubator/<project> <area>/<project>` back into its research area.
2. Add it to the area README and to `RESEARCH.md`'s hypothesis → experiment map.
3. Wire it into the offline `make ci` smoke tests (use the harness `sim` provider).

## Note on papers

`papers/latent-topologies-multimodal.tex` describes `topologies/`. It is kept in `papers/`
but its results are not yet reproducible from committed code — treat it as a design/position
paper until the project graduates.
