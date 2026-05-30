# Papers

LaTeX sources for Hidden Layer research papers, plus a build pipeline that keeps cited
numbers in sync with the code.

## Reproducible numbers (the important part)

Papers should cite results that come **from experiments**, not hand-typed figures that
silently go stale. A *bound* paper declares a generator in `generate_results.py` that
runs its (offline, deterministic) experiment and writes LaTeX into `generated/`:

- `generated/<paper>_macros.tex` — `\newcommand` per `(arm, metric)`, `\input` in the preamble
- `generated/<paper>_table.tex` — a results table, `\input` in the body
- `figures/<paper>_*.png` — optional figures (only if matplotlib is installed)

Regenerate everything:

```bash
make -C papers results        # Python only; no LaTeX toolchain required
```

Every `\mc...` macro in `metacognition-calibration.tex`, for example, expands to a number
produced by `theory_of_mind.metacognition` running on the offline `sim` provider, so the
paper is reproducible on any machine.

## Building PDFs

```bash
make -C papers pdf                          # build all (requires latexmk + TeX)
make -C papers metacognition-calibration.pdf
```

If `latexmk` isn't installed, `results` still runs and the build prints install guidance.
A shared bibliography lives in `references.bib` (papers reference it via
`\bibliography{references}`).

## Adding a bound paper

1. Write `your-paper.tex`; `\input{generated/your-paper_macros.tex}` in the preamble.
2. Add a `generate_your_paper()` function to `generate_results.py` and register it in
   `GENERATORS`. Build `rows` (one dict per arm) and emit macros/table with
   `harness.analysis.latex_macros(...)` and `harness.analysis.to_latex(...)`.
3. Keep it deterministic (seed everything; the `sim` provider is offline) so
   `make results` is reproducible.

## Status

| Paper | Bound to runs? |
|---|---|
| `metacognition-calibration.tex` | ✅ generated from `theory_of_mind.metacognition` |
| `multi-agent-coordination-cost.tex` | ✅ generated via `harness.run_experiment` (coordination overhead) |
| `multi-agent-coordination.tex` | ✗ prose (quality claims; needs a live provider to bind) |
| `selphi-theory-of-mind.tex` | ✗ prose |
| `model-introspection.tex` | ✗ prose (references `figures/`) |
| `latent-lens-sae.tex` | ✗ prose |
| `steerability-adherence.tex` | ✗ prose |
| `ai-to-ai-communication.tex` | ✗ prose |
| `latent-topologies-multimodal.tex` | ✗ prose (project now in `incubator/`) |

The unbound papers cite hand-written numbers; binding them via `harness.run_experiment`
+ `harness.analysis` is the path to making them reproducible.
